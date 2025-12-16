# LangSmith Integration Specification

## Purpose

Enable full observability of WINK background agents through LangSmith. This
specification covers telemetry/tracing integration and prompt management via
LangSmith Hub.

## Guiding Principles

- **Non-invasive instrumentation**: Telemetry hooks into existing event bus
  infrastructure without requiring changes to business logic.
- **Decoupled from critical path**: Network calls to LangSmith run
  asynchronously to avoid blocking prompt evaluation.
- **Bidirectional prompt management**: Override system supports both push
  (publish to Hub) and pull (fetch from Hub) workflows.
- **Graceful degradation**: LangSmith unavailability does not break agent
  execution.
- **Composable with LangSmith SDK**: Works alongside `@traceable` decorator
  and native LangSmith integrations (e.g., `configure_claude_agent_sdk()`).

## Auto-Instrumentation

WINK provides a single-call configuration function following the pattern
established by LangSmith's native integrations:

```python
from weakincentives.contrib.langsmith import configure_wink

# Enable automatic tracing at application start
configure_wink()

# All WINK evaluations are now traced to LangSmith
response = adapter.evaluate(prompt, session=session)
```

### How It Works

`configure_wink()` patches the `InProcessEventBus` class to automatically
attach telemetry handlers to every new bus instance:

```mermaid
flowchart LR
    Configure["configure_wink()"] --> Patch["Patch InProcessEventBus.__init__"]
    Patch --> NewBus["New EventBus Created"]
    NewBus --> Attach["Auto-attach LangSmithTelemetryHandler"]
    Attach --> Events["Events flow to LangSmith"]
```

### Configuration Options

```python
configure_wink(
    # API settings (fall back to LANGCHAIN_* env vars)
    api_key="...",
    project="my-project",

    # Tracing behavior
    tracing_enabled=True,
    trace_sample_rate=1.0,

    # Hub integration
    hub_enabled=True,

    # Advanced
    async_upload=True,
    flush_on_exit=True,
)
```

### Composing with LangSmith's Claude Agent SDK Integration

When using the `ClaudeAgentSDKAdapter`, you can combine WINK's event-based
tracing with LangSmith's native Claude Agent SDK instrumentation:

```python
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from weakincentives.contrib.langsmith import configure_wink

# Enable both integrations
configure_claude_agent_sdk()  # Traces Claude SDK internals
configure_wink()               # Traces WINK prompt lifecycle

# Claude Agent SDK traces nest under WINK traces
adapter = ClaudeAgentSDKAdapter(model="claude-sonnet-4-5-20250929")
response = adapter.evaluate(prompt, session=session)
```

**Trace Hierarchy:**

```
WINK: PromptRendered (run_type="chain")
  └─ LangSmith: Claude SDK Query (run_type="llm")
       └─ LangSmith: Tool Use (run_type="tool")
  └─ WINK: ToolInvoked (run_type="tool")
  └─ WINK: PromptExecuted (updates parent run)
```

### Integrating with `@traceable`

WINK traces compose naturally with LangSmith's `@traceable` decorator for
custom application code:

```python
from langsmith import traceable
from weakincentives.contrib.langsmith import configure_wink

configure_wink()

@traceable(name="process_user_request")
def process_request(user_input: str) -> str:
    # Custom pre-processing (traced)
    processed = preprocess(user_input)

    # WINK evaluation (automatically traced as child)
    response = adapter.evaluate(
        prompt.bind(input=processed),
        session=session,
    )

    # Custom post-processing (traced)
    return postprocess(response.output)
```

**Resulting Trace:**

```
process_user_request (run_type="chain")
  └─ preprocess (if @traceable)
  └─ WINK: my_prompt (run_type="chain")
       └─ Provider Call (run_type="llm")
       └─ Tool: search (run_type="tool")
  └─ postprocess (if @traceable)
```

```mermaid
flowchart TB
    subgraph WINK["WINK Runtime"]
        EventBus["Event Bus"]
        Session["Session"]
        Adapter["Provider Adapter"]
        OverrideStore["Override Store"]
    end

    subgraph LangSmith["LangSmith Platform"]
        Tracing["Tracing"]
        Hub["Prompt Hub"]
    end

    EventBus -->|PromptRendered<br/>ToolInvoked<br/>PromptExecuted| Tracing
    OverrideStore <-->|resolve/upsert| Hub
```

## Integration Surface

### Event Bus Telemetry (Primary Hook)

The event bus provides the primary integration point for tracing. Subscribers
receive lifecycle events without modifying adapter or prompt code.

**Available Events:**

| Event | When Fired | Key Fields |
|-------|------------|------------|
| `PromptRendered` | After render, before provider call | `prompt_ns`, `prompt_key`, `prompt_name`, `adapter`, `rendered_prompt`, `descriptor` |
| `ToolInvoked` | After each tool handler | `name`, `params`, `result`, `usage`, `call_id` |
| `PromptExecuted` | After final parse | `result`, `usage`, `prompt_name` |

**Mapping to LangSmith Runs:**

```
PromptRendered  → Run(run_type="chain", name=prompt_name)
  └─ Provider Call → Run(run_type="llm", parent_run_id=...)
       └─ ToolInvoked → Run(run_type="tool", parent_run_id=...)
  └─ PromptExecuted → Update parent run with outputs
```

### Override Store Protocol (Prompt Hub)

Custom `PromptOverridesStore` implementations can fetch and persist overrides
via LangSmith Hub, enabling centralized prompt management.

**Store Protocol:**

```python
class PromptOverridesStore(Protocol):
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...
```

**Hub Mapping:**

| WINK Concept | LangSmith Hub Concept |
|--------------|----------------------|
| `ns/prompt_key` | Prompt name |
| `tag` | Commit hash or alias |
| `SectionOverride.body` | Prompt template content |
| `ToolOverride.description` | Tool description in template |

## Architecture

### Telemetry Layer

```mermaid
sequenceDiagram
    participant A as Adapter
    participant B as EventBus
    participant T as TelemetryHandler
    participant Q as AsyncQueue
    participant L as LangSmith Client

    A->>B: publish(PromptRendered)
    B->>T: handle(event)
    T->>Q: enqueue(run_create)
    Q-->>L: batch upload (async)

    A->>B: publish(ToolInvoked)
    B->>T: handle(event)
    T->>Q: enqueue(run_create)

    A->>B: publish(PromptExecuted)
    B->>T: handle(event)
    T->>Q: enqueue(run_update)
    Q-->>L: batch upload (async)
```

**Key Design Decisions:**

1. **Async upload queue**: Events are queued and uploaded in batches to avoid
   blocking the evaluation loop.
1. **Trace context propagation**: A trace ID generated at `PromptRendered`
   links all subsequent runs in a single trace.
1. **Graceful failure**: Queue overflow or upload failures are logged but do
   not raise to callers.

### Prompt Hub Layer

```mermaid
flowchart LR
    subgraph Render["Prompt Render"]
        Descriptor["Compute Descriptor"]
        Resolve["resolve(descriptor, tag)"]
        Apply["Apply Overrides"]
    end

    subgraph Hub["LangSmith Hub"]
        Cache["Local Cache"]
        API["Hub API"]
    end

    Descriptor --> Resolve
    Resolve --> Cache
    Cache -->|miss| API
    API --> Cache
    Cache --> Apply
```

**Caching Strategy:**

- **TTL-based invalidation**: Cached overrides expire after configurable
  duration (default: 5 minutes).
- **Tag-aware**: `latest` tag bypasses cache for freshness; versioned tags
  cache indefinitely.
- **Fail-open on network errors**: Use cached value or skip overrides rather
  than failing evaluation.

## Claude Agent SDK Adapter Integration

The `ClaudeAgentSDKAdapter` presents unique tracing considerations due to its
hook-based architecture and delegation to Claude Code's native tools.

### Dual Tracing Strategy

When using the Claude Agent SDK adapter, traces can be captured at two levels:

```mermaid
flowchart TB
    subgraph WINK["WINK Event Bus"]
        PR["PromptRendered"]
        TI["ToolInvoked (from hooks)"]
        PE["PromptExecuted"]
    end

    subgraph LS["LangSmith Claude SDK Integration"]
        Query["sdk.query() → LLM Run"]
        Tool["Tool Use → Tool Run"]
    end

    subgraph Combined["Combined Trace"]
        Root["WINK Chain Run"]
        LLM["Claude SDK LLM Runs"]
        Tools["Tool Runs (both sources)"]
    end

    PR --> Root
    Query --> LLM
    TI --> Tools
    Tool --> Tools
    PE --> Root
```

### Hook-to-Run Mapping

The Claude Agent SDK adapter's hooks map to LangSmith runs:

| SDK Hook | WINK Event | LangSmith Run |
|----------|------------|---------------|
| `PreToolUse` | (none - internal) | Child span start |
| `PostToolUse` | `ToolInvoked` | Tool run complete |
| `Stop` | `PromptExecuted` | Chain run complete |

### Deduplication

When both WINK and LangSmith's native Claude SDK integration are enabled, tool
invocations may be reported twice. The telemetry handler deduplicates based on
`call_id`:

```python
class LangSmithTelemetryHandler:
    def _on_tool_invoked(self, event: ToolInvoked) -> None:
        # Skip if LangSmith native integration already traced this call
        if self._is_traced_by_native_integration(event.call_id):
            return
        self._create_tool_run(event)
```

### MCP Tool Bridge Tracing

Custom WINK tools bridged via MCP are traced through WINK's event bus:

```python
# Custom tool with handler
@dataclass(frozen=True)
class SearchParams:
    query: str

def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[str]:
    # Handler execution
    results = do_search(params.query)
    return ToolResult(message="Found results", value=results, success=True)

# Tool definition
search_tool = Tool(
    name="search",
    description="Search the knowledge base",
    params=SearchParams,
    handler=search_handler,
)

# When invoked via MCP bridge, ToolInvoked event is published
# and appears in LangSmith as a tool run
```

### Native Tool Tracing

Claude Code's native tools (Read, Write, Bash, etc.) are traced via:

1. **LangSmith's `configure_claude_agent_sdk()`**: Captures at SDK level
1. **WINK's `PostToolUse` hook**: Publishes `ToolInvoked` events

To avoid duplication, configure one or the other, or use the deduplication
logic above.

### Recommended Configuration

For most use cases, use both integrations with WINK tool tracing disabled:

```python
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from weakincentives.contrib.langsmith import configure_wink

configure_claude_agent_sdk()  # Detailed Claude SDK internals
configure_wink(
    project="my-agent",
    trace_native_tools=False,  # Let Claude SDK handle tool tracing
)
```

### Trace Context Correlation

**Critical**: For traces to appear unified in LangSmith, both integrations
must share the same trace context. WINK achieves this by setting up the
LangSmith run context before invoking the Claude Agent SDK:

```mermaid
sequenceDiagram
    participant W as WINK Telemetry
    participant C as ContextVar
    participant S as Claude SDK
    participant L as LangSmith

    W->>L: Create chain run (PromptRendered)
    L-->>W: run_id, trace_id
    W->>C: Set run context (parent_run_id)
    W->>S: adapter.evaluate() → sdk.query()
    S->>C: Read parent context
    S->>L: Create LLM run (parent=chain run)
    S->>L: Create tool runs (parent=LLM run)
    W->>L: Update chain run (PromptExecuted)
```

**Implementation:**

```python
class LangSmithTelemetryHandler:
    def _on_prompt_rendered(self, event: PromptRendered) -> None:
        # Create parent run
        run = self._client.create_run(
            name=event.prompt_name,
            run_type="chain",
            inputs={"rendered_prompt": event.rendered_prompt},
        )

        # Set context so Claude SDK integration inherits this parent
        langsmith_context.set_parent_run(run.id, run.trace_id)

        # Store for later update
        self._active_runs[event.session_id] = run
```

The `langsmith` SDK uses `contextvars` for trace propagation. As long as WINK
sets the parent context before `sdk.query()` is called, all Claude SDK traces
automatically nest under the WINK chain run.

**Resulting Unified Trace:**

```
my_prompt (chain) ← WINK
  ├─ claude-sdk-query (llm) ← Claude SDK
  │   ├─ Read (tool) ← Claude SDK
  │   ├─ Write (tool) ← Claude SDK
  │   └─ Bash (tool) ← Claude SDK
  └─ [completion metadata] ← WINK (PromptExecuted)
```

**Session ID Propagation:**

WINK's `session_id` is included as metadata on all runs:

```python
run = self._client.create_run(
    name=event.prompt_name,
    run_type="chain",
    metadata={
        "wink_session_id": str(event.session_id),
        "prompt_ns": event.prompt_ns,
        "prompt_key": event.prompt_key,
    },
)
```

This enables querying all runs for a session in LangSmith:
`metadata.wink_session_id = "uuid-here"`

### Isolation and Tracing

When using `IsolationConfig`, traces still flow normally since the telemetry
handler runs in the parent process, not the isolated SDK subprocess:

```python
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            # Tracing still works - events published via hooks
        ),
    ),
)
```

## Configuration

### LangSmithConfig

```python
@dataclass(slots=True, frozen=True)
class LangSmithConfig:
    """Configuration for LangSmith integration."""

    # API settings
    api_key: str | None = None  # Falls back to LANGCHAIN_API_KEY env
    api_url: str = "https://api.smith.langchain.com"
    project: str | None = None  # Falls back to LANGCHAIN_PROJECT env

    # Telemetry settings
    tracing_enabled: bool = True
    trace_sample_rate: float = 1.0  # 0.0-1.0, for high-volume scenarios
    async_upload: bool = True
    upload_batch_size: int = 100
    upload_interval_seconds: float = 1.0
    max_queue_size: int = 10000

    # Hub settings
    hub_enabled: bool = True
    cache_ttl_seconds: float = 300.0  # 5 minutes
    cache_versioned_indefinitely: bool = True
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_API_KEY` | LangSmith API key | (required) |
| `LANGCHAIN_PROJECT` | Default project name | `"default"` |
| `LANGCHAIN_TRACING_V2` | Enable tracing | `"true"` |
| `LANGCHAIN_ENDPOINT` | API endpoint | `"https://api.smith.langchain.com"` |

## Implementation Components

### LangSmithTelemetryHandler

Event handler that captures WINK events and creates LangSmith runs.

```python
class LangSmithTelemetryHandler:
    """Subscribes to WINK events and publishes to LangSmith."""

    def __init__(
        self,
        config: LangSmithConfig,
        *,
        client: Client | None = None,  # For testing
    ) -> None: ...

    def attach(self, bus: EventBus) -> None:
        """Subscribe to all telemetry events."""

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe from all telemetry events."""

    def flush(self, *, timeout: float | None = None) -> None:
        """Block until pending uploads complete."""

    # Internal handlers
    def _on_prompt_rendered(self, event: PromptRendered) -> None: ...
    def _on_tool_invoked(self, event: ToolInvoked) -> None: ...
    def _on_prompt_executed(self, event: PromptExecuted) -> None: ...
```

**Trace Context Management:**

```python
@dataclass(slots=True)
class TraceContext:
    """Tracks parent-child relationships for a single evaluation."""
    trace_id: UUID
    root_run_id: UUID
    current_run_id: UUID
    session_id: UUID | None

# Thread-local storage for active contexts
_active_contexts: ContextVar[dict[UUID, TraceContext]] = ContextVar("langsmith_contexts")
```

### RunTree Integration

For advanced scenarios requiring manual control, the telemetry handler exposes
the underlying `RunTree` for direct manipulation:

```python
from langsmith.run_trees import RunTree
from weakincentives.contrib.langsmith import get_current_run_tree

# Within a traced context
run_tree = get_current_run_tree()
if run_tree:
    # Add custom child run
    with run_tree.create_child(
        name="custom_step",
        run_type="chain",
        inputs={"key": "value"},
    ) as child:
        result = do_custom_step()
        child.end(outputs={"result": result})

    # Add metadata to current run
    run_tree.add_metadata({"custom_key": "custom_value"})

    # Add tags
    run_tree.add_tags(["tag1", "tag2"])
```

**Use Cases:**

- Adding custom spans for non-WINK operations within tool handlers
- Attaching domain-specific metadata for filtering in LangSmith UI
- Creating structured traces for complex multi-step tool implementations

### LangSmithPromptOverridesStore

Override store backed by LangSmith Hub.

```python
class LangSmithPromptOverridesStore(PromptOverridesStore):
    """Fetch and persist prompt overrides via LangSmith Hub."""

    def __init__(
        self,
        config: LangSmithConfig,
        *,
        client: Client | None = None,
        fallback_store: PromptOverridesStore | None = None,
    ) -> None: ...

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Fetch override from Hub, with caching."""

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Publish override to Hub."""

    def pull(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Pull prompt from Hub without descriptor (for initial sync)."""

    def push(
        self,
        prompt: Prompt[object],
        *,
        tag: str = "latest",
        commit_message: str | None = None,
    ) -> str:
        """Push current prompt to Hub, returning commit hash."""
```

**Hub ↔ Override Mapping:**

```python
def _hub_prompt_to_override(
    hub_prompt: HubPrompt,
    descriptor: PromptDescriptor,
) -> PromptOverride:
    """Convert LangSmith Hub prompt to WINK override format."""

def _override_to_hub_prompt(
    override: PromptOverride,
    descriptor: PromptDescriptor,
) -> HubPrompt:
    """Convert WINK override to LangSmith Hub prompt format."""
```

## Usage Examples

### Basic Telemetry

```python
from weakincentives.contrib.langsmith import (
    LangSmithConfig,
    LangSmithTelemetryHandler,
)
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.adapters.openai import OpenAIAdapter

# Configure
config = LangSmithConfig(
    project="my-agent",
    tracing_enabled=True,
)

# Setup
bus = InProcessEventBus()
session = Session(bus=bus)
adapter = OpenAIAdapter(model="gpt-4o")

# Attach telemetry
telemetry = LangSmithTelemetryHandler(config)
telemetry.attach(bus)

try:
    # Evaluate - traces automatically sent to LangSmith
    response = adapter.evaluate(prompt, session=session)
finally:
    # Ensure all traces are uploaded
    telemetry.flush()
    telemetry.detach(bus)
```

### Prompt Hub Integration

```python
from weakincentives.contrib.langsmith import (
    LangSmithConfig,
    LangSmithPromptOverridesStore,
)
from weakincentives.prompt import Prompt

config = LangSmithConfig(hub_enabled=True)
store = LangSmithPromptOverridesStore(config)

# Create prompt with Hub-backed overrides
prompt = Prompt(
    template,
    overrides_store=store,
    overrides_tag="production",  # or "latest", commit hash, etc.
)

# Changes in LangSmith Hub automatically apply on next evaluation
response = adapter.evaluate(prompt, session=session)

# Push local changes to Hub
commit_hash = store.push(prompt, tag="staging", commit_message="Improved instructions")
```

### Full Agent Observability

```python
from weakincentives.contrib.langsmith import (
    LangSmithConfig,
    LangSmithTelemetryHandler,
    LangSmithPromptOverridesStore,
)
from weakincentives import MainLoop

config = LangSmithConfig(
    project="production-agent",
    tracing_enabled=True,
    hub_enabled=True,
)

# Shared telemetry handler
telemetry = LangSmithTelemetryHandler(config)

# Hub-backed override store
store = LangSmithPromptOverridesStore(config)

class ObservableAgentLoop(MainLoop[UserRequest, AgentOutput]):
    def __init__(self, adapter: ProviderAdapter[AgentOutput]) -> None:
        bus = InProcessEventBus()
        super().__init__(adapter=adapter, bus=bus)

        # Attach telemetry
        telemetry.attach(bus)

        # Configure prompts with Hub overrides
        self._prompt = Prompt(
            agent_template,
            overrides_store=store,
            overrides_tag="production",
        )

    def shutdown(self) -> None:
        telemetry.flush(timeout=5.0)
        telemetry.detach(self._bus)
```

## Trace Correlation

### Session-Based Correlation

Use `session_id` to correlate traces across multiple evaluations:

```python
session = Session(bus=bus, session_id=uuid4())  # Explicit ID

# All evaluations in this session share the session_id in LangSmith
response1 = adapter.evaluate(prompt1, session=session)
response2 = adapter.evaluate(prompt2, session=session)

# Query in LangSmith: session_id="..."
```

### Custom Metadata

Add custom tags and metadata via session tags:

```python
session = Session(
    bus=bus,
    tags={
        "user_id": user.id,
        "request_source": "api",
        "environment": "production",
    },
)

# Tags propagate to all LangSmith runs in this session
```

### Nested Traces

Tool handlers that invoke sub-evaluations automatically nest:

```python
def research_tool(params: ResearchParams, *, context: ToolContext) -> ToolResult[str]:
    # This evaluation appears as a child run in LangSmith
    sub_response = context.adapter.evaluate(
        research_prompt,
        session=context.session,  # Same session maintains trace context
    )
    return ToolResult(message="done", value=sub_response.output, success=True)
```

## Error Handling

### Telemetry Failures

```python
# Telemetry failures are logged but don't raise
try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError:
    # This is a prompt error, not a telemetry error
    raise

# Telemetry errors appear in logs:
# WARNING - langsmith_upload_failed: Failed to upload 3 runs: ConnectionError
```

### Hub Failures

```python
class LangSmithPromptOverridesStore:
    def resolve(self, descriptor, tag):
        try:
            return self._fetch_from_hub(descriptor, tag)
        except LangSmithAPIError:
            # Log warning, try cache
            cached = self._cache.get(descriptor, tag)
            if cached:
                return cached
            # Fall back to local store if configured
            if self._fallback_store:
                return self._fallback_store.resolve(descriptor, tag)
            # Skip overrides rather than fail
            return None
```

## Events

### LangSmith-Specific Events

```python
@FrozenDataclass()
class LangSmithTraceStarted:
    trace_id: UUID
    session_id: UUID | None
    project: str
    created_at: datetime

@FrozenDataclass()
class LangSmithTraceCompleted:
    trace_id: UUID
    run_count: int
    total_tokens: int
    trace_url: str | None
    created_at: datetime

@FrozenDataclass()
class LangSmithUploadFailed:
    trace_id: UUID | None
    error: str
    retry_count: int
    created_at: datetime
```

These events are published to the session event bus for custom handling.

## Performance Considerations

### Async Upload Queue

- Events are queued immediately (non-blocking)
- Background thread uploads in batches
- Configurable batch size and interval
- Graceful overflow handling (oldest events dropped with warning)

### Caching

- Hub prompts cached with configurable TTL
- Versioned tags cached indefinitely (immutable)
- Cache invalidation on `upsert()` calls

### Sampling

For high-volume scenarios, enable trace sampling:

```python
config = LangSmithConfig(
    trace_sample_rate=0.1,  # 10% of traces
)
```

Sampling decision made at `PromptRendered` and propagated to all child runs.

## Testing

### Mock Client

```python
from weakincentives.contrib.langsmith.testing import MockLangSmithClient

mock_client = MockLangSmithClient()
telemetry = LangSmithTelemetryHandler(config, client=mock_client)

# After evaluation
assert mock_client.runs_created == 3  # 1 chain + 1 llm + 1 tool
assert mock_client.last_run.name == "my_prompt"
```

### Fixtures

```python
# tests/helpers/langsmith.py
@pytest.fixture
def langsmith_config():
    return LangSmithConfig(
        api_key="test-key",
        project="test-project",
        async_upload=False,  # Sync for deterministic tests
    )

@pytest.fixture
def mock_hub():
    return MockLangSmithHub()
```

## Limitations

- **Synchronous WINK runtime**: Telemetry upload runs on background threads to
  avoid blocking, but the WINK event loop itself is synchronous.
- **No mid-evaluation updates**: Traces are created/updated at event
  boundaries, not during streaming.
- **Hub schema constraints**: Complex WINK prompt structures may require
  flattening for Hub storage.
- **Claude Agent SDK deduplication**: When using both
  `configure_claude_agent_sdk()` and `configure_wink()`, careful configuration
  is needed to avoid duplicate traces.
- **Isolated subprocess tracing**: When using `IsolationConfig`, LangSmith's
  native Claude SDK integration cannot trace the isolated subprocess directly;
  use WINK's hook-based tracing instead.

## Future Considerations

- **Streaming telemetry**: Support for token-level streaming events when WINK
  adds streaming support.
- **Evaluation integration**: Dataset-driven testing and automated prompt
  optimization via LangSmith experiments.
- **Multi-project support**: Route different prompt namespaces to different
  LangSmith projects.
- **Cost tracking**: Aggregate token costs per prompt/tool in LangSmith
  dashboards.
