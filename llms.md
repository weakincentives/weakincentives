# Weak Incentives (WINK)

WINK is an open source toolkit for developing and optimizing side-effect-free
background agents (e.g., research, review, and coding agents) that translate
end-user instructions into deterministic actions. The library is designed for
human developers, but this README is written so automated coding agents
(OpenAI Codex web/CLI, Claude Code, Cursor background agents, etc.) can be
productive consumers of the API surface without depending on internals.

The public API centers on typed prompts, declarative tool contracts,
inspectable sessions, and provider-agnostic adapters so you can keep
determinism, observability, and safety front and center while iterating on
agent behaviors.

This document is the package README published to PyPI. It focuses on the
supported, public API surface you can depend on when wiring WINK into your own
agent or orchestration system.

## Architecture Overview

An agent harness in WINK wires together five core components:

1. **PromptTemplate** (`weakincentives.prompt.PromptTemplate`): Immutable
   blueprint defining sections, tools, and structured output schema. Import
   from `weakincentives.prompt`.

1. **Prompt** (`weakincentives.Prompt`): Wraps a template with parameter
   bindings and optional overrides. Pass params to the constructor or call
   `.bind()` to attach additional params before evaluation.

1. **Session** (`weakincentives.runtime.Session`): Event-driven state
   container that records all prompt renders, tool invocations, and custom
   state. State changes flow through pure reducers. Creates its own `EventBus`
   internally (access via `session.event_bus`).

1. **ProviderAdapter** (`OpenAIAdapter`, `LiteLLMAdapter`): Bridges prompts to
   LLM providers. Call `adapter.evaluate(prompt, session=session)` to execute.

1. **Tool handlers**: Functions with signature
   `(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]` that
   implement side effects when the model requests tool calls.

**Data flow**: PromptTemplate → Prompt (with bound params) →
`adapter.evaluate()` → LLM response → Tool calls dispatched → ToolResult
returned → Session updated → Events published

## Installation

WINK targets Python 3.12+. Install the core library:

```bash
pip install weakincentives
```

Optional extras enable specific providers or tooling:

- `pip install "weakincentives[openai]"` for the OpenAI adapter.
- `pip install "weakincentives[litellm]"` for the LiteLLM adapter.
- `pip install "weakincentives[claude-agent-sdk]"` for the Claude Agent SDK
  adapter.
- `pip install "weakincentives[asteval]"` to enable the sandboxed Python eval
  tool.
- `pip install "weakincentives[podman]"` for Podman-based sandboxes.
- `pip install "weakincentives[wink]"` for the demo CLI (`wink`).

The Claude Agent SDK adapter also requires the Claude Code CLI:
`npm install -g @anthropic-ai/claude-code`

## Key Concepts

- **Prompts** (`weakincentives.prompt.Prompt`): Composable, dataclass-driven
  blueprints that render deterministic model inputs while exposing tool
  contracts.
- **Tools** (`weakincentives.prompt.Tool`): Declarative descriptions of
  capabilities the model can invoke. Tools surface type-checked handlers and
  renderable schemas.
- **Sessions** (`weakincentives.runtime.Session`): Event-driven state
  containers that record every prompt render and tool invocation as immutable
  events. State changes flow through pure functions called "reducers", keeping
  state deterministic and inspectable.
- **Adapters** (`weakincentives.adapters.ProviderAdapter`): Bridges to model
  providers that negotiate tool calls and structured outputs without locking
  you into a single vendor.
- **Structured Output** (`parse_structured_output`): JSON-schema-backed
  parsing that turns model responses into typed dataclass instances.
- **Overrides** (`PromptOverride`, `LocalPromptOverridesStore`): Hash-based
  prompt overrides that let you refine prompt text safely in version control.

## Public API

- `weakincentives`: Curated entrypoints for building prompts, tools, and
  sessions.
  - Classes and functions:
    - `Budget`: Resource envelope combining time and token limits.
    - `BudgetExceededError`: Exception raised when a budget limit is breached.
    - `BudgetTracker`: Thread-safe tracker for cumulative token usage against
      a Budget.
    - `Deadline`: Immutable value object describing a wall-clock expiration.
    - `DeadlineExceededError`: Exception raised when a deadline is exceeded.
    - `FrozenDataclass`: Decorator providing immutable dataclass utilities
      (copy, asdict, normalization).
    - `JSONValue`: Type alias for JSON-compatible primitives, objects, and
      arrays.
    - `MarkdownSection`: Render markdown content using `string.Template`.
    - `Prompt`: Coordinate prompt sections and their parameter bindings.
    - `PromptResponse`: Structured result emitted by an adapter evaluation.
    - `StructuredLogger`: Logger adapter enforcing a minimal structured event
      schema.
    - `SupportsDataclass`: Protocol satisfied by dataclass types and
      instances.
    - `Tool`: Describe a callable tool exposed by prompt sections.
    - `ToolContext`: Immutable container exposing prompt execution state to
      handlers.
    - `ToolHandler`: Callable protocol implemented by tool handlers.
    - `ToolResult`: Structured response emitted by a tool handler.
    - `ToolValidationError`: Raised when tool parameters fail validation
      checks.
    - `WinkError`: Base class for all weakincentives exceptions.
    - `configure_logging`: Configure the root logger with sensible defaults.
    - `get_logger`: Return a `StructuredLogger` scoped to a name.
    - `parse_structured_output`: Parse a model response into the structured
      output type declared by the prompt.
  - Modules: `adapters`, `cli`, `contrib`, `deadlines`, `debug`, `optimizers`,
    `prompt`, `runtime`, `serde`, `types`.
- `weakincentives.adapters`: Provider integrations, configuration, and
  throttling primitives.
  - Constants: `CLAUDE_AGENT_SDK_ADAPTER_NAME`, `LITELLM_ADAPTER_NAME`,
    `OPENAI_ADAPTER_NAME`.
  - Types:
    - `AdapterName`: Type alias for adapter names.
    - `PromptEvaluationError`: Raised when evaluation against a provider
      fails.
    - `PromptResponse`: Structured result emitted by an adapter evaluation.
    - `ProviderAdapter`: Abstract base class describing the synchronous
      adapter contract.
    - `SessionProtocol`: Protocol describing the session interface required by
      adapters.
    - `ThrottleError`: Raised when a provider throttles a request.
    - `ThrottlePolicy`: Configuration for automatic retry/backoff on
      throttling.
  - Configuration:
    - `LLMConfig`: Base configuration for common LLM parameters (temperature,
      max_tokens, top_p, etc.).
    - `OpenAIClientConfig`: Configuration for OpenAI client instantiation
      (api_key, base_url, timeout).
    - `OpenAIModelConfig`: OpenAI-specific model configuration extending
      LLMConfig.
    - `LiteLLMClientConfig`: Configuration for LiteLLM client instantiation.
    - `LiteLLMModelConfig`: LiteLLM-specific model configuration extending
      LLMConfig.
    - `ClaudeAgentSDKClientConfig`: Configuration for Claude Agent SDK
      (permission_mode, cwd, max_turns, isolation).
    - `ClaudeAgentSDKModelConfig`: Claude Agent SDK model configuration
      extending LLMConfig.
  - Factory: `new_throttle_policy`: Factory for creating throttle policies.
  - Claude Agent SDK Isolation (`weakincentives.adapters.claude_agent_sdk`):
    - `IsolationConfig`: Hermetic isolation configuration (network_policy,
      sandbox, env, api_key).
    - `NetworkPolicy`: Network access constraints (allowed_domains). Use
      `NetworkPolicy.no_network()` for API-only.
    - `SandboxConfig`: OS-level sandboxing (enabled, writable_paths,
      readable_paths, bash_auto_allow).
    - `EphemeralHome`: Temporary HOME directory for isolation (auto-created
      when IsolationConfig is set).
    - `PermissionMode`: Literal type for SDK permission levels ("default",
      "acceptEdits", "plan", "bypassPermissions").
  - Claude Agent SDK Workspace (`weakincentives.adapters.claude_agent_sdk`):
    - `ClaudeAgentWorkspaceSection`: Section that materializes host files into
      a temp directory for SDK access.
    - `HostMount`: Configuration for mounting host paths (host_path,
      mount_path, include_glob, exclude_glob, max_bytes).
    - `HostMountPreview`: Preview of mount contents before materialization.
    - `WorkspaceBudgetExceededError`: Raised when mount exceeds max_bytes.
    - `WorkspaceSecurityError`: Raised when accessing paths outside
      allowed_host_roots.
- `weakincentives.prompt`: Prompt authoring, rendering, and override helpers.
  - Authoring:
    - `PromptTemplate`: Immutable prompt blueprint with sections (import from
      here).
    - `MarkdownSection`: Render markdown content using `string.Template`.
    - `Prompt`: Coordinate prompt sections and their parameter bindings.
    - `RenderedPrompt`: Result of rendering a prompt.
    - `Section`: Base class for prompt sections.
    - `SectionNode`: Node in section tree.
    - `SectionPath`: Path to a section.
    - `SectionVisibility`: Enum controlling how a section is rendered (`FULL`,
      `SUMMARY`).
    - `Tool`: Describe a callable tool exposed by prompt sections.
    - `ToolContext`: Immutable container exposing prompt execution state to
      handlers.
    - `ToolExample`: Representative invocation for a tool documenting inputs
      and outputs.
    - `ToolHandler`: Callable protocol implemented by tool handlers.
    - `ToolRenderableResult`: Protocol for tool results that can be rendered.
    - `ToolResult`: Structured response emitted by a tool handler.
    - `SupportsDataclass`: Protocol satisfied by dataclass types and
      instances.
    - `SupportsDataclassOrNone`: Protocol for dataclass types or None.
    - `SupportsToolResult`: Protocol for tool results.
    - `PromptProtocol`: Protocol for prompts.
    - `PromptTemplateProtocol`: Protocol for prompt templates.
    - `RenderedPromptProtocol`: Protocol for rendered prompts.
    - `ProviderAdapterProtocol`: Protocol for provider adapters.
  - Composition:
    - `OpenSectionsParams`: Parameters for progressive disclosure of sections.
  - Overrides:
    - `LocalPromptOverridesStore`: Store for local prompt overrides.
    - `PromptDescriptor`: Descriptor for a prompt.
    - `PromptLike`: Protocol for objects that look like prompts.
    - `PromptOverride`: Override for a prompt.
    - `PromptOverridesError`: Raised when prompt overrides fail.
    - `PromptOverridesStore`: Protocol for prompt override stores.
    - `SectionDescriptor`: Descriptor for a section.
    - `SectionOverride`: Override for a section.
    - `ToolDescriptor`: Descriptor for a tool.
    - `ToolOverride`: Override for a tool.
    - `hash_json`: Hash a JSON value.
    - `hash_text`: Hash a text value.
  - Structured output and validation:
    - `OutputParseError`: Raised when structured output parsing fails.
    - `StructuredOutputConfig`: Configuration for structured output.
    - `parse_structured_output`: Parse a model response into the structured
      output type declared by the prompt.
    - `PromptError`: Base class for prompt errors.
    - `PromptRenderError`: Raised when prompt rendering fails.
    - `PromptValidationError`: Raised when prompt validation fails.
    - `VisibilityExpansionRequired`: Raised when model requests expansion of
      summarized sections.
- `weakincentives.runtime`: Session, event, and orchestration primitives.
  - Logging:
    - `StructuredLogger`: Logger adapter enforcing a minimal structured event
      schema.
    - `configure_logging`: Configure the root logger with sensible defaults.
    - `get_logger`: Return a `StructuredLogger` scoped to a name.
  - Events:
    - `EventBus`: Interface for publishing events.
    - `HandlerFailure`: Event emitted when a handler fails.
    - `InProcessEventBus`: Simple in-process event bus.
    - `PromptExecuted`: Event emitted when a prompt is executed.
    - `PromptRendered`: Event emitted when a prompt is rendered.
    - `PublishResult`: Result of publishing an event.
    - `TokenUsage`: Token usage data from provider responses.
    - `ToolInvoked`: Event emitted when a tool is invoked.
  - Main loop orchestration:
    - `MainLoop`: Abstract base class for standardized agent workflow
      orchestration.
    - `MainLoopConfig`: Configuration for default deadline/budget.
    - `MainLoopRequest`: Event requesting execution with optional constraints.
    - `MainLoopCompleted`: Success event published via bus.
    - `MainLoopFailed`: Failure event published via bus.
  - Session ledger:
    - `DataEvent`: Event carrying data.
    - `ReducerContext`: Context for reducers.
    - `ReducerContextProtocol`: Protocol for reducer context.
    - `ReducerEvent`: Type alias for reducer events (dataclasses).
    - `Session`: Immutable event ledger with Redux-like reducers.
    - `SessionProtocol`: Protocol for sessions.
    - `Snapshot`: Session snapshot.
    - `SnapshotProtocol`: Protocol for session snapshots.
    - `SnapshotRestoreError`: Raised when snapshot restoration fails.
    - `SnapshotSerializationError`: Raised when snapshot serialization fails.
    - `TypedReducer`: Reducer for typed state.
    - `append`: Append an event to a session.
    - `build_reducer_context`: Build a reducer context.
    - `iter_sessions_bottom_up`: Iterate over sessions bottom-up.
    - `QueryBuilder`: Fluent query builder for session slices.
    - `replace_latest`: Replace the latest value in a session.
    - `replace_latest_by`: Replace the latest value in a session by key.
    - `upsert_by`: Upsert a value in a session by key.
- `weakincentives.optimizers`: Prompt optimization algorithms and utilities.
  - Protocol and base classes:
    - `PromptOptimizer`: Protocol for prompt optimization algorithms.
    - `BasePromptOptimizer`: Abstract base class for prompt optimizers.
    - `OptimizerConfig`: Base configuration dataclass with `accepts_overrides`
      field.
  - Context and results:
    - `OptimizationContext`: Immutable context bundle with adapter, event bus,
      deadline, and overrides.
    - `OptimizationResult`: Generic result container with response, artifact,
      and metadata.
    - `WorkspaceDigestResult`: Result of workspace digest optimization.
    - `PersistenceScope`: Enum for artifact storage location (`SESSION`,
      `GLOBAL`).
  - Concrete implementations live in `weakincentives.contrib.optimizers`.
  - Events:
    - `OptimizationStarted`: Event emitted when optimizer begins work.
    - `OptimizationCompleted`: Event emitted on successful completion.
    - `OptimizationFailed`: Event emitted when optimization raises exception.
- `weakincentives.contrib`: Optional, domain-specific tools and optimizers.
  - `weakincentives.contrib.tools`: Planning (`PlanningToolsSection`, `Plan`),
    VFS (`VfsToolsSection`, `VirtualFileSystem`, `HostMount`), workspace
    digest (`WorkspaceDigestSection`), and (with extras) `AstevalSection` /
    `PodmanSandboxSection`.
  - `weakincentives.contrib.optimizers`: Concrete optimizers (currently
    `WorkspaceDigestOptimizer`).
- `weakincentives.serde`: Dataclass serialization helpers.
  - `clone`: Clone a dataclass.
  - `dump`: Dump a dataclass to JSON-compatible types.
  - `parse`: Parse JSON-compatible types into a dataclass.
  - `schema`: Generate a JSON schema for a dataclass.
- `weakincentives.types`: JSON typing helpers.
  - `ContractResult`: Result of a contract check.
  - `JSONArray`: Type alias for JSON arrays.
  - `JSONArrayT`: Type variable for JSON arrays.
  - `JSONObject`: Type alias for JSON objects.
  - `JSONObjectT`: Type variable for JSON objects.
  - `JSONValue`: Type alias for JSON-compatible primitives, objects, and
    arrays.
  - `ParseableDataclassT`: Type variable for parseable dataclasses.
- `weakincentives.dbc`: Design-by-contract utilities.
  - `dbc_active`: Return `True` when DbC checks should run.
  - `dbc_enabled`: Context manager to temporarily enable DbC.
  - `disable_dbc`: Force DbC enforcement off.
  - `enable_dbc`: Force DbC enforcement on.
  - `ensure`: Validate postconditions once the callable returns or raises.
  - `invariant`: Enforce invariants before and after public method calls.
  - `pure`: Validate that the wrapped callable behaves like a pure function.
  - `require`: Validate preconditions before invoking the wrapped callable.
  - `skip_invariant`: Mark a method so invariants are not evaluated around it.
- `weakincentives.cli`: CLI entrypoints, notably the `wink` module.

## Agent-facing operational notes

- WINK does not run unattended background agents by itself. It provides
  deterministic primitives that research/review/coding agents (or humans)
  drive explicitly via prompts, tool handlers, and adapters.
- Rendering is side-effect-free: `Prompt.render()` produces a typed
  `RenderedPrompt` containing message content, declared tools, and any
  structured-output schema, but does not contact providers until you pass it
  to an adapter.
- Tool handlers are synchronous callables; use them to gate filesystem or
  network access and to enforce policy before applying patches. Handlers
  accept the typed params plus a keyword-only `context` and return
  `ToolResult` instances
  (`ToolResult(message=..., value=..., success=True/False)`) to keep session
  logs consistent.
- `PromptResponse` carries the prompt name, rendered text, and parsed output
  (when structured output is requested) so you can safely resume after partial
  failures or retries.
- Sessions are immutable ledgers: reducers consume `PromptRendered`,
  `PromptExecuted`, and `ToolInvoked` events that include `event_id`,
  `session_id`, timestamps, and provider metadata so you can join prompt and
  tool flows deterministically.

## Quickstart Snippets

### Minimal harness setup

```python
from dataclasses import dataclass
from weakincentives import MarkdownSection, Prompt
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class TaskResponse:
    summary: str
    next_steps: list[str]

template = PromptTemplate[TaskResponse](
    ns="myapp/tasks", key="task-agent", name="task-agent",
    sections=[MarkdownSection(title="Instructions", template="...", key="instructions")],
)

session = Session()  # Creates event bus internally (access via session.event_bus)
adapter = OpenAIAdapter(model="gpt-4o-mini")
response = adapter.evaluate(Prompt(template), session=session)
result: TaskResponse = response.output
```

### Defining a tool handler

```python
from weakincentives import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class PatchArgs:
    path: str
    diff: str

@dataclass(slots=True, frozen=True)
class PatchResult:
    applied: bool

def apply_patch(params: PatchArgs, *, context: ToolContext) -> ToolResult[PatchResult]:
    # context.session, context.deadline, context.event_bus available
    return ToolResult(message="Applied", value=PatchResult(applied=True), success=True)

patch_tool = Tool[PatchArgs, PatchResult](
    name="apply_patch", description="Apply a unified diff.", handler=apply_patch
)
```

### Attaching tools to sections

```python
section = MarkdownSection(
    title="Instructions", template="Use apply_patch to edit files.",
    key="instructions", tools=(patch_tool,),
)
```

### Multi-turn with session state

```python
response = adapter.evaluate(prompt, session=session)
session[TaskResponse].append(response.output)  # Store result
later = session[TaskResponse].latest()  # Retrieve later
```

### Error handling

```python
from weakincentives.adapters import PromptEvaluationError

try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError as exc:
    print(exc.phase, exc.prompt_name)  # "request"/"response"/"tool"/"budget"
```

## Prompt Authoring (`weakincentives.prompt`)

### PromptTemplate and Prompt

```python
from weakincentives.prompt import PromptTemplate

template = PromptTemplate[OutputType](
    ns="myapp/agents", key="my-agent", name="my-agent",
    sections=[...],
)
prompt = Prompt(template).bind(MyParams(value="..."))  # Bind returns self
```

### MarkdownSection with parameters

Use `${param}` syntax for dynamic content:

```python
section = MarkdownSection[TaskParams](
    title="Task", template="Objective: ${objective}", key="task",
    default_params=TaskParams(objective=""),
)
```

### Tool.wrap helper

Creates a Tool using the function's `__name__` and docstring:

```python
def search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    """Search for content."""  # Becomes tool description
    return ToolResult(message="Done", value=SearchResult(...), success=True)

search_tool = Tool.wrap(search)  # name="search", description="Search for content."
```

### ToolContext fields

Available in tool handlers via `context`:

- `context.session` - Current Session
- `context.deadline` - Optional Deadline (check with `deadline.remaining()`)
- `context.event_bus` - EventBus for publishing
- `context.prompt` / `context.rendered_prompt` / `context.adapter`

### ToolResult fields

```python
ToolResult(message="...", value=MyResult(...), success=True, exclude_value_from_context=False)
```

### Additional components

- **`parse_structured_output`**: Parse model response into typed dataclass
- **Overrides**: `LocalPromptOverridesStore` for hash-scoped prompt
  refinements

## Adapter Layer (`weakincentives.adapters`)

### OpenAI and LiteLLM adapters

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import OpenAIModelConfig, OpenAIClientConfig

# Basic usage
adapter = OpenAIAdapter(model="gpt-4o-mini")  # Native JSON schema by default

# With typed configuration
adapter = OpenAIAdapter(
    model="gpt-4o",
    model_config=OpenAIModelConfig(temperature=0.7, max_tokens=4096),
    client_config=OpenAIClientConfig(timeout=30.0),
)

# LiteLLM for multi-provider support
adapter = LiteLLMAdapter(model="claude-3-sonnet-20240229")  # Any LiteLLM model
```

### Claude Agent SDK adapter

The Claude Agent SDK adapter provides Claude's full agentic capabilities
through the official `claude-agent-sdk` package. Unlike OpenAI/LiteLLM
adapters, this runs Claude Code as a subprocess with native tools (Read,
Write, Bash, Glob, Grep).

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

# Create workspace section that materializes host files
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/path/to/project",
            mount_path="project",
            include_glob=("*.py", "*.md"),
            exclude_glob=("*.pyc", "__pycache__/*"),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/path/to",),
)

# Configure with hermetic isolation
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",  # Auto-approve all tools
        cwd=str(workspace.temp_dir),          # Working directory
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),  # API-only access
            sandbox=SandboxConfig(
                enabled=True,
                readable_paths=(str(workspace.temp_dir),),
            ),
        ),
    ),
)

# Evaluate prompt (add workspace section to prompt template)
response = adapter.evaluate(prompt, session=session)

# Clean up temp directory when done
workspace.cleanup()
```

#### Isolation modes

```python
# Minimal isolation (development)
adapter = ClaudeAgentSDKAdapter(model="claude-sonnet-4-5-20250929")

# Hermetic with specific domains (documentation access)
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("docs.python.org", "peps.python.org"),
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)

# Full lockdown (sensitive data)
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
            include_host_env=False,  # Don't inherit environment
        ),
    ),
)
```

#### MCP tool bridging

Custom weakincentives tools with handlers are automatically bridged to the SDK
via MCP servers. The adapter creates an MCP server for tools from prompt
sections:

```python
from weakincentives.contrib.tools import PlanningToolsSection

# Planning tools are bridged as MCP tools
template = PromptTemplate[Result](
    ns="app", key="agent",
    sections=(
        MarkdownSection(title="Task", template="...", key="task"),
        PlanningToolsSection(session=session),  # planning_* tools
        workspace,  # No tools - just provides workspace info
    ),
)
```

### ProviderAdapter.evaluate() signature

```python
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,                           # Optional timeout
    budget=budget,                               # Token/time limits
    budget_tracker=budget_tracker,               # Shared tracker across evaluations
)
```

Progressive disclosure is managed via session state:

```python
from weakincentives.prompt import (
    SectionVisibility,
    SetVisibilityOverride,
    VisibilityOverrides,
)

session[VisibilityOverrides].apply(
    SetVisibilityOverride(path=("details",), visibility=SectionVisibility.FULL)
)
```

### PromptResponse fields

```python
response = adapter.evaluate(prompt, session=session)
response.output       # Parsed dataclass
response.text         # Raw text
response.prompt_name  # Prompt identifier
```

### Throttling

```python
from weakincentives.adapters import ThrottleError

try:
    response = adapter.evaluate(prompt, session=session)
except ThrottleError as exc:
    # exc.kind, exc.retry_after, exc.attempts
    raise
```

## Runtime & Events (`weakincentives.runtime`)

- **`Session`**: Immutable event ledger with Redux-like reducers. Feed events
  in with `append(session, event)` or convenience selectors like
  `replace_latest` and `upsert_by`. `Snapshot`/`SnapshotProtocol` provide
  persistence helpers.
- **Slice accessor API**: Use `session[T]` for reading and writing state
  slices. Methods include `latest()`, `all()`, `where()`, `seed()`, `clear()`.
- **Reducers**: Use `TypedReducer` with `ReducerContext` to manage typed state
  slices through event-driven mutations.
- **Events**: `PromptExecuted` and `ToolInvoked` events capture every model
  exchange. `EventBus`/`InProcessEventBus` publish events to reducers and
  custom observers. `HandlerFailure` and `PublishResult` offer backpressure
  and error reporting controls.
- **MainLoop**: Abstract orchestrator for agent workflows with automatic
  visibility expansion handling and budget tracking.
- **Logging**: `configure_logging()` wires a structured logger; `get_logger`
  retrieves a module-level logger. `StructuredLogger` is a protocol you can
  implement for custom sinks.

## Contributed Tool Sections (`weakincentives.contrib.tools`)

### VfsToolsSection - Sandboxed file operations

```python
from weakincentives.contrib.tools import VfsToolsSection, HostMount, VfsPath
vfs = VfsToolsSection(
    session=session,
    mounts=(HostMount(host_path="./repo", mount_path=VfsPath(("workspace",)),
                      include_glob=("*.py",), exclude_glob=("*.pyc",), max_bytes=600_000),),
    allowed_host_roots=(Path("."),),
)
```

Tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `rm`

### PlanningToolsSection - Multi-step planning

```python
from weakincentives.contrib.tools import PlanningToolsSection, PlanningStrategy
planning = PlanningToolsSection(session=session, strategy=PlanningStrategy.PLAN_ACT_REFLECT)
```

Tools: `planning_setup_plan`, `planning_read_plan`, `planning_add_step`,
`planning_update_step`

### WorkspaceDigestSection

```python
digest = WorkspaceDigestSection(session=session)  # Renders workspace summary
```

## Session State Management

### Query API

```python
latest = session[MyType].latest()
all_items = session[MyType].all()
filtered = session[MyType].where(lambda x: x.status == "done")
exists = session[MyType].exists()
```

### Dispatch API

```python
# Broadcast dispatch - runs ALL reducers for the event type
session.apply(AddStep(step="x"))

# Targeted dispatch - runs only reducers for the specified slice
session[Plan].apply(AddStep(step="x"))
```

### Mutation API

```python
# Initialize or replace slice values (bypasses reducers)
session[Plan].seed(initial_plan)

# Append value using default reducer
session[Plan].append(new_step)

# Register reducer for custom event types
session[Plan].register(AddStep, my_reducer)

# Remove items from a slice
session[Plan].clear()                         # Clear all
session[Plan].clear(lambda p: p.done)         # Clear matching

# Global operations
session.reset()                               # Clear all slices
session.restore(snapshot)                     # Restore from snapshot
```

### Legacy helpers (still available)

```python
from weakincentives.runtime import append, replace_latest

session = append(session, my_data)
session = replace_latest(session, MyType, updated)
```

### In tool handlers

```python
def handler(params, *, context: ToolContext) -> ToolResult:
    plan = context.session[Plan].latest()
    # Tool handlers can read session; adapters record ToolInvoked events
```

## MainLoop Orchestration

`MainLoop` standardizes agent workflow orchestration: receive request, build
prompt, evaluate, handle visibility expansion, publish result. Implementations
define only the domain-specific factories.

### Implementing a MainLoop

```python
from weakincentives.runtime import MainLoop, MainLoopConfig, Session
from weakincentives.prompt import Prompt, PromptTemplate

class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def __init__(
        self, *, adapter: ProviderAdapter[ReviewResult], bus: EventBus
    ) -> None:
        super().__init__(
            adapter=adapter,
            bus=bus,
            config=MainLoopConfig(budget=Budget(max_total_tokens=50000)),
        )
        self._template = PromptTemplate[ReviewResult](
            ns="reviews", key="code-review", sections=[...],
        )

    def create_prompt(self, request: ReviewRequest) -> Prompt[ReviewResult]:
        return Prompt(self._template).bind(ReviewParams.from_request(request))

    def create_session(self) -> Session:
        return Session(bus=self._bus, tags={"loop": "code-review"})
```

### Direct execution

```python
loop = CodeReviewLoop(adapter=adapter, bus=bus)
response, session = loop.execute(ReviewRequest(...))
```

### Bus-driven execution

```python
from weakincentives.runtime import MainLoopRequest, MainLoopCompleted, MainLoopFailed

# MainLoop auto-subscribes to MainLoopRequest in __init__.

# Handle results
bus.subscribe(MainLoopCompleted, lambda e: print(f"Done: {e.response}"))
bus.subscribe(MainLoopFailed, lambda e: print(f"Failed: {e.error}"))

# Submit request with optional per-request constraints
bus.publish(MainLoopRequest(
    request=ReviewRequest(...),
    budget=Budget(max_total_tokens=10000),  # Overrides config default
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
))
```

### Visibility expansion handling

MainLoop automatically handles `VisibilityExpansionRequired` exceptions by
accumulating visibility overrides and retrying evaluation. A shared
`BudgetTracker` enforces limits cumulatively across retries.

## Event Subscription

```python
from weakincentives.runtime import PromptRendered, PromptExecuted, ToolInvoked, TokenUsage

session = Session()

# Subscribe to events
session.event_bus.subscribe(ToolInvoked, lambda e: print(e.name))
session.event_bus.subscribe(PromptExecuted, lambda e: print(e.usage))

# Unsubscribe handler (returns True if found and removed)
handler = lambda e: print(e)
session.event_bus.subscribe(PromptRendered, handler)
session.event_bus.unsubscribe(PromptRendered, handler)
```

## Session Snapshots

```python
# Capture session state
snapshot = session.snapshot()

# Restore from snapshot
session.restore(snapshot)

# Serialize for persistence
snapshot_json = snapshot.to_json()
restored = Snapshot.from_json(snapshot_json)
```

## Deadlines

```python
from datetime import datetime, timedelta, UTC
from weakincentives import Deadline

deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
# In handlers: if deadline.remaining() <= timedelta(0): ...
```

## Budgets

Budgets combine time and token limits into a single resource envelope:

```python
from datetime import datetime, timedelta, UTC
from weakincentives import Budget, BudgetTracker, BudgetExceededError, Deadline

# Create a budget with deadline and token limits
budget = Budget(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10)),
    max_total_tokens=100_000,
    max_input_tokens=80_000,
    max_output_tokens=20_000,
)

# Track usage across evaluations
tracker = BudgetTracker(budget=budget)
tracker.record_cumulative("eval-1", usage)  # Record TokenUsage from response
tracker.check()  # Raises BudgetExceededError if any limit breached
```

## Serialization (`weakincentives.serde`)

```python
from weakincentives.serde import dump, parse, schema, clone
data = dump(my_dataclass)           # To JSON-compatible dict
obj = parse(MyDataclass, data)      # From dict
json_schema = schema(MyDataclass)   # JSON schema
copy = clone(my_dataclass)          # Deep clone
```

## Additional Patterns

### Hierarchical sections

```python
root = MarkdownSection(title="Root", template="...", key="root", children=[
    MarkdownSection(title="Child", template="...", key="child"),
])
```

### Tool examples

```python
tool = Tool[P, R](name="search", description="...", handler=h, examples=(
    ToolExample(description="Find X", input=P(...), output=R(...)),
))
```

### Design-by-contract

```python
from weakincentives.dbc import require, ensure
@require(lambda p: p.query, "Query required")
def handler(params, *, context): ...
```

### Section visibility (progressive disclosure)

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

# Section with summary for progressive disclosure
section = MarkdownSection(
    title="Details",
    template="Full detailed content...",
    key="details",
    summary="Brief summary of the section",
    visibility=SectionVisibility.SUMMARY,  # Show summary by default
)
```

### Prompt optimizers

```python
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.optimizers import OptimizationContext, PersistenceScope

context = OptimizationContext(
    adapter=adapter,
    event_bus=session.event_bus,
    overrides_store=overrides_store,
)
optimizer = WorkspaceDigestOptimizer(context, store_scope=PersistenceScope.SESSION)
result = optimizer.optimize(prompt, session=session)
# result.digest contains the workspace summary
```

## CLI

```bash
pip install "weakincentives[wink]"
wink --help
```

## Example

See `code_reviewer_example.py` in the repository for a complete production
harness demonstrating all patterns: structured types, tool handlers, built-in
sections (VFS, Planning), event subscription, and prompt overrides.

## Versioning & Stability

- Public APIs are the objects exported from `weakincentives` and the
  submodules documented above.
- Adapters are optional; include only the extras you need.
- Keep `StructuredOutputConfig`, tool schemas, and overrides in version
  control so your agents remain deterministic and auditable.

## License

Apache License 2.0. See `LICENSE` for details.
