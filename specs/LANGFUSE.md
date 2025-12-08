# Langfuse Integration Specification

## Purpose

This specification defines how to integrate Langfuse observability into WINK
applications. Langfuse provides LLM tracing, analytics, prompt management, and
evaluation capabilities. The integration subscribes to the existing event bus
and forwards telemetry to Langfuse without modifying core adapter or session
code.

## Guiding Principles

- **Non-invasive**: Integration hooks into the existing `EventBus` subscription
  model; no changes to adapters or session internals required.
- **Opt-in**: Langfuse integration is an optional add-on. Applications that do
  not configure Langfuse incur no runtime overhead.
- **Fail-safe**: Network failures or Langfuse unavailability must not interrupt
  prompt evaluation. Errors are logged and isolated.
- **Complete traces**: Each prompt evaluation maps to a Langfuse trace with
  nested generations and spans for tool calls, providing end-to-end visibility.

## Core Concepts

### Langfuse Primitives

| Langfuse Concept | WINK Mapping |
|------------------|--------------|
| **Trace** | One `MainLoop.execute()` or `adapter.evaluate()` invocation |
| **Generation** | A single provider request/response cycle |
| **Span** | Tool execution within a generation |
| **Event** | Discrete occurrences (errors, custom markers) |

### Event Mapping

The integration subscribes to these WINK events:

| WINK Event | Langfuse Action |
|------------|-----------------|
| `PromptRendered` | Start trace, record prompt input |
| `ToolInvoked` | Create span within current generation |
| `PromptExecuted` | End generation, record output and usage |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Adapter   │───▶│  EventBus   │───▶│ LangfuseSubscriber  │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                  │               │
│                                                  ▼               │
│                                        ┌─────────────────────┐  │
│                                        │  Langfuse Client    │  │
│                                        │  (Background Flush) │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │   Langfuse Cloud    │
                                        │   or Self-Hosted    │
                                        └─────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse project secret key |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse project public key |
| `LANGFUSE_HOST` | No | Custom host for self-hosted (default: cloud) |
| `LANGFUSE_ENABLED` | No | Set to `"false"` to disable (default: enabled if keys present) |
| `LANGFUSE_FLUSH_INTERVAL` | No | Background flush interval in seconds (default: 5) |
| `LANGFUSE_FLUSH_AT` | No | Batch size before flush (default: 15) |
| `LANGFUSE_DEBUG` | No | Enable debug logging (default: false) |

### Programmatic Configuration

```python
from weakincentives.integrations.langfuse import LangfuseConfig, LangfuseSubscriber

config = LangfuseConfig(
    secret_key="sk-lf-...",
    public_key="pk-lf-...",
    host="https://cloud.langfuse.com",  # or self-hosted URL
    enabled=True,
    flush_interval_seconds=5.0,
    flush_at=15,
    debug=False,
    release="v1.0.0",         # Optional: application version
    sample_rate=1.0,          # Optional: trace sampling (0.0-1.0)
)
```

### Configuration Dataclass

```python
@FrozenDataclass()
class LangfuseConfig:
    """Configuration for Langfuse integration."""

    secret_key: str | None = None
    public_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True
    flush_interval_seconds: float = 5.0
    flush_at: int = 15
    debug: bool = False
    release: str | None = None
    sample_rate: float = 1.0
    tags: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> LangfuseConfig:
        """Load configuration from environment variables."""
        ...
```

## Integration Components

### LangfuseSubscriber

The primary integration point that subscribes to the event bus:

```python
class LangfuseSubscriber:
    """Event bus subscriber that forwards telemetry to Langfuse."""

    def __init__(
        self,
        config: LangfuseConfig | None = None,
        *,
        client: Langfuse | None = None,
    ) -> None:
        """Initialize the subscriber.

        Args:
            config: Langfuse configuration. Loads from environment if None.
            client: Pre-configured Langfuse client for testing.
        """
        ...

    def attach(self, bus: EventBus) -> None:
        """Subscribe to all relevant events on the bus."""
        ...

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe from all events on the bus."""
        ...

    def flush(self) -> None:
        """Force flush pending events to Langfuse."""
        ...

    def shutdown(self) -> None:
        """Flush and close the Langfuse client."""
        ...
```

### Trace Context

The subscriber maintains trace context to correlate events:

```python
@FrozenDataclass()
class TraceContext:
    """Context linking WINK events to Langfuse traces."""

    trace_id: str
    session_id: UUID | None
    prompt_name: str
    adapter: AdapterName
    started_at: datetime
    generation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Context is keyed by `(session_id, prompt_name)` to handle concurrent
evaluations. When `session_id` is `None`, the `event_id` from `PromptRendered`
serves as the correlation key.

## Event Handlers

### PromptRendered Handler

Creates a new trace and generation:

```python
def _handle_prompt_rendered(self, event: PromptRendered) -> None:
    """Start a Langfuse trace when a prompt is rendered."""

    trace = self._client.trace(
        id=str(event.event_id),
        name=event.prompt_name or f"{event.prompt_ns}/{event.prompt_key}",
        session_id=str(event.session_id) if event.session_id else None,
        metadata={
            "prompt_ns": event.prompt_ns,
            "prompt_key": event.prompt_key,
            "adapter": event.adapter,
            "render_inputs": self._serialize_inputs(event.render_inputs),
        },
        tags=list(self._config.tags),
        release=self._config.release,
    )

    generation = trace.generation(
        name=f"{event.prompt_name or event.prompt_key}/generation",
        input=event.rendered_prompt,
        model=self._extract_model(event.adapter),
        metadata={"descriptor": self._serialize_descriptor(event.descriptor)},
    )

    self._store_context(event, trace, generation)
```

### ToolInvoked Handler

Creates a span within the current generation:

```python
def _handle_tool_invoked(self, event: ToolInvoked) -> None:
    """Record tool execution as a Langfuse span."""

    context = self._get_context(event.session_id, event.prompt_name)
    if context is None:
        return  # No active trace

    generation = self._get_generation(context)
    if generation is None:
        return

    span = generation.span(
        name=f"tool/{event.name}",
        input=self._serialize_params(event.params),
        output=event.rendered_output,
        metadata={
            "call_id": event.call_id,
            "success": getattr(event.result, "success", None),
        },
    )

    if event.usage:
        span.update(
            usage={
                "input": event.usage.input_tokens,
                "output": event.usage.output_tokens,
                "total": event.usage.total_tokens,
            }
        )

    span.end()
```

### PromptExecuted Handler

Completes the generation and trace:

```python
def _handle_prompt_executed(self, event: PromptExecuted) -> None:
    """Complete the Langfuse generation when evaluation finishes."""

    context = self._get_context(event.session_id, event.prompt_name)
    if context is None:
        return

    generation = self._get_generation(context)
    if generation is None:
        return

    output = self._serialize_output(event.result)

    usage_dict = None
    if event.usage:
        usage_dict = {
            "input": event.usage.input_tokens,
            "output": event.usage.output_tokens,
            "total": event.usage.total_tokens,
        }
        if event.usage.cached_tokens:
            usage_dict["cached"] = event.usage.cached_tokens

    generation.end(
        output=output,
        usage=usage_dict,
        level="DEFAULT",
        status_message="completed",
    )

    trace = self._get_trace(context)
    if trace:
        trace.update(
            output=output,
            metadata={**context.metadata, "completed": True},
        )

    self._clear_context(event.session_id, event.prompt_name)
```

## Error Handling

### Network Failures

```python
def _safe_operation(self, operation: Callable[[], T]) -> T | None:
    """Execute a Langfuse operation with error isolation."""

    try:
        return operation()
    except Exception as error:
        self._logger.warning(
            "Langfuse operation failed.",
            event="langfuse_operation_failed",
            context={"error": str(error)},
        )
        return None
```

### PromptEvaluationError Handling

When prompt evaluation fails, the integration should record the error:

```python
def record_error(
    self,
    *,
    session_id: UUID | None,
    prompt_name: str,
    error: Exception,
) -> None:
    """Record an error for an active trace."""

    context = self._get_context(session_id, prompt_name)
    if context is None:
        return

    generation = self._get_generation(context)
    if generation:
        generation.end(
            level="ERROR",
            status_message=str(error),
            metadata={"error_type": type(error).__name__},
        )

    trace = self._get_trace(context)
    if trace:
        trace.update(
            level="ERROR",
            status_message=str(error),
        )

    self._clear_context(session_id, prompt_name)
```

## Usage Patterns

### Basic Integration

```python
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.integrations.langfuse import LangfuseSubscriber

# Setup
bus = InProcessEventBus()
langfuse = LangfuseSubscriber()  # Loads config from environment
langfuse.attach(bus)

# Use normally
adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, bus=bus, session=session)

# Cleanup
langfuse.shutdown()
```

### With MainLoop

```python
from weakincentives.runtime import MainLoop
from weakincentives.integrations.langfuse import LangfuseSubscriber

class MyLoop(MainLoop[Request, Output]):
    def __init__(self, *, adapter, bus):
        super().__init__(adapter=adapter, bus=bus)
        self._langfuse = LangfuseSubscriber()
        self._langfuse.attach(bus)

    def shutdown(self):
        self._langfuse.shutdown()
```

### With Context Manager

```python
from weakincentives.integrations.langfuse import langfuse_tracing

with langfuse_tracing(bus) as langfuse:
    response = adapter.evaluate(prompt, bus=bus, session=session)
# Automatically flushes and shuts down
```

### Custom Metadata

```python
langfuse = LangfuseSubscriber(
    config=LangfuseConfig(
        tags=("production", "customer-facing"),
        release="v2.1.0",
    )
)

# Or add metadata per-trace via session tags
session = Session(
    bus=bus,
    tags={
        "langfuse.user_id": user_id,
        "langfuse.conversation_id": conversation_id,
    },
)
```

### Sampling

```python
# Only trace 10% of requests
config = LangfuseConfig.from_env()
config = replace(config, sample_rate=0.1)
langfuse = LangfuseSubscriber(config=config)
```

## Session Tag Conventions

The integration recognizes these session tag keys for enriching Langfuse traces:

| Tag Key | Langfuse Field | Description |
|---------|----------------|-------------|
| `langfuse.user_id` | `user_id` | End-user identifier |
| `langfuse.session_id` | `session_id` | Conversation/session ID |
| `langfuse.metadata.*` | `metadata` | Arbitrary key-value pairs |
| `langfuse.tags` | `tags` | Additional trace tags |

```python
session = Session(
    bus=bus,
    tags={
        "langfuse.user_id": "user_123",
        "langfuse.metadata.customer_tier": "enterprise",
        "langfuse.tags": ("high-priority", "beta-feature"),
    },
)
```

## Serialization

### Input Serialization

Render inputs (dataclass instances) are serialized via `serde.serialize`:

```python
def _serialize_inputs(
    self,
    inputs: tuple[SupportsDataclass, ...],
) -> list[dict[str, Any]]:
    """Serialize render inputs for Langfuse metadata."""

    return [serialize(item) for item in inputs]
```

### Output Serialization

Structured outputs use the same serialization:

```python
def _serialize_output(self, result: PromptResponse[Any]) -> dict[str, Any]:
    """Serialize prompt response for Langfuse output."""

    if result.output is not None and is_dataclass_instance(result.output):
        return serialize(result.output)
    if result.text:
        return {"text": result.text}
    return {}
```

## Thread Safety

- `LangfuseSubscriber` uses a `threading.RLock` to protect trace context state.
- The Langfuse Python SDK handles background flushing in a separate thread.
- Context lookup and mutation are atomic operations.
- Multiple threads can publish events concurrently without corruption.

```python
class LangfuseSubscriber:
    def __init__(self, ...):
        self._lock = RLock()
        self._contexts: dict[ContextKey, TraceContext] = {}

    def _store_context(self, ...):
        with self._lock:
            self._contexts[key] = context

    def _get_context(self, ...):
        with self._lock:
            return self._contexts.get(key)
```

## Testing

### Unit Tests

Mock the Langfuse client to verify event handling:

```python
def test_prompt_rendered_creates_trace():
    mock_client = Mock(spec=Langfuse)
    subscriber = LangfuseSubscriber(client=mock_client)

    event = PromptRendered(
        prompt_ns="test",
        prompt_key="example",
        prompt_name="test_prompt",
        adapter="openai",
        session_id=uuid4(),
        render_inputs=(),
        rendered_prompt="Hello, world!",
        created_at=datetime.now(UTC),
    )

    subscriber._handle_prompt_rendered(event)

    mock_client.trace.assert_called_once()
    call_kwargs = mock_client.trace.call_args.kwargs
    assert call_kwargs["name"] == "test_prompt"
```

### Integration Tests

Use Langfuse's local development mode:

```python
@pytest.mark.integration
def test_full_trace_lifecycle():
    config = LangfuseConfig(
        host="http://localhost:3000",
        secret_key="sk-lf-test",
        public_key="pk-lf-test",
    )
    subscriber = LangfuseSubscriber(config=config)

    # ... execute prompt and verify trace in Langfuse
```

### Fixtures

```python
@pytest.fixture
def langfuse_subscriber():
    """Provide a test subscriber with mocked client."""

    mock_client = Mock(spec=Langfuse)
    subscriber = LangfuseSubscriber(client=mock_client)
    yield subscriber
    subscriber.shutdown()
```

## Logging

The integration uses structured logging consistent with `specs/LOGGING.md`:

| Event | Level | Context Fields |
|-------|-------|----------------|
| `langfuse_attached` | INFO | `bus_type` |
| `langfuse_trace_started` | DEBUG | `trace_id`, `prompt_name`, `session_id` |
| `langfuse_generation_ended` | DEBUG | `trace_id`, `generation_id`, `usage` |
| `langfuse_operation_failed` | WARNING | `error`, `operation` |
| `langfuse_flush_completed` | DEBUG | `event_count` |
| `langfuse_shutdown` | INFO | `pending_events` |

## Limitations

- **Async not supported**: The integration is synchronous-only, matching WINK's
  threading model. Async adapters would require a separate implementation.
- **No prompt management**: This integration focuses on observability. Langfuse
  prompt versioning integration is out of scope.
- **Single generation per evaluation**: Multi-turn conversations within a single
  `evaluate()` call create one generation with multiple tool spans. Each retry
  in the inner loop is not separately tracked.
- **Sampling is per-trace**: Once a trace is sampled out, all nested events are
  also excluded. Per-event sampling is not supported.
- **No cost tracking**: Token-to-cost conversion requires model pricing data
  that is not currently available in WINK events.

## Future Considerations

- **Langfuse Prompt Management**: Pull prompt templates from Langfuse and use
  them in WINK prompts. This would require a `LangfusePromptStore` that
  implements a prompt override interface.
- **Evaluation Integration**: Send structured outputs to Langfuse for automated
  evaluation pipelines.
- **Dataset Generation**: Capture prompt/response pairs for fine-tuning datasets.
- **Cost Tracking**: Add model pricing configuration to compute costs from token
  usage.

## Dependencies

The integration requires the `langfuse` Python SDK:

```toml
[project.optional-dependencies]
langfuse = ["langfuse>=2.0.0"]
```

Import with feature check:

```python
try:
    from langfuse import Langfuse
except ImportError as error:
    raise ImportError(
        "Langfuse integration requires the 'langfuse' extra. "
        "Install with: pip install weakincentives[langfuse]"
    ) from error
```
