# RunContext Specification

Immutable execution metadata for distributed tracing and request correlation.

**Source:** `src/weakincentives/runtime/run_context.py`

## RunContext

```python
@FrozenDataclass()
class RunContext:
    run_id: UUID           # Fresh per execution
    request_id: UUID       # Stable across retries
    session_id: UUID | None
    attempt: int = 1       # From Message.delivery_count
    worker_id: str = ""
    trace_id: str | None   # OpenTelemetry trace ID
    span_id: str | None

    def to_log_context(self) -> dict[str, str | int | None]: ...
```

## Field Semantics

| Field | Lifecycle | Purpose |
|-------|-----------|---------|
| `run_id` | Fresh per execution | Unique execution identifier |
| `request_id` | Stable across retries | Logical request correlation |
| `session_id` | Set by MainLoop | Session correlation |
| `attempt` | From Message.delivery_count | Retry tracking |
| `worker_id` | From MainLoop | Worker identification |
| `trace_id` | Passed through | Distributed trace correlation |
| `span_id` | Passed through | Span within trace |

### run_id vs request_id

```python
# First attempt
run_context.run_id = UUID("aaa...")  # fresh
run_context.request_id = UUID("bbb...")
run_context.attempt = 1

# Retry after visibility timeout
run_context.run_id = UUID("ccc...")  # fresh (different)
run_context.request_id = UUID("bbb...")  # same
run_context.attempt = 2
```

## Data Flow

```
MainLoop._handle_message()
  └─ _build_run_context()
  └─ adapter.evaluate(run_context=...)
       └─ PromptRendered(run_context=...)
       └─ ToolExecutor(run_context=...)
            └─ ToolContext(run_context=...)
            └─ ToolInvoked(run_context=...)
       └─ PromptExecuted(run_context=...)
  └─ MainLoopResult(run_context=...)
```

## Integration Points

### ToolContext

```python
def my_handler(params, *, context: ToolContext) -> ToolResult:
    if context.run_context:
        log = logger.bind(**context.run_context.to_log_context())
```

### Telemetry Events

All events include optional `run_context`: `PromptRendered`, `PromptExecuted`, `ToolInvoked`.

### OpenTelemetry Integration

```python
with tracer.start_as_current_span("process_request") as span:
    ctx = trace.get_current_span().get_span_context()
    run_context = RunContext(
        trace_id=format(ctx.trace_id, "032x"),
        span_id=format(ctx.span_id, "016x"),
    )
```

## Invariants

1. **Immutability**: All fields read-only after creation
2. **Fresh run_id per execution**: Never reused
3. **Stable run_id during execution**: Same from start through result
4. **Preserved trace context**: `trace_id`/`span_id` pass through unchanged
5. **Optional everywhere**: All integration points accept `run_context | None`
