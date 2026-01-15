# RunContext Specification

## Purpose

`RunContext` provides immutable execution metadata for distributed tracing,
request correlation, and debugging. Flows from `MainLoop` through adapters to
tool handlers and telemetry events.

**Implementation:** `src/weakincentives/runtime/run_context.py`

## Core Type

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `UUID` | Fresh per execution (auto-generated) |
| `request_id` | `UUID` | Stable across retries (from MainLoopRequest) |
| `session_id` | `UUID \| None` | Session correlation |
| `attempt` | `int` | Delivery count (1 = first) |
| `worker_id` | `str` | Worker identification |
| `trace_id` | `str \| None` | Distributed trace correlation |
| `span_id` | `str \| None` | Span within trace |

## Data Flow

```
MainLoop._handle_message()
  └─ _build_run_context()
  └─ _execute(run_context=...)
       └─ adapter.evaluate(run_context=...)
            └─ PromptRendered(run_context=...)
            └─ ToolExecutor → ToolContext(run_context=...)
            └─ ToolInvoked(run_context=...)
            └─ PromptExecuted(run_context=...)
  └─ MainLoopResult(run_context=...)
```

## Integration Points

### MainLoopRequest/Result

Requests include optional `run_context` with trace context. MainLoop builds
full context with execution-specific values.

### ToolContext

Tool handlers access via `context.run_context`:

```python
def my_handler(params, *, context: ToolContext) -> ToolResult:
    if context.run_context:
        log = logger.bind(**context.run_context.to_log_context())
```

### Telemetry Events

All events include optional `run_context`: `PromptRendered`, `ToolInvoked`,
`PromptExecuted`.

## run_id vs request_id

| Field | Lifecycle | Purpose |
|-------|-----------|---------|
| `run_id` | Fresh per execution | Unique execution identifier |
| `request_id` | Stable across retries | Logical request correlation |

## OpenTelemetry Integration

```python
with tracer.start_as_current_span("process_request") as span:
    ctx = trace.get_current_span().get_span_context()
    run_context = RunContext(
        trace_id=format(ctx.trace_id, "032x"),
        span_id=format(ctx.span_id, "016x"),
    )
```

## Invariants

1. **Immutable**: All fields read-only after creation
2. **Fresh run_id per execution**: Never reused across attempts
3. **Stable run_id during execution**: Same across all events in one execution
4. **Preserved trace context**: `trace_id`/`span_id` pass through unchanged

## Related Specifications

- `specs/MAIN_LOOP.md` - Request processing
- `specs/MAILBOX.md` - Message delivery and retries
- `specs/LOGGING.md` - Structured logging
