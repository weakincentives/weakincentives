# RunContext Specification

## Purpose

`RunContext` provides immutable execution metadata for distributed tracing,
request correlation, and debugging. Flows from `AgentLoop` through adapters to
tool handlers and telemetry events.

**Implementation:** `src/weakincentives/runtime/run_context.py`

## Core Type

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `UUID` | Fresh per execution (auto-generated) |
| `request_id` | `UUID` | Stable across retries (from AgentLoopRequest) |
| `session_id` | `UUID \| None` | Session correlation |
| `attempt` | `int` | Delivery count (1 = first) |
| `worker_id` | `str` | Worker identification |
| `trace_id` | `str \| None` | Distributed trace correlation |
| `span_id` | `str \| None` | Span within trace |

## Data Flow

```
AgentLoop._handle_message()
  └─ _build_run_context()
  └─ bind_run_context(logger, run_context)  → log with run_id/request_id/...
  └─ _execute(run_context=...)
       └─ adapter.evaluate(run_context=...)
            └─ InnerLoopConfig(
                 logger_override=bind_run_context(logger, run_context),
                 run_context=run_context
               )
            └─ InnerLoop._prepare() → uses logger_override with context
            └─ PromptRendered(run_context=...)
            └─ ToolExecutor → ToolContext(run_context=...)
            └─ ToolInvoked(run_context=...)
            └─ PromptExecuted(run_context=...)
  └─ _dead_letter() → logs with run_context bound
  └─ AgentLoopResult(run_context=...)
```

## Integration Points

### AgentLoopRequest/Result

Requests include optional `run_context` with trace context. AgentLoop builds
full context with execution-specific values.

### Logger Binding

The `bind_run_context` helper (at `src/weakincentives/runtime/logging.py`) binds
RunContext fields to a structured logger, enabling consistent tracing across
the entire request lifecycle. All downstream logging inherits the run context
fields automatically.

**Adapter Integration**: Adapters bind run_context to `logger_override` when
creating `InnerLoopConfig`. All downstream logging (InnerLoop, ToolExecutor,
tool handlers) automatically inherits the run context fields.

### ToolContext

Tool handlers access `context.run_context` and bind it to their logger via
`run_context.to_log_context()` for consistent tracing.

### Telemetry Events

All events include optional `run_context`: `PromptRendered`, `ToolInvoked`,
`PromptExecuted`.

## run_id vs request_id

| Field | Lifecycle | Purpose |
|-------|-----------|---------|
| `run_id` | Fresh per execution | Unique execution identifier |
| `request_id` | Stable across retries | Logical request correlation |

## OpenTelemetry Integration

Populate `RunContext.trace_id` and `span_id` from the active OTel span context
(format as 32-hex and 16-hex strings respectively) to correlate WINK logs with
distributed traces.

## Invariants

1. **Immutable**: All fields read-only after creation
1. **Fresh run_id per execution**: Never reused across attempts
1. **Stable run_id during execution**: Same across all events in one execution
1. **Preserved trace context**: `trace_id`/`span_id` pass through unchanged
1. **Logger binding**: Adapters bind run_context to logger_override for consistent tracing

## Related Specifications

- `specs/AGENT_LOOP.md` - Request processing
- `specs/MAILBOX.md` - Message delivery and retries
- `specs/LOGGING.md` - Structured logging
