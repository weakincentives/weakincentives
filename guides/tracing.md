# Tracing and Correlation

*Canonical spec: [specs/RUN_CONTEXT.md](../specs/RUN_CONTEXT.md)*

WINK provides built-in request correlation and optional distributed tracing
integration. Every request gets correlation IDs automatically, and you can
connect these to external tracing systems like OpenTelemetry when needed.

## RunContext: The Core Abstraction

`RunContext` is an immutable dataclass that captures execution metadata for a
single request run. It flows through the system from MainLoop to tool handlers
and telemetry events.

```python nocheck
from weakincentives.runtime import RunContext

run_context = RunContext(
    worker_id="worker-1",
    trace_id="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    span_id="00f067aa0ba902b7",
)
```

**Fields:**

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | `UUID` | Fresh for each execution attempt (auto-generated) |
| `request_id` | `UUID` | Stable across retries of the same logical request |
| `session_id` | `UUID \| None` | Session correlation |
| `attempt` | `int` | Delivery count (1 = first attempt) |
| `worker_id` | `str` | Identifier for the processing worker |
| `trace_id` | `str \| None` | Distributed trace correlation (e.g., OpenTelemetry) |
| `span_id` | `str \| None` | Span within the trace |

## Correlation IDs: run_id vs request_id

WINK distinguishes between two correlation IDs:

- **`run_id`**: Fresh UUID for each execution attempt. If a request fails and is
  retried, the retry gets a new `run_id`.
- **`request_id`**: Stable UUID for the logical request. Preserved across all
  retry attempts.

This separation enables both fine-grained debugging (which specific execution
had the error?) and logical correlation (what happened to this request across
all attempts?).

```
Request A (request_id: abc-123)
├── Attempt 1 (run_id: run-001) → Failed
├── Attempt 2 (run_id: run-002) → Failed
└── Attempt 3 (run_id: run-003) → Succeeded
```

## How Context Flows Through the System

When MainLoop processes a message, it builds a `RunContext` and threads it
through the entire execution:

```
MainLoop._handle_message()
  └─ _build_run_context()        → Creates RunContext with fresh run_id
  └─ bind_run_context(logger)    → All logs include correlation fields
  └─ adapter.evaluate(run_context=...)
       └─ PromptRendered(run_context=...)   → Telemetry event
       └─ ToolInvoked(run_context=...)      → Telemetry event
       └─ PromptExecuted(run_context=...)   → Telemetry event
  └─ MainLoopResult(run_context=...)        → Response includes context
```

Every log entry and telemetry event carries the same correlation IDs, making it
easy to reconstruct what happened during a request.

## Structured Logging Integration

The `bind_run_context` helper binds all RunContext fields to a structured
logger. Every log entry from the bound logger includes the correlation fields:

```python nocheck
from weakincentives.runtime import RunContext, get_logger
from weakincentives.runtime.logging import bind_run_context

logger = get_logger(__name__)
run_context = RunContext(worker_id="worker-1")

# Bind context to logger
log = bind_run_context(logger, run_context)

# All logs now include run_id, request_id, attempt, worker_id
log.info("Processing started", event="request.start")
```

**Output (JSON mode):**

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "level": "INFO",
  "event": "request.start",
  "message": "Processing started",
  "run_id": "a1b2c3d4-...",
  "request_id": "e5f6g7h8-...",
  "attempt": 1,
  "worker_id": "worker-1"
}
```

## Accessing Context in Tool Handlers

Tool handlers can access the RunContext via `ToolContext`:

```python nocheck
from weakincentives.tools import ToolContext, ToolResult

def my_handler(params: MyParams, *, context: ToolContext) -> ToolResult[str]:
    if context.run_context:
        # Log with correlation context
        log = context.logger.bind(**context.run_context.to_log_context())
        log.info("Executing tool", event="tool.my_handler")

    return ToolResult.ok("done")
```

## Telemetry Events

All telemetry events carry `run_context` for downstream analysis:

- **`PromptRendered`**: Emitted when a prompt is rendered
- **`ToolInvoked`**: Emitted when a tool is called (includes params and result)
- **`PromptExecuted`**: Emitted when evaluation completes (includes token usage)

You can subscribe to these events through the session dispatcher and forward
them to your tracing backend:

```python nocheck
from weakincentives.runtime import PromptExecuted, ToolInvoked

def handle_event(event):
    if isinstance(event, PromptExecuted) and event.run_context:
        # Forward to your tracing system
        send_to_tracer(
            trace_id=event.run_context.trace_id,
            span_id=event.run_context.span_id,
            operation="prompt.executed",
            metadata={"prompt": event.prompt_name, "tokens": event.usage},
        )
```

## OpenTelemetry Integration

WINK's `trace_id` and `span_id` fields are designed for OpenTelemetry
compatibility. Pass trace context from your existing instrumentation:

```python nocheck
from opentelemetry import trace
from weakincentives.runtime import RunContext, MainLoopRequest

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_request") as span:
    ctx = span.get_span_context()

    # Create RunContext with OpenTelemetry trace IDs
    run_context = RunContext(
        trace_id=format(ctx.trace_id, "032x"),
        span_id=format(ctx.span_id, "016x"),
        worker_id="worker-1",
    )

    # Pass to MainLoop via request
    request = MainLoopRequest(
        payload=my_payload,
        run_context=run_context,
    )
```

The trace context propagates through the entire execution and appears in all
logs and telemetry events, enabling end-to-end distributed traces.

## Injecting Context from HTTP Headers

When receiving requests from upstream services, extract trace context from
headers:

```python nocheck
from opentelemetry.propagate import extract
from opentelemetry import trace

def handle_http_request(request):
    # Extract trace context from incoming headers
    ctx = extract(request.headers)

    with trace.get_tracer(__name__).start_span("handle", context=ctx) as span:
        span_ctx = span.get_span_context()

        run_context = RunContext(
            trace_id=format(span_ctx.trace_id, "032x"),
            span_id=format(span_ctx.span_id, "016x"),
        )

        # Process with MainLoop
        result = loop.execute(payload, run_context=run_context)
```

## Multi-Worker Correlation

In distributed deployments, `worker_id` identifies which worker processed each
request:

```python nocheck
import socket
import os

worker_id = f"{socket.gethostname()}-{os.getpid()}"
run_context = RunContext(worker_id=worker_id)
```

MainLoop sets this automatically using the pattern `{hostname}-{pid}`. All logs
from a worker include this identifier, making it easy to correlate logs across
multiple workers.

## Dead Letter Queue Correlation

When requests fail after all retries, they're moved to the dead letter queue.
The `DeadLetter` preserves `trace_id` for error correlation:

```python nocheck
from weakincentives.runtime import DeadLetter

# When processing dead letters, trace_id links back to the original trace
for dead_letter in dlq.receive():
    print(f"Failed request: {dead_letter.request_id}")
    print(f"Trace ID: {dead_letter.trace_id}")  # For distributed trace lookup
    print(f"Attempts: {dead_letter.delivery_count}")
    print(f"Error: {dead_letter.last_error}")
```

## Retry Tracking

The `attempt` field tracks retry attempts, mapping to mailbox delivery count:

```python nocheck
# In logs, you'll see attempt incrementing on retries:
# {"run_id": "run-001", "request_id": "abc-123", "attempt": 1, ...}  # First try
# {"run_id": "run-002", "request_id": "abc-123", "attempt": 2, ...}  # Retry
# {"run_id": "run-003", "request_id": "abc-123", "attempt": 3, ...}  # Final retry
```

Use this to identify flaky operations that succeed on retry vs. persistent
failures.

## Best Practices

1. **Always log with bound context**: Use `bind_run_context` to ensure all logs
   include correlation fields.

1. **Preserve trace context across boundaries**: When making outbound HTTP calls
   or publishing to queues, include trace IDs for end-to-end correlation.

1. **Use request_id for business correlation**: When you need to answer "what
   happened to request X?", query by `request_id` to see all attempts.

1. **Use run_id for debugging specific executions**: When investigating a
   specific failure, query by `run_id` to see exactly what happened in that
   attempt.

1. **Set worker_id for distributed debugging**: In multi-worker deployments,
   worker_id helps identify which instance processed a request.

## Example: Full Tracing Setup

```python nocheck
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from weakincentives.runtime import MainLoop, RunContext, configure_logging

# Configure OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Configure WINK logging
configure_logging(level="INFO", json_mode=True)


class TracedLoop(MainLoop[Request, Response]):
    def execute(self, request, **kwargs):
        with tracer.start_as_current_span("agent.execute") as span:
            ctx = span.get_span_context()

            run_context = RunContext(
                trace_id=format(ctx.trace_id, "032x"),
                span_id=format(ctx.span_id, "016x"),
                worker_id=self._worker_id,
            )

            return super().execute(request, run_context=run_context, **kwargs)
```

## Next Steps

- [Debugging](debugging.md): Use debug bundles to inspect execution
- [Orchestration](orchestration.md): Learn about MainLoop
- [Lifecycle](lifecycle.md): Health checks and watchdog monitoring
