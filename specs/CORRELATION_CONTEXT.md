# Correlation Context Specification

This document specifies the run-level correlation context system for
propagating identifiers across logs, telemetry events, and mailbox messages.

## Overview

Correlation context provides a structured mechanism to trace requests across
distributed components. Every operation within a run shares a common set of
identifiers that flow through:

- **Logs**: Structured log entries carry correlation fields
- **Telemetry events**: Runtime events include correlation metadata
- **Mailbox messages**: Message headers propagate context across boundaries

## Context Identifiers

### Identifier Hierarchy

```
run_id                    # Identifies a complete run (may span retries)
├── attempt               # Retry counter within a run (0-indexed)
├── request_id            # Identifies a single MainLoop request
│   └── session_id        # Session processing the request
└── trace_id              # Distributed tracing root (W3C compatible)
    └── span_id           # Current operation span
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Stable identifier for a logical run. Survives retries. Format: UUID v4 or caller-provided. |
| `attempt` | `int` | Zero-indexed retry counter. Increments on each retry of the same `run_id`. |
| `request_id` | `UUID` | Unique identifier for a MainLoop request. Generated per request. |
| `session_id` | `UUID` | Session processing the current request. Already exists in Session. |
| `trace_id` | `str` | 32-character lowercase hex string (W3C Trace Context). |
| `span_id` | `str` | 16-character lowercase hex string (W3C Trace Context). |

### CorrelationContext Type

```python
from dataclasses import dataclass, field
from datetime import datetime, UTC
from uuid import UUID, uuid4

@dataclass(slots=True, frozen=True)
class CorrelationContext:
    """Immutable correlation context propagated through a run."""

    run_id: str
    attempt: int = 0
    request_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex[:16])
    baggage: tuple[tuple[str, str], ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def with_span(self, span_id: str | None = None) -> "CorrelationContext":
        """Create child context with new span_id, preserving trace_id."""
        return CorrelationContext(
            run_id=self.run_id,
            attempt=self.attempt,
            request_id=self.request_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            span_id=span_id or uuid4().hex[:16],
            baggage=self.baggage,
        )

    def with_session(self, session_id: UUID) -> "CorrelationContext":
        """Bind context to a session."""
        return CorrelationContext(
            run_id=self.run_id,
            attempt=self.attempt,
            request_id=self.request_id,
            session_id=session_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            baggage=self.baggage,
        )

    def with_attempt(self, attempt: int) -> "CorrelationContext":
        """Create context for a retry attempt."""
        return CorrelationContext(
            run_id=self.run_id,
            attempt=attempt,
            request_id=uuid4(),  # New request per attempt
            session_id=None,     # New session per attempt
            trace_id=self.trace_id,
            span_id=uuid4().hex[:16],
            baggage=self.baggage,
        )
```

### Baggage

Baggage carries application-specific key-value pairs across boundaries:

```python
context = CorrelationContext(
    run_id="abc-123",
    baggage=(
        ("tenant_id", "acme-corp"),
        ("environment", "production"),
    ),
)
```

Baggage keys must be ASCII lowercase alphanumeric with hyphens. Values must be
URL-safe strings. Total baggage size must not exceed 8192 bytes when serialized.

## Propagation Rules

### Creation

Correlation context is created at run entry points:

1. **MainLoop.execute()**: Creates context if not provided
2. **CLI entrypoint**: Creates context from environment or generates fresh
3. **Message receipt**: Extracts context from message headers

```python
# MainLoop creates context for new requests
def execute(
    self,
    request: UserRequestT,
    *,
    correlation: CorrelationContext | None = None,
) -> MainLoopResult[OutputT]:
    correlation = correlation or CorrelationContext(run_id=uuid4().hex)
    ...
```

### Inheritance

Child operations inherit context from parents:

| Parent | Child | Inherited Fields |
|--------|-------|------------------|
| MainLoop | Session | All fields, session_id bound |
| Session | Tool call | All fields, new span_id |
| Session | Child session | All fields, new session_id |
| Message send | Message receive | All fields via headers |

### Thread Safety

CorrelationContext is immutable (`frozen=True`). Mutations create new instances.
No thread-local storage is used. Context flows explicitly through function
parameters and return values.

## Logging Integration

### StructuredLogger Binding

Bind correlation context to loggers at scope boundaries:

```python
from weakincentives.runtime.logging import get_logger

logger = get_logger(__name__)

# Bind at request scope
request_logger = logger.bind(
    run_id=context.run_id,
    attempt=context.attempt,
    request_id=str(context.request_id),
    trace_id=context.trace_id,
    span_id=context.span_id,
)

# All logs from request_logger include correlation fields
request_logger.info("Processing request", event="request_started")
```

### Log Schema

Correlation fields appear in the `context` object of structured logs:

```json
{
  "event": "tool_executed",
  "context": {
    "run_id": "abc-123",
    "attempt": 0,
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "session_id": "660e8400-e29b-41d4-a716-446655440001",
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "span_id": "00f067aa0ba902b7",
    "tool": "read_file",
    "prompt_name": "code_review"
  }
}
```

### Required Binding Points

| Location | When to Bind |
|----------|--------------|
| `MainLoop.execute()` | On request start |
| `Session.__init__()` | On session creation |
| `ProviderAdapter.evaluate()` | On prompt evaluation |
| `ToolExecutor.invoke()` | On tool invocation |

## Telemetry Event Integration

### Event Base Extension

All runtime events carry correlation context:

```python
@dataclass(slots=True, frozen=True)
class BaseEvent:
    """Base for all telemetry events."""

    event_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Correlation fields
    run_id: str | None = None
    attempt: int | None = None
    request_id: UUID | None = None
    session_id: UUID | None = None
    trace_id: str | None = None
    span_id: str | None = None
```

### Event Types

Existing events extended with correlation fields:

```python
@dataclass(slots=True, frozen=True)
class PromptRendered(BaseEvent):
    prompt_name: str
    adapter: AdapterName
    render_inputs: tuple[Any, ...] = ()

@dataclass(slots=True, frozen=True)
class ToolInvoked(BaseEvent):
    prompt_name: str
    adapter: AdapterName
    tool_name: str
    invocation_id: str
    result_type: ToolResultType
    token_usage: TokenUsage | None = None

@dataclass(slots=True, frozen=True)
class PromptExecuted(BaseEvent):
    prompt_name: str
    adapter: AdapterName
    token_usage: TokenUsage
    output: Any = None
```

### Event Dispatch

Adapters populate correlation fields when dispatching events:

```python
def _dispatch_prompt_executed(
    self,
    prompt_name: str,
    token_usage: TokenUsage,
    output: Any,
    context: CorrelationContext,
) -> None:
    self._bus.dispatch(
        PromptExecuted(
            prompt_name=prompt_name,
            adapter=self.adapter_name,
            token_usage=token_usage,
            output=output,
            run_id=context.run_id,
            attempt=context.attempt,
            request_id=context.request_id,
            session_id=context.session_id,
            trace_id=context.trace_id,
            span_id=context.span_id,
        )
    )
```

## Mailbox Integration

### Message Headers

Messages carry correlation context in a headers field:

```python
@dataclass(slots=True)
class Message[T, R]:
    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    reply_to: Mailbox[R, None] | None = None
    headers: MessageHeaders | None = None


@dataclass(slots=True, frozen=True)
class MessageHeaders:
    """Correlation headers for message propagation."""

    run_id: str | None = None
    attempt: int | None = None
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    baggage: str | None = None  # URL-encoded key=value pairs

    @classmethod
    def from_context(cls, context: CorrelationContext) -> "MessageHeaders":
        """Create headers from correlation context."""
        return cls(
            run_id=context.run_id,
            attempt=context.attempt,
            request_id=str(context.request_id),
            trace_id=context.trace_id,
            span_id=context.span_id,
            baggage=cls._encode_baggage(context.baggage),
        )

    def to_context(self) -> CorrelationContext:
        """Reconstruct correlation context from headers."""
        return CorrelationContext(
            run_id=self.run_id or uuid4().hex,
            attempt=self.attempt or 0,
            request_id=UUID(self.request_id) if self.request_id else uuid4(),
            trace_id=self.trace_id or uuid4().hex,
            span_id=self.span_id or uuid4().hex[:16],
            baggage=self._decode_baggage(self.baggage),
        )

    @staticmethod
    def _encode_baggage(baggage: tuple[tuple[str, str], ...]) -> str:
        return ",".join(f"{k}={v}" for k, v in baggage)

    @staticmethod
    def _decode_baggage(encoded: str | None) -> tuple[tuple[str, str], ...]:
        if not encoded:
            return ()
        pairs = []
        for item in encoded.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                pairs.append((k, v))
        return tuple(pairs)
```

### Send with Context

Mailbox.send() accepts correlation context:

```python
class Mailbox(Protocol[T, R]):
    def send(
        self,
        body: T,
        *,
        reply_to: Mailbox[R, None] | None = None,
        correlation: CorrelationContext | None = None,
    ) -> None:
        """Send message with optional correlation context."""
        ...
```

### Redis Implementation

Redis mailbox stores headers as JSON in a dedicated field:

```python
# Message structure in Redis stream
{
    "body": "<json-encoded-body>",
    "reply_to": "<mailbox-reference>",
    "headers": "{\"run_id\":\"abc\",\"trace_id\":\"...\",\"span_id\":\"...\"}"
}
```

### Header Propagation

When processing a message, extract and propagate context:

```python
async def process_message(message: Message[T, R]) -> None:
    # Extract correlation context from message headers
    context = (
        message.headers.to_context()
        if message.headers
        else CorrelationContext(run_id=uuid4().hex)
    )

    # Bind to logger
    logger = get_logger(__name__).bind(
        run_id=context.run_id,
        request_id=str(context.request_id),
        trace_id=context.trace_id,
    )

    # Process with context
    result = await handle(message.body, correlation=context)

    # Reply preserves context (new span)
    if message.reply_to:
        message.reply_to.send(
            result,
            correlation=context.with_span(),
        )
```

## MainLoop Integration

### MainLoopRequest Extension

```python
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Correlation context
    run_id: str | None = None
    attempt: int = 0
    trace_id: str | None = None
    span_id: str | None = None
    baggage: tuple[tuple[str, str], ...] = ()

    def correlation_context(self) -> CorrelationContext:
        """Build correlation context from request fields."""
        return CorrelationContext(
            run_id=self.run_id or uuid4().hex,
            attempt=self.attempt,
            request_id=self.request_id,
            trace_id=self.trace_id or uuid4().hex,
            span_id=self.span_id or uuid4().hex[:16],
            baggage=self.baggage,
        )
```

### MainLoopResult Extension

```python
@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    request_id: UUID
    output: OutputT | None = None
    error: str | None = None
    session_id: UUID | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Correlation echo
    run_id: str | None = None
    attempt: int | None = None
    trace_id: str | None = None
```

## W3C Trace Context Compatibility

### traceparent Header

For HTTP-based transports, support W3C Trace Context format:

```
traceparent: 00-<trace_id>-<span_id>-<trace_flags>
             │   │          │         └─ 2 hex chars (sampled flag)
             │   │          └─ 16 hex chars (span_id)
             │   └─ 32 hex chars (trace_id)
             └─ version (always "00")
```

Example:
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

### Parsing and Formatting

```python
import re

TRACEPARENT_PATTERN = re.compile(
    r"^00-([a-f0-9]{32})-([a-f0-9]{16})-([a-f0-9]{2})$"
)

def parse_traceparent(header: str) -> tuple[str, str, bool] | None:
    """Parse W3C traceparent header.

    Returns (trace_id, span_id, sampled) or None if invalid.
    """
    match = TRACEPARENT_PATTERN.match(header)
    if not match:
        return None
    trace_id, span_id, flags = match.groups()
    sampled = int(flags, 16) & 0x01 == 1
    return trace_id, span_id, sampled


def format_traceparent(
    trace_id: str,
    span_id: str,
    sampled: bool = True,
) -> str:
    """Format W3C traceparent header."""
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"
```

### baggage Header

Baggage uses W3C Baggage format:

```
baggage: tenant_id=acme-corp,environment=production
```

## Resource Registry Integration

### CorrelationContext as Resource

Register correlation context in the resource registry:

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

def create_registry(context: CorrelationContext) -> ResourceRegistry:
    return ResourceRegistry.of(
        Binding(
            CorrelationContext,
            lambda r: context,
            scope=Scope.SINGLETON,
        ),
        # Other bindings...
    )
```

### Tool Access

Tools access correlation context through ToolContext:

```python
def my_tool_handler(
    params: MyParams,
    *,
    context: ToolContext,
) -> ToolResult[MyOutput]:
    correlation = context.resources.get(CorrelationContext)
    logger = get_logger(__name__).bind(
        run_id=correlation.run_id,
        span_id=correlation.span_id,
    )
    logger.info("Tool executing", event="tool_started")
    ...
```

## Session Tags

Session tags include correlation fields for inspection:

```python
session = Session(
    bus=bus,
    tags={
        "run_id": context.run_id,
        "attempt": str(context.attempt),
        "trace_id": context.trace_id,
    },
)
```

Tags are available in session snapshots and debug UI.

## Error Handling

### Missing Context

When correlation context is unavailable, generate fresh identifiers:

```python
def ensure_context(
    context: CorrelationContext | None,
) -> CorrelationContext:
    """Ensure valid correlation context exists."""
    if context is not None:
        return context
    return CorrelationContext(run_id=uuid4().hex)
```

### Invalid Headers

Invalid or malformed headers are logged and replaced with fresh context:

```python
def parse_headers_safe(
    headers: MessageHeaders | None,
    logger: StructuredLogger,
) -> CorrelationContext:
    """Parse headers with fallback to fresh context."""
    if headers is None:
        return CorrelationContext(run_id=uuid4().hex)
    try:
        return headers.to_context()
    except (ValueError, TypeError) as e:
        logger.warning(
            "Invalid correlation headers, generating fresh context",
            event="correlation_parse_failed",
            error=str(e),
        )
        return CorrelationContext(run_id=uuid4().hex)
```

## Testing

### Context Propagation Tests

Verify context flows through all integration points:

```python
def test_context_propagates_to_events(bus: InProcessDispatcher) -> None:
    """Correlation context appears in dispatched events."""
    events: list[PromptExecuted] = []
    bus.subscribe(PromptExecuted, events.append)

    context = CorrelationContext(
        run_id="test-run",
        trace_id="a" * 32,
        span_id="b" * 16,
    )

    # Execute with context
    adapter.evaluate(prompt, session=session, correlation=context)

    assert len(events) == 1
    assert events[0].run_id == "test-run"
    assert events[0].trace_id == "a" * 32


def test_context_propagates_through_mailbox() -> None:
    """Correlation context survives message send/receive."""
    context = CorrelationContext(run_id="test-run")

    mailbox.send(body, correlation=context)
    message = mailbox.receive()

    received_context = message.headers.to_context()
    assert received_context.run_id == "test-run"
    assert received_context.trace_id == context.trace_id


def test_child_span_preserves_trace_id() -> None:
    """with_span() creates new span under same trace."""
    parent = CorrelationContext(
        run_id="test-run",
        trace_id="parent-trace",
        span_id="parent-span",
    )

    child = parent.with_span()

    assert child.trace_id == "parent-trace"
    assert child.span_id != "parent-span"
    assert child.run_id == "test-run"
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
def test_context_roundtrip_through_headers(run_id: str, attempt: int) -> None:
    """Context survives header serialization roundtrip."""
    original = CorrelationContext(run_id=run_id, attempt=attempt)
    headers = MessageHeaders.from_context(original)
    restored = headers.to_context()

    assert restored.run_id == original.run_id
    assert restored.attempt == original.attempt
    assert restored.trace_id == original.trace_id
```

## Migration

### Backward Compatibility

Existing code without correlation context continues to work:

1. All correlation fields are optional with sensible defaults
2. Events without correlation fields remain valid
3. Messages without headers are processed normally

### Adoption Path

1. Add `CorrelationContext` type and `MessageHeaders`
2. Extend `MainLoopRequest`/`MainLoopResult` with correlation fields
3. Update adapters to accept and propagate context
4. Extend events with correlation fields
5. Update mailbox implementations to handle headers
6. Add logger binding at scope boundaries

## Observability Recommendations

### Metrics

Emit metrics with correlation dimensions:

```python
# Example: request latency by run
metrics.histogram(
    "request_latency_seconds",
    latency,
    tags={
        "run_id": context.run_id,
        "attempt": str(context.attempt),
    },
)
```

### Distributed Tracing

Export spans to tracing backends (Jaeger, Zipkin, OTLP):

```python
# Span creation with correlation context
span = tracer.start_span(
    "evaluate_prompt",
    trace_id=context.trace_id,
    parent_span_id=context.span_id,
    attributes={
        "run_id": context.run_id,
        "session_id": str(context.session_id),
    },
)
```

### Log Aggregation

Query logs by correlation identifiers:

```sql
-- Find all logs for a run
SELECT * FROM logs
WHERE context->>'run_id' = 'abc-123'
ORDER BY timestamp;

-- Trace request through system
SELECT * FROM logs
WHERE context->>'trace_id' = '4bf92f3577b34da6a3ce929d0e0e4736'
ORDER BY timestamp;
```

## Summary

Correlation context provides unified tracing across the weakincentives runtime:

| Surface | Integration Point | Fields |
|---------|-------------------|--------|
| Logs | `logger.bind()` | All correlation fields in `context` |
| Events | Event dataclass fields | run_id, attempt, request_id, session_id, trace_id, span_id |
| Mailbox | `MessageHeaders` | Serialized context in message headers |
| MainLoop | Request/Result fields | run_id, attempt, trace_id, span_id, baggage |
| Resources | `CorrelationContext` binding | Full context available to tools |
| Session | Tags | run_id, attempt, trace_id for inspection |

The design preserves immutability, explicit propagation, and W3C compatibility
while integrating cleanly with existing patterns.
