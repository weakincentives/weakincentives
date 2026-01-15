# Dead Letter Queue Specification

## Purpose

DLQs capture messages that cannot be processed after repeated attempts. Prevents
poison messages from blocking queues while preserving them for inspection.

**Implementation:** `src/weakincentives/runtime/dlq.py`

**Use DLQ for:** Poison message isolation, failure forensics, manual remediation.

**Not for:** Transient failures (use backoff), rate limiting, validation errors.

## Core Types

### DLQPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mailbox` | `Mailbox[DeadLetter[T], None]` | - | Destination |
| `max_delivery_count` | `int` | `5` | Attempts before dead-lettering |
| `include_errors` | `frozenset[type] \| None` | `None` | Immediate dead-letter errors |
| `exclude_errors` | `frozenset[type] \| None` | `None` | Never dead-letter errors |

`should_dead_letter(message, error)` method determines action. Override for
custom logic.

### DeadLetter

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | `str` | Original ID |
| `body` | `T` | Original message |
| `source_mailbox` | `str` | Source queue name |
| `delivery_count` | `int` | Delivery attempts |
| `last_error` | `str` | Error string |
| `last_error_type` | `str` | Qualified type name |
| `dead_lettered_at` | `datetime` | Timestamp |
| `first_received_at` | `datetime` | First delivery |
| `request_id` | `UUID \| None` | Request ID |
| `reply_to` | `str \| None` | Reply mailbox name |
| `trace_id` | `str \| None` | Tracing correlation |

## MainLoop Integration

```python
main_loop = MainLoop(
    adapter=adapter,
    requests=requests,
    dlq=DLQPolicy(mailbox=dead_letters, max_delivery_count=5),
)
```

### Execution Flow

```
receive() → process() → reply() → acknowledge()
                │
                ▼ (exception)
          should_dead_letter()?
                │
        ┌───────┴───────┐
        ▼ No            ▼ Yes
    nack(backoff)   send_to_dlq() → acknowledge()
```

## EvalLoop Integration

Same pattern as MainLoop, wrapping `EvalRequest` in `DeadLetter`.

## Usage Examples

### Basic Setup

```python
dlq = DLQPolicy(mailbox=dead_letters, max_delivery_count=5)
```

### Immediate Dead-Lettering

```python
dlq = DLQPolicy(
    mailbox=dead_letters,
    include_errors=frozenset({ValidationError, ContentPolicyViolation}),
)
```

### Exclude Transient Errors

```python
dlq = DLQPolicy(
    mailbox=dead_letters,
    exclude_errors=frozenset({RateLimitError, TimeoutError}),
)
```

### Custom Policy

```python
@dataclass(frozen=True)
class ErrorBudgetPolicy[T, R](DLQPolicy[T, R]):
    error_budget: float = 0.5

    def should_dead_letter(self, message, error) -> bool:
        if message.delivery_count >= self.max_delivery_count:
            return True
        return get_error_rate(message.body) > self.error_budget
```

## Processing Dead Letters

### DLQ Consumer

```python
def process_dead_letters(dlq, handler):
    for msg in dlq.receive(wait_time_seconds=20):
        try:
            handler(msg.body)
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=3600)  # 1 hour backoff
```

### Replay to Source

```python
def replay_handler(dead_letter, source, reply_resolver):
    reply_to = reply_resolver.resolve(dead_letter.reply_to) if dead_letter.reply_to else None
    source.send(dead_letter.body, reply_to=reply_to)
```

## LoopGroup Integration

```python
dlq_consumer = DLQConsumer(mailbox=dead_letters, handler=alert_handler)
group = LoopGroup(loops=[main_loop, dlq_consumer], health_port=8080)
```

## Error Classification

| Category | Examples | Behavior |
|----------|----------|----------|
| Retriable | `RateLimitError`, `TimeoutError` | Backoff retry |
| Non-Retriable | `ValidationError`, `AuthenticationError` | Immediate DLQ |
| Ambiguous | `ProviderError` | Depends on context |

## Observability

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `dlq.messages.sent` | Counter | Sent to DLQ |
| `dlq.messages.processed` | Counter | Processed from DLQ |
| `dlq.queue.depth` | Gauge | Current DLQ size |

## Best Practices

- **Naming**: `{source}-dlq` pattern
- **Retention**: DLQ longer than source (30 days vs 3 days)
- **Avoid loops**: DLQ handler should never re-dead-letter
- **Idempotent replay**: Deduplicate by request_id

## Limitations

- **No automatic replay**: Requires explicit setup
- **No cross-backend DLQ**: Same backend type required
- **No transformation**: Replay sends original as-is
- **Single DLQ per loop**: Cannot route errors to different DLQs

## Related Specifications

- `specs/MAILBOX.md` - Mailbox protocol
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/LIFECYCLE.md` - LoopGroup coordination
