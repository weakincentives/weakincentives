# Dead Letter Queue Specification

Capture messages that fail after repeated attempts. Prevents poison messages
from blocking queues while preserving them for inspection.

**Source:** `src/weakincentives/runtime/dlq.py`

## When to Use

**DLQ:** Poison message isolation, failure forensics, manual remediation.

**Not for:** Transient failures (use backoff), rate limiting (use throttling),
validation errors (reject immediately).

## DLQPolicy

**Definition:** `runtime/dlq.py:DLQPolicy`

```python
DLQPolicy[T, R](
    mailbox: Mailbox[DeadLetter[T], None],
    max_delivery_count: int = 5,
    include_errors: frozenset[type[Exception]] | None = None,  # Immediate DLQ
    exclude_errors: frozenset[type[Exception]] | None = None,  # Never DLQ
)
```

Override `should_dead_letter(message, error)` for custom logic.

## DeadLetter

**Definition:** `runtime/dlq.py:DeadLetter`

| Field | Purpose |
|-------|---------|
| `message_id` | Original ID |
| `body` | Original payload |
| `source_mailbox` | Origin queue name |
| `delivery_count` | Attempts before DLQ |
| `last_error` | Error string |
| `last_error_type` | Qualified type name |
| `dead_lettered_at` | Timestamp |

## Integration

```python
main_loop = MainLoop(
    adapter=adapter,
    requests=requests,
    dlq=DLQPolicy(mailbox=dead_letters, max_delivery_count=5),
)
```

```
receive() → process() → reply() → acknowledge()
               │
               ▼ (exception)
         should_dead_letter()?
               │
       No: nack(backoff)   Yes: send_to_dlq() → acknowledge()
```

## Error Classification

| Type | Behavior |
|------|----------|
| `MailboxConnectionError`, `RateLimitError`, `TimeoutError` | Retry (exclude) |
| `ValidationError`, `ContentPolicyViolation` | Immediate DLQ (include) |

## Processing DLQ

```python
for msg in dlq.receive():
    alert(msg.body)        # Or replay to source
    msg.acknowledge()
```

Avoid DLQ loops: handler failures should log and discard, not re-DLQ.

## DLQConsumer

**Definition:** `runtime/dlq.py:DLQConsumer`

Runnable for LoopGroup integration:

```python
group = LoopGroup(loops=[main_loop, DLQConsumer(mailbox=dlq, handler=alert)])
```

## Observability

| Metric | Purpose |
|--------|---------|
| `dlq.messages.sent` | Messages dead-lettered |
| `dlq.queue.depth` | Current DLQ size |
| `dlq.message.age_seconds` | Time before processing |

## Limitations

- No automatic replay (requires explicit setup)
- No cross-backend DLQ (same backend type required)
- Single DLQ per loop (no error-based routing)
