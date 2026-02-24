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
| `include_errors` | `frozenset[type[Exception]] \| None` | `None` | Immediate dead-letter errors |
| `exclude_errors` | `frozenset[type[Exception]] \| None` | `None` | Never dead-letter errors |

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

## AgentLoop Integration

Pass `dlq=DLQPolicy(mailbox=..., max_delivery_count=5)` to `AgentLoop` or
`EvalLoop`. See `src/weakincentives/runtime/agent_loop.py`.

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

Same pattern as AgentLoop, wrapping `EvalRequest` in `DeadLetter`.

## Usage Examples

**Common patterns** at `src/weakincentives/runtime/dlq.py`:

- **Basic**: `DLQPolicy(mailbox=dead_letters, max_delivery_count=5)`
- **Immediate dead-lettering**: Use `include_errors` for errors that should
  never be retried (e.g., `ValidationError`, `ContentPolicyViolation`)
- **Exclude transient errors**: Use `exclude_errors` to prevent retryable
  errors (e.g., `RateLimitError`, `TimeoutError`) from dead-lettering
- **Custom policy**: Subclass `DLQPolicy` and override `should_dead_letter()`
  for error budget or context-aware logic

## Processing Dead Letters

**DLQ Consumer**: Poll the dead letter mailbox, invoke a handler per message,
acknowledge on success, nack with long backoff (e.g., 1 hour) on failure.

**Replay to Source**: Resolve the original `reply_to` mailbox from the
`DeadLetter.reply_to` name (if present) via a `MailboxResolver`, then
re-send `dead_letter.body` to the source mailbox.

**LoopGroup**: Run a `DLQConsumer` alongside `AgentLoop` within the same
`LoopGroup` to alert on dead-lettered messages.

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
- `specs/AGENT_LOOP.md` - AgentLoop orchestration
- `specs/LIFECYCLE.md` - LoopGroup coordination
