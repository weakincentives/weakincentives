# Mailbox Specification

## Purpose

`Mailbox[T, R]` provides typed message queue with SQS-compatible semantics:
point-to-point delivery, visibility timeout, explicit acknowledgment.
Core at `runtime/mailbox/`.

**Use Mailbox for:** Durable processing, work distribution, cross-process communication.
**Use Dispatcher for:** Telemetry, in-process events, fire-and-forget.

## Core Types

### Mailbox Protocol

At `runtime/mailbox/protocol.py`:

| Method | Description |
| --- | --- |
| `name` | Unique identifier |
| `send(body, reply_to)` | Enqueue message, return ID |
| `receive(max_messages, visibility_timeout, wait_time_seconds)` | Receive messages |
| `purge()` | Delete all messages |
| `approximate_count()` | Approximate queue depth |
| `close()` | Release resources |
| `closed` | True if closed |

### Message

| Field | Description |
| --- | --- |
| `id` | Message identifier |
| `body` | Typed payload |
| `receipt_handle` | Current delivery handle |
| `delivery_count` | Delivery attempts |
| `enqueued_at` | Enqueue timestamp |
| `reply_to` | Reply mailbox reference |

| Method | Description |
| --- | --- |
| `reply(body)` | Send to reply_to mailbox |
| `acknowledge()` | Delete message (finalizes) |
| `nack(visibility_timeout)` | Return to queue (finalizes) |
| `extend_visibility(timeout)` | Extend invisibility |
| `is_finalized` | True if ack'd or nack'd |

### Errors

| Error | Description |
| --- | --- |
| `ReceiptHandleExpiredError` | Handle no longer valid |
| `MailboxFullError` | Queue capacity exceeded |
| `SerializationError` | Cannot serialize body |
| `MailboxConnectionError` | Backend unreachable |
| `ReplyNotAvailableError` | Cannot resolve reply_to |
| `MessageFinalizedError` | Already ack'd/nack'd |

## Message Lifecycle

```
send() → Queued → receive() → Invisible → acknowledge() → Deleted
                                 ↓
                         timeout expires → Redelivery
                                 ↓
                         nack() → Delayed Redelivery
```

**At-least-once delivery.** Consumers must be idempotent.

## Reply Pattern

Workers reply via `msg.reply()` which sends to `reply_to` mailbox. Multiple
replies allowed before finalization. After `acknowledge()` or `nack()`,
`reply()` raises `MessageFinalizedError`.

### Example

```python
# Client
requests.send(MainLoopRequest(request=data), reply_to=responses)

# Worker
for msg in requests.receive(visibility_timeout=300):
    result = process(msg.body)
    msg.reply(result)
    msg.acknowledge()
```

## MainLoop Integration

MainLoop receives requests mailbox; response routing derives from `reply_to`.

### Request/Response Types

| Type | Fields |
| --- | --- |
| `MainLoopRequest[T]` | request, budget, deadline, request_id, created_at |
| `MainLoopResult[T]` | request_id, output, error, session_id, completed_at |

### Error Handling Guidelines

1. `visibility_timeout` > max execution time
2. Send response before acknowledging
3. Use `delivery_count` for backoff: `min(60 * delivery_count, 900)`
4. Dead-letter after N retries: `delivery_count > 5` → log and acknowledge

## Implementations

| Implementation | Backend | Use Case |
| --- | --- | --- |
| `InMemoryMailbox` | Dict | Testing, single process |
| `RedisMailbox` | Redis | Multi-process, self-hosted |
| `SQSMailbox` | AWS SQS | Production, managed |

### Backend Comparison

| Aspect | SQS Standard | SQS FIFO | Redis | InMemory |
| --- | --- | --- | --- | --- |
| Ordering | Best-effort | Strict | FIFO | FIFO |
| Long poll max | 20s | 20s | ∞ | ∞ |
| Visibility max | 12h | 12h | ∞ | ∞ |
| Count accuracy | ~1min lag | ~1min | Exact | Exact |
| Durability | Replicated | Replicated | Config | None |

### RedisMailbox Data Structures

```
{queue:name}:pending    # LIST - awaiting delivery
{queue:name}:invisible  # ZSET - in-flight, scored by expiry
{queue:name}:data       # HASH - message ID → payload
{queue:name}:meta       # HASH - delivery counts
```

Hash tags ensure co-location in Redis Cluster.

## Test Utilities

| Utility | Purpose |
| --- | --- |
| `NullMailbox` | Drops all messages |
| `CollectingMailbox` | Stores sent messages for inspection |
| `FakeMailbox` | Controllable behavior for edge cases |

## Limitations

- **At-least-once only**: No exactly-once; consumers must be idempotent
- **Ordering varies**: SQS Standard best-effort; use FIFO or Redis for strict
- **No built-in DLQ**: Configure via `DLQConfig`; see `specs/DLQ.md`
- **No deduplication**: Handle at application level
- **No transactions**: Send and receive independent

## Related Specifications

- `specs/DLQ.md` - Dead letter queue configuration
- `specs/MAILBOX_RESOLVER.md` - Service discovery
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/RESOURCE_REGISTRY.md` - DI container
