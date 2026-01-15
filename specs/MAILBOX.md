# Mailbox Specification

Typed message queue with SQS-compatible semantics: point-to-point delivery,
visibility timeout, explicit acknowledgment.

**Source:** `src/weakincentives/runtime/mailbox/`

## When to Use

**Mailbox:** Durable processing, work distribution, cross-process, acknowledgment needed.

**Dispatcher:** Telemetry, in-process events, fire-and-forget broadcasts.

## Mailbox Protocol

**Definition:** `runtime/mailbox/_types.py:Mailbox`

| Method | Purpose |
|--------|---------|
| `send(body, reply_to=None)` | Enqueue, return message ID |
| `receive(...)` | Get messages (become invisible) |
| `purge()` | Delete all messages |
| `approximate_count()` | Queue depth (eventually consistent) |
| `close()` | Release resources |

## Message

**Definition:** `runtime/mailbox/_types.py:Message`

| Method | Purpose |
|--------|---------|
| `reply(body)` | Send to `reply_to` mailbox |
| `acknowledge()` | Delete (finalize) |
| `nack(visibility_timeout)` | Return to queue (finalize) |
| `extend_visibility(timeout)` | Extend processing time |

Multiple replies allowed before finalization.

## Lifecycle

```
send() → Queued → receive() → Invisible → acknowledge() → Deleted
                      │
                      └── timeout/nack ─────→ Redelivered
```

**At-least-once delivery.** Consumers must be idempotent.

## Reply Pattern

```python
# Client
requests.send(Request(...), reply_to=responses)
for msg in responses.receive(): handle(msg.body); msg.acknowledge()

# Worker
for msg in requests.receive():
    msg.reply(process(msg.body))
    msg.acknowledge()
```

## Errors

| Error | Cause |
|-------|-------|
| `ReceiptHandleExpiredError` | Handle no longer valid |
| `MailboxFullError` | Capacity exceeded |
| `MessageFinalizedError` | Already ack/nack'd |

## Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemoryMailbox` | Dict | Testing |
| `RedisMailbox` | Redis | Multi-process |
| `SQSMailbox` | AWS SQS | Production |

## Test Utilities

```python
NullMailbox()        # Drops all messages
CollectingMailbox()  # Stores for inspection
FakeMailbox()        # Controllable behavior
```

## Limitations

- At-least-once only (no exactly-once)
- Ordering varies by backend
- No built-in DLQ (use `DLQPolicy`)
- No deduplication (handle at app level)
