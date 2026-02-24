# Mailbox Specification

## Purpose

`Mailbox[T, R]` provides typed message queue with SQS-compatible semantics:
point-to-point delivery, visibility timeout, explicit acknowledgment.
Core at `src/weakincentives/runtime/mailbox/`.

**Use Mailbox for:** Durable processing, work distribution, cross-process communication.
**Use Dispatcher for:** Telemetry, in-process events, fire-and-forget.

## Core Types

### Mailbox Protocol

At `src/weakincentives/runtime/mailbox/_types.py`:

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
| `MailboxResolutionError` | Cannot resolve identifier |
| `InvalidParameterError` | Timeout parameters out of range |

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

### In-Memory Implementation

Direct reference stored—no resolution needed. `reply_to` receives a direct
reference to the responses mailbox. See
`src/weakincentives/runtime/mailbox/_in_memory.py`.

### Redis Implementation

`reply_to` serializes to mailbox name. `MailboxResolver` reconstructs the
instance on the worker side. RedisMailbox is in `contrib`:
`from weakincentives.contrib.mailbox import RedisMailbox, RedisMailboxFactory`.

## Service Discovery: MailboxResolver

**Implementation:** `src/weakincentives/runtime/mailbox/_resolver.py`

**Use for:** Redis/distributed backends where mailbox names are serialized.
**Not for:** In-memory mailboxes (store direct references).

### MailboxResolver Protocol

| Method | Description |
|--------|-------------|
| `resolve(identifier)` | Resolve to mailbox, raises on failure |
| `resolve_optional(identifier)` | Resolve or return None |

### Resolver Types

| Type | Description |
|------|-------------|
| `CompositeResolver` | Registry first, then factory fallback |
| `RegistryResolver` | Static registry only |

### MailboxFactory Protocol

`create(identifier: str) -> Mailbox[R, None]` — creates a new mailbox for the
given identifier.

**InMemoryMailboxFactory** at `src/weakincentives/runtime/mailbox/_in_memory.py`:
returns `InMemoryMailbox(name=identifier)`.

**RedisMailboxFactory** at `src/weakincentives/contrib/mailbox/_redis.py`:
accepts `client`, `body_type`, and `default_ttl` (3 days); returns a configured
`RedisMailbox`.

### Dynamic Reply Queues

Clients create a unique per-request reply mailbox (e.g., `f"client-{uuid4()}"`),
pass it as `reply_to` on send, and the resolver factory creates the matching
mailbox on the worker side.

### Multi-Tenant

Use queue naming conventions for tenant isolation (e.g.,
`f"tenant-{tenant_id}:requests"`) with a shared `CompositeResolver` and
`RedisMailboxFactory`.

### Comparison with ResourceResolver

| Aspect | ResourceResolver | MailboxResolver |
|--------|------------------|-----------------|
| Key type | `type[T]` | `str` |
| Purpose | DI container | Service discovery |
| Resolution | Static bindings | Dynamic lookup + factory |
| Caching | Scope-aware | Optional |

## AgentLoop Integration

AgentLoop receives requests mailbox; response routing derives from `reply_to`.

### Request/Response Types

| Type | Fields |
| --- | --- |
| `AgentLoopRequest[T]` | request, budget, deadline, request_id, created_at |
| `AgentLoopResult[T]` | request_id, output, error, session_id, completed_at |

### Error Handling Guidelines

1. `visibility_timeout` > max execution time
1. Send response before acknowledging
1. Use `delivery_count` for backoff: `min(60 * delivery_count, 900)`
1. Dead-letter after N retries: `delivery_count > 5` → log and acknowledge

## Implementations

| Implementation | Location | Backend | Use Case |
| --- | --- | --- | --- |
| `InMemoryMailbox` | `runtime/mailbox/` | Dict | Testing, single process |
| `InMemoryMailboxFactory` | `runtime/mailbox/` | Dict | Reply routing factory |
| `RedisMailbox` | `contrib/mailbox/` | Redis | Multi-process, self-hosted |
| `RedisMailboxFactory` | `contrib/mailbox/` | Redis | Reply routing factory |

### Backend Comparison

| Aspect | Redis | InMemory |
| --- | --- | --- |
| Ordering | FIFO | FIFO |
| Long poll max | Unlimited | Unlimited |
| Visibility max | 43200s (12h) | 43200s (12h) |
| Count accuracy | Exact | Exact |
| Durability | Config | None |
| Reply resolution | Resolver | Direct reference |

### RedisMailbox Data Structures

```
{queue:name}:pending    # LIST - awaiting delivery
{queue:name}:invisible  # ZSET - in-flight, scored by expiry
{queue:name}:data       # HASH - message ID → payload
{queue:name}:meta       # HASH - delivery counts
```

Hash tags ensure co-location in Redis Cluster.

## Test Utilities

All at `src/weakincentives/runtime/mailbox/_testing.py`:

| Utility | Purpose |
| --- | --- |
| `NullMailbox` | Drops all messages on send, returns empty on receive |
| `CollectingMailbox` | Stores sent messages in `.sent` list for inspection |
| `FakeMailbox` | Full in-memory impl with `expire_handle()`, `set_connection_error()`, `inject_message()` |

## Limitations

- **At-least-once only**: No exactly-once; consumers must be idempotent
- **FIFO ordering only**: Both implementations provide strict ordering
- **No built-in DLQ**: See `specs/DLQ.md` for dead letter queue design
- **No deduplication**: Handle at application level
- **No transactions**: Send and receive independent
- **Resolver per-type**: Use separate resolvers for heterogeneous mailboxes
- **No resolver caching**: Factory creates new mailbox each time

## Related Specifications

- `specs/DLQ.md` - Dead letter queue configuration
- `specs/AGENT_LOOP.md` - AgentLoop orchestration
- `specs/RESOURCE_REGISTRY.md` - DI container
