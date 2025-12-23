# Mailbox Specification

## Purpose

`Mailbox` provides a message queue abstraction with SQS-compatible semantics.
Unlike the pub/sub `Dispatcher`, Mailbox delivers messages point-to-point with
visibility timeout and acknowledgment.

For request/response patterns, use two Mailbox instances: one for requests, one
for responses. Correlate by message ID.

## Core Abstractions

### Mailbox

```python
class Mailbox(Protocol[T]):
    def send(self, body: T, *, delay_seconds: int = 0) -> str: ...
    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]: ...
    def purge(self) -> int: ...
    def approximate_count(self) -> int: ...
```

### Message

```python
@dataclass(frozen=True, slots=True)
class Message[T]:
    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    attributes: Mapping[str, str]

    def acknowledge(self) -> bool: ...
    def nack(self, *, visibility_timeout: int = 0) -> bool: ...
    def extend_visibility(self, timeout: int) -> bool: ...
```

## Lifecycle

```
send() → Queued → receive() → Invisible → acknowledge() → Deleted
                                  ↓
                             nack() or timeout → Queued (redelivery)
```

## SQS Semantics

The Mailbox abstraction mirrors AWS SQS behavior:

- `send()` → `SendMessage`
- `receive()` → `ReceiveMessage`
- `acknowledge()` → `DeleteMessage`
- `extend_visibility()` / `nack()` → `ChangeMessageVisibility`
- `purge()` → `PurgeQueue`
- `approximate_count()` → `GetQueueAttributes`

### Key Behaviors

1. **At-least-once delivery**: Messages may be delivered more than once.
   Consumers must be idempotent.

1. **Visibility timeout**: After `receive()`, the message becomes invisible to
   other consumers. If not acknowledged before timeout, the message returns to
   the queue.

1. **Long polling**: `wait_time_seconds > 0` enables long polling. The call
   blocks until messages arrive or timeout expires.

1. **Delay**: `delay_seconds` defers message visibility after send.

1. **Receipt handle**: Each delivery generates a unique receipt handle.
   Operations require the current handle; stale handles fail.

1. **Ordering varies by backend**: SQS provides best-effort ordering only. Redis
   guarantees FIFO within a single queue. See backend-specific sections below.

1. **Approximate count**: The count is eventually consistent, not exact.

## MainLoop Integration

MainLoop uses two Mailbox instances for request/response communication:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT]],
        responses: Mailbox[MainLoopResult[OutputT]],
    ) -> None: ...

    def run(self, *, max_iterations: int | None = None) -> None:
        while should_continue:
            for msg in self._requests.receive(
                visibility_timeout=300,  # Must exceed max execution time
                wait_time_seconds=20,
            ):
                try:
                    result = self._execute(msg.body)
                    self._responses.send(MainLoopResult(
                        request_id=msg.body.request_id,
                        response=result,
                    ))
                    msg.acknowledge()
                except Exception as e:
                    try:
                        self._responses.send(MainLoopResult(
                            request_id=msg.body.request_id,
                            error=e,
                        ))
                        msg.acknowledge()
                    except Exception:
                        # Response send failed - requeue for retry
                        msg.nack(visibility_timeout=60)
```

**Error handling notes:**

- Set `visibility_timeout` higher than maximum expected execution time
- Send response before acknowledging to avoid lost responses on crash
- If response send fails, nack the request to retry later
- For long-running tasks, call `msg.extend_visibility()` periodically

### Request/Response Events

```python
@dataclass(frozen=True, slots=True)
class MainLoopRequest(Generic[UserRequestT]):
    request: UserRequestT
    request_id: UUID = field(default_factory=uuid4)
    budget: Budget | None = None
    deadline: Deadline | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class MainLoopResult(Generic[OutputT]):
    request_id: UUID
    response: PromptResponse[OutputT] | None = None
    error: Exception | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Client Usage

```python
# Send request
request_id = uuid4()
requests.send(MainLoopRequest(
    request=my_request,
    request_id=request_id,
))

# Poll for response with timeout
deadline = time.time() + timeout_seconds
while time.time() < deadline:
    for msg in responses.receive(wait_time_seconds=5):
        if msg.body.request_id == request_id:
            msg.acknowledge()
            return msg.body
        # Not our message - return immediately for other consumers
        msg.nack(visibility_timeout=0)
raise TimeoutError(f"No response within {timeout_seconds}s")
```

**Response queue strategies:**

- **Shared queue with filtering**: Multiple clients poll the same queue and
  filter by `request_id`. Unmatched messages are nacked immediately. Simple but
  causes contention under high concurrency.

- **Per-client response queues**: Each client uses a dedicated response queue.
  No filtering needed. Better for high-throughput scenarios but requires queue
  lifecycle management.

## Implementations

| Implementation | Backend |
|----------------|---------|
| `InMemoryMailbox` | Dict + deque |
| `RedisMailbox` | Redis lists + sorted sets |
| `SQSMailbox` | AWS SQS |

### InMemory

- Thread-safe with `RLock`
- Condition variables for long-poll blocking
- Background thread for visibility timeout expiry
- Useful for testing and single-process deployments

### Redis

Supports both standalone Redis and Redis Cluster deployments.

#### Data Structures

```
{queue:name}:pending    # LIST - pending messages (LPUSH/RPOP)
{queue:name}:invisible  # SORTED SET - invisible messages by expiry time
{queue:name}:data       # HASH - message ID → serialized message
```

The `{queue:name}` hash tag ensures all keys for a queue hash to the same slot,
required for Cluster mode and harmless in standalone mode.

#### Operations

- `send()` → `HSET` data + `LPUSH` pending
- `receive()` → `BRPOP` pending + `ZADD` invisible
- `acknowledge()` → `ZREM` invisible + `HDEL` data
- `nack()` → `ZREM` invisible + `LPUSH` pending
- `extend_visibility()` → `ZADD` invisible (update score)

**Visibility reaper:** Background task scans `invisible` sorted set for expired
entries and moves them back to `pending`.

#### Standalone vs Cluster

| Behavior | Standalone | Cluster |
|----------|------------|---------|
| Key distribution | Single node | Sharded by hash slot |
| Lua scripts | Any keys | Same-slot keys only |
| BRPOP | Single list | Single list per slot |
| Ordering | FIFO guaranteed | FIFO per queue only |
| Failover | Manual or Sentinel | Automatic |

#### Redis Cluster Requirements

1. **Hash tags mandatory**: All keys for a queue must include the same hash tag
   (e.g., `{queue:name}`) to ensure co-location on the same shard. Without
   hash tags, multi-key operations fail with `CROSSSLOT` error.

1. **Lua script constraints**: Scripts that touch multiple keys only work when
   all keys hash to the same slot. The hash tag pattern ensures this.

1. **No cross-queue atomicity**: Operations spanning multiple queues (different
   hash tags) cannot be atomic. Each queue operates independently.

1. **BRPOP behavior**: In Cluster mode, `BRPOP` works normally since it
   operates on a single list. The client library handles slot routing.

1. **Replication lag**: During failover, recently written messages on the old
   primary may be lost if not yet replicated. Configure `min-replicas-to-write`
   for stronger durability guarantees.

1. **Scaling**: Adding/removing shards triggers slot migration. In-flight
   messages remain safe but `READONLY` errors may occur briefly during
   migration. Retry with backoff.

#### Configuration

```python
# Standalone
RedisMailbox(
    name="requests",
    client=Redis(host="localhost", port=6379),
)

# Cluster
RedisMailbox(
    name="requests",
    client=RedisCluster(host="localhost", port=7000),
)
```

The `RedisMailbox` implementation auto-detects cluster mode and adjusts
behavior accordingly. No code changes required beyond the client type.

### SQS

Direct mapping to SQS API. No additional data structures needed.

## SQS vs Redis: Semantic Differences

**Message retention:**

- SQS: 4 days default (configurable 1-14 days)
- Redis: Until acknowledged or Redis eviction

**Max message size:**

- SQS: 256 KB
- Redis: Limited by Redis maxmemory

**Long poll max:**

- SQS: 20 seconds
- Redis: Unlimited (BRPOP)

**Visibility timeout max:**

- SQS: 12 hours
- Redis: Unlimited

**Approximate count:**

- SQS: Eventually consistent (~1 min lag)
- Redis: Exact (LLEN + ZCARD)

**Purge cooldown:**

- SQS: 60 seconds between purges
- Redis: None

**Ordering:**

- SQS: Best-effort (not guaranteed)
- Redis standalone: FIFO guaranteed
- Redis Cluster: FIFO per queue (each queue on one shard)

**Message deduplication:**

- SQS: None (standard queue)
- Redis: None

### Redis-Specific Considerations

1. **No native TTL per message**: Message expiry requires application-level
   reaper. Use sorted set scores for expiry timestamps.

1. **Atomic visibility**: Lua scripts ensure atomic receive (RPOP + ZADD) and
   acknowledge (ZREM + HDEL). Works in Cluster mode when keys share a hash tag.

1. **Persistence**: Redis persistence (RDB/AOF) affects durability. With no
   persistence, messages lost on restart. In Cluster mode, also configure
   `min-replicas-to-write` for cross-replica durability.

## Concurrency

### Multiple Consumers

Multiple processes can safely poll the same request queue. Visibility timeout
ensures each message is processed by exactly one consumer at a time. If a
consumer crashes, the message returns to the queue after timeout.

### Thread Safety

All implementations are safe for concurrent access:

- `InMemoryMailbox`: Thread-safe via `RLock`
- `RedisMailbox`: Thread-safe (Redis operations are atomic)
- `SQSMailbox`: Thread-safe (SQS handles concurrency server-side)

Multiple threads can call `receive()` concurrently on the same `Mailbox`
instance.

### Performance

For high-throughput scenarios:

- Increase `max_messages` in `receive()` to reduce round-trips
- With Redis, operations use pipelining internally where possible
- Monitor `invisible` set size to detect stuck messages (consumer crashes)
- Consider per-client response queues to avoid filtering overhead

## Mailbox vs Dispatcher

| Aspect | Dispatcher | Mailbox |
|--------|------------|---------|
| Pattern | Pub/sub broadcast | Point-to-point |
| Delivery | All subscribers | One consumer |
| Acknowledgment | None | Required |
| Use case | Telemetry | Request processing |

Both coexist: `Dispatcher` for observability events, `Mailbox` for MainLoop
orchestration.

## Errors

```python
class MailboxError(WinkError): ...
class ReceiptHandleExpiredError(MailboxError): ...
class MailboxFullError(MailboxError): ...
class SerializationError(MailboxError): ...
```

## Testing Utilities

- `NullMailbox`: Drops all messages on send, returns empty on receive
- `CollectingMailbox`: Stores sent messages for assertion

### Testing Request/Response Patterns

```python
# Setup
requests = InMemoryMailbox[MainLoopRequest[MyRequest]]("requests")
responses = InMemoryMailbox[MainLoopResult[MyOutput]]("responses")
loop = MyLoop(adapter=adapter, requests=requests, responses=responses)

# Send request
request_id = uuid4()
requests.send(MainLoopRequest(request=MyRequest(...), request_id=request_id))

# Run single iteration
loop.run(max_iterations=1)

# Assert response
msgs = responses.receive()
assert len(msgs) == 1
assert msgs[0].body.request_id == request_id
assert msgs[0].body.error is None
msgs[0].acknowledge()
```

## Limitations

- At-least-once delivery only (all backends; no exactly-once semantics)
- Ordering varies by backend (SQS best-effort, Redis FIFO per queue)
- No message grouping or deduplication
- No dead-letter queue (implement at application level)
