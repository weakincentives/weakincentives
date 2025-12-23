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

1. **No ordering guarantee**: Messages may arrive out of order. Do not rely on
   FIFO semantics.

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
                visibility_timeout=300,
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
                    self._responses.send(MainLoopResult(
                        request_id=msg.body.request_id,
                        error=e,
                    ))
                    msg.acknowledge()
```

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

# Poll for response
while True:
    for msg in responses.receive(wait_time_seconds=5):
        if msg.body.request_id == request_id:
            result = msg.body
            msg.acknowledge()
            break
    else:
        continue
    break
```

For concurrent clients polling the same response queue, each client must filter
by `request_id`. Unrelated messages should be left unacknowledged to return to
the queue after visibility timeout.

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

Uses Redis data structures to approximate SQS semantics:

```
queue:{name}:pending    # LIST - pending messages (LPUSH/RPOP)
queue:{name}:invisible  # SORTED SET - invisible messages by expiry time
queue:{name}:data       # HASH - message ID → serialized message
```

**Operations:**

- `send()` → `HSET` data + `LPUSH` pending
- `receive()` → `BRPOP` pending + `ZADD` invisible
- `acknowledge()` → `ZREM` invisible + `HDEL` data
- `nack()` → `ZREM` invisible + `LPUSH` pending
- `extend_visibility()` → `ZADD` invisible (update score)

**Visibility reaper:** Background task scans `invisible` sorted set for expired
entries and moves them back to `pending`.

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
- Redis: FIFO within single instance

**Message deduplication:**

- SQS: None (standard queue)
- Redis: None

### Redis-Specific Considerations

1. **No native TTL per message**: Message expiry requires application-level
   reaper. Use sorted set scores for expiry timestamps.

1. **Atomic visibility**: Lua scripts ensure atomic receive (RPOP + ZADD) and
   acknowledge (ZREM + HDEL).

1. **Cluster mode**: With Redis Cluster, all keys for a queue must hash to the
   same slot. Use hash tags: `{queue:name}:pending`.

1. **Persistence**: Redis persistence (RDB/AOF) affects durability. With no
   persistence, messages lost on restart.

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

## Limitations

- At-least-once delivery only (no exactly-once)
- No FIFO ordering guarantee (except Redis single-instance)
- No message grouping or deduplication
- No dead-letter queue (implement at application level)
