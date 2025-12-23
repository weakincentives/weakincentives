# Mailbox Specification

## Purpose

`Mailbox` provides a message queue abstraction with SQS-compatible semantics for
durable, at-least-once message delivery between processes. Unlike the pub/sub
`Dispatcher`, Mailbox delivers messages point-to-point with visibility timeout
and explicit acknowledgment.

**When to use Mailbox:**

- Durable request processing that survives process restarts
- Work distribution across multiple consumers
- Cross-process communication (distributed deployments)
- Tasks requiring acknowledgment and retry on failure

**When to use Dispatcher:**

- Telemetry and observability events
- In-process event notifications
- Fire-and-forget broadcasts to multiple subscribers

## Core Abstractions

### Mailbox

```python
class Mailbox(Protocol[T]):
    """Point-to-point message queue with visibility timeout semantics."""

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message, optionally delaying visibility.

        Args:
            body: Message payload (must be serializable).
            delay_seconds: Seconds before message becomes visible (0-900).

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded (backend-specific).
            SerializationError: Body cannot be serialized.
        """
        ...

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Receive messages from the queue.

        Received messages become invisible to other consumers for
        ``visibility_timeout`` seconds. Messages must be acknowledged
        before timeout expires or they return to the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration (0-20). Zero returns
                immediately; positive values block until messages arrive
                or timeout expires.

        Returns:
            Sequence of messages (may be empty if no messages available
            or long poll timed out).
        """
        ...

    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Approximate count of messages deleted.

        Note:
            SQS enforces 60-second cooldown between purges.
            Redis has no cooldown.
        """
        ...

    def approximate_count(self) -> int:
        """Return approximate number of messages in the queue.

        The count includes both visible and invisible messages.
        Value is eventually consistent (SQS ~1 minute lag, Redis exact).
        """
        ...
```

### Message

```python
@dataclass(frozen=True, slots=True)
class Message(Generic[T]):
    """A received message with delivery metadata and lifecycle methods."""

    id: str
    """Unique message identifier within the queue."""

    body: T
    """Deserialized message payload."""

    receipt_handle: str
    """Opaque handle for this specific delivery. Changes on each delivery.
    Required for acknowledge/nack/extend operations."""

    delivery_count: int
    """Number of times this message has been received. First delivery = 1.
    Use for dead-letter logic or debugging redelivery issues."""

    enqueued_at: datetime
    """Timestamp when message was originally sent (UTC)."""

    attributes: Mapping[str, str]
    """Backend-specific message attributes (e.g., SQS MessageAttributes)."""

    def acknowledge(self) -> bool:
        """Delete the message from the queue.

        Call after successfully processing the message. The receipt handle
        must still be valid (message not timed out or already acknowledged).

        Returns:
            True if acknowledged successfully, False if receipt handle expired.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        ...

    def nack(self, *, visibility_timeout: int = 0) -> bool:
        """Return message to queue immediately or after delay.

        Use when processing fails and the message should be retried.
        Setting ``visibility_timeout=0`` makes the message immediately
        visible to other consumers.

        Args:
            visibility_timeout: Seconds before message becomes visible again.

        Returns:
            True if nacked successfully, False if receipt handle expired.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        ...

    def extend_visibility(self, timeout: int) -> bool:
        """Extend the visibility timeout for long-running processing.

        Call periodically during long processing to prevent timeout.
        The new timeout is relative to now, not the original receive time.

        Args:
            timeout: New visibility timeout in seconds from now.

        Returns:
            True if extended successfully, False if receipt handle expired.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        ...
```

## Message Lifecycle

```
send() ──► Queued ──► receive() ──► Invisible ──► acknowledge() ──► Deleted
              │                          │
              │                          ▼
              │                     timeout expires
              │                          │
              ◄──────────────────────────┘
                    (redelivery)
              │
              │                     nack()
              │                       │
              ◄───────────────────────┘
                    (immediate or delayed redelivery)
```

**State definitions:**

- **Queued**: Message visible and available for receive.
- **Invisible**: Message received but not yet acknowledged. Other consumers
  cannot see it.
- **Deleted**: Message permanently removed after acknowledgment.

## SQS Semantics

The Mailbox abstraction mirrors AWS SQS behavior for portability:

| Mailbox Method | SQS API |
|----------------|---------|
| `send()` | `SendMessage` |
| `receive()` | `ReceiveMessage` |
| `acknowledge()` | `DeleteMessage` |
| `nack()` | `ChangeMessageVisibility` (to 0 or specified) |
| `extend_visibility()` | `ChangeMessageVisibility` |
| `purge()` | `PurgeQueue` |
| `approximate_count()` | `GetQueueAttributes` |

### Key Behaviors

**At-least-once delivery.** Messages may be delivered more than once. This
occurs when:

- Consumer crashes after receiving but before acknowledging
- Visibility timeout expires during processing
- Network issues cause duplicate sends

Consumers must be idempotent. Use `message.id` for deduplication if needed.

**Visibility timeout.** After `receive()`, the message becomes invisible to
other consumers for the specified duration. The message remains in the queue
until explicitly acknowledged. If the consumer fails to acknowledge before
timeout, the message becomes visible again and may be delivered to another
consumer.

**Long polling.** `wait_time_seconds > 0` blocks the receive call until:

- At least one message arrives, or
- The wait duration expires

Long polling reduces empty responses and API calls. Use 20 seconds (SQS maximum)
for efficient polling loops.

**Delay.** `delay_seconds` on send defers message visibility. The message is
queued immediately but remains invisible until the delay expires. Use for
scheduled tasks or rate limiting.

**Receipt handle.** Each delivery generates a unique receipt handle. Operations
(acknowledge, nack, extend) require the current handle. After timeout or
redelivery, the old handle becomes invalid.

**Ordering.** Ordering guarantees vary by backend:

- SQS Standard: Best-effort ordering (not guaranteed)
- SQS FIFO: Strict ordering within message groups
- Redis: FIFO guaranteed within a single queue
- InMemory: FIFO guaranteed

**Approximate count.** The count is eventually consistent. SQS may lag by ~1
minute. Redis provides exact counts. Use for monitoring, not precise control
flow.

## MainLoop Integration

`MainLoop` currently uses `ControlDispatcher` for request/response events
(see `src/weakincentives/runtime/main_loop.py:147`). For distributed deployments,
Mailbox provides durable message delivery with acknowledgment semantics.

### Pattern: Two Mailboxes

Use separate mailboxes for requests and responses:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT]],
        responses: Mailbox[MainLoopResult[OutputT]],
    ) -> None: ...
```

Request and response are correlated by `request_id`. This pattern supports:

- Multiple workers processing requests concurrently
- Client recovery after disconnect (poll response queue)
- Dead-letter handling for failed requests

### Request/Response Events

The existing `MainLoopRequest` dataclass (`src/weakincentives/runtime/main_loop.py:50-68`):

```python
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

The unified result type for all MainLoop responses:

```python
@dataclass(frozen=True, slots=True)
class MainLoopResult(Generic[OutputT]):
    """Response to a MainLoopRequest."""

    request_id: UUID
    """Correlates with MainLoopRequest.request_id."""

    response: PromptResponse[OutputT] | None = None
    """Present on success."""

    error: str | None = None
    """Error message on failure."""

    session_id: UUID | None = None
    """Session that processed the request (if available)."""

    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Worker Loop Pattern

```python
def run(self, *, max_iterations: int | None = None) -> None:
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        for msg in self._requests.receive(
            visibility_timeout=300,  # 5 min - must exceed max execution time
            wait_time_seconds=20,    # Long poll for efficiency
        ):
            try:
                response, session = self._execute(msg.body)
                self._responses.send(MainLoopResult(
                    request_id=msg.body.request_id,
                    response=response,
                    session_id=session.session_id,
                ))
                msg.acknowledge()
            except Exception as e:
                self._handle_failure(msg, e)
        iterations += 1
```

### Error Handling

```python
def _handle_failure(self, msg: Message[MainLoopRequest], error: Exception) -> None:
    try:
        # Attempt to send error response
        self._responses.send(MainLoopResult(
            request_id=msg.body.request_id,
            error=str(error),
        ))
        msg.acknowledge()  # Error response sent - don't retry
    except Exception:
        # Response send failed - nack for retry with backoff
        msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

**Error handling guidelines:**

1. **Set visibility_timeout higher than max execution time.** If processing
   takes up to 4 minutes, set visibility to 5+ minutes.

2. **Send response before acknowledging.** If you acknowledge first and crash
   before sending the response, the client never receives a reply.

3. **Use delivery_count for backoff.** Exponential backoff on failure prevents
   tight retry loops: `min(60 * delivery_count, 900)` gives 1, 2, 3... up to
   15 minute delays.

4. **Consider dead-letter handling.** After N retries (e.g., `delivery_count > 5`),
   acknowledge the message and log for manual intervention instead of infinite
   retry.

5. **Extend visibility for long tasks.** For processing that may exceed the
   visibility timeout, spawn a background task to call `extend_visibility()`
   periodically.

### Client Usage

```python
# Send request and wait for response
request_id = uuid4()
requests.send(MainLoopRequest(
    request=my_request,
    request_id=request_id,
))

# Poll for response
deadline = time.time() + timeout_seconds
while time.time() < deadline:
    for msg in responses.receive(wait_time_seconds=5):
        if msg.body.request_id == request_id:
            msg.acknowledge()
            if msg.body.error:
                raise RemoteError(msg.body.error)
            return msg.body.response
        # Not our message - return to queue for other clients
        msg.nack(visibility_timeout=0)
raise TimeoutError(f"No response within {timeout_seconds}s")
```

### Response Queue Strategies

**Shared queue with filtering:**

All clients poll a single response queue and filter by `request_id`. Simple but
causes contention under high concurrency. Unmatched messages are nacked
immediately with `visibility_timeout=0`.

**Per-client response queues:**

Each client creates a dedicated response queue (e.g., `responses:{client_id}`).
Workers include `reply_to` in requests. No filtering needed. Better for
high-throughput but requires queue lifecycle management.

## Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemoryMailbox` | Dict + deque | Testing, single process |
| `RedisMailbox` | Redis lists + sorted sets | Multi-process, self-hosted |
| `SQSMailbox` | AWS SQS | Production, managed infrastructure |

### InMemoryMailbox

Thread-safe implementation using Python primitives:

- `RLock` for thread synchronization
- `Condition` variables for long-poll blocking
- `deque` for message queue
- Background `Thread` for visibility timeout expiry

**Characteristics:**

- Messages lost on process restart
- No persistence
- Useful for tests and development
- Exact message counts

```python
mailbox: Mailbox[MyEvent] = InMemoryMailbox(name="events")
```

### RedisMailbox

Supports both standalone Redis and Redis Cluster deployments.

#### Data Structures

```
{queue:name}:pending    # LIST - messages awaiting delivery (LPUSH/RPOP)
{queue:name}:invisible  # ZSET - in-flight messages scored by expiry timestamp
{queue:name}:data       # HASH - message ID → serialized message data
{queue:name}:meta       # HASH - delivery counts and metadata
```

The `{queue:name}` prefix (with curly braces) is a Redis hash tag ensuring all
keys for a queue hash to the same slot. Required for Cluster mode; harmless in
standalone mode.

#### Operations

| Operation | Redis Commands |
|-----------|----------------|
| `send()` | `HSET data`, `LPUSH pending` |
| `receive()` | `BRPOP pending`, `ZADD invisible` |
| `acknowledge()` | `ZREM invisible`, `HDEL data`, `HDEL meta` |
| `nack()` | `ZREM invisible`, `LPUSH pending` |
| `extend_visibility()` | `ZADD invisible XX` (update score) |
| `purge()` | `DEL pending invisible data meta` |
| `approximate_count()` | `LLEN pending` + `ZCARD invisible` |

#### Visibility Reaper

Background task scans the `invisible` sorted set for expired entries:

```python
async def _reap_expired(self) -> None:
    while True:
        now = time.time()
        # Find messages with expiry score <= now
        expired = await self._redis.zrangebyscore(
            f"{{{self._name}}}:invisible",
            "-inf",
            now,
            start=0,
            num=100,
        )
        for msg_id in expired:
            # Atomic: remove from invisible, push to pending
            await self._requeue(msg_id)
        await asyncio.sleep(1)
```

#### Atomic Operations with Lua

Multi-key operations use Lua scripts for atomicity:

```lua
-- Atomic receive: RPOP from pending, ZADD to invisible
local msg_id = redis.call('RPOP', KEYS[1])  -- pending
if not msg_id then return nil end
redis.call('ZADD', KEYS[2], ARGV[1], msg_id)  -- invisible with expiry score
local data = redis.call('HGET', KEYS[3], msg_id)  -- data
local count = redis.call('HINCRBY', KEYS[4], msg_id, 1)  -- meta: increment delivery count
return {msg_id, data, count}
```

#### Standalone vs Cluster

| Behavior | Standalone | Cluster |
|----------|------------|---------|
| Key distribution | Single node | Sharded by hash slot |
| Lua scripts | Any keys | Same-slot keys only |
| BRPOP | Works normally | Works (single list) |
| Ordering | FIFO guaranteed | FIFO per queue |
| Failover | Manual or Sentinel | Automatic |

#### Redis Cluster Requirements

**Hash tags are mandatory.** All keys for a queue must include the same hash
tag (e.g., `{queue:myqueue}`) to ensure co-location on the same shard:

```python
# Correct - all keys hash to same slot
"{queue:requests}:pending"
"{queue:requests}:invisible"
"{queue:requests}:data"

# Wrong - keys may land on different shards
"queue:requests:pending"
"queue:requests:invisible"
```

Without hash tags, multi-key Lua scripts fail with `CROSSSLOT` error.

**No cross-queue atomicity.** Operations spanning multiple queues (different
hash tags) cannot be atomic. Each queue operates independently. If you need
atomic operations across queues, use a single queue with message type field.

**Replication lag.** During primary failover, recently written messages may be
lost if not yet replicated to the new primary. For stronger durability:

```
min-replicas-to-write 1
min-replicas-max-lag 10
```

This blocks writes until at least one replica acknowledges.

**Slot migration.** Adding/removing shards triggers slot migration. In-flight
messages remain safe but `READONLY` errors may occur briefly. Retry with backoff.

#### Configuration

```python
from redis import Redis
from redis.cluster import RedisCluster

# Standalone
mailbox = RedisMailbox(
    name="requests",
    client=Redis(host="localhost", port=6379),
)

# Cluster
mailbox = RedisMailbox(
    name="requests",
    client=RedisCluster(host="localhost", port=7000),
)
```

`RedisMailbox` auto-detects cluster mode from the client type.

### SQSMailbox

Direct mapping to SQS API. No additional data structures.

```python
import boto3

sqs = boto3.client("sqs")
mailbox = SQSMailbox(
    queue_url="https://sqs.us-east-1.amazonaws.com/123456789/my-queue",
    client=sqs,
)
```

**SQS-specific considerations:**

- Create queue via AWS console, CloudFormation, or Terraform
- Use Standard queues for throughput, FIFO for ordering
- Configure dead-letter queue in AWS for failed messages
- MessageAttributes map to `Message.attributes`

## Backend Semantic Differences

| Aspect | SQS Standard | SQS FIFO | Redis | InMemory |
|--------|--------------|----------|-------|----------|
| **Ordering** | Best-effort | Strict (per group) | FIFO | FIFO |
| **Deduplication** | None | 5-min window | None | None |
| **Retention** | 4 days (1-14) | 4 days (1-14) | Until ack | Until ack |
| **Max message size** | 256 KB | 256 KB | Redis maxmemory | Unlimited |
| **Long poll max** | 20 seconds | 20 seconds | Unlimited (BRPOP) | Unlimited |
| **Visibility max** | 12 hours | 12 hours | Unlimited | Unlimited |
| **Approximate count lag** | ~1 minute | ~1 minute | Exact | Exact |
| **Purge cooldown** | 60 seconds | 60 seconds | None | None |
| **Durability** | Replicated | Replicated | Configurable | None |

### Choosing a Backend

**Use InMemoryMailbox when:**

- Writing tests
- Single-process development
- Messages can be lost on restart

**Use RedisMailbox when:**

- Multi-process deployment on same network
- Self-hosted infrastructure
- Need FIFO ordering guarantees
- Sub-millisecond latency required

**Use SQSMailbox when:**

- AWS infrastructure
- Managed service preferred
- Cross-region durability needed
- Integration with AWS services (Lambda, SNS)

## Concurrency

### Multiple Consumers

Multiple processes can safely poll the same request queue:

```
                    ┌─────────────┐
                    │   Mailbox   │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐     ┌─────────┐
      │ Worker  │     │ Worker  │     │ Worker  │
      │   #1    │     │   #2    │     │   #3    │
      └─────────┘     └─────────┘     └─────────┘
```

Visibility timeout ensures each message is processed by exactly one consumer at
a time. If a consumer crashes, the message returns to the queue after timeout.

### Thread Safety

All implementations are safe for concurrent access from multiple threads:

- `InMemoryMailbox`: Thread-safe via `RLock`
- `RedisMailbox`: Thread-safe (Redis operations are atomic)
- `SQSMailbox`: Thread-safe (SQS handles concurrency server-side)

A single `Mailbox` instance can be shared across threads.

### Async Support

Mailbox implementations provide both sync and async interfaces:

```python
# Sync
messages = mailbox.receive(wait_time_seconds=20)

# Async
messages = await mailbox.receive_async(wait_time_seconds=20)
```

For async Redis, use `redis.asyncio.Redis` as the client.

### Performance Tuning

**Batch receive.** Increase `max_messages` to reduce round-trips:

```python
for msg in mailbox.receive(max_messages=10, wait_time_seconds=20):
    process(msg)
    msg.acknowledge()
```

**Pipeline acknowledgments.** With Redis, use pipelining for batch acks:

```python
messages = mailbox.receive(max_messages=10)
# Process all messages first
for msg in messages:
    process(msg)
# Then ack in batch (internal pipelining)
for msg in messages:
    msg.acknowledge()
```

**Monitor invisible queue.** High `invisible` set size indicates stuck messages
(slow consumers or crashes). Alert when size exceeds threshold.

## Errors

```python
class MailboxError(WinkError):
    """Base class for mailbox-related errors."""

class ReceiptHandleExpiredError(MailboxError):
    """Receipt handle no longer valid.

    Raised when acknowledge, nack, or extend_visibility is called
    with an expired receipt handle. This occurs when:
    - Visibility timeout expired before operation
    - Message was already acknowledged
    - Message was redelivered (new receipt handle issued)
    """

class MailboxFullError(MailboxError):
    """Queue capacity exceeded.

    SQS: 120,000 in-flight messages (standard) or 20,000 (FIFO)
    Redis: Depends on maxmemory configuration
    InMemory: Configurable max_size parameter
    """

class SerializationError(MailboxError):
    """Message body cannot be serialized or deserialized.

    The body must be JSON-serializable via the standard serde module.
    Complex objects should use FrozenDataclass for automatic serialization.
    """

class MailboxConnectionError(MailboxError):
    """Cannot connect to backend.

    Redis: Connection refused, timeout, authentication failure
    SQS: AWS credentials invalid, network unreachable
    """
```

## Testing Utilities

### NullMailbox

Drops all messages on send. Returns empty on receive. For tests where you need
a Mailbox interface but don't care about message delivery.

```python
mailbox: Mailbox[Event] = NullMailbox()
mailbox.send(Event(...))  # Silently dropped
assert mailbox.receive() == []
assert mailbox.approximate_count() == 0
```

### CollectingMailbox

Stores all sent messages for inspection. Useful for asserting what was sent.

```python
mailbox: CollectingMailbox[Event] = CollectingMailbox()
mailbox.send(Event(type="a"))
mailbox.send(Event(type="b"))

assert len(mailbox.sent) == 2
assert mailbox.sent[0].type == "a"
```

### FakeMailbox

Full in-memory implementation with controllable behavior for testing edge cases.

```python
mailbox: FakeMailbox[Event] = FakeMailbox()

# Simulate receipt handle expiry
msg = mailbox.receive()[0]
mailbox.expire_handle(msg.receipt_handle)
with pytest.raises(ReceiptHandleExpiredError):
    msg.acknowledge()

# Simulate connection failure
mailbox.set_connection_error(MailboxConnectionError("Redis down"))
with pytest.raises(MailboxConnectionError):
    mailbox.send(Event(...))
```

### Testing MainLoop with Mailbox

```python
def test_mainloop_processes_request():
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

## Mailbox vs Dispatcher

| Aspect | Dispatcher | Mailbox |
|--------|------------|---------|
| **Pattern** | Pub/sub broadcast | Point-to-point |
| **Delivery** | All subscribers | One consumer |
| **Acknowledgment** | None | Required |
| **Durability** | In-memory only | Backend-dependent |
| **Cross-process** | No | Yes |
| **Ordering** | Synchronous dispatch | Backend-dependent |
| **Use case** | Events, telemetry | Request processing |

Both coexist in a typical deployment:

- **Dispatcher**: Session events, logging, metrics, in-process coordination
- **Mailbox**: Request queue, response delivery, distributed work

See `ControlDispatcher` in `src/weakincentives/runtime/events/_types.py` for the
dispatcher interface.

## Limitations

**At-least-once delivery only.** No exactly-once semantics. Consumers must be
idempotent or implement their own deduplication.

**Ordering varies by backend.** SQS Standard provides best-effort only. For
strict ordering, use SQS FIFO or Redis.

**No message grouping.** Unlike SQS FIFO message groups, messages are processed
in queue order without partitioning. Use separate queues for independent
processing streams.

**No built-in dead-letter queue.** Implement at application level using
`delivery_count`:

```python
if msg.delivery_count > 5:
    log.error("Message exceeded retry limit", msg_id=msg.id)
    dead_letter_mailbox.send(msg.body)
    msg.acknowledge()
```

**No message deduplication.** Duplicate sends result in duplicate messages.
Deduplicate at the application level if needed.

**Single consumer per message.** No competing consumer groups or partitions.
For parallel processing, use multiple queues or implement sharding.

**No transactions.** Send and receive are independent operations. Use sagas or
compensating transactions for distributed workflows.
