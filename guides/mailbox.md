# Mailbox

*Canonical spec: [specs/MAILBOX.md](../specs/MAILBOX.md)*

A `Mailbox` is WINK's message queue abstraction with a key design principle:

> **Durable, at-least-once delivery with explicit acknowledgment.**

Unlike the fire-and-forget `Dispatcher`, mailboxes guarantee that messages are
not lost: they remain in the queue until explicitly acknowledged. Visibility
timeout semantics (borrowed from AWS SQS) allow automatic retry when consumers
crash or stall.

## When to Use Mailbox

**Use Mailbox for:**

- Durable request processing that survives process restarts
- Work distribution across multiple consumers
- Cross-process communication (distributed deployments)
- Tasks requiring acknowledgment and retry on failure
- Request/response patterns with reply routing

**Use Dispatcher for:**

- Telemetry and observability events
- In-process event notifications
- Fire-and-forget broadcasts to multiple subscribers

Mental model: **Mailbox is a work queue; Dispatcher is a notification bus.**

## Core Concepts

### Message Lifecycle

```
send() → Queued → receive() → Invisible → acknowledge() → Deleted
                                 ↓
                         timeout expires → Redelivery
                                 ↓
                         nack() → Delayed Redelivery
```

When a consumer calls `receive()`, the message becomes *invisible* to other
consumers for a configurable timeout. The consumer must call `acknowledge()` to
permanently delete the message, or `nack()` to return it to the queue.

If the consumer crashes or the visibility timeout expires, the message
automatically becomes visible again for redelivery. This is **at-least-once
delivery**—consumers must be idempotent.

### The Message Type

A received message contains:

```python nocheck
@dataclass
class Message[T, R]:
    id: str                    # Unique within the queue
    body: T                    # Your typed payload
    receipt_handle: str        # Handle for this delivery
    delivery_count: int        # How many times delivered (starts at 1)
    enqueued_at: datetime      # When originally sent
    reply_to: Mailbox[R, None] | None  # For request/response patterns
```

**Lifecycle methods on Message:**

- `acknowledge()`: Delete the message (processing succeeded)
- `nack(visibility_timeout=0)`: Return to queue (processing failed)
- `extend_visibility(timeout)`: Keep working longer
- `reply(body)`: Send response to `reply_to` mailbox

### The Mailbox Protocol

```python nocheck
class Mailbox[T, R](Protocol):
    @property
    def name(self) -> str: ...
    @property
    def closed(self) -> bool: ...

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str: ...
    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]: ...
    def purge(self) -> int: ...
    def approximate_count(self) -> int: ...
    def close(self) -> None: ...
```

Type parameters:

- `T`: Message body type
- `R`: Reply type (use `None` if no replies expected)

## Basic Usage

### InMemoryMailbox

For development and testing, `InMemoryMailbox` provides the full mailbox
contract without external dependencies:

```python nocheck
from weakincentives.runtime.mailbox import InMemoryMailbox

# Create a mailbox for work items
work_queue: InMemoryMailbox[str, None] = InMemoryMailbox(name="work")

# Producer: send work
work_queue.send("task-1")
work_queue.send("task-2")

# Consumer: process work
for msg in work_queue.receive(max_messages=10, visibility_timeout=60):
    try:
        process(msg.body)
        msg.acknowledge()
    except Exception:
        msg.nack(visibility_timeout=30)  # Retry after 30 seconds
```

**Characteristics:**

- Thread-safe via internal locking
- FIFO ordering guaranteed
- Exact message counts
- No persistence (lost on process restart)
- Background reaper thread handles visibility timeouts

### Visibility Timeout

The `visibility_timeout` parameter controls how long a message stays invisible
after `receive()`. Choose based on your processing time:

```python nocheck
# Short tasks: 30 seconds (default)
messages = queue.receive(visibility_timeout=30)

# Long tasks: extend as you work
messages = queue.receive(visibility_timeout=300)
for msg in messages:
    # Periodically extend if processing takes time
    msg.extend_visibility(300)
    do_work(msg.body)
    msg.acknowledge()
```

Valid range: 0 to 43,200 seconds (12 hours).

### Long Polling

Use `wait_time_seconds` to avoid busy-waiting:

```python nocheck
# Block up to 20 seconds for messages
messages = queue.receive(wait_time_seconds=20)

# Returns immediately with whatever is available (or empty)
messages = queue.receive(wait_time_seconds=0)
```

## Request/Response Pattern

Mailboxes support request/response via the `reply_to` parameter:

```python nocheck
from dataclasses import dataclass
from weakincentives.runtime.mailbox import InMemoryMailbox

@dataclass(slots=True, frozen=True)
class Request:
    query: str

@dataclass(slots=True, frozen=True)
class Response:
    answer: str

# Create both mailboxes
requests: InMemoryMailbox[Request, Response] = InMemoryMailbox(name="requests")
responses: InMemoryMailbox[Response, None] = InMemoryMailbox(name="responses")

# Client: send request with reply destination
requests.send(Request(query="What is 2+2?"), reply_to=responses)

# Worker: process and reply
for msg in requests.receive(visibility_timeout=60):
    try:
        answer = compute(msg.body.query)
        msg.reply(Response(answer=answer))  # Sends to responses mailbox
        msg.acknowledge()
    except Exception:
        msg.nack(visibility_timeout=30)

# Client: receive response
for msg in responses.receive():
    print(msg.body.answer)
    msg.acknowledge()
```

**Key behaviors:**

- `reply()` sends directly to the `reply_to` mailbox
- Multiple replies allowed before `acknowledge()` or `nack()`
- After finalization, `reply()` raises `MessageFinalizedError`
- If no `reply_to` was set, `reply()` raises `ReplyNotAvailableError`

## Redis Implementation

For production deployments, `RedisMailbox` provides durable, distributed
message queuing:

```python nocheck
from redis import Redis
from weakincentives.contrib.mailbox import RedisMailbox

client = Redis(host="localhost", port=6379)

requests: RedisMailbox[Request, Response] = RedisMailbox(
    name="requests",
    client=client,
    body_type=Request,  # Optional: enables typed deserialization
)

try:
    requests.send(Request(query="hello"))
    for msg in requests.receive(visibility_timeout=60):
        process(msg.body)
        msg.acknowledge()
finally:
    requests.close()  # Stops reaper thread (doesn't close Redis client)
```

**Characteristics:**

- Durable (configurable Redis persistence)
- Supports Redis Cluster via hash tags
- Exact message counts
- Atomic operations via Lua scripts
- Uses Redis server TIME (no client clock skew)
- Auto-expiring keys (default 3 days TTL, refreshed on activity)

### Redis Data Structures

Each queue uses four Redis keys with hash tags for cluster compatibility:

```
{queue:name}:pending    # LIST - messages awaiting delivery
{queue:name}:invisible  # ZSET - in-flight, scored by expiry timestamp
{queue:name}:data       # HASH - message ID → serialized body
{queue:name}:meta       # HASH - delivery counts, timestamps, handles
```

### Reply Resolution in Redis

Unlike `InMemoryMailbox` which stores direct references, `RedisMailbox`
serializes `reply_to` as a queue name. A `MailboxResolver` reconstructs the
mailbox on receive:

```python nocheck
from weakincentives.contrib.mailbox import RedisMailbox, RedisMailboxFactory
from weakincentives.runtime.mailbox import CompositeResolver

# Factory creates mailboxes for any queue name
factory = RedisMailboxFactory(client=redis_client)
resolver = CompositeResolver(registry={}, factory=factory)

requests = RedisMailbox(
    name="requests",
    client=redis_client,
    reply_resolver=resolver,  # Resolves reply_to names to mailboxes
)

# If you don't specify a resolver, RedisMailbox creates one automatically
# using a RedisMailboxFactory with the same client
```

## Service Discovery: Resolvers

Resolvers map string identifiers to mailbox instances. This is essential for
distributed systems where mailbox references must be serialized.

### RegistryResolver

Static mapping for known mailboxes:

```python nocheck
from weakincentives.runtime.mailbox import RegistryResolver

registry = {"responses": responses_mailbox}
resolver = RegistryResolver(registry)

mailbox = resolver.resolve("responses")  # Returns responses_mailbox
resolver.resolve("unknown")  # Raises MailboxResolutionError
```

### CompositeResolver

Combines registry with dynamic factory fallback:

```python nocheck
from weakincentives.runtime.mailbox import CompositeResolver
from weakincentives.contrib.mailbox import RedisMailboxFactory

resolver = CompositeResolver(
    registry={"known": existing_mailbox},
    factory=RedisMailboxFactory(client=redis_client),
)

resolver.resolve("known")    # Returns existing_mailbox from registry
resolver.resolve("dynamic")  # Creates new RedisMailbox via factory
```

### Dynamic Reply Queues

Clients can create unique reply queues:

```python nocheck
from uuid import uuid4

# Each client gets its own reply queue
client_id = uuid4()
my_responses = RedisMailbox(
    name=f"client-{client_id}",
    client=redis_client,
)

requests.send(Request(...), reply_to=my_responses)

# Worker's resolver creates RedisMailbox for "client-{uuid}" via factory
```

## Error Handling

### Error Types

| Error | When Raised |
|-------|-------------|
| `ReceiptHandleExpiredError` | Handle expired (timeout, already acked, redelivered) |
| `MailboxFullError` | Queue capacity exceeded |
| `SerializationError` | Body cannot be serialized/deserialized |
| `MailboxConnectionError` | Cannot connect to backend |
| `ReplyNotAvailableError` | No `reply_to` on message |
| `MessageFinalizedError` | `reply()` after `acknowledge()`/`nack()` |
| `MailboxResolutionError` | Cannot resolve mailbox identifier |

### Handling Receipt Handle Expiration

Receipt handles expire when:

- Visibility timeout expires (message redelivered with new handle)
- Message was already acknowledged
- Message was redelivered to another consumer

```python nocheck
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

for msg in queue.receive(visibility_timeout=30):
    try:
        do_long_work(msg.body)
        msg.acknowledge()
    except ReceiptHandleExpiredError:
        # Message was redelivered; another consumer may have it
        # Our work was wasted, but the message isn't lost
        pass
```

### Dead Letter Logic

Use `delivery_count` to implement dead-letter patterns:

```python nocheck
MAX_DELIVERIES = 5

for msg in queue.receive():
    if msg.delivery_count > MAX_DELIVERIES:
        # Too many failures; log and discard
        log_dead_letter(msg)
        msg.acknowledge()
        continue

    try:
        process(msg.body)
        msg.acknowledge()
    except Exception as e:
        # Exponential backoff: 60s, 120s, 180s, ...
        delay = min(60 * msg.delivery_count, 900)
        msg.nack(visibility_timeout=delay)
```

## Testing Utilities

WINK provides mailbox implementations for different testing scenarios:

### NullMailbox

Drops all messages silently:

```python nocheck
from weakincentives.runtime.mailbox import NullMailbox

mailbox: NullMailbox[Event, None] = NullMailbox()
mailbox.send(Event(...))  # Silently dropped
assert mailbox.receive() == []
assert mailbox.approximate_count() == 0
```

### CollectingMailbox

Stores sent messages for inspection:

```python nocheck
from weakincentives.runtime.mailbox import CollectingMailbox

mailbox: CollectingMailbox[Event, None] = CollectingMailbox()
mailbox.send(Event(type="a"))
mailbox.send(Event(type="b"))

assert len(mailbox.sent) == 2
assert mailbox.sent[0].type == "a"
```

### FakeMailbox

Full implementation with controllable failures:

```python nocheck
from weakincentives.runtime.mailbox import FakeMailbox, MailboxConnectionError

mailbox: FakeMailbox[Event, None] = FakeMailbox()

# Simulate receipt handle expiry
msg = mailbox.receive()[0]
mailbox.expire_handle(msg.receipt_handle)
with pytest.raises(ReceiptHandleExpiredError):
    msg.acknowledge()

# Simulate connection failure
mailbox.set_connection_error(MailboxConnectionError("Redis down"))
with pytest.raises(MailboxConnectionError):
    mailbox.send(Event(...))

mailbox.clear_connection_error()  # Restore normal operation

# Inject messages directly for test setup
mailbox.inject_message(Event(...), delivery_count=3)
```

## MainLoop Integration

`MainLoop` uses mailboxes for request processing. See
[Orchestration](orchestration.md) for details:

```python nocheck
from weakincentives.runtime import MainLoop, MainLoopRequest

loop = MainLoop(
    prompt=my_prompt,
    adapter=my_adapter,
    requests=requests_mailbox,  # Receives MainLoopRequest[T]
)

# Responses route via reply_to in each request
loop.run()
```

## Best Practices

1. **Set visibility_timeout > max processing time**: Prevents duplicate
   processing from timeout-triggered redelivery.

1. **Send response before acknowledging**: If you ack first and crash before
   sending the response, the client never gets their reply.

1. **Make consumers idempotent**: At-least-once delivery means you may process
   the same message twice.

1. **Use delivery_count for backoff**: Implement exponential backoff on retries
   to avoid overwhelming failed dependencies.

1. **Always close mailboxes**: `close()` stops background reaper threads.
   Forgetting this can cause resource leaks.

1. **Use type hints**: `body_type` on `RedisMailbox` enables proper
   deserialization; without it you get raw dicts.

## Next Steps

- [Orchestration](orchestration.md): Use MainLoop with mailboxes
- [Lifecycle](lifecycle.md): Health checks and graceful shutdown
- [Sessions](sessions.md): State management with reducers
