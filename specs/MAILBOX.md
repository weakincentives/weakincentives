# Mailbox Specification

## Purpose

`Mailbox` provides a message queue abstraction with SQS/Redis-compatible
semantics for `MainLoop` communication. Unlike the pub/sub `Dispatcher`, Mailbox
delivers messages point-to-point with visibility timeout and acknowledgment.

`Reply` provides a future-like abstraction for request/response patterns,
backed by `ReplyStore` for persistence across processes.

## Core Abstractions

### Mailbox

```python
class Mailbox(Protocol[T, R]):
    def send(self, body: T, *, delay_seconds: int = 0) -> str: ...
    def send_expecting_reply(self, body: T, *, reply_timeout: float = 300) -> Reply[R]: ...
    def receive(self, *, max_messages: int = 1, visibility_timeout: int = 30,
                wait_time_seconds: int = 0) -> Sequence[Message[T, R]]: ...
    def purge(self) -> int: ...
    def approximate_count(self) -> int: ...
```

### Message

```python
@dataclass(frozen=True, slots=True)
class Message[T, R]:
    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    attributes: Mapping[str, str]
    reply_channel: ReplyChannel[R] | None

    def expects_reply(self) -> bool: ...
    def reply(self, value: R) -> None: ...      # Writes to ReplyStore, then acknowledges
    def acknowledge(self) -> bool: ...          # Raises ReplyExpectedError if expects_reply
    def nack(self, *, visibility_timeout: int = 0) -> bool: ...
    def extend_visibility(self, timeout: int) -> bool: ...
```

**Note:** `reply()` writes the response to `ReplyStore` before acknowledging the
message. This ensures the response is persisted before the message is removed
from the queue. If reply storage fails, the message remains in-flight and will
be redelivered after visibility timeout.

### Reply

Caller's future-like handle for awaiting responses:

```python
class Reply(Protocol[T]):
    id: str
    def wait(self, *, timeout: float | None = None) -> T: ...
    def poll(self) -> T | None: ...
    def is_ready(self) -> bool: ...
    def is_cancelled(self) -> bool: ...
    def cancel(self) -> bool: ...
```

### ReplyChannel

Consumer's write-once channel for sending responses:

```python
class ReplyChannel(Protocol[T]):
    def send(self, value: T) -> None: ...
    def is_open(self) -> bool: ...
```

### ReplyStore

Backing storage for reply state with TTL and atomic consume:

```python
class ReplyStore(Protocol[T]):
    def create(self, id: str, *, ttl: float) -> bool: ...
    def resolve(self, id: str, value: T) -> bool: ...
    def cancel(self, id: str) -> bool: ...
    def get(self, id: str) -> ReplyEntry[T] | None: ...
    def consume(self, id: str) -> ReplyEntry[T] | None: ...
    def delete(self, id: str) -> bool: ...
    def scan_expired(self, *, limit: int = 100) -> Sequence[str]: ...
    def cleanup_expired(self, *, limit: int = 100) -> int: ...

@dataclass(frozen=True, slots=True)
class ReplyEntry[T]:
    id: str
    value: T | None
    state: ReplyState  # pending | resolved | cancelled | expired
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None
```

## Lifecycles

### Message

```
send() → Queued → receive() → Invisible → acknowledge() → Deleted
                                   ↓
                              nack() or timeout → Queued (redelivery)
```

### Reply

```
create() → Pending → resolve() → Resolved → consume() → Deleted
               ↓
          cancel() → Cancelled
               ↓
          TTL expires → Expired → cleanup_expired() → Deleted
```

## MainLoop Integration

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(self, *, adapter: ProviderAdapter[OutputT],
                 requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]) -> None: ...

    def run(self, *, max_iterations: int | None = None) -> None:
        while should_continue:
            for msg in self._requests.receive(visibility_timeout=300, wait_time_seconds=20):
                result = self._execute(msg.body)
                if msg.expects_reply:
                    msg.reply(result)  # Implies acknowledgment
                else:
                    msg.acknowledge()
```

### Client Usage

```python
# Request with reply
reply = mailbox.send_expecting_reply(MainLoopRequest(request=my_request))
result = reply.wait(timeout=60)

# Fire-and-forget
mailbox.send(MainLoopRequest(request=background_task))
```

## Implementations

| Component | InMemory | Redis | AWS |
|-----------|----------|-------|-----|
| **Mailbox** | `InMemoryMailbox` | `RedisMailbox` | `SQSMailbox` |
| **ReplyStore** | `InMemoryReplyStore` | `RedisReplyStore` | `DynamoDBReplyStore` |

### InMemory

- Dict + deque with RLock synchronization
- Condition variables for long-poll and reply blocking
- Background reaper for visibility timeout and TTL

### Redis

- Mailbox: LPUSH/BRPOP + sorted set for visibility timeout
- ReplyStore: Hash per entry + sorted set for expiry scanning
- Lua scripts for atomic operations
- BLPOP on notification channel for reply.wait()

### AWS

- Mailbox: SQS with native visibility timeout and long polling
- ReplyStore: DynamoDB with native TTL, conditional writes
- Polling-based consume (no push notification)

## Mailbox vs Dispatcher

| Aspect | Dispatcher | Mailbox |
|--------|------------|---------|
| Pattern | Pub/sub broadcast | Point-to-point |
| Delivery | All subscribers | One consumer |
| Acknowledgment | None | Required |
| Response | N/A | Via Reply |
| Use case | Telemetry | Request processing |

Both coexist: `Dispatcher` for observability events, `Mailbox` for MainLoop
orchestration.

## Errors

```python
# Mailbox errors
class MailboxError(WinkError): ...
class ReceiptHandleExpiredError(MailboxError): ...
class MailboxFullError(MailboxError): ...
class SerializationError(MailboxError): ...

# Reply errors
class ReplyError(WinkError): ...
class ReplyTimeoutError(ReplyError): ...
class ReplyCancelledError(ReplyError): ...
class ReplyAlreadySentError(ReplyError): ...
class ReplyExpectedError(ReplyError): ...     # acknowledge() called when reply expected
class NoReplyChannelError(ReplyError): ...
```

## Testing Utilities

- `NullMailbox`: Drops messages, returns non-resolving replies
- `ImmediateReply`: Resolves instantly with preset value

## Limitations

- At-least-once delivery only (no exactly-once)
- No FIFO ordering guarantee
- Single response per request via Reply
- Reply timeout handled by caller (`ReplyTimeoutError`)
