# Mailbox Specification

## Purpose

`Mailbox[T, R]` provides a typed message queue with SQS-compatible semantics:
point-to-point delivery, visibility timeout, and explicit acknowledgment.

```python
# Request mailbox: receives MainLoopRequest, replies with MainLoopResult
requests: Mailbox[MainLoopRequest, MainLoopResult]

# Response mailbox: receives MainLoopResult, no replies
responses: Mailbox[MainLoopResult, None]
```

**Use Mailbox for:** Durable request processing, work distribution, cross-process
communication, tasks requiring acknowledgment.

**Use Dispatcher for:** Telemetry, in-process events, fire-and-forget broadcasts.

## Core Types

### Mailbox

```python
class Mailbox(Protocol[T, R]):
    """Point-to-point message queue.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).
    """

    def send(self, body: T, *, reply_to: str | None = None) -> str:
        """Enqueue a message.

        Args:
            body: Message payload (must be serializable).
            reply_to: Identifier for response mailbox. Workers resolve this
                via Message.reply().

        Returns:
            Message ID.

        Raises:
            MailboxFullError: Queue capacity exceeded.
            SerializationError: Body cannot be serialized.
        """
        ...

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Receive messages.

        Messages become invisible for `visibility_timeout` seconds.
        Must acknowledge before timeout or message returns to queue.

        Args:
            max_messages: Maximum to receive (1-10).
            visibility_timeout: Invisibility duration in seconds.
            wait_time_seconds: Long poll duration (0 = immediate return).

        Returns:
            Sequence of messages (may be empty).
        """
        ...

    def purge(self) -> int:
        """Delete all messages. Returns approximate count deleted."""
        ...

    def approximate_count(self) -> int:
        """Approximate message count (eventually consistent)."""
        ...

    def close(self) -> None:
        """Close the mailbox and release resources.

        After closing:
        - receive() returns empty immediately
        - send() behavior is implementation-defined
        """
        ...

    @property
    def closed(self) -> bool:
        """Return True if the mailbox has been closed."""
        ...
```

### Message

```python
@dataclass(slots=True)
class Message(Generic[T, R]):
    """Received message with lifecycle methods."""

    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    reply_to: str | None

    def reply(self, body: R) -> str:
        """Send reply to reply_to destination. Multiple replies allowed."""
        ...

    def acknowledge(self) -> None:
        """Delete message. Finalizes - no further replies allowed."""
        ...

    def nack(self, *, visibility_timeout: int = 0) -> None:
        """Return to queue. Finalizes - no further replies allowed."""
        ...

    def extend_visibility(self, timeout: int) -> None:
        """Extend visibility for long-running processing."""
        ...

    @property
    def is_finalized(self) -> bool:
        """True if acknowledged or nacked."""
        ...
```

### Errors

```python
class MailboxError(WinkError): ...
class ReceiptHandleExpiredError(MailboxError): ...  # Handle no longer valid
class MailboxFullError(MailboxError): ...           # Queue capacity exceeded
class SerializationError(MailboxError): ...         # Cannot serialize body
class MailboxConnectionError(MailboxError): ...     # Backend unreachable
class ReplyNotAvailableError(MailboxError): ...     # Cannot resolve reply_to
class MessageFinalizedError(MailboxError): ...      # Message already ack/nack'd
```

## Message Lifecycle

```
send() ──► Queued ──► receive() ──► Invisible ──► acknowledge() ──► Deleted
              │                          │
              │                          ▼
              │                    timeout expires
              │                          │
              ◄──────────────────────────┘ (redelivery)
              │
              │                    nack(visibility_timeout=N)
              ◄────────────────────┘ (delayed redelivery)
```

**At-least-once delivery.** Messages may be delivered multiple times if
visibility expires or consumer crashes. Consumers must be idempotent.

**Receipt handle.** Each delivery generates a unique handle. Operations
require the current handle; after timeout, old handles become invalid.

## Reply Pattern

Workers send responses via `Message.reply()`, which resolves the destination
mailbox internally. Multiple replies are permitted before acknowledgment.

### Setup

```python
# Registry for resolving reply_to identifiers
registry: dict[str, Mailbox] = {}

# Request mailbox with resolver
requests: Mailbox[MainLoopRequest, MainLoopResult] = InMemoryMailbox(
    name="requests",
    reply_resolver=registry.get,
)
```

**Important:** Resolvers should cache mailbox instances to avoid resource leaks.
See `specs/MAILBOX_RESOLVER.md` for resolver patterns.

```python
@cache
def redis_resolver(reply_to: str) -> RedisMailbox[MainLoopResult, None]:
    return RedisMailbox(name=reply_to, client=redis_client)
```

### Client

```python
# Create dedicated response queue
responses: Mailbox[MainLoopResult, None] = InMemoryMailbox(name="my-responses")
registry["my-responses"] = responses

# Send request
requests.send(MainLoopRequest(request=data), reply_to="my-responses")

# Collect response
for msg in responses.receive(wait_time_seconds=20):
    result = msg.body
    msg.acknowledge()
```

### Worker

```python
for msg in requests.receive(visibility_timeout=300, wait_time_seconds=20):
    try:
        result = process(msg.body)
        msg.reply(result)
        msg.acknowledge()
    except ReplyNotAvailableError:
        log.warning("No reply_to", msg_id=msg.id)
        msg.acknowledge()
    except Exception as e:
        # Backoff retry
        msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

### Multiple Replies

Messages support multiple replies before finalization. This enables
progress reporting, streaming results, or multi-part responses:

```python
for msg in requests.receive(visibility_timeout=600):
    try:
        # Stream progress updates
        for i, chunk in enumerate(process_chunks(msg.body)):
            msg.reply(Progress(step=i, data=chunk))

        # Send final result
        msg.reply(Complete(summary=summarize()))
        msg.acknowledge()

    except Exception as e:
        msg.nack(visibility_timeout=60)
```

### Reply After Finalization

Once a message is acknowledged or nacked, further replies raise
`MessageFinalizedError`:

```python
msg = requests.receive()[0]
msg.reply(Result(value=1))  # OK
msg.acknowledge()
msg.reply(Result(value=2))  # Raises MessageFinalizedError
```

This prevents:

- Sending replies to a deleted message (after ack)
- Sending replies that race with redelivery (after nack)

### Eval Run Collection

All samples specify the same `reply_to`. Results collect into one mailbox
regardless of which worker processes each sample:

```python
run_id = uuid4()
results: Mailbox[MainLoopResult, None] = InMemoryMailbox(name=f"eval-{run_id}")
registry[f"eval-{run_id}"] = results

for sample in dataset:
    requests.send(MainLoopRequest(request=sample.input), reply_to=f"eval-{run_id}")

collected = []
while len(collected) < len(dataset):
    for msg in results.receive(wait_time_seconds=20):
        collected.append(msg.body)
        msg.acknowledge()
```

## MainLoop Integration

MainLoop takes a single requests mailbox. Response routing derives from each
message's `reply_to`.

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        resources: ResourceRegistry | None = None,
    ) -> None: ...
```

### Request/Response Types

```python
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    request_id: UUID
    output: OutputT | None = None
    error: str | None = None
    session_id: UUID | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Error Handling Guidelines

1. **Set visibility_timeout > max execution time.** 5 minutes for 4-minute tasks.
1. **Send response before acknowledging.** Crash-safe ordering.
1. **Use delivery_count for backoff.** `min(60 * delivery_count, 900)` = 1-15 min.
1. **Dead-letter after N retries.** `delivery_count > 5` → log and acknowledge.

## Implementations

| Implementation | Backend | Use Case |
| ----------------- | ------- | ---------------------------------- |
| `InMemoryMailbox` | Dict | Testing, single process |
| `RedisMailbox` | Redis | Multi-process, self-hosted |
| `SQSMailbox` | AWS SQS | Production, managed infrastructure |

### Backend Differences

| Aspect | SQS Standard | SQS FIFO | Redis | InMemory |
| -------------- | ------------ | ---------- | ------ | -------- |
| Ordering | Best-effort | Strict | FIFO | FIFO |
| Long poll max | 20 sec | 20 sec | ∞ | ∞ |
| Visibility max | 12 hours | 12 hours | ∞ | ∞ |
| Count accuracy | ~1 min lag | ~1 min | Exact | Exact |
| Durability | Replicated | Replicated | Config | None |

### RedisMailbox Data Structures

```
{queue:name}:pending    # LIST - awaiting delivery
{queue:name}:invisible  # ZSET - in-flight, scored by expiry
{queue:name}:data       # HASH - message ID → payload
{queue:name}:meta       # HASH - delivery counts
```

Hash tags (`{queue:name}`) ensure all keys co-locate in Redis Cluster.

## Testing

### Test Utilities

```python
# Drops all messages
mailbox: Mailbox[Event, None] = NullMailbox()

# Stores sent messages for inspection
mailbox: CollectingMailbox[Event, None] = CollectingMailbox()
assert mailbox.sent[0].type == "a"

# Controllable behavior for edge cases
mailbox: FakeMailbox[Event, None] = FakeMailbox()
mailbox.expire_handle(msg.receipt_handle)  # Simulate expiry
mailbox.set_connection_error(...)          # Simulate failure
```

### Testing Reply

```python
def test_reply_sends_to_resolved_mailbox():
    responses = CollectingMailbox()
    requests = InMemoryMailbox(reply_resolver={"responses": responses}.get)

    requests.send("hello", reply_to="responses")
    msg = requests.receive()[0]
    msg.reply("world")
    msg.acknowledge()

    assert responses.sent == ["world"]


def test_multiple_replies_allowed():
    msg = receive_message(reply_to="responses")
    msg.reply(1)
    msg.reply(2)
    msg.reply(3)
    msg.acknowledge()  # OK - replies before finalization


def test_reply_after_finalization_raises():
    msg = receive_message(reply_to="responses")
    msg.acknowledge()

    with pytest.raises(MessageFinalizedError):
        msg.reply("too late")
```

## Limitations

- **At-least-once only.** No exactly-once. Consumers must be idempotent.
- **Ordering varies.** SQS Standard is best-effort. Use FIFO or Redis for strict.
- **No built-in DLQ.** Implement via `delivery_count` threshold.
- **No deduplication.** Handle at application level if needed.
- **No transactions.** Send and receive are independent operations.

## Related Specifications

- `specs/MAILBOX_RESOLVER.md` - Service discovery for mailbox instances
- `specs/MAIN_LOOP.md` - MainLoop orchestration using mailboxes
- `specs/RESOURCE_REGISTRY.md` - DI container for lifecycle-scoped resources
