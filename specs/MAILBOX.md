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

### ReplyRoutes

```python
@dataclass(frozen=True, slots=True)
class ReplyRoutes:
    """Type-based reply routing table.

    Routes replies to different mailboxes based on the response type.
    Supports inheritance: if no exact match, checks parent types in MRO order.

    Attributes:
        default: Fallback identifier when no type-specific route matches.
        routes: Mapping from response types to mailbox identifiers.
    """

    default: str | None = None
    routes: Mapping[type, str] = field(default_factory=dict)

    def route_for(self, body: object) -> str:
        """Determine the mailbox identifier for a reply body.

        Resolution order:
        1. Exact type match in routes
        2. Parent type match (MRO order, excluding object)
        3. Default route

        Args:
            body: The reply payload.

        Returns:
            Mailbox identifier string.

        Raises:
            NoRouteError: No matching route and no default configured.
        """
        body_type = type(body)

        # Exact match
        if body_type in self.routes:
            return self.routes[body_type]

        # Check parent types (MRO order)
        for parent in body_type.__mro__[1:]:
            if parent is object:
                break
            if parent in self.routes:
                return self.routes[parent]

        # Fallback to default
        if self.default is not None:
            return self.default

        raise NoRouteError(body_type)

    @classmethod
    def single(cls, identifier: str) -> "ReplyRoutes":
        """Create routes with a single default destination."""
        return cls(default=identifier)

    @classmethod
    def typed(
        cls,
        routes: Mapping[type, str],
        *,
        default: str | None = None,
    ) -> "ReplyRoutes":
        """Create routes with type-specific destinations."""
        return cls(default=default, routes=routes)
```

**Examples:**

```python
# Single destination (simple case)
routes = ReplyRoutes.single("client-123")

# Type-based routing
routes = ReplyRoutes.typed(
    {
        SuccessResult: "client-123:success",
        ErrorResult: "client-123:errors",
        ProgressUpdate: "client-123:progress",
    },
    default="client-123:other",
)

# Inheritance-aware routing
@dataclass
class BaseError: ...

@dataclass
class ValidationError(BaseError): ...

@dataclass
class TimeoutError(BaseError): ...

routes = ReplyRoutes.typed({BaseError: "errors"})
routes.route_for(ValidationError(...))  # → "errors" (matches parent)
```

### Mailbox

```python
class Mailbox(Protocol[T, R]):
    """Point-to-point message queue.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).
    """

    def send(self, body: T, *, reply_routes: ReplyRoutes | None = None) -> str:
        """Enqueue a message.

        Args:
            body: Message payload (must be serializable).
            reply_routes: Routing table for responses. Workers use this
                to determine where to send replies based on response type.

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
    reply_routes: ReplyRoutes | None

    def reply(self, body: R) -> str:
        """Send reply to type-appropriate destination.

        Routes the reply based on body type using the message's ReplyRoutes.
        Multiple replies are allowed before finalization.

        Args:
            body: Response payload.

        Returns:
            Message ID of the sent reply.

        Raises:
            MessageFinalizedError: Message already acknowledged or nacked.
            NoRouteError: No route matches body type.
            ReplyNotAvailableError: No reply_routes configured or
                resolver cannot find the destination mailbox.
        """
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
class ReplyNotAvailableError(MailboxError): ...     # Cannot resolve destination
class MessageFinalizedError(MailboxError): ...      # Message already ack/nack'd
class NoRouteError(MailboxError):                   # No route for reply type
    """No route matches the reply body type.

    Raised when ReplyRoutes.route_for() cannot find:
    - Exact type match in routes
    - Parent type match in routes
    - Default route
    """
    body_type: type
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

Workers send responses via `Message.reply()`, which routes based on the
response type and resolves the destination mailbox internally.

### Setup

```python
# Registry for resolving route identifiers to mailboxes
registry: dict[str, Mailbox] = {
    "client-123:success": InMemoryMailbox(name="success"),
    "client-123:errors": InMemoryMailbox(name="errors"),
    "client-123:progress": InMemoryMailbox(name="progress"),
}

# Request mailbox with resolver
requests: Mailbox[Request, SuccessResult | ErrorResult | ProgressUpdate] = (
    InMemoryMailbox(
        name="requests",
        reply_resolver=RegistryResolver(registry),
    )
)
```

### Client

```python
# Define response types
@dataclass(frozen=True)
class SuccessResult:
    value: int

@dataclass(frozen=True)
class ErrorResult:
    message: str
    code: int

@dataclass(frozen=True)
class ProgressUpdate:
    step: int
    total: int

# Create dedicated response queues
success_queue: Mailbox[SuccessResult, None] = InMemoryMailbox(name="success")
error_queue: Mailbox[ErrorResult, None] = InMemoryMailbox(name="errors")
progress_queue: Mailbox[ProgressUpdate, None] = InMemoryMailbox(name="progress")

registry["client-123:success"] = success_queue
registry["client-123:errors"] = error_queue
registry["client-123:progress"] = progress_queue

# Send request with type-based routing
requests.send(
    Request(data="process this"),
    reply_routes=ReplyRoutes.typed(
        {
            SuccessResult: "client-123:success",
            ErrorResult: "client-123:errors",
            ProgressUpdate: "client-123:progress",
        },
    ),
)

# Collect responses from appropriate queues
for msg in success_queue.receive(wait_time_seconds=20):
    print(f"Success: {msg.body.value}")
    msg.acknowledge()

for msg in error_queue.receive():
    print(f"Error: {msg.body.message}")
    msg.acknowledge()
```

### Worker

```python
for msg in requests.receive(visibility_timeout=300, wait_time_seconds=20):
    try:
        # Progress updates go to progress queue
        msg.reply(ProgressUpdate(step=1, total=3))
        msg.reply(ProgressUpdate(step=2, total=3))
        msg.reply(ProgressUpdate(step=3, total=3))

        # Final result goes to success queue
        result = process(msg.body)
        msg.reply(SuccessResult(value=result))
        msg.acknowledge()

    except ValidationError as e:
        # Errors go to error queue
        msg.reply(ErrorResult(message=str(e), code=400))
        msg.acknowledge()

    except NoRouteError:
        log.warning("No route for reply type", msg_id=msg.id)
        msg.acknowledge()

    except Exception as e:
        # Backoff retry
        msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

### Single Destination (Simple Case)

For simple request/response without type-based routing:

```python
# Client
requests.send(
    Request(data="hello"),
    reply_routes=ReplyRoutes.single("client-123"),
)

# Worker - all replies go to same destination
for msg in requests.receive():
    msg.reply(SuccessResult(value=42))  # → client-123
    msg.acknowledge()
```

### Inheritance-Based Routing

Routes support type inheritance via MRO lookup:

```python
@dataclass(frozen=True)
class BaseResult: ...

@dataclass(frozen=True)
class SuccessResult(BaseResult):
    value: int

@dataclass(frozen=True)
class PartialResult(BaseResult):
    partial: list[int]

# Route all BaseResult subtypes to same destination
routes = ReplyRoutes.typed(
    {BaseResult: "client-123:results"},
    default="client-123:other",
)

# Both match via parent type
routes.route_for(SuccessResult(value=1))    # → "client-123:results"
routes.route_for(PartialResult(partial=[]))  # → "client-123:results"
```

### Reply After Finalization

Once a message is acknowledged or nacked, further replies raise
`MessageFinalizedError`:

```python
msg = requests.receive()[0]
msg.reply(SuccessResult(value=1))  # OK
msg.acknowledge()
msg.reply(SuccessResult(value=2))  # Raises MessageFinalizedError
```

This prevents:

- Sending replies to a deleted message (after ack)
- Sending replies that race with redelivery (after nack)

### Eval Run Collection

Evaluation runs can route successes and failures to different queues:

```python
run_id = uuid4()

# Separate queues for analysis
successes: Mailbox[EvalSuccess, None] = InMemoryMailbox(name=f"eval-{run_id}-pass")
failures: Mailbox[EvalFailure, None] = InMemoryMailbox(name=f"eval-{run_id}-fail")

registry[f"eval-{run_id}:success"] = successes
registry[f"eval-{run_id}:failure"] = failures

# Submit all samples
for sample in dataset:
    requests.send(
        EvalRequest(sample=sample),
        reply_routes=ReplyRoutes.typed({
            EvalSuccess: f"eval-{run_id}:success",
            EvalFailure: f"eval-{run_id}:failure",
        }),
    )

# Process results by outcome
passed = list(drain(successes))
failed = list(drain(failures))
print(f"Pass rate: {len(passed) / len(dataset):.1%}")
```

## MainLoop Integration

MainLoop takes a single requests mailbox. Response routing derives from each
message's `reply_routes`.

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
{queue:name}:meta       # HASH - delivery counts, reply routes
```

Hash tags (`{queue:name}`) ensure all keys co-locate in Redis Cluster.

### ReplyRoutes Serialization

`ReplyRoutes` serializes to JSON for storage in message metadata:

```python
# Wire format
{
    "default": "client-123",
    "routes": {
        "myapp.models.SuccessResult": "client-123:success",
        "myapp.models.ErrorResult": "client-123:errors"
    }
}
```

Type keys use fully-qualified names (`module.ClassName`) for unambiguous
deserialization. The resolver reconstructs type objects via import at
receive time.

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

### Testing Type-Based Routing

```python
def test_reply_routes_to_type_specific_mailbox():
    success_mb = CollectingMailbox()
    error_mb = CollectingMailbox()

    registry = {
        "success": success_mb,
        "errors": error_mb,
    }
    requests = InMemoryMailbox(reply_resolver=RegistryResolver(registry))

    requests.send(
        "hello",
        reply_routes=ReplyRoutes.typed({
            SuccessResult: "success",
            ErrorResult: "errors",
        }),
    )

    msg = requests.receive()[0]
    msg.reply(SuccessResult(value=42))
    msg.reply(ErrorResult(message="oops", code=500))
    msg.acknowledge()

    assert len(success_mb.sent) == 1
    assert success_mb.sent[0].value == 42
    assert len(error_mb.sent) == 1
    assert error_mb.sent[0].message == "oops"


def test_reply_routes_inheritance():
    results_mb = CollectingMailbox()
    requests = InMemoryMailbox(
        reply_resolver=RegistryResolver({"results": results_mb}),
    )

    requests.send(
        "hello",
        reply_routes=ReplyRoutes.typed({BaseResult: "results"}),
    )

    msg = requests.receive()[0]
    msg.reply(SuccessResult(value=1))  # Matches via parent
    msg.reply(PartialResult(partial=[2, 3]))  # Matches via parent
    msg.acknowledge()

    assert len(results_mb.sent) == 2


def test_no_route_raises():
    requests = InMemoryMailbox(
        reply_resolver=RegistryResolver({"other": CollectingMailbox()}),
    )

    requests.send(
        "hello",
        reply_routes=ReplyRoutes.typed({SuccessResult: "success"}),  # No ErrorResult route
    )

    msg = requests.receive()[0]

    with pytest.raises(NoRouteError) as exc:
        msg.reply(ErrorResult(message="oops", code=500))

    assert exc.value.body_type is ErrorResult


def test_default_route_fallback():
    default_mb = CollectingMailbox()
    requests = InMemoryMailbox(
        reply_resolver=RegistryResolver({"default": default_mb}),
    )

    requests.send(
        "hello",
        reply_routes=ReplyRoutes.typed(
            {SuccessResult: "success"},
            default="default",
        ),
    )

    msg = requests.receive()[0]
    msg.reply(UnknownType())  # Falls back to default
    msg.acknowledge()

    assert len(default_mb.sent) == 1


def test_reply_after_finalization_raises():
    requests = InMemoryMailbox(
        reply_resolver=RegistryResolver({"r": CollectingMailbox()}),
    )
    requests.send("hello", reply_routes=ReplyRoutes.single("r"))

    msg = requests.receive()[0]
    msg.acknowledge()

    with pytest.raises(MessageFinalizedError):
        msg.reply(SuccessResult(value=1))
```

## Limitations

- **At-least-once only.** No exactly-once. Consumers must be idempotent.
- **Ordering varies.** SQS Standard is best-effort. Use FIFO or Redis for strict.
- **No built-in DLQ.** Implement via `delivery_count` threshold.
- **No deduplication.** Handle at application level if needed.
- **No transactions.** Send and receive are independent operations.
- **Type serialization requires importable types.** Anonymous or local classes
  cannot be used as route keys.

## Related Specifications

- `specs/MAILBOX_RESOLVER.md` - Service discovery for mailbox instances
- `specs/MAIN_LOOP.md` - MainLoop orchestration using mailboxes
- `specs/RESOURCE_REGISTRY.md` - DI container for lifecycle-scoped resources
