# Mailbox Resolver Specification

## Purpose

`MailboxResolver` provides service discovery for mailbox instances, enabling
dynamic reply routing via string identifiers. This abstraction sits between
the `Message.reply_mailbox()` API and backend-specific mailbox construction.

```python
# Worker resolves reply destination from message metadata
for msg in requests.receive(visibility_timeout=300):
    result = process(msg.body)
    msg.reply_mailbox().send(result)  # Resolves reply_to → Mailbox
    msg.acknowledge()
```

**Use MailboxResolver for:** Dynamic reply routing, multi-tenant mailbox
discovery, backend-agnostic mailbox construction from string identifiers.

**Use ResourceRegistry for:** Static dependency injection with scope-aware
lifecycle management (singletons, per-tool instances).

## Comparison with ResourceResolver

| Aspect | ResourceResolver | MailboxResolver |
|--------|------------------|-----------------|
| Key type | `type[T]` | `str` |
| Purpose | DI container | Service discovery |
| Resolution | Static bindings | Dynamic lookup + factory |
| Caching | Scope-aware (singleton/tool) | Optional, leak-prevention |
| Configuration | `Binding` objects | `MailboxFactory` + registry |

The mailbox resolver is a **factory + registry pattern**, not a DI container.
It does not participate in dependency graphs or lifecycle scoping.

## Core Types

### MailboxResolver

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MailboxResolver[T](Protocol):
    """Resolves string identifiers to Mailbox instances.

    Implementations may use registries, factories, or both.
    Caching is implementation-specific but recommended to prevent
    resource leaks from repeated resolution of the same identifier.
    """

    def resolve(self, identifier: str) -> Mailbox[T]:
        """Resolve an identifier to a mailbox instance.

        Args:
            identifier: String identifier for the mailbox (e.g., queue name,
                URI, or registered key).

        Returns:
            Mailbox instance for the identifier.

        Raises:
            MailboxResolutionError: Cannot resolve identifier.
        """
        ...

    def resolve_optional(self, identifier: str) -> Mailbox[T] | None:
        """Resolve if possible, return None otherwise."""
        ...
```

### MailboxFactory

```python
from typing import Protocol

class MailboxFactory[T](Protocol):
    """Creates mailbox instances from string identifiers.

    Factories handle backend-specific construction. They do not cache;
    caching is the resolver's responsibility.
    """

    def create(self, identifier: str) -> Mailbox[T]:
        """Create a new mailbox for the given identifier.

        Args:
            identifier: Backend-specific identifier (queue name, URI, etc.).

        Returns:
            New mailbox instance.

        Raises:
            MailboxConnectionError: Cannot connect to backend.
            ValueError: Invalid identifier format.
        """
        ...
```

### CompositeResolver

```python
@dataclass(slots=True)
class CompositeResolver[T]:
    """Combines a registry with a factory for dynamic resolution.

    Resolution order:
    1. Check registry for pre-registered mailbox
    2. Fall back to factory for dynamic creation
    3. Cache factory-created instances (if caching enabled)

    Example::

        registry: dict[str, Mailbox[Event]] = {}
        factory = RedisMailboxFactory(client=redis_client)
        resolver = CompositeResolver(registry=registry, factory=factory)

        # Pre-register known mailboxes
        registry["responses"] = InMemoryMailbox(name="responses")

        # Dynamic resolution via factory
        mailbox = resolver.resolve("worker-123")  # Creates RedisMailbox
    """

    registry: Mapping[str, Mailbox[T]]
    """Pre-registered mailboxes looked up first."""

    factory: MailboxFactory[T] | None = None
    """Optional factory for dynamic creation. None disables fallback."""

    cache: dict[str, Mailbox[T]] = field(default_factory=dict)
    """Cache for factory-created instances. Prevents resource leaks."""

    def resolve(self, identifier: str) -> Mailbox[T]:
        # Check registry first
        if identifier in self.registry:
            return self.registry[identifier]

        # Check cache
        if identifier in self.cache:
            return self.cache[identifier]

        # Fall back to factory
        if self.factory is None:
            raise MailboxResolutionError(identifier)

        mailbox = self.factory.create(identifier)
        self.cache[identifier] = mailbox
        return mailbox

    def resolve_optional(self, identifier: str) -> Mailbox[T] | None:
        try:
            return self.resolve(identifier)
        except MailboxResolutionError:
            return None
```

### RegistryResolver

```python
@dataclass(slots=True, frozen=True)
class RegistryResolver[T]:
    """Simple resolver backed by a static registry.

    Use when all mailboxes are known at configuration time.
    No factory fallback, no dynamic creation.

    Example::

        registry = {
            "requests": InMemoryMailbox(name="requests"),
            "responses": InMemoryMailbox(name="responses"),
        }
        resolver = RegistryResolver(registry=registry)
    """

    registry: Mapping[str, Mailbox[T]]

    def resolve(self, identifier: str) -> Mailbox[T]:
        if identifier not in self.registry:
            raise MailboxResolutionError(identifier)
        return self.registry[identifier]

    def resolve_optional(self, identifier: str) -> Mailbox[T] | None:
        return self.registry.get(identifier)
```

### Errors

```python
class MailboxResolutionError(MailboxError):
    """Cannot resolve mailbox identifier.

    Raised when:
    - Identifier not in registry and no factory configured
    - Factory cannot create mailbox for identifier
    """

    identifier: str
    """The identifier that could not be resolved."""
```

## Message Integration

### Updated Message Type

```python
@dataclass(frozen=True, slots=True)
class Message[T]:
    """Received message with lifecycle methods and reply support."""

    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    attributes: Mapping[str, str]
    reply_to: str | None = None
    """Identifier for reply mailbox. Resolved via reply_mailbox()."""

    _acknowledge_fn: Callable[[], None] = field(repr=False, compare=False)
    _nack_fn: Callable[[int], None] = field(repr=False, compare=False)
    _extend_fn: Callable[[int], None] = field(repr=False, compare=False)
    _reply_resolver: MailboxResolver[object] | None = field(
        default=None, repr=False, compare=False
    )
    """Resolver bound at receive time for reply_mailbox()."""

    def reply_mailbox[R](self) -> Mailbox[R]:
        """Resolve reply_to to a Mailbox for sending responses.

        Returns:
            Mailbox instance for the reply_to identifier.

        Raises:
            ReplyMailboxUnavailableError: No reply_to set or resolution failed.
        """
        if self.reply_to is None:
            raise ReplyMailboxUnavailableError("No reply_to specified")

        if self._reply_resolver is None:
            raise ReplyMailboxUnavailableError("No resolver configured")

        try:
            return cast(Mailbox[R], self._reply_resolver.resolve(self.reply_to))
        except MailboxResolutionError as e:
            raise ReplyMailboxUnavailableError(
                f"Cannot resolve reply_to '{self.reply_to}': {e}"
            ) from e
```

### Updated Mailbox Protocol

```python
@runtime_checkable
class Mailbox[T](Protocol):
    """Point-to-point message queue with visibility timeout semantics."""

    def send(self, body: T, *, delay_seconds: int = 0, reply_to: str | None = None) -> str:
        """Enqueue a message with optional reply routing.

        Args:
            body: Message payload.
            delay_seconds: Seconds before message becomes visible.
            reply_to: Identifier for response mailbox. Workers resolve
                this via Message.reply_mailbox().

        Returns:
            Message ID.
        """
        ...

    # ... other methods unchanged
```

### Updated InMemoryMailbox

```python
@dataclass(slots=True)
class InMemoryMailbox[T]:
    """Thread-safe in-memory mailbox implementation."""

    name: str = "default"
    max_size: int | None = None
    reply_resolver: MailboxResolver[object] | None = None
    """Resolver for reply_mailbox() on received messages."""

    # ... internal state ...

    def send(self, body: T, *, delay_seconds: int = 0, reply_to: str | None = None) -> str:
        """Enqueue a message with optional reply routing."""
        # Store reply_to in message metadata
        ...

    def receive(self, ...) -> Sequence[Message[T]]:
        # Bind resolver to each message
        message = Message(
            id=in_flight.id,
            body=in_flight.body,
            reply_to=in_flight.reply_to,
            _reply_resolver=self.reply_resolver,
            # ... other fields ...
        )
        ...
```

## Backend Factories

### InMemoryMailboxFactory

```python
@dataclass(slots=True, frozen=True)
class InMemoryMailboxFactory[T]:
    """Factory for in-memory mailboxes.

    Creates InMemoryMailbox instances keyed by identifier.
    Useful for testing and single-process scenarios.
    """

    reply_resolver: MailboxResolver[object] | None = None
    """Resolver to bind to created mailboxes."""

    max_size: int | None = None
    """Default max_size for created mailboxes."""

    def create(self, identifier: str) -> InMemoryMailbox[T]:
        return InMemoryMailbox(
            name=identifier,
            max_size=self.max_size,
            reply_resolver=self.reply_resolver,
        )
```

### RedisMailboxFactory

```python
@dataclass(slots=True, frozen=True)
class RedisMailboxFactory[T]:
    """Factory for Redis-backed mailboxes.

    Creates RedisMailbox instances using the provided client.
    Queue names are prefixed to avoid collisions.
    """

    client: RedisClient
    """Redis client for backend operations."""

    prefix: str = "wink:queue:"
    """Prefix for Redis keys."""

    reply_resolver: MailboxResolver[object] | None = None
    """Resolver to bind to created mailboxes."""

    item_type: type[T] | None = None
    """Type for deserialization. Required if T is not runtime-inspectable."""

    def create(self, identifier: str) -> RedisMailbox[T]:
        return RedisMailbox(
            name=identifier,
            client=self.client,
            key_prefix=self.prefix,
            item_type=self.item_type,
            reply_resolver=self.reply_resolver,
        )
```

### URIMailboxFactory

```python
@dataclass(slots=True, frozen=True)
class URIMailboxFactory[T]:
    """Factory that parses URI schemes to select backend.

    Supports:
    - memory://name → InMemoryMailbox
    - redis://host:port/name → RedisMailbox
    - sqs://region/queue-name → SQSMailbox

    Example::

        factory = URIMailboxFactory(
            redis_client=redis_client,
            sqs_client=sqs_client,
        )
        mailbox = factory.create("redis://localhost/responses")
    """

    redis_client: RedisClient | None = None
    sqs_client: SQSClient | None = None
    reply_resolver: MailboxResolver[object] | None = None

    def create(self, identifier: str) -> Mailbox[T]:
        parsed = urlparse(identifier)

        match parsed.scheme:
            case "memory":
                return InMemoryMailbox(
                    name=parsed.netloc or parsed.path,
                    reply_resolver=self.reply_resolver,
                )
            case "redis":
                if self.redis_client is None:
                    raise ValueError("Redis client not configured")
                return RedisMailbox(
                    name=parsed.path.lstrip("/"),
                    client=self.redis_client,
                    reply_resolver=self.reply_resolver,
                )
            case "sqs":
                if self.sqs_client is None:
                    raise ValueError("SQS client not configured")
                return SQSMailbox(
                    queue_url=f"https://sqs.{parsed.netloc}.amazonaws.com{parsed.path}",
                    client=self.sqs_client,
                    reply_resolver=self.reply_resolver,
                )
            case _:
                raise ValueError(f"Unknown scheme: {parsed.scheme}")
```

## Usage Patterns

### Basic Reply Pattern

```python
# Setup
registry: dict[str, Mailbox] = {}
responses: Mailbox[Result] = InMemoryMailbox(name="responses")
registry["responses"] = responses

resolver = RegistryResolver(registry=registry)
requests: Mailbox[Request] = InMemoryMailbox(
    name="requests",
    reply_resolver=resolver,
)

# Client
requests.send(Request(data="..."), reply_to="responses")

# Worker
for msg in requests.receive():
    result = process(msg.body)
    msg.reply_mailbox().send(result)
    msg.acknowledge()
```

### Dynamic Reply Queues

```python
# Setup with factory fallback
registry: dict[str, Mailbox] = {}
factory = RedisMailboxFactory(client=redis_client)
resolver = CompositeResolver(registry=registry, factory=factory)

requests: Mailbox[Request] = RedisMailbox(
    name="requests",
    client=redis_client,
    reply_resolver=resolver,
)

# Client creates dedicated response queue
client_id = uuid4()
response_queue = f"client-{client_id}"
# No need to register - factory will create on demand

requests.send(Request(...), reply_to=response_queue)

# Worker resolves dynamically
for msg in requests.receive():
    result = process(msg.body)
    msg.reply_mailbox().send(result)  # Factory creates RedisMailbox
    msg.acknowledge()
```

### Eval Run Collection

```python
# All samples route to same eval mailbox
run_id = uuid4()
eval_queue = f"eval-{run_id}"

for sample in dataset:
    requests.send(
        MainLoopRequest(request=sample.input),
        reply_to=eval_queue,
    )

# Collect results
eval_mailbox = resolver.resolve(eval_queue)
collected = []
while len(collected) < len(dataset):
    for msg in eval_mailbox.receive(wait_time_seconds=20):
        collected.append(msg.body)
        msg.acknowledge()
```

### Multi-Tenant Isolation

```python
# Per-tenant resolver with isolated factories
def create_tenant_resolver(tenant_id: str) -> MailboxResolver:
    prefix = f"tenant-{tenant_id}:"
    return CompositeResolver(
        registry={},
        factory=RedisMailboxFactory(
            client=redis_client,
            prefix=prefix,
        ),
    )

# Tenants cannot access each other's mailboxes
tenant_a_resolver = create_tenant_resolver("a")
tenant_b_resolver = create_tenant_resolver("b")
```

## MainLoop Integration

MainLoop evolves to use a single requests mailbox with reply-based routing:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT]],
        resources: ResourceRegistry | None = None,
    ) -> None:
        """Initialize MainLoop with request mailbox.

        Response routing is determined by each message's reply_to field.
        The requests mailbox must have a reply_resolver configured.
        """
        ...

    def _handle_message(
        self, msg: Message[MainLoopRequest[UserRequestT]]
    ) -> None:
        request_event = msg.body

        try:
            response, session = self._execute(request_event)
            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                output=response.output,
                session_id=session.session_id,
            )
        except Exception as exc:
            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                error=str(exc),
            )

        # Route response via reply_mailbox
        try:
            msg.reply_mailbox().send(result)
        except ReplyMailboxUnavailableError:
            log.warning("No reply_to for request", request_id=request_event.request_id)

        msg.acknowledge()
```

## ResourceRegistry Integration

MailboxResolver can be registered as a resource for DI:

```python
registry = ResourceRegistry.of(
    Binding(RedisClient, lambda r: create_redis_client()),
    Binding(
        MailboxResolver,
        lambda r: CompositeResolver(
            registry={},
            factory=RedisMailboxFactory(client=r.get(RedisClient)),
        ),
    ),
)

# Tools can request the resolver
def my_tool(params: Params, *, context: ToolContext) -> ToolResult:
    resolver = context.resources.get(MailboxResolver)
    mailbox = resolver.resolve("notifications")
    mailbox.send(Notification(...))
    ...
```

This keeps MailboxResolver as a focused abstraction while allowing it to
participate in the broader resource lifecycle.

## Testing

### FakeMailboxResolver

```python
@dataclass(slots=True)
class FakeMailboxResolver[T]:
    """Test double for mailbox resolution.

    Tracks resolution calls and allows configuration of behavior.
    """

    mailboxes: dict[str, Mailbox[T]] = field(default_factory=dict)
    resolution_log: list[str] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)

    def resolve(self, identifier: str) -> Mailbox[T]:
        self.resolution_log.append(identifier)

        if identifier in self.fail_on:
            raise MailboxResolutionError(identifier)

        if identifier not in self.mailboxes:
            self.mailboxes[identifier] = CollectingMailbox()

        return self.mailboxes[identifier]

    def resolve_optional(self, identifier: str) -> Mailbox[T] | None:
        try:
            return self.resolve(identifier)
        except MailboxResolutionError:
            return None
```

### Testing Message Reply

```python
def test_reply_mailbox_resolves():
    resolver = FakeMailboxResolver()
    msg = Message(
        id="1",
        body="test",
        receipt_handle="h1",
        delivery_count=1,
        enqueued_at=datetime.now(UTC),
        reply_to="responses",
        _reply_resolver=resolver,
    )

    mailbox = msg.reply_mailbox()
    assert mailbox is resolver.mailboxes["responses"]
    assert resolver.resolution_log == ["responses"]


def test_reply_mailbox_without_reply_to():
    msg = Message(
        id="1",
        body="test",
        receipt_handle="h1",
        delivery_count=1,
        enqueued_at=datetime.now(UTC),
        reply_to=None,
    )

    with pytest.raises(ReplyMailboxUnavailableError):
        msg.reply_mailbox()
```

## Caching Considerations

### Why Cache Factory-Created Mailboxes?

Without caching, each `resolve()` call creates a new mailbox instance:

```python
# BAD: Creates new RedisMailbox each time
for msg in requests.receive():
    reply = msg.reply_mailbox()  # New connection, new threads
    reply.send(result)
    # Leaked mailbox with active connections!
```

With caching:

```python
# GOOD: Reuses cached mailbox
resolver = CompositeResolver(registry={}, factory=factory, cache={})

for msg in requests.receive():
    reply = msg.reply_mailbox()  # Returns cached instance
    reply.send(result)
    # No leak - same mailbox reused
```

### Cache Eviction

The default `CompositeResolver` does not evict. For long-running processes
with many dynamic queues, consider:

1. **LRU cache** with max size
2. **TTL-based eviction** for stale entries
3. **Explicit cleanup** via `resolver.evict(identifier)`

```python
@dataclass(slots=True)
class LRUCompositeResolver[T]:
    """Composite resolver with LRU cache eviction."""

    registry: Mapping[str, Mailbox[T]]
    factory: MailboxFactory[T]
    max_cache_size: int = 1000
    _cache: OrderedDict[str, Mailbox[T]] = field(default_factory=OrderedDict)

    def resolve(self, identifier: str) -> Mailbox[T]:
        if identifier in self.registry:
            return self.registry[identifier]

        if identifier in self._cache:
            self._cache.move_to_end(identifier)
            return self._cache[identifier]

        mailbox = self.factory.create(identifier)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()

        self._cache[identifier] = mailbox
        return mailbox
```

## Limitations

- **No distributed cache**: Each process has its own cache. Shared state
  requires external coordination (Redis, database).
- **No health checks**: Cached mailboxes may have stale connections. Consider
  periodic validation or connection pooling at the backend level.
- **Single type parameter**: Resolver is typed for one message type. Use
  separate resolvers or cast for heterogeneous mailboxes.
- **Synchronous only**: Resolution is blocking. Async resolution not supported.

## Future Considerations

- **URI-based resolution**: Parse `redis://host/queue` or `sqs://region/name`
  for backend selection (sketched in URIMailboxFactory).
- **Metrics**: Resolution latency, cache hit rate, factory call count.
- **Health monitoring**: Periodic liveness checks on cached mailboxes.
- **Distributed cache**: Redis-backed resolver cache for multi-process deployments.

## Related Specifications

- `specs/MAILBOX.md` - Core mailbox abstraction and semantics
- `specs/RESOURCE_REGISTRY.md` - DI container for lifecycle-scoped resources
- `specs/MAIN_LOOP.md` - MainLoop orchestration using mailboxes
