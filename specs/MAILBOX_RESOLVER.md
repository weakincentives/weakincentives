# Mailbox Resolver Specification

## Purpose

`MailboxResolver` provides service discovery for mailbox instances, enabling
dynamic reply routing via string identifiers. This abstraction sits between
the `Message.reply()` API and backend-specific mailbox construction.

```python
# Worker sends reply - routing and resolution happen internally
for msg in requests.receive(visibility_timeout=300):
    result = process(msg.body)
    msg.reply(result)  # Routes via ReplyRoutes, resolves via MailboxResolver
    msg.acknowledge()
```

**Use MailboxResolver for:** Dynamic reply routing, multi-tenant mailbox
discovery, backend-agnostic mailbox construction from string identifiers.

**Use ResourceRegistry for:** Static dependency injection with scope-aware
lifecycle management (singletons, per-tool instances).

## How Reply Routing Works

When `Message.reply(body)` is called, two steps occur:

1. **Route selection**: `ReplyRoutes.route_for(body)` determines the mailbox
   identifier based on the body's type
1. **Mailbox resolution**: `MailboxResolver.resolve(identifier)` returns the
   target mailbox

```
reply(body)
    │
    ▼
ReplyRoutes.route_for(body)  →  identifier (str)
    │
    ▼
MailboxResolver.resolve(identifier)  →  Mailbox
    │
    ▼
mailbox.send(body)
```

This separation allows:

- **Sender control**: Client specifies type→identifier mapping via ReplyRoutes
- **Worker flexibility**: Worker doesn't need to know mailbox locations
- **Backend agnosticism**: Same ReplyRoutes work with any resolver backend

## Comparison with ResourceResolver

| Aspect | ResourceResolver | MailboxResolver |
| ------------- | ---------------------------- | ----------------------------- |
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
@runtime_checkable
class MailboxResolver(Protocol[R]):
    """Resolves string identifiers to Mailbox instances.

    Type parameter R is the body type for resolved mailboxes.
    Resolved mailboxes are Mailbox[R, None] (send-only, no nested replies).
    """

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        """Resolve an identifier to a mailbox instance.

        Args:
            identifier: String identifier from ReplyRoutes.route_for().

        Returns:
            Mailbox instance for sending replies.

        Raises:
            MailboxResolutionError: Cannot resolve identifier.
        """
        ...

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        """Resolve if possible, return None otherwise.

        Use when resolution failure is expected and should not raise.
        """
        ...
```

### MailboxFactory

```python
class MailboxFactory(Protocol[R]):
    """Creates mailbox instances from string identifiers.

    Factories are stateless creators - they don't cache. Caching is the
    resolver's responsibility.
    """

    def create(self, identifier: str) -> Mailbox[R, None]:
        """Create a new mailbox for the given identifier.

        Created mailboxes are send-only (no reply resolution) to prevent
        resource leaks from ephemeral reply destinations.
        """
        ...
```

### RegistryResolver

```python
@dataclass(slots=True, frozen=True)
class RegistryResolver(Generic[R]):
    """Simple resolver backed by a static registry.

    Use when all mailbox destinations are known at configuration time.
    """

    registry: Mapping[str, Mailbox[R, None]]

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        if identifier not in self.registry:
            raise MailboxResolutionError(identifier)
        return self.registry[identifier]

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        return self.registry.get(identifier)
```

### CompositeResolver

```python
@dataclass(slots=True, frozen=True)
class CompositeResolver(Generic[R]):
    """Combines a registry with a factory for dynamic resolution.

    Resolution order:
    1. Check registry for pre-registered mailbox (fast path)
    2. Fall back to factory for dynamic creation (lazy path)

    Factory-created mailboxes are cached in the registry to prevent
    resource leaks from repeated creation of the same identifier.
    """

    registry: MutableMapping[str, Mailbox[R, None]]
    factory: MailboxFactory[R] | None = None

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        # Fast path: already in registry
        if identifier in self.registry:
            return self.registry[identifier]

        # Lazy path: create via factory and cache
        if self.factory is None:
            raise MailboxResolutionError(identifier)

        mailbox = self.factory.create(identifier)
        self.registry[identifier] = mailbox
        return mailbox

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        try:
            return self.resolve(identifier)
        except MailboxResolutionError:
            return None
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

`Message.reply()` combines route selection with mailbox resolution:

```python
@dataclass(slots=True)
class Message(Generic[T, R]):
    reply_routes: ReplyRoutes | None

    # Bound at receive() time by the mailbox
    _reply_fn: Callable[[str, R], str]
    _finalized: bool = False

    def reply(self, body: R) -> str:
        if self._finalized:
            raise MessageFinalizedError(...)
        if self.reply_routes is None:
            raise ReplyNotAvailableError("No reply routes configured")

        # Step 1: Route selection (type-based)
        identifier = self.reply_routes.route_for(body)

        # Step 2: Resolution and send (via bound callback)
        return self._reply_fn(identifier, body)

    def acknowledge(self) -> None:
        self._acknowledge_fn()
        self._finalized = True
```

Mailboxes bind their resolver to messages at receive time:

```python
class InMemoryMailbox(Generic[T, R]):
    reply_resolver: MailboxResolver[R] | None

    def _reply(self, identifier: str, body: R) -> str:
        """Resolve identifier and send reply."""
        if self.reply_resolver is None:
            raise ReplyNotAvailableError("No resolver configured")

        try:
            mailbox = self.reply_resolver.resolve(identifier)
        except MailboxResolutionError as e:
            raise ReplyNotAvailableError(f"Cannot resolve '{identifier}'") from e

        return mailbox.send(body)

    def receive(self, ...) -> Sequence[Message[T, R]]:
        # Bind resolver to each message
        return [
            Message(
                ...,
                reply_routes=in_flight.reply_routes,
                _reply_fn=self._reply,
            )
            for in_flight in received
        ]
```

## Backend Factories

Factories create mailboxes from identifiers. They don't cache—that's the
resolver's job.

```python
@dataclass(slots=True, frozen=True)
class InMemoryMailboxFactory(Generic[R]):
    """Creates in-memory mailboxes (testing/single-process)."""

    def create(self, identifier: str) -> InMemoryMailbox[R, None]:
        return InMemoryMailbox(name=identifier, _send_only=True)


@dataclass(slots=True, frozen=True)
class RedisMailboxFactory(Generic[R]):
    """Creates Redis-backed mailboxes (production/distributed)."""

    client: RedisClient
    prefix: str = "wink:queue:"

    def create(self, identifier: str) -> RedisMailbox[R, None]:
        return RedisMailbox(
            name=f"{self.prefix}{identifier}",
            client=self.client,
            _send_only=True,
        )
```

**Send-only mode**: Factory-created mailboxes use `_send_only=True` to disable
reply resolution and reaper threads. This prevents resource leaks from
ephemeral reply destinations that may be created dynamically.

### URI-Based Factory

For backend selection based on identifier scheme:

```python
@dataclass(slots=True, frozen=True)
class URIMailboxFactory(Generic[R]):
    """Parses memory://, redis://, sqs:// schemes."""

    redis_client: RedisClient | None = None
    sqs_client: SQSClient | None = None

    def create(self, identifier: str) -> Mailbox[R, None]:
        parsed = urlparse(identifier)
        match parsed.scheme:
            case "memory":
                return InMemoryMailbox(name=parsed.path, _send_only=True)
            case "redis":
                if self.redis_client is None:
                    raise MailboxResolutionError(identifier)
                return RedisMailbox(
                    name=parsed.path,
                    client=self.redis_client,
                    _send_only=True,
                )
            case "sqs":
                if self.sqs_client is None:
                    raise MailboxResolutionError(identifier)
                return SQSMailbox(
                    queue_url=parsed.path,
                    client=self.sqs_client,
                )
            case _:
                raise MailboxResolutionError(identifier)
```

Usage with URI-based routing:

```python
routes = ReplyRoutes.typed({
    SuccessResult: "redis://results:success",
    ErrorResult: "redis://results:errors",
    ProgressUpdate: "memory://progress",  # Local-only progress
})
```

## Usage Patterns

### Basic Type-Based Routing

```python
# Setup
registry = {
    "success": InMemoryMailbox(name="success"),
    "errors": InMemoryMailbox(name="errors"),
}
requests = InMemoryMailbox(reply_resolver=RegistryResolver(registry))

# Client sends with type-based routes
requests.send(
    Request(...),
    reply_routes=ReplyRoutes.typed({
        SuccessResult: "success",
        ErrorResult: "errors",
    }),
)

# Worker replies - routing is automatic
for msg in requests.receive():
    try:
        result = process(msg.body)
        msg.reply(SuccessResult(value=result))  # → "success" mailbox
    except ProcessingError as e:
        msg.reply(ErrorResult(message=str(e)))  # → "errors" mailbox
    msg.acknowledge()
```

### Progress Streaming with Type Routing

```python
routes = ReplyRoutes.typed({
    ProgressUpdate: "client-123:progress",
    SuccessResult: "client-123:results",
    ErrorResult: "client-123:results",
})

requests.send(Request(...), reply_routes=routes)

# Worker streams progress to dedicated queue
for msg in requests.receive():
    for i, chunk in enumerate(process_chunks(msg.body)):
        msg.reply(ProgressUpdate(step=i, data=chunk))  # → progress queue

    msg.reply(SuccessResult(final=summarize()))  # → results queue
    msg.acknowledge()
```

### Dynamic Reply Queues

```python
# Factory creates mailboxes on demand
resolver = CompositeResolver(
    registry={},
    factory=RedisMailboxFactory(client=redis_client),
)
requests = RedisMailbox(reply_resolver=resolver)

# Each client uses unique reply destinations
client_id = uuid4()
requests.send(
    Request(...),
    reply_routes=ReplyRoutes.typed({
        SuccessResult: f"client-{client_id}:success",
        ErrorResult: f"client-{client_id}:errors",
    }),
)

# Worker resolves dynamically - factory creates and caches
for msg in requests.receive():
    msg.reply(SuccessResult(...))  # Factory creates "client-xxx:success"
    msg.acknowledge()
```

### Multi-Tenant Isolation

```python
def tenant_resolver(tenant_id: str) -> CompositeResolver:
    """Create resolver with tenant-specific mailbox prefix."""
    return CompositeResolver(
        registry={},
        factory=RedisMailboxFactory(prefix=f"tenant-{tenant_id}:"),
    )

# Tenant A's results go to "tenant-A:success", "tenant-A:errors"
# Tenant B's results go to "tenant-B:success", "tenant-B:errors"
```

### Inheritance-Based Routing

```python
@dataclass(frozen=True)
class BaseError: ...

@dataclass(frozen=True)
class ValidationError(BaseError): ...

@dataclass(frozen=True)
class TimeoutError(BaseError): ...

# All errors route to same destination via parent type
routes = ReplyRoutes.typed({
    SuccessResult: "results:success",
    BaseError: "results:errors",  # Catches all error subtypes
})

# Both ValidationError and TimeoutError resolve to "results:errors"
```

## MainLoop Integration

MainLoop uses a single requests mailbox; response routing via `reply_routes`:

```python
class MainLoop:
    def __init__(self, adapter, requests: Mailbox): ...

    def _handle_message(self, msg):
        try:
            result = self._execute(msg.body)
            msg.reply(result)  # Routes based on result type
        except Exception as e:
            msg.reply(ErrorResult(message=str(e)))  # Routes to error destination
        msg.acknowledge()
```

## ResourceRegistry Integration

MailboxResolver can be a DI-managed singleton:

```python
registry = ResourceRegistry.of(
    Binding(
        MailboxResolver,
        lambda r: CompositeResolver(
            registry={},
            factory=RedisMailboxFactory(client=r.get(RedisClient)),
        ),
    ),
)

# Tools access via context
def my_tool(params, *, context):
    resolver = context.resources.get(MailboxResolver)
    notifications = resolver.resolve("notifications")
    notifications.send(Notification(...))
```

## Testing

### Testing Route Selection

```python
def test_route_for_exact_match():
    routes = ReplyRoutes.typed({
        SuccessResult: "success",
        ErrorResult: "errors",
    })

    assert routes.route_for(SuccessResult(value=1)) == "success"
    assert routes.route_for(ErrorResult(message="x")) == "errors"


def test_route_for_inheritance():
    routes = ReplyRoutes.typed({BaseError: "errors"})

    assert routes.route_for(ValidationError()) == "errors"
    assert routes.route_for(TimeoutError()) == "errors"


def test_route_for_default_fallback():
    routes = ReplyRoutes.typed(
        {SuccessResult: "success"},
        default="other",
    )

    assert routes.route_for(UnknownType()) == "other"


def test_route_for_no_match_raises():
    routes = ReplyRoutes.typed({SuccessResult: "success"})

    with pytest.raises(NoRouteError) as exc:
        routes.route_for(ErrorResult(message="x"))

    assert exc.value.body_type is ErrorResult
```

### Testing Resolution

```python
def test_registry_resolver():
    mailbox = CollectingMailbox()
    resolver = RegistryResolver({"dest": mailbox})

    assert resolver.resolve("dest") is mailbox

    with pytest.raises(MailboxResolutionError):
        resolver.resolve("unknown")


def test_composite_resolver_caches_factory_results():
    registry = {}
    factory = InMemoryMailboxFactory()
    resolver = CompositeResolver(registry=registry, factory=factory)

    mb1 = resolver.resolve("dest")
    mb2 = resolver.resolve("dest")

    assert mb1 is mb2  # Same instance (cached)
    assert "dest" in registry
```

### Testing End-to-End Reply

```python
def test_reply_routes_to_correct_mailbox():
    success_mb = CollectingMailbox()
    error_mb = CollectingMailbox()

    resolver = RegistryResolver({
        "success": success_mb,
        "errors": error_mb,
    })
    requests = InMemoryMailbox(reply_resolver=resolver)

    requests.send(
        "process this",
        reply_routes=ReplyRoutes.typed({
            SuccessResult: "success",
            ErrorResult: "errors",
        }),
    )

    msg = requests.receive()[0]
    msg.reply(SuccessResult(value=42))
    msg.reply(ErrorResult(message="warning"))
    msg.acknowledge()

    assert len(success_mb.sent) == 1
    assert success_mb.sent[0].value == 42
    assert len(error_mb.sent) == 1
    assert error_mb.sent[0].message == "warning"
```

## Limitations

- **No caching in factories**: Factories create new instances each call.
  CompositeResolver provides caching; RegistryResolver requires pre-population.
- **Single type parameter**: Resolver is typed for one reply body type. Use
  `MailboxResolver[SuccessResult | ErrorResult]` for union types.
- **Type serialization**: ReplyRoutes serialize type keys as fully-qualified
  names. Types must be importable at deserialization time.

## Related Specifications

- `specs/MAILBOX.md` - Core mailbox abstraction, ReplyRoutes, and semantics
- `specs/RESOURCE_REGISTRY.md` - DI container for lifecycle-scoped resources
- `specs/MAIN_LOOP.md` - MainLoop orchestration using mailboxes
