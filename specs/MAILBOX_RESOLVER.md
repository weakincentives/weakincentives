# Mailbox Resolver Specification

## Purpose

`MailboxResolver` provides service discovery for mailbox instances, enabling
dynamic reply routing via string identifiers. This abstraction sits between
the `Message.reply()` API and backend-specific mailbox construction.

```python
# Worker sends reply - resolution happens internally
for msg in requests.receive(visibility_timeout=300):
    result = process(msg.body)
    msg.reply(result)  # Resolves reply_to → Mailbox internally
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

`Message.reply()` resolves the mailbox internally and tracks finalization:

```python
def reply(self, body: R) -> str:
    if self._finalized:
        raise MessageFinalizedError(...)
    mailbox = self._resolver.resolve(self.reply_to)
    return mailbox.send(body)

def acknowledge(self) -> None:
    self._acknowledge_fn()
    self._finalized = True
```

Mailboxes bind their resolver to messages at receive time:

```python
requests = InMemoryMailbox(reply_resolver=resolver)
msg = requests.receive()[0]  # msg._resolver = resolver
msg.reply(result)            # Uses bound resolver
```

## Backend Factories

Factories create mailboxes from identifiers. They don't cache—that's the
resolver's job.

```python
class InMemoryMailboxFactory[T]:
    def create(self, identifier: str) -> InMemoryMailbox[T]:
        return InMemoryMailbox(name=identifier)

class RedisMailboxFactory[T]:
    client: RedisClient
    prefix: str = "wink:queue:"

    def create(self, identifier: str) -> RedisMailbox[T]:
        return RedisMailbox(name=identifier, client=self.client)
```

URI-based factory for backend selection:

```python
class URIMailboxFactory[T]:
    """Parses memory://, redis://, sqs:// schemes."""

    def create(self, identifier: str) -> Mailbox[T]:
        match urlparse(identifier).scheme:
            case "memory": return InMemoryMailbox(...)
            case "redis":  return RedisMailbox(...)
            case "sqs":    return SQSMailbox(...)
```

## Usage Patterns

### Basic Reply

```python
# Setup
registry = {"responses": InMemoryMailbox()}
requests = InMemoryMailbox(reply_resolver=RegistryResolver(registry))

# Client
requests.send(Request(...), reply_to="responses")

# Worker
for msg in requests.receive():
    msg.reply(process(msg.body))
    msg.acknowledge()
```

### Multiple Replies (Progress Streaming)

```python
for msg in requests.receive():
    msg.reply(Progress(step=1))
    msg.reply(Progress(step=2))
    msg.reply(Complete(result=...))
    msg.acknowledge()
```

### Dynamic Reply Queues

```python
# Factory creates mailboxes on demand
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(...))
requests = RedisMailbox(reply_resolver=resolver)

# Client uses unique reply queue
requests.send(Request(...), reply_to=f"client-{uuid4()}")

# Worker resolves dynamically - factory creates if not cached
for msg in requests.receive():
    msg.reply(result)  # Resolves "client-xxx" → new RedisMailbox
    msg.acknowledge()
```

### Multi-Tenant Isolation

```python
def tenant_resolver(tenant_id: str) -> CompositeResolver:
    return CompositeResolver(
        registry={},
        factory=RedisMailboxFactory(prefix=f"tenant-{tenant_id}:"),
    )
```

## MainLoop Integration

MainLoop uses a single requests mailbox; response routing via `reply_to`:

```python
class MainLoop:
    def __init__(self, adapter, requests: Mailbox): ...

    def _handle_message(self, msg):
        result = self._execute(msg.body)
        msg.reply(result)
        msg.acknowledge()
```

## ResourceRegistry Integration

MailboxResolver can be a DI-managed singleton:

```python
registry = ResourceRegistry.of(
    Binding(MailboxResolver, lambda r: CompositeResolver(...)),
)

# Tools access via context
def my_tool(params, *, context):
    resolver = context.resources.get(MailboxResolver)
    resolver.resolve("notifications").send(Notification(...))
```

## Testing

### FakeMailboxResolver

```python
class FakeMailboxResolver[T]:
    mailboxes: dict[str, Mailbox[T]]  # Auto-creates CollectingMailbox
    resolution_log: list[str]          # Tracks resolve() calls
    fail_on: set[str]                  # Identifiers that raise
```

### Testing Reply

```python
def test_reply_resolves_and_sends():
    resolver = FakeMailboxResolver()
    msg = make_message(reply_to="responses", resolver=resolver)

    msg.reply("hello")

    assert resolver.resolution_log == ["responses"]
    assert resolver.mailboxes["responses"].sent == ["hello"]


def test_reply_after_ack_raises():
    msg = make_message(reply_to="responses")
    msg.acknowledge()

    with pytest.raises(MessageFinalizedError):
        msg.reply("too late")
```

## Caching Considerations

Without caching, each `reply()` to a dynamic queue creates a new mailbox
(new connections, threads, resources). `CompositeResolver` caches
factory-created instances to prevent leaks.

For long-running processes with many dynamic queues:
- **LRU cache** with max size
- **TTL-based eviction** for stale entries
- **Explicit cleanup** via `resolver.evict(identifier)`

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
