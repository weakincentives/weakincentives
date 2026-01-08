# Mailbox Resolver Specification

## Purpose

`MailboxResolver` enables reconstruction of mailbox instances from string
identifiers in distributed systems. This is primarily used by Redis mailboxes
where the `reply_to` mailbox reference cannot be serialized directly—only
the mailbox name is stored in Redis.

```python
# Direct mailbox references (in-memory)
requests.send("hello", reply_to=responses)  # Direct reference
msg.reply(result)  # Calls responses.send() directly

# Redis: name serialized, resolver reconstructs mailbox on receive
redis_requests.send("hello", reply_to=redis_responses)  # Stores "responses"
msg.reply(result)  # Resolver reconstructs mailbox from stored name
```

**Use MailboxResolver for:** Redis/distributed mailbox backends where mailbox
names are serialized and must be reconstructed on receive.

**Use direct Mailbox references for:** In-memory mailboxes where the actual
mailbox instance can be stored and passed to Message.reply_to.

**Use ResourceRegistry for:** Static dependency injection with scope-aware
lifecycle management (singletons, per-tool instances).

## Comparison with ResourceResolver

| Aspect | ResourceResolver | MailboxResolver |
| ------------- | ---------------------------- | --------------------------- |
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
class MailboxResolver[R](Protocol):
    """Resolves string identifiers to Mailbox instances.

    Type parameter R matches the reply type from Mailbox[T, R].
    """

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        """Resolve an identifier to a mailbox instance."""
        ...

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        """Resolve if possible, return None otherwise."""
        ...
```

### MailboxFactory

```python
class MailboxFactory[R](Protocol):
    """Creates mailbox instances from string identifiers."""

    def create(self, identifier: str) -> Mailbox[R, None]:
        """Create a new mailbox for the given identifier."""
        ...
```

### CompositeResolver

```python
@dataclass(slots=True, frozen=True)
class CompositeResolver[R]:
    """Combines a registry with a factory for dynamic resolution.

    Resolution order:
    1. Check registry for pre-registered mailbox
    2. Fall back to factory for dynamic creation
    """

    registry: Mapping[str, Mailbox[R, None]]
    factory: MailboxFactory[R] | None = None

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        if identifier in self.registry:
            return self.registry[identifier]
        if self.factory is None:
            raise MailboxResolutionError(identifier)
        return self.factory.create(identifier)
```

### RegistryResolver

```python
@dataclass(slots=True, frozen=True)
class RegistryResolver[R]:
    """Simple resolver backed by a static registry."""

    registry: Mapping[str, Mailbox[R, None]]

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        if identifier not in self.registry:
            raise MailboxResolutionError(identifier)
        return self.registry[identifier]
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

`Message.reply()` sends directly to the `reply_to` mailbox instance:

```python
def reply(self, body: R) -> str:
    if self._finalized:
        raise MessageFinalizedError(...)
    if self.reply_to is None:
        raise ReplyNotAvailableError(...)
    return self.reply_to.send(body)  # Direct mailbox call

def acknowledge(self) -> None:
    self._acknowledge_fn()
    self._finalized = True
```

For in-memory mailboxes, the `reply_to` mailbox is stored directly:

```python
# In-memory: direct reference stored
requests = InMemoryMailbox(name="requests")
responses = InMemoryMailbox(name="responses")
requests.send("hello", reply_to=responses)
msg = requests.receive()[0]  # msg.reply_to is responses mailbox
msg.reply(result)            # Calls responses.send(result) directly
```

For Redis mailboxes, the resolver reconstructs the mailbox from the stored name:

```python
# Redis: name serialized, resolver reconstructs on receive
redis_requests = RedisMailbox(name="requests", reply_resolver=resolver)
requests.send("hello", reply_to=redis_responses)  # Stores "responses" name
msg = requests.receive()[0]  # resolver.resolve("responses") → mailbox
msg.reply(result)            # Calls reconstructed mailbox.send(result)
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

### Basic Reply (In-Memory)

```python
# Setup - direct mailbox references
requests = InMemoryMailbox(name="requests")
responses = InMemoryMailbox(name="responses")

# Client - pass mailbox reference
requests.send(Request(...), reply_to=responses)

# Worker - reply_to is the actual mailbox instance
for msg in requests.receive():
    msg.reply(process(msg.body))  # Sends directly to responses
    msg.acknowledge()
```

### Basic Reply (Redis)

```python
# Setup - resolver needed for reconstruction
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(client))
requests = RedisMailbox(name="requests", client=client, reply_resolver=resolver)
responses = RedisMailbox(name="responses", client=client)

# Client - pass mailbox reference (name serialized to Redis)
requests.send(Request(...), reply_to=responses)

# Worker - resolver reconstructs mailbox from stored name
for msg in requests.receive():
    msg.reply(process(msg.body))  # Resolved mailbox receives reply
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

### Dynamic Reply Queues (Redis)

```python
# Factory creates mailboxes on demand for unique client queues
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(client))
requests = RedisMailbox(name="requests", reply_resolver=resolver)

# Client creates unique reply mailbox
client_responses = RedisMailbox(name=f"client-{uuid4()}", client=client)
requests.send(Request(...), reply_to=client_responses)

# Worker - factory reconstructs from stored name
for msg in requests.receive():
    msg.reply(result)  # Factory creates mailbox for "client-xxx"
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

### Testing Reply (In-Memory)

```python
def test_reply_sends_to_mailbox():
    responses = CollectingMailbox(name="responses")
    requests = InMemoryMailbox(name="requests")

    requests.send("hello", reply_to=responses)
    msg = requests.receive()[0]
    msg.reply("world")
    msg.acknowledge()

    assert responses.sent == ["world"]


def test_reply_after_ack_raises():
    responses = CollectingMailbox(name="responses")
    requests = InMemoryMailbox(name="requests")

    requests.send("hello", reply_to=responses)
    msg = requests.receive()[0]
    msg.acknowledge()

    with pytest.raises(MessageFinalizedError):
        msg.reply("too late")
```

## Limitations

- **Redis-specific**: In-memory mailboxes don't need resolvers—they store
  direct mailbox references. Resolvers are only needed for distributed
  backends where mailbox instances can't be serialized.
- **No caching**: Factory creates new mailbox on each resolution. Callers
  should cache or use registry for repeated access to same identifier.
- **Single type parameter**: Resolver is typed for one reply type. Use
  separate resolvers for heterogeneous mailboxes.

## Future Considerations

- **Caching**: LRU or TTL-based caching to prevent resource leaks from
  repeated factory creation of the same identifier.
- **URI-based resolution**: Parse `redis://host/queue` or `sqs://region/name`
  for backend selection (sketched in URIMailboxFactory).
- **Metrics**: Resolution latency, factory call count.

## Related Specifications

- `specs/MAILBOX.md` - Core mailbox abstraction and semantics
- `specs/RESOURCE_REGISTRY.md` - DI container for lifecycle-scoped resources
- `specs/MAIN_LOOP.md` - MainLoop orchestration using mailboxes
