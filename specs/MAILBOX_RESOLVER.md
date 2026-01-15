# Mailbox Resolver Specification

## Purpose

`MailboxResolver` reconstructs mailbox instances from string identifiers in
distributed systems. Used by Redis mailboxes where `reply_to` references
serialize to names, not actual mailbox instances.

**Implementation:** `src/weakincentives/runtime/mailbox/_resolver.py`

**Use for:** Redis/distributed backends where mailbox names are serialized.

**Not for:** In-memory mailboxes (store direct references).

## Comparison with ResourceResolver

| Aspect | ResourceResolver | MailboxResolver |
|--------|------------------|-----------------|
| Key type | `type[T]` | `str` |
| Purpose | DI container | Service discovery |
| Resolution | Static bindings | Dynamic lookup + factory |
| Caching | Scope-aware | Optional |

## Core Types

### MailboxResolver Protocol

| Method | Description |
|--------|-------------|
| `resolve(identifier)` | Resolve to mailbox, raises on failure |
| `resolve_optional(identifier)` | Resolve or return None |

### MailboxFactory Protocol

```python
def create(self, identifier: str) -> Mailbox[R, None]:
    """Create new mailbox for identifier."""
```

### CompositeResolver

Resolution order:
1. Check registry for pre-registered mailbox
2. Fall back to factory for dynamic creation

### RegistryResolver

Simple resolver backed by static registry only.

### Errors

`MailboxResolutionError`: Cannot resolve identifier (not in registry, no factory).

## Message Integration

`Message.reply()` sends directly to `reply_to` mailbox:

**In-memory:** Direct reference stored.

**Redis:** Resolver reconstructs from stored name.

```python
# Redis setup
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(client))
requests = RedisMailbox(name="requests", reply_resolver=resolver)
```

## Backend Factories

```python
class RedisMailboxFactory[T]:
    client: RedisClient
    prefix: str = "wink:queue:"

    def create(self, identifier: str) -> RedisMailbox[T]:
        return RedisMailbox(name=identifier, client=self.client)
```

URI-based factory for scheme selection: `memory://`, `redis://`, `sqs://`

## Usage Patterns

### Basic Reply (In-Memory)

```python
requests = InMemoryMailbox(name="requests")
responses = InMemoryMailbox(name="responses")
requests.send(Request(...), reply_to=responses)  # Direct reference

for msg in requests.receive():
    msg.reply(process(msg.body))  # Direct call
    msg.acknowledge()
```

### Basic Reply (Redis)

```python
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(client))
requests = RedisMailbox(name="requests", reply_resolver=resolver)

for msg in requests.receive():
    msg.reply(process(msg.body))  # Resolver reconstructs mailbox
    msg.acknowledge()
```

### Dynamic Reply Queues

```python
# Client creates unique reply mailbox
client_responses = RedisMailbox(name=f"client-{uuid4()}", client=client)
requests.send(Request(...), reply_to=client_responses)

# Worker - factory creates mailbox for "client-xxx"
```

### Multi-Tenant

```python
def tenant_resolver(tenant_id: str) -> CompositeResolver:
    return CompositeResolver(
        registry={},
        factory=RedisMailboxFactory(prefix=f"tenant-{tenant_id}:"),
    )
```

## MainLoop Integration

MainLoop uses single requests mailbox; response routing via `reply_to`:

```python
def _handle_message(self, msg):
    result = self._execute(msg.body)
    msg.reply(result)
    msg.acknowledge()
```

## ResourceRegistry Integration

```python
registry = ResourceRegistry.of(
    Binding(MailboxResolver, lambda r: CompositeResolver(...)),
)
```

## Limitations

- **Redis-specific**: In-memory doesn't need resolvers
- **No caching**: Factory creates new mailbox each time
- **Single type parameter**: Use separate resolvers for heterogeneous mailboxes

## Related Specifications

- `specs/MAILBOX.md` - Core mailbox abstraction
- `specs/RESOURCE_REGISTRY.md` - DI container
- `specs/MAIN_LOOP.md` - MainLoop orchestration
