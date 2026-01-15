# Mailbox Resolver Specification

Reconstructs mailbox instances from string identifiers in distributed systems.

**Source:** `src/weakincentives/runtime/mailbox/_resolver.py`

## Purpose

Redis mailboxes serialize only the mailbox name, not the instance. The resolver reconstructs mailboxes from stored names on receive.

```python
# In-memory: direct reference stored
requests.send("hello", reply_to=responses)
msg.reply(result)  # responses.send(result) directly

# Redis: name serialized, resolver reconstructs
redis_requests.send("hello", reply_to=redis_responses)  # Stores "responses"
msg.reply(result)  # resolver.resolve("responses") → mailbox
```

## Core Types

### MailboxResolver Protocol

```python
class MailboxResolver[R](Protocol):
    def resolve(self, identifier: str) -> Mailbox[R, None]: ...
    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None: ...
```

### CompositeResolver

```python
CompositeResolver[R](
    registry: Mapping[str, Mailbox[R, None]],  # Pre-registered mailboxes
    factory: MailboxFactory[R] | None = None,  # Dynamic creation fallback
)
```

Resolution order: registry first, then factory.

### RegistryResolver

```python
RegistryResolver[R](registry: Mapping[str, Mailbox[R, None]])
```

Simple lookup, no factory fallback.

## Comparison with ResourceRegistry

| Aspect | ResourceRegistry | MailboxResolver |
|--------|------------------|-----------------|
| Key type | `type[T]` | `str` |
| Purpose | DI container | Service discovery |
| Resolution | Static bindings | Dynamic lookup + factory |

## Usage Patterns

### Redis Reply Pattern

```python
resolver = CompositeResolver(registry={}, factory=RedisMailboxFactory(client))
requests = RedisMailbox(name="requests", client=client, reply_resolver=resolver)

requests.send(Request(...), reply_to=responses)
for msg in requests.receive():
    msg.reply(process(msg.body))  # Resolver reconstructs mailbox
    msg.acknowledge()
```

### Dynamic Reply Queues

```python
# Factory creates mailboxes on demand
client_responses = RedisMailbox(name=f"client-{uuid4()}", client=client)
requests.send(Request(...), reply_to=client_responses)
```

## Limitations

- **Redis-specific**: In-memory mailboxes store direct references
- **No caching**: Factory creates new instance per resolution
- **Single type parameter**: Use separate resolvers for heterogeneous mailboxes
