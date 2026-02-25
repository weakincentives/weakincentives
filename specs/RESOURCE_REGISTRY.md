# Resource Registry Specification

## Purpose

Dependency injection with scope-aware lifecycle management. Core at `src/weakincentives/resources/`.

## Principles

- **Lazy by default**: Construct on first access
- **Explicit scopes**: Every binding declares lifetime
- **Dependency resolution**: Automatic graph resolution
- **Cycle detection**: Fast failure with clear errors
- **Immutable configuration**: Only scope caches mutable
- **Prompt-owned lifecycle**: Managed via context manager

## Module Structure

```
src/weakincentives/resources/
├── __init__.py     # Public API
├── scope.py        # Scope enum
├── binding.py      # Binding dataclass, Provider type
├── protocols.py    # ResourceResolver, Closeable, PostConstruct, Snapshotable
├── errors.py       # Error hierarchy
├── registry.py     # ResourceRegistry (immutable)
└── context.py      # ScopedResourceContext (mutable)
```

## Scopes

At `src/weakincentives/resources/scope.py`:

| Scope | Lifetime |
| --- | --- |
| `SINGLETON` | One per prompt context |
| `TOOL_CALL` | Fresh per tool invocation |
| `PROTOTYPE` | Fresh every access |

### Selection Guidelines

| Scope | When | Examples |
| --- | --- | --- |
| SINGLETON | Expensive, stateless/thread-safe | HTTP clients, config |
| TOOL_CALL | Fresh state per tool | Request tracers |
| PROTOTYPE | Cheap, independent instances | Builders, buffers |

## Core Types

### Binding

At `src/weakincentives/resources/binding.py`:

| Field | Description |
| --- | --- |
| `protocol` | Type this binding satisfies |
| `provider` | Factory `Callable[[ResourceResolver], T]` |
| `scope` | Instance lifetime (default SINGLETON) |
| `eager` | Initialize during `start()` |
| `provided` | Pre-constructed instance, accessible without context manager |

Factory: `Binding.instance(protocol, value)` for pre-constructed instances.

### ResourceResolver Protocol

At `src/weakincentives/resources/protocols.py`:

| Method | Description |
| --- | --- |
| `get(protocol)` | Resolve or raise `UnboundResourceError` |
| `get_optional(protocol)` | Resolve or return None |

### ResourceRegistry

At `src/weakincentives/resources/registry.py`:

| Method | Description |
| --- | --- |
| `of(*bindings)` | Create from bindings |
| `build(mapping)` | Create from pre-constructed instances |
| `merge(other, strict)` | Compose registries |
| `conflicts(other)` | Find overlapping bindings |
| `open()` | Context manager for resource lifecycle |
| `binding_for(protocol)` | Return the binding for a protocol, or None |

### ScopedResourceContext

At `src/weakincentives/resources/context.py`:

| Method | Description |
| --- | --- |
| `get(protocol)` | Resolve resource |
| `get_optional(protocol)` | Resolve if bound |
| `start()` | Initialize eager singletons |
| `close()` | Dispose Closeable resources |
| `tool_scope()` | Context manager for TOOL_CALL scope |

## Lifecycle Protocols

### Snapshotable

At `src/weakincentives/resources/protocols.py`. `snapshot(tag=)` / `restore(snapshot)`
for rollback. Used by `InMemoryFilesystem` (structural sharing) and `HostFilesystem`
(git commits).

### Closeable

Resources requiring cleanup implement `close() -> None`; closed in reverse
instantiation order.

### PostConstruct

Resources needing post-construction initialization implement `post_construct() -> None`.
Failures prevent caching and are wrapped in `ProviderError`.

## Error Hierarchy

| Error | Description |
| --- | --- |
| `ResourceError` | Base class |
| `UnboundResourceError` | No binding for protocol |
| `CircularDependencyError` | Cycle detected with path |
| `DuplicateBindingError` | Same protocol bound twice |
| `ProviderError` | Provider raised exception |

## Prompt Integration

Resources collected from (lowest to highest precedence):

1. `PromptTemplate.resources`
1. Section `resources()` methods (depth-first)
1. `bind(resources=...)` at bind time

### Usage

Use `prompt.resources` as a context manager; access resources via `.get(Protocol)`.
Pass extra bindings at bind time via `bind(resources={Protocol: instance})`.
See `src/weakincentives/prompt/prompt.py` for `PromptResources`.

## Transaction Patterns

`tool_transaction(session, resource_context, tag)` from `runtime/transactions.py`
provides a context manager that snapshots session and resource state on entry and
restores it on failure.

## Testing Patterns

Build test registries via `ResourceRegistry.build({Protocol: mock_instance})` to
replace implementations. Verify cleanup via resource `.closed` attribute.

## Limitations

- **Synchronous only**: No async providers
- **No conditional bindings**: Use explicit registry construction
- **No interception**: No AOP-style interceptors
- **No named bindings**: Use wrapper types
- **Snapshot scope**: Only SINGLETON resources snapshotted

## Future Considerations

- Modules for grouping bindings
- Qualifiers for multiple implementations
- Async providers
- Health checks
- Metrics instrumentation
