# Dependency Injection with Resources

*Canonical spec: [specs/RESOURCE_REGISTRY.md](../specs/RESOURCE_REGISTRY.md)*

WINK provides a lightweight dependency injection system through the `resources`
module. It manages object lifecycles, resolves dependencies automatically, and
integrates with tool execution for proper cleanup and rollback.

## Why Dependency Injection?

Agents need access to external services—HTTP clients, filesystems, databases,
clocks. Hardcoding these dependencies makes testing difficult and configuration
inflexible. WINK's resource system:

- **Enables testing**: Inject mocks and fakes without global state
- **Manages lifecycles**: Automatic cleanup in correct order
- **Scopes resources**: Fresh instances per tool call when needed
- **Prevents cycles**: Detects circular dependencies at resolution time

## Core Concepts

### Bindings

A `Binding` describes how to construct a resource and how long it lives:

```python nocheck
from weakincentives.resources import Binding, Scope

# Simple binding with provider function
config_binding = Binding(
    Config,                         # Protocol/type this satisfies
    lambda r: Config.from_env(),    # Factory receiving resolver
    scope=Scope.SINGLETON,          # Lifetime (default)
)

# Binding with dependencies
client_binding = Binding(
    HTTPClient,
    lambda r: HTTPClient(r.get(Config).base_url),  # Resolve Config first
)

# Pre-constructed instance
clock_binding = Binding.instance(Clock, SystemClock())
```

The provider function receives a `ResourceResolver` to access other resources,
enabling automatic dependency resolution.

### Scopes

Scopes control when resources are created and how long they live:

| Scope | Behavior | Use Case |
| --- | --- | --- |
| `SINGLETON` | Created once, cached for session | HTTP clients, config, shared services |
| `TOOL_CALL` | Fresh per tool invocation, disposed after | Request tracers, per-call context |
| `PROTOTYPE` | Fresh every access, never cached | Stateless builders, query factories |

Choose scope based on the resource's characteristics:

```python nocheck
from weakincentives.resources import Binding, Scope

# Expensive to create, thread-safe → SINGLETON
Binding(HTTPClient, make_client, scope=Scope.SINGLETON)

# Needs fresh state per tool → TOOL_CALL
Binding(RequestTracer, lambda r: RequestTracer(), scope=Scope.TOOL_CALL)

# Cheap, independent instances → PROTOTYPE
Binding(QueryBuilder, lambda r: QueryBuilder(), scope=Scope.PROTOTYPE)
```

### Registry

A `ResourceRegistry` holds an immutable collection of bindings:

```python nocheck
from weakincentives.resources import ResourceRegistry, Binding

# From explicit bindings
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Binding(Filesystem, lambda r: InMemoryFilesystem()),
)

# From pre-constructed instances (auto-wrapped as SINGLETON)
registry = ResourceRegistry.build({
    Config: config,
    HTTPClient: http_client,
    Filesystem: fs,
})
```

Registries can be merged for composition:

```python nocheck
base_registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
)

extended_registry = base_registry.merge(
    ResourceRegistry.of(
        Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    )
)
```

### Resolution Context

The registry creates a `ScopedResourceContext` that manages resolution and
lifecycle:

```python nocheck
from weakincentives.resources import ResourceRegistry

registry = ResourceRegistry.of(...)

# Context manager handles startup and cleanup
with registry.open() as ctx:
    config = ctx.get(Config)        # Resolves lazily, caches
    client = ctx.get(HTTPClient)    # Resolves Config first

    # Optional resolution (None if unbound)
    tracer = ctx.get_optional(Tracer)
# All Closeable resources cleaned up here
```

## Lifecycle Protocols

Resources can implement protocols for lifecycle management.

### Closeable

Resources needing cleanup implement `Closeable`:

```python nocheck
from weakincentives.resources import Closeable

@dataclass
class DatabasePool(Closeable):
    url: str
    _connections: list[Connection] = field(default_factory=list)

    def close(self) -> None:
        """Called when scope ends."""
        for conn in self._connections:
            conn.close()
        self._connections.clear()
```

Resources are closed in **reverse instantiation order**, ensuring dependencies
outlive their dependents.

### PostConstruct

For initialization that should happen after construction but before use:

```python nocheck
from weakincentives.resources import PostConstruct, Closeable

@dataclass
class DatabaseClient(PostConstruct, Closeable):
    url: str
    _pool: Pool | None = None

    def post_construct(self) -> None:
        """Called after construction, before caching."""
        self._pool = create_pool(self.url)
        self._pool.verify_connection()  # Fail fast if unreachable

    def close(self) -> None:
        if self._pool:
            self._pool.close()
```

If `post_construct()` raises, the resource is not cached and `close()` is
called for cleanup.

### Snapshotable

Resources supporting transactional rollback implement `Snapshotable`:

```python nocheck
from weakincentives.resources import Snapshotable

@dataclass
class InMemoryFilesystem(Snapshotable[FSSnapshot]):
    def snapshot(self, *, tag: str | None = None) -> FSSnapshot:
        """Capture current state."""
        return FSSnapshot(files=self._files.copy())

    def restore(self, snapshot: FSSnapshot) -> None:
        """Revert to captured state."""
        self._files = snapshot.files.copy()
```

This enables automatic rollback when tools fail—the filesystem returns to its
state before the tool executed.

## Tool-Call Scoped Resources

When tools need fresh instances that don't persist between invocations, use
`TOOL_CALL` scope:

```python nocheck
from weakincentives.resources import Binding, Scope

registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),  # SINGLETON
    Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
)

with registry.open() as ctx:
    # Singleton: same instance everywhere
    config1 = ctx.get(Config)
    config2 = ctx.get(Config)
    assert config1 is config2

    # Tool scope: fresh instances, automatic cleanup
    with ctx.tool_scope() as resolver:
        tracer1 = resolver.get(Tracer)
        # Use tracer during tool execution
    # tracer1.close() called automatically

    with ctx.tool_scope() as resolver:
        tracer2 = resolver.get(Tracer)
        assert tracer1 is not tracer2  # Fresh instance
```

## Prompt Integration

Resources integrate with the prompt system through `prompt.resources`:

```python nocheck
prompt = Prompt(template).bind(
    Params(...),
    resources={
        Clock: SystemClock(),
        Filesystem: InMemoryFilesystem(),
    },
)

# Enter resource context for lifecycle management
with prompt.resources:
    fs = prompt.resources.get(Filesystem)
    result = adapter.evaluate(prompt, session=session)
# Resources cleaned up
```

Resources are collected from multiple sources (lowest to highest precedence):

1. `PromptTemplate.resources` — base resources from template
1. Section `resources()` methods — resources declared by sections
1. `bind(resources=...)` — resources provided at bind time

## Error Handling

The resource system provides clear errors for common problems:

| Error | Cause |
| --- | --- |
| `UnboundResourceError` | Requested a protocol with no binding |
| `CircularDependencyError` | A depends on B depends on A |
| `DuplicateBindingError` | Same protocol bound twice (in strict mode) |
| `ProviderError` | Provider function raised an exception |

Circular dependencies are detected at resolution time with a clear cycle path:

```python nocheck
# This will fail with CircularDependencyError
registry = ResourceRegistry.of(
    Binding(A, lambda r: A(r.get(B))),  # A needs B
    Binding(B, lambda r: B(r.get(A))),  # B needs A → cycle!
)
```

## Testing Patterns

### Replace Implementations

Inject test doubles without changing production code:

```python nocheck
# Production registry
prod_registry = ResourceRegistry.of(
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Binding(Filesystem, lambda r: HostFilesystem()),
)

# Test registry with mocks
test_registry = ResourceRegistry.build({
    HTTPClient: MockHTTPClient(responses={"GET /api": "ok"}),
    Filesystem: InMemoryFilesystem(),
    Config: Config(url="http://test"),
})

# Same code, different resources
with test_registry.open() as ctx:
    service = ctx.get(MyService)
    result = service.do_something()
    # Verify against mock
```

### Verify Lifecycle

Test that resources are properly initialized and cleaned up:

```python nocheck
class TrackedResource(Closeable):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

registry = ResourceRegistry.of(
    Binding(TrackedResource, lambda r: TrackedResource()),
)

with registry.open() as ctx:
    resource = ctx.get(TrackedResource)
    assert not resource.closed

assert resource.closed  # Verified after context exits
```

### Test Eager Initialization

Verify eager bindings are created during startup:

```python nocheck
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env(), eager=True),
)

with registry.open() as ctx:
    # Config already in cache from start()
    assert Config in ctx.singleton_cache
```

## Best Practices

### Do

- **Use protocols**: Bind to protocols/ABCs, not concrete classes
- **Keep providers pure**: Providers should only construct and wire
- **Fail fast**: Use `post_construct()` for validation that should fail early
- **Close gracefully**: Implement `Closeable` for any resource holding connections

### Avoid

- **Avoid TOOL_CALL for expensive resources**: Creation cost multiplied by tool calls
- **Avoid circular dependencies**: Refactor to break cycles
- **Avoid global state**: Inject everything explicitly
- **Avoid side effects in providers**: Use `post_construct()` instead

## Limitations

The current implementation has intentional constraints:

- **Synchronous only**: No async providers or resolution
- **No conditional bindings**: Construct registry programmatically
- **No interception**: No AOP-style middleware
- **No named bindings**: Use wrapper types to distinguish
- **Snapshot scope**: Only SINGLETON resources participate in snapshots

## Next Steps

- [Tools](tools.md): Learn how tools use resources
- [Sessions](sessions.md): Understand state management
- [Testing](testing.md): More testing patterns for resources
