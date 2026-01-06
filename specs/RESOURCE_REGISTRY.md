# Resource Registry Specification

## Purpose

This specification defines `weakincentives.resources`, a module providing
dependency injection with scope-aware lifecycle management. It enables complex
agent architectures where resources have dependencies on each other and require
different lifetimes (session-scoped singletons vs. per-tool-call instances).

## Guiding Principles

- **Lazy by default**: Resources are constructed on first access, not upfront.
- **Explicit scopes**: Every binding declares its lifetime; no implicit behavior.
- **Dependency resolution**: Resources can depend on other resources; the
  registry resolves the graph automatically.
- **Cycle detection**: Circular dependencies fail fast with clear errors.
- **Immutable configuration**: The registry is immutable; only scope caches are
  mutable.
- **Prompt-owned lifecycle**: Resource contexts are owned by prompts and managed
  via context manager.
- **Clean API**: Simple, focused interfaces without legacy cruft.

```mermaid
flowchart TB
    subgraph Configuration["Registry Configuration (Immutable)"]
        Bindings["Bindings<br/>(protocol → provider + scope)"]
    end

    subgraph Prompt["Prompt Lifecycle"]
        Enter["__enter__()"]
        Create["create_context()"]
        Start["start()"]
        Use["Tool execution"]
        Close["close()"]
        Exit["__exit__()"]
    end

    subgraph Resolution["Resolution (per request)"]
        Get["get(Protocol)"] --> Check{"Cached?"}
        Check -->|Yes| Return["Return instance"]
        Check -->|No| Resolve["Invoke provider"]
        Resolve --> Deps["Resolve dependencies"]
        Deps --> Construct["Construct instance"]
        Construct --> Cache{"Scope?"}
        Cache -->|SINGLETON| SingletonCache["Session cache"]
        Cache -->|TOOL_CALL| ToolCache["Tool-call cache"]
        Cache -->|PROTOTYPE| NoCache["No cache"]
        SingletonCache --> Return
        ToolCache --> Return
        NoCache --> Return
    end

    subgraph Scopes["Scope Lifecycles"]
        Singleton["SINGLETON<br/>Lives for prompt context"]
        ToolCall["TOOL_CALL<br/>Fresh per invocation"]
        Prototype["PROTOTYPE<br/>Fresh every access"]
    end

    Configuration --> Prompt
    Prompt --> Resolution
    Enter --> Create --> Start --> Use --> Close --> Exit
```

## Module Structure

```
weakincentives/resources/
├── __init__.py      # Public API exports
├── scope.py         # Scope enum
├── binding.py       # Binding dataclass, Provider type alias
├── protocols.py     # ResourceResolver, Closeable, PostConstruct, Snapshotable
├── errors.py        # Error hierarchy
├── registry.py      # ResourceRegistry (immutable config)
└── context.py       # ScopedResourceContext (mutable resolution)
```

> **Note:** Transactional snapshot/restore is handled by `runtime.transactions`
> module, which provides `CompositeSnapshot` combining session and resource
> snapshots. The resource context provides access to snapshotable resources
> via `singleton_cache` for the transactions module to iterate.

## Scopes

Resources declare their lifetime via `Scope`:

```python
class Scope(Enum):
    """Determines instance lifetime and caching behavior."""

    SINGLETON = "singleton"
    """One instance per prompt context. Created on first access, reused thereafter."""

    TOOL_CALL = "tool_call"
    """Fresh instance per tool invocation. Disposed after tool completes."""

    PROTOTYPE = "prototype"
    """Fresh instance on every access. Never cached."""
```

### Scope Selection Guidelines

| Scope | Use When | Examples |
| ----------- | ----------------------------------------------------------- | ----------------------------------------- |
| `SINGLETON` | Expensive to create, stateless or thread-safe | HTTP clients, connection pools, config |
| `TOOL_CALL` | Needs fresh state per tool, or tracks tool-specific context | Request tracers, tool-scoped transactions |
| `PROTOTYPE` | Cheap to create, each caller needs independent instance | Builders, temporary buffers |

## Core Types

### Binding

`Binding[T]` describes how to obtain an instance of protocol `T`:

```python
Provider = Callable[[ResourceResolver], T]

@dataclass(slots=True, frozen=True)
class Binding[T]:
    """Describes how to construct a resource and its lifetime."""

    protocol: type[T]
    """The protocol type this binding satisfies."""

    provider: Provider[T]
    """Factory function that constructs the instance."""

    scope: Scope = Scope.SINGLETON
    """Lifetime of constructed instances."""

    eager: bool = False
    """If True, instantiate during context startup (SINGLETON only)."""

    @staticmethod
    def instance[U](protocol: type[U], value: U) -> Binding[U]:
        """Create a binding for a pre-constructed instance.

        Returns an eager SINGLETON binding that returns the given instance.
        This is the canonical way to register existing objects.
        """
        ...
```

Provider signature:

```python
def my_provider(resolver: ResourceResolver) -> MyService:
    # Request dependencies from resolver
    config = resolver.get(Config)
    http = resolver.get(HTTPClient)
    return MyService(config=config, http=http)
```

### ResourceResolver Protocol

Passed to providers for dependency resolution:

```python
@runtime_checkable
class ResourceResolver(Protocol):
    """Protocol for resolving dependencies during construction."""

    def get[T](self, protocol: type[T]) -> T:
        """Return the resource for the given protocol.

        Raises:
            UnboundResourceError: No binding exists for the protocol.
            CircularDependencyError: Dependency cycle detected.
        """
        ...

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Return the resource if bound, None otherwise."""
        ...
```

### ResourceRegistry

Immutable configuration of resource bindings:

```python
@dataclass(slots=True, frozen=True)
class ResourceRegistry:
    """Immutable configuration of resource bindings."""

    _bindings: Mapping[type[object], Binding[object]]

    @staticmethod
    def of(*bindings: Binding[object]) -> ResourceRegistry:
        """Construct a registry from bindings.

        Raises:
            DuplicateBindingError: Same protocol bound twice.
        """
        ...

    @staticmethod
    def build(mapping: Mapping[type[object], object]) -> ResourceRegistry:
        """Convenience method to create a registry from pre-constructed instances.

        Equivalent to calling of() with Binding.instance() for each entry.
        None values are filtered out.
        """
        ...

    def __contains__(self, protocol: type[object]) -> bool:
        """Check if protocol has a binding."""
        ...

    def __len__(self) -> int:
        """Return number of bindings."""
        ...

    def __iter__(self) -> Iterator[type[object]]:
        """Iterate over bound protocol types."""
        ...

    def binding_for[T](self, protocol: type[T]) -> Binding[T] | None:
        """Return the binding for a protocol, or None if unbound."""
        ...

    def merge(
        self, other: ResourceRegistry, *, strict: bool = False
    ) -> ResourceRegistry:
        """Merge registries; other takes precedence on conflicts.

        Args:
            other: Registry to merge with.
            strict: If True, raise DuplicateBindingError on conflicts.

        Raises:
            DuplicateBindingError: If strict=True and registries share protocols.
        """
        ...

    def conflicts(self, other: ResourceRegistry) -> frozenset[type[object]]:
        """Return protocols bound in both registries."""
        ...

    def eager_bindings(self) -> Sequence[Binding[object]]:
        """Return all bindings marked as eager."""
        ...

    def create_context(
        self,
        *,
        singleton_cache: dict[type[object], object] | None = None,
    ) -> ScopedResourceContext:
        """Create a scoped resolution context."""
        ...
```

### ScopedResourceContext

Mutable context that manages scope caches and performs resolution:

```python
@dataclass(slots=True)
class ScopedResourceContext:
    """Scoped resolution context with lifecycle management."""

    registry: ResourceRegistry
    """Immutable registry configuration."""

    singleton_cache: dict[type[object], object]
    """Cache for SINGLETON-scoped resources."""

    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return resource for protocol.

        Raises:
            UnboundResourceError: No binding exists.
            CircularDependencyError: Dependency cycle detected.
            ProviderError: Provider raised an exception.
        """
        ...

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve if bound, return None otherwise."""
        ...

    def start(self) -> None:
        """Initialize context and instantiate eager singletons."""
        ...

    def close(self) -> None:
        """Dispose all instantiated resources implementing Closeable."""
        ...

    @contextmanager
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter a tool-call scope.

        Resources with TOOL_CALL scope are fresh within this context
        and disposed on exit.
        """
        ...
```

> **Note:** Snapshot/restore operations are handled by the `runtime.transactions`
> module via `create_snapshot()` and `restore_snapshot()` functions, which work
> with `CompositeSnapshot` (combining session and resource state).

## Lifecycle Protocols

### Snapshotable

Resources that support transactional semantics implement `Snapshotable`:

```python
class Snapshotable(Protocol[SnapshotT]):
    """Protocol for state containers that support snapshot and restore."""

    def snapshot(self, *, tag: str | None = None) -> SnapshotT:
        """Capture current state as an immutable snapshot."""
        ...

    def restore(self, snapshot: SnapshotT) -> None:
        """Restore state from a snapshot."""
        ...
```

**Implementations:**

| Component | Snapshot Type | Storage Strategy |
| ------------------- | -------------------- | ------------------------------ |
| `InMemoryFilesystem` | `FilesystemSnapshot` | Structural sharing of file dicts |
| `HostFilesystem` | `FilesystemSnapshot` | Git commits |

### Closeable

Resources that need cleanup implement `Closeable`:

```python
@runtime_checkable
class Closeable(Protocol):
    """Protocol for resources requiring cleanup."""

    def close(self) -> None:
        """Release resources. Called when scope ends."""
        ...
```

### PostConstruct

Resources needing initialization after construction:

```python
@runtime_checkable
class PostConstruct(Protocol):
    """Protocol for post-construction initialization."""

    def post_construct(self) -> None:
        """Called after construction, before caching.

        Failures here prevent the resource from being cached
        and are wrapped in ProviderError.
        """
        ...
```

## Error Hierarchy

```python
class ResourceError(WinkError, RuntimeError):
    """Base class for resource resolution errors."""


class UnboundResourceError(ResourceError, LookupError):
    """No binding exists for the requested protocol."""
    protocol: type[object]


class CircularDependencyError(ResourceError):
    """Circular dependency detected during resolution."""
    cycle: tuple[type[object], ...]


class DuplicateBindingError(ResourceError, ValueError):
    """Same protocol bound multiple times."""
    protocol: type[object]


class ProviderError(ResourceError):
    """Provider raised an exception during construction."""
    protocol: type[object]
    cause: BaseException
```

## Integration with Prompts

Prompts own their resource lifecycle. The typical pattern:

```python
# 1. Define template with resources
template = PromptTemplate[Output](
    ns="example",
    key="task",
    sections=[
        WorkspaceSection(filesystem=LocalFilesystem("/workspace")),
        MarkdownSection(title="Task", template="...", key="task"),
    ],
    resources=ResourceRegistry.of(
        Binding(HTTPClient, lambda r: HTTPClient(timeout=30)),
    ),
)

# 2. Bind parameters (optionally add runtime resources)
prompt = Prompt(template).bind(
    Params(...),
    resources=ResourceRegistry.build({Clock: SystemClock()}),
)

# 3. Use as context manager
with prompt.resources:
    # Resources initialized via prompt.resources.start()
    fs = prompt.resources.get(Filesystem)
    http = prompt.resources.get(HTTPClient)

    # Tool execution with transactional rollback
    # (handled automatically by adapters via runtime.transactions)
    result = adapter.evaluate(prompt, session=session)

# Resources cleaned up via prompt.resources.close()
```

### Resource Collection

Prompts collect resources from multiple sources in precedence order:

1. **Template resources** - `PromptTemplate.resources` (lowest precedence)
1. **Section resources** - Each section's `resources()` method (depth-first)
1. **Bind-time resources** - Passed to `prompt.bind(resources=...)` (highest)

Later sources override earlier on conflicts. To detect conflicts during
development, use `registry.conflicts(other)` to inspect overlaps or
`registry.merge(other, strict=True)` to raise on duplicates.

## Usage Examples

### Basic Usage

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

# Define bindings with providers
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
)

# Create resolution context (open() manages start/close automatically)
with registry.open() as ctx:
    http = ctx.get(HTTPClient)  # Lazily constructs Config, then HTTPClient
```

### Pre-Constructed Instances

```python
# Use Binding.instance() for pre-constructed resources
fs = InMemoryFilesystem()
tracker = BudgetTracker(budget=Budget(max_tokens=1000))

registry = ResourceRegistry.of(
    Binding.instance(Filesystem, fs),
    Binding.instance(BudgetTracker, tracker),
    Binding(Service, lambda r: Service(r.get(Filesystem))),
)

# Or use the convenience build() method
registry = ResourceRegistry.build({
    Filesystem: fs,
    BudgetTracker: tracker,
})
```

### Snapshot and Restore

Transactional snapshot/restore is handled by the `runtime.transactions` module:

```python
from weakincentives.runtime.transactions import (
    create_snapshot,
    restore_snapshot,
    tool_transaction,
)

# Using tool_transaction context manager (recommended)
with prompt.resources:
    with tool_transaction(session, prompt.resources.context, tag="my_tool") as snapshot:
        result = execute_tool(...)
        if not result.success:
            restore_snapshot(session, prompt.resources.context, snapshot)

# Manual snapshot/restore
with prompt.resources:
    snapshot = create_snapshot(session, prompt.resources.context, tag="before_tool")
    try:
        result = execute_tool(...)
    except Exception:
        restore_snapshot(session, prompt.resources.context, snapshot)
        raise
```

### Tool-Call Scoped Resources

```python
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(
        RequestTracer,
        lambda r: RequestTracer(request_id=uuid4()),
        scope=Scope.TOOL_CALL,
    ),
)

# Each tool scope gets fresh TOOL_CALL resources
with registry.open() as ctx:
    with ctx.tool_scope() as resolver:
        tracer1 = resolver.get(RequestTracer)

    with ctx.tool_scope() as resolver:
        tracer2 = resolver.get(RequestTracer)

    assert tracer1 is not tracer2  # Fresh instances
```

### Eager Initialization

```python
# Validate configuration at startup
registry = ResourceRegistry.of(
    Binding(
        Config,
        lambda r: Config.from_env(),  # May raise on invalid config
        eager=True,  # Instantiate during start()
    ),
)

# open() calls start() automatically - raises here if config invalid
with registry.open() as ctx:
    pass  # Config already validated
```

### Registry Composition

```python
base = ResourceRegistry.of(
    Binding(Config, lambda r: Config(env="prod")),
)

test_override = ResourceRegistry.of(
    Binding(Config, lambda r: Config(env="test")),
)

# Default: silently override
merged = base.merge(test_override)  # test_override wins

# Strict mode: detect conflicts
try:
    merged = base.merge(test_override, strict=True)
except DuplicateBindingError as e:
    print(f"Conflict on {e.protocol}")

# Inspect conflicts without raising
conflicts = base.conflicts(test_override)
if conflicts:
    print(f"Warning: {len(conflicts)} bindings will be overridden")
```

## Acceptance Criteria

### Lazy Construction

```python
def test_lazy_construction():
    constructed = []

    def make_service(r: ResourceResolver) -> Service:
        constructed.append("service")
        return Service()

    registry = ResourceRegistry.of(Binding(Service, make_service))

    assert constructed == []

    with registry.open() as ctx:
        _ = ctx.get(Service)

    assert constructed == ["service"]
```

### Dependency Resolution

```python
def test_dependency_resolution():
    registry = ResourceRegistry.of(
        Binding(Config, lambda r: Config(value=42)),
        Binding(Service, lambda r: Service(config=r.get(Config))),
    )

    with registry.open() as ctx:
        service = ctx.get(Service)
        assert service.config.value == 42
```

### Cycle Detection

```python
def test_circular_dependency_detected():
    registry = ResourceRegistry.of(
        Binding(A, lambda r: A(b=r.get(B))),
        Binding(B, lambda r: B(a=r.get(A))),
    )

    with registry.open() as ctx:
        with pytest.raises(CircularDependencyError) as exc:
            ctx.get(A)
        assert A in exc.value.cycle
        assert B in exc.value.cycle
```

### Singleton Caching

```python
def test_singleton_cached():
    call_count = 0

    def make_service(r: ResourceResolver) -> Service:
        nonlocal call_count
        call_count += 1
        return Service()

    registry = ResourceRegistry.of(Binding(Service, make_service))

    with registry.open() as ctx:
        s1 = ctx.get(Service)
        s2 = ctx.get(Service)

        assert s1 is s2
        assert call_count == 1
```

### Tool-Call Scope Isolation

```python
def test_tool_call_fresh_per_scope():
    counter = itertools.count()

    registry = ResourceRegistry.of(
        Binding(Tracer, lambda r: Tracer(id=next(counter)), scope=Scope.TOOL_CALL)
    )

    with registry.open() as ctx:
        with ctx.tool_scope() as r1:
            t1 = r1.get(Tracer)

        with ctx.tool_scope() as r2:
            t2 = r2.get(Tracer)

        assert t1.id == 0
        assert t2.id == 1
```

### Resource Cleanup

```python
def test_closeable_disposed():
    registry = ResourceRegistry.of(
        Binding(CloseableResource, lambda r: CloseableResource())
    )

    with registry.open() as ctx:
        resource = ctx.get(CloseableResource)
        assert resource.closed is False
    # Context manager calls close() on exit
    assert resource.closed is True
```

### Reverse Order Cleanup

```python
def test_close_reverse_order():
    closed_order = []

    registry = ResourceRegistry.of(
        Binding(ResourceA, lambda r: ResourceA(on_close=lambda: closed_order.append("A"))),
        Binding(ResourceB, lambda r: ResourceB(a=r.get(ResourceA), on_close=lambda: closed_order.append("B"))),
    )

    with registry.open() as ctx:
        _ = ctx.get(ResourceB)  # Constructs A, then B
    # Context manager calls close() in reverse instantiation order
    assert closed_order == ["B", "A"]
```

### Snapshot and Restore

```python
def test_snapshot_restore():
    from weakincentives.runtime.transactions import (
        create_snapshot,
        restore_snapshot,
    )

    fs = InMemoryFilesystem()
    registry = ResourceRegistry.build({Filesystem: fs})
    session = Session(bus=InProcessDispatcher())
    prompt = Prompt(template).bind(resources=registry)

    with prompt.resources:
        fs.write("data.txt", "before")
        snapshot = create_snapshot(session, prompt.resources.context, tag="test")

        fs.write("data.txt", "after")
        assert fs.read("data.txt").content == "after"

        restore_snapshot(session, prompt.resources.context, snapshot)
        assert fs.read("data.txt").content == "before"
```

## Limitations

- **Synchronous only**: Resolution is single-threaded; async providers not
  supported.
- **No conditional bindings**: Cannot bind different implementations based on
  runtime conditions (use explicit registry construction instead).
- **No interception**: No AOP-style interceptors on resource access.
- **No named bindings**: Use wrapper types if you need multiple implementations
  of the same protocol.
- **Snapshot scope**: Only SINGLETON resources are snapshotted; TOOL_CALL and
  PROTOTYPE resources are not tracked.

## Future Considerations

The following are explicitly out of scope but may be added later:

- **Modules**: Grouping related bindings for composition.
- **Qualifiers**: Built-in support for distinguishing multiple implementations.
- **Async providers**: `async def provider(...)` with `await ctx.get_async(T)`.
- **Health checks**: Protocol for resource health monitoring.
- **Metrics**: Instrumentation for resolution timing and cache hit rates.
