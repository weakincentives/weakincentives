# Plan: Refactor Prompt Resources & Introduce Module System

## Current State Analysis

### What exists today

The resources system has three layers:
1. **`resources/`** — `Binding`, `Scope`, `ResourceRegistry`, `ScopedResourceContext`, lifecycle protocols
2. **`prompt/_prompt_resources.py`** — `PromptResources` (context manager + accessor proxy)
3. **`prompt/prompt.py`** — `Prompt._collected_resources()`, `Prompt.bind(resources=...)`, `Prompt.resources` property

### Identified problems

**P1: `PromptResources` is a dual-purpose object with confused identity.**
It's both a context manager (lifecycle) and a resource accessor (resolution). Every call
to `prompt.resources` creates a *new* `PromptResources` instance (not cached). The
`__enter__` returns `Self` instead of `ScopedResourceContext`, which means `with prompt.resources as ctx`
gives you the proxy, not the real context.

**P2: `_collected_resources()` is called redundantly.**
Called once in `__enter__()` to build the context, then again in *every* `get()`/`get_optional()`
call to check for pre-provided instances. The collection performs N registry merges each time.

**P3: The `Binding.provided` field creates a two-track resolution system.**
Pre-provided instances bypass the context entirely (can be accessed outside `with`), while
factory-constructed resources require being inside `with`. This special-casing leaks into
`PromptResources.get()` which has to check `provided` before delegating to context.

**P4: Section `configure(builder)` returns full `ResourceRegistry`.**
Every section that wants to contribute a single binding creates an entire `ResourceRegistry`.
The typical pattern is just `ResourceRegistry.build({Filesystem: self._filesystem})` — heavy
for a single instance binding.

**P5: `pyright: ignore[reportPrivateUsage]` littered through `PromptResources`.**
PromptResources reaches into `_resource_context`, `_collected_resources()` on Prompt.
Uses a `_PromptForResources` protocol to "formalize" this, but it's still reaching into
private state.

**P6: No module/bundle concept.**
Related bindings (e.g., workspace = Filesystem + Git + TempDirs) can't be grouped as a
reusable unit. Each section independently returns its own registry, and the prompt does
an ad-hoc merge chain. There's no way to express "this set of bindings goes together."

**P7: Context lifecycle is tangled with Prompt.**
The `_resource_context` field lives on `Prompt`, but it's only set/read by `PromptResources`.
Adapters do `with prompt.resource_scope():` but the real owner of the lifecycle should be more
explicit.

---

## Design: Module System

### Core idea

Introduce `ResourceModule` as a first-class unit of binding configuration, replacing the
current pattern where sections return bare `ResourceRegistry` objects and the prompt does
ad-hoc merging.

### New types

```python
class ResourceModule(Protocol):
    """A unit of resource configuration. Like Guice's Module."""

    def configure(self, registry: RegistryBuilder) -> None:
        """Contribute bindings to the builder."""
        ...


class RegistryBuilder:
    """Accumulates bindings from modules. Like Guice's Binder."""

    def bind(self, protocol: type[T], provider: Provider[T],
             scope: Scope = Scope.SINGLETON, eager: bool = False) -> None: ...

    def bind_instance(self, protocol: type[T], instance: T) -> None: ...

    def install(self, module: ResourceModule) -> None: ...

    def build(self) -> ResourceRegistry: ...
```

### Changes to Section

```python
class Section:
    # BEFORE: returns ResourceRegistry
    def resources(self) -> ResourceRegistry: ...

    # AFTER: contributes to builder (or returns modules)
    def modules(self) -> Sequence[ResourceModule]: ...
    # Default: ()
```

Alternatively, keep `resources()` but change its return type to accept both:
```python
    def resources(self) -> ResourceRegistry | Sequence[ResourceModule]: ...
```

### Changes to PromptResources

**Split the dual-purpose object:**

1. `Prompt.resources` becomes a cached property returning a `PromptResources` that is
   purely an accessor (get/get_optional/tool_scope). No context management.

2. Context management moves to `Prompt.open_resources()` which returns a context manager
   yielding `ScopedResourceContext`.

```python
class Prompt:
    @contextmanager
    def open_resources(self) -> Iterator[ScopedResourceContext]:
        """Enter resource lifecycle. Resources available via self.resources after entry."""
        registry = self._collected_resources()
        ctx = registry._create_context()
        ctx.start()
        self._resource_context = ctx
        try:
            yield ctx
        finally:
            ctx.close()
            self._resource_context = None

    @property
    def resources(self) -> PromptResources:
        """Accessor to active resource context. Must be inside open_resources()."""
        return PromptResources(self)
```

### Eliminate `Binding.provided`

The `provided` field exists so adapters can do `prompt.resources.get(Filesystem)` outside
the context (for guardrails checks). Instead:

- Eager singletons are resolved at `start()` time — already the case.
- Outside-context access should go through the prompt directly:
  `prompt.filesystem()` already exists as a dedicated accessor.

The few callsites that do `prompt.resources.get(Filesystem)` outside context with
try/except can instead use `prompt.filesystem()` or check via the section directly.

---

## Implementation Steps

### Step 1: Introduce `ResourceModule` protocol and `RegistryBuilder`

Add to `resources/`:
- `module.py` — `ResourceModule` protocol
- `builder.py` — `RegistryBuilder` class

`ResourceRegistry.from_modules(*modules)` as a new factory method.

### Step 2: Cache `_collected_resources()` result

Fix P2: The collected registry should be computed once and cached (invalidated on `bind()`).
This eliminates the per-`get()` overhead.

### Step 3: Clean up `PromptResources` interface

Fix P1/P5:
- Make `prompt.resources` a cached `PromptResources` that reads `_resource_context`
- Remove the context manager from `PromptResources`
- Add `Prompt.open_resources()` as the lifecycle context manager
- Remove `_PromptForResources` protocol (no longer needed if we clean up the interface)

### Step 4: Migrate Section.configure(builder) to modules

Fix P4/P6:
- Add `Section.modules()` returning `Sequence[ResourceModule]`
- Deprecate or keep `Section.configure(builder)` as a compatibility shim
- Update `WorkspaceSection` to return a module

### Step 5: Remove `Binding.provided`

Fix P3:
- Remove the `provided` field from `Binding`
- Remove `Binding.instance()` special-casing of `provided`
- Update the few callsites that access resources outside context
- Eager singletons still work through `start()` + cache

### Step 6: Update adapters and tests

- Update all adapter `with prompt.resource_scope():` → `with prompt.open_resources():`
- Update guardrail code that accesses resources outside context
- Update all tests

---

## Migration Strategy

This can be done incrementally:
1. Steps 1-2 are additive (no breaking changes)
2. Step 3 can introduce `open_resources()` alongside existing `with prompt.resource_scope():`
3. Step 4 is additive (modules + existing configure(builder) coexist)
4. Step 5 requires coordinated update of callsites
5. Step 6 follows naturally

---

## Open Questions

1. **Should `ResourceModule` be a Protocol or an ABC?** Protocol is lighter and more
   Pythonic, but ABC allows `install()` helper method.

2. **Should we keep `Section.configure(builder)` alongside `modules()`?** Having both is
   redundant. Could treat `configure(builder)` as sugar that wraps a simple module.

3. **Is `Binding.provided` removal worth the churn?** The alternative is to just
   fix the caching issue (Step 2) and live with two-track resolution.

4. **Should `RegistryBuilder` support `override` semantics explicitly?** Currently
   `merge(strict=False)` handles this implicitly.
