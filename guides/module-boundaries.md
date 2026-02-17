# Module Boundaries

*Canonical spec: [specs/MODULE_BOUNDARIES.md](../specs/MODULE_BOUNDARIES.md)*

This guide explains WINK's layer architecture and module dependency rules.
Understanding these boundaries helps you place new code correctly and avoid
the coupling problems that make agent codebases difficult to maintain.

## Why Layers Exist

Agent systems compose many concerns: type definitions, serialization,
prompt rendering, provider integration, session management, and
user-facing features. Without clear boundaries, these concerns entangle.
A change to an adapter leaks into the prompt system. A type definition
pulls in runtime machinery. A CLI command depends on provider internals.

Layering prevents this by enforcing a single rule: **dependencies flow
downward, never upward.** Lower layers know nothing about higher layers.
Higher layers compose lower layers freely. This constraint keeps each
layer independently testable, replaceable, and understandable.

## The Four Layers

WINK organizes code into four layers, each with a clear purpose:

| Layer | Packages | Purpose |
|-------|----------|---------|
| Foundation | `types`, `errors`, `dataclasses`, `dbc`, `deadlines`, `budget`, `clock`, `experiment` | Base types, contracts, time, and utilities that depend only on stdlib |
| Core | `runtime`, `prompt`, `resources`, `filesystem`, `serde`, `skills`, `formal`, `debug`, `optimizers` | Library primitives that implement WINK's main abstractions |
| Adapters | `adapters` | Provider integrations that bridge prompts to execution harnesses |
| High-Level | `contrib`, `evals`, `cli`, `docs` | User-facing features built on everything below |

### Dependency Direction

Each layer may import from layers below it, never above:

| Layer | Can Import From | Cannot Import From |
|-------|-----------------|-------------------|
| Foundation | stdlib only | Core, Adapters, High-Level |
| Core | Foundation, other Core | Adapters, High-Level |
| Adapters | Foundation, Core | High-Level |
| High-Level | any lower layer | -- |

This is the fundamental constraint. Everything else follows from it.

## The Mental Model

Think of the layers as concentric rings. Foundation sits at the center,
stable and dependency-free. Each outer ring adds capabilities but never
reaches inward to modify what it depends on.

**Foundation is bedrock.** It defines the vocabulary: error types, clock
protocols, budget tracking, design-by-contract decorators. These types
appear everywhere, so they must depend on nothing. When you change
Foundation, everything above may be affected. Keep it small and stable.

**Core is the engine.** It implements prompts, sessions, resources,
serialization -- the machinery that makes WINK work. Core modules may
depend on each other (e.g., `runtime` uses `prompt` types) and on
Foundation. Core should never know which provider will execute a prompt.

**Adapters are bridges.** They translate between WINK's abstractions and
specific providers (Claude Agent SDK, Codex App Server). Adapters depend
on Core to access prompts, tools, and sessions. They should not depend
on user-facing features, and user-facing code should not depend on a
specific adapter.

**High-Level is the surface.** CLI commands, evaluation frameworks,
contributed tools -- anything a user interacts with directly. This layer
composes everything below and has the most freedom, but also the most
churn.

## What Goes Wrong Without Boundaries

When layers are violated, specific problems emerge:

**Circular imports.** Module A imports from Module B, which imports from
Module A. Python raises `ImportError` at startup. This is the most
visible symptom, but not the worst.

**Hidden coupling.** A Foundation type imports a Core module "just for
one helper." Now every consumer of that Foundation type transitively
depends on the Core module, its dependencies, and their dependencies.
Test setup becomes complex. Refactoring becomes risky.

**Adapter lock-in.** Core code imports an adapter to access a
convenience function. Now the prompt system only works with that one
provider. The entire point of the adapter layer -- provider
portability -- is defeated.

**Cascading changes.** A change in High-Level code forces changes in
Core. Core changes force Foundation changes. Instead of changes flowing
outward (where they belong), they propagate inward, destabilizing the
most depended-upon code.

## Breaking Circular Dependencies

Two patterns handle cases where a lower layer needs to reference a
higher-layer type.

### TYPE_CHECKING Imports

When a Foundation or Core module needs a type annotation from a higher
layer, use a `TYPE_CHECKING` guard:

```python nocheck
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.adapters.core import SomeAdapter

def process(adapter: SomeAdapter) -> None:
    ...
```

The import only runs during static analysis. At runtime, the lower layer
has no dependency on the higher layer. This is acceptable for type hints
but not for runtime behavior -- you cannot instantiate, call methods on,
or otherwise use the imported type at runtime.

### Protocols

When a lower layer needs to *use* a higher-layer object at runtime,
define a protocol that captures the required interface:

```python nocheck
from typing import Protocol

class Evaluator(Protocol):
    def evaluate(self, prompt: Prompt) -> Response: ...
```

The lower layer depends on the protocol. The higher layer provides a
concrete implementation. No import needed. This is WINK's preferred
pattern for decoupling -- it keeps the dependency graph clean and makes
the required interface explicit.

**When to use which:**

| Situation | Pattern |
|-----------|---------|
| Type annotation only, no runtime use | `TYPE_CHECKING` import |
| Need to call methods at runtime | Protocol |
| Need to construct instances | Refactor -- the code is in the wrong layer |

If you find yourself needing to construct a higher-layer object in a
lower layer, that is a signal the code belongs in the higher layer.
Move it up rather than working around the boundary.

## Private Modules

Modules prefixed with `_` (e.g., `_internal.py`) are implementation
details of their package. They must not be imported outside that
package.

```
weakincentives/
  runtime/
    _reducers.py    # Private to runtime/
    session.py      # Public -- can be imported by other packages
    __init__.py     # Reexports public API
```

The `__init__.py` file controls what a package exposes. Consumers import
from the package, not from private modules:

```python nocheck
# Correct
from weakincentives.runtime import Session

# Wrong -- reaches into private implementation
from weakincentives.runtime._reducers import apply_op
```

Private modules let you restructure internals without breaking
consumers. Respect the boundary even when the private module has
exactly the function you need -- it may change without notice.

## Where Does My Code Go?

When adding new code, ask these questions in order:

1. **Does it depend only on stdlib?** If yes, it likely belongs in
   Foundation. Examples: a new error type, a time utility, a contract
   decorator.

1. **Does it implement a core WINK abstraction?** If it works with
   prompts, sessions, resources, or serialization, it belongs in Core.

1. **Does it integrate with an external provider?** Adapter layer.
   Keep provider-specific code isolated here.

1. **Is it a user-facing feature?** CLI commands, evaluation helpers,
   contributed tool suites belong in High-Level.

1. **Does it need to import from Adapters or High-Level?** If a Core
   module needs to, reconsider the design. Either the code belongs in a
   higher layer, or you need a protocol to invert the dependency.

## Validation

Module boundaries are enforced automatically by `make check` via three
dedicated checkers in the toolchain:

| Violation | Description |
|-----------|-------------|
| `LAYER_VIOLATION` | Lower layer imports higher layer at runtime |
| `PRIVATE_MODULE_LEAK` | Private module imported outside its package |
| `CIRCULAR_DEPENDENCY` | Import cycle between packages |
| `REDUNDANT_REEXPORT` | `__init__.py` imports both module and its items |

`TYPE_CHECKING` imports are explicitly allowed and do not count as
layer violations. Dynamic imports via `importlib.import_module()` bypass
static analysis and require manual review.

## Next Steps

- [Code Quality](code-quality.md): Quality gates and enforcement
- [Resources](resources.md): Dependency injection across layers
- [Adapters](adapters.md): How the adapter layer works
