# Module Boundaries Specification

## Purpose

Module organization and import patterns ensuring clear separation, predictable
dependency flow, minimal coupling, and maintainable growth.

## Guiding Principles

- **Foundation layers dependency-free**: Core types have no higher-layer dependencies
- **Private modules stay private**: `_` prefix means implementation detail
- **Reexports serve aggregation**: `__init__.py` aggregates, not duplicates
- **Type-only imports don't violate layers**: `TYPE_CHECKING` allows type hints
- **Cross-cutting concerns need justification**: Document private imports

## Architecture Layers

```
┌─────────────────────────────────────┐
│       HIGH-LEVEL (Layer 4)          │  User-facing features
│  contrib, evals, cli, docs          │
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       ADAPTERS (Layer 3)            │  Provider integrations
│  adapters                           │
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       CORE (Layer 2)                │  Library primitives
│  runtime, prompt, resources,        │
│  filesystem, serde, skills, formal, │
│  debug, optimizers                  │
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       FOUNDATION (Layer 1)          │  Base types & utilities
│  types, errors, dataclasses, dbc,   │
│  deadlines, budget, clock,          │
│  experiment                         │
└─────────────────────────────────────┘
```

### Layer Rules

| Layer | Can Import From | Cannot Import From |
|-------|-----------------|-------------------|
| Foundation | stdlib only | core, adapters, high-level |
| Core | foundation, other core | adapters, high-level |
| Adapters | foundation, core | high-level |
| High-Level | any lower layer | - |

## Import Rules

### 1. Layer Violations

Lower layers MUST NOT import from higher layers at runtime. Use `TYPE_CHECKING`
for type-only imports.

### 2. Private Module Leaks

Private modules (`_foo.py`) MUST NOT be imported outside their package.

### 3. Circular Dependencies

Packages MUST NOT have circular import dependencies. Use protocols or
`TYPE_CHECKING` to break cycles.

### 4. Redundant Reexports

`__init__.py` should not both import a submodule AND reexport items from it.

## Acceptable Exceptions

| Pattern | Justification |
|---------|---------------|
| `TYPE_CHECKING` imports | Type hints without runtime dependencies |
| Cross-cutting private imports | When public exports cause circular deps |
| Lazy imports | Break import cycles or defer expensive imports |
| Protocol imports | Interfaces without dependencies |

## Validation

Module boundary validation is enforced through code review and type checking.
Use these violation categories during review:

### Violation Types

| Type | Description |
|------|-------------|
| `LAYER_VIOLATION` | Lower layer imports higher layer |
| `PRIVATE_MODULE_LEAK` | Private module imported outside package |
| `CIRCULAR_DEPENDENCY` | Import cycle detected |
| `REDUNDANT_REEXPORT` | Module and items both reexported |

### Review Notes

- `TYPE_CHECKING` imports are acceptable for type-only dependencies
- String-based `importlib.import_module()` calls bypass static analysis
- Manual review required for dynamic imports and runtime-constructed imports

## Migration Patterns

**Fixing layer violations** (illustrative pattern):

```python
# Use TYPE_CHECKING to import higher-layer types for annotations only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from weakincentives.adapters.core import SomeAdapter
```

**Fixing circular dependencies** (illustrative pattern):

```python
# Define a protocol with minimal interface instead of importing concrete class
class SomeProtocol(Protocol):
    def method(self) -> Result: ...
```

## Related Specifications

- `specs/ADAPTERS.md` - Adapter layer architecture
- `specs/PROMPTS.md` - Prompt system in core layer
- `specs/SESSIONS.md` - Runtime session management
