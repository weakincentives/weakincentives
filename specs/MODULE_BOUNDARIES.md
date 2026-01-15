# Module Boundaries Specification

Module organization and import patterns with layered architecture.

**Validation:** `scripts/validate_module_boundaries.py`

## Architecture Layers

```
┌─────────────────────────────────────┐
│       HIGH-LEVEL (Layer 4)          │  contrib, evals, cli
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       ADAPTERS (Layer 3)            │  adapters
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       CORE (Layer 2)                │  runtime, prompt, resources,
│                                     │  filesystem, serde, skills
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       FOUNDATION (Layer 1)          │  types, errors, dataclasses,
│                                     │  dbc, deadlines, budget
└─────────────────────────────────────┘
```

## Layer Rules

| Layer | Can Import From | Cannot Import From |
|-------|-----------------|-------------------|
| Foundation | stdlib only | core, adapters, high-level |
| Core | foundation | adapters, high-level |
| Adapters | foundation, core | high-level |
| High-Level | all lower layers | — |

## Import Rules

1. **Layer Violations**: Lower layers MUST NOT import from higher layers
2. **Private Modules**: `_foo.py` MUST NOT be imported outside its package
3. **Circular Dependencies**: Packages MUST NOT have import cycles
4. **Redundant Reexports**: Don't both import submodule AND reexport its items

## Acceptable Exceptions

### TYPE_CHECKING Imports

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from weakincentives.adapters.core import ProviderAdapter  # OK
```

### Cross-Cutting Private Imports

When public exports would cause circular dependencies:

```python
from weakincentives.prompt._visibility import SectionVisibility  # Justified
```

### Lazy Imports

```python
def get_podman_workspace():
    from weakincentives.contrib.tools.podman import PodmanWorkspace
    return PodmanWorkspace()
```

## Validation

```bash
python scripts/validate_module_boundaries.py
make validate-modules
```

### Violation Types

| Type | Meaning |
|------|---------|
| `LAYER_VIOLATION` | Lower layer imports from higher layer |
| `PRIVATE_MODULE_LEAK` | Private module imported outside package |
| `CIRCULAR_DEPENDENCY` | Import cycle detected |
| `REDUNDANT_REEXPORT` | Module and items both reexported |

## Fixing Violations

### Layer Violation → TYPE_CHECKING

```python
# Before
from weakincentives.adapters.core import PromptResponse

# After
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from weakincentives.adapters.core import PromptResponse
```

### Circular Dependency → Protocols

```python
from typing import Protocol
class SessionProtocol(Protocol): ...
```

## Limitations

- TYPE_CHECKING imports currently flagged as violations (acceptable)
- String-based `importlib.import_module()` not detected
