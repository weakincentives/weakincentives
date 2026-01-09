# Module Boundaries Specification

## Purpose

This specification defines the module organization and import patterns for the
weakincentives library. The architecture uses layered modules with explicit
dependency flow to ensure:

- **Clear separation of concerns**: Each layer has distinct responsibilities
- **Predictable dependency flow**: Higher layers depend on lower layers, never
  the reverse
- **Minimal coupling**: Modules expose only what callers need
- **Maintainable growth**: New features fit into the existing structure

The validation script (`scripts/validate_module_boundaries.py`) enforces these
boundaries automatically.

## Guiding Principles

- **Foundation layers are dependency-free**: Core types and utilities have no
  dependencies on higher-layer abstractions
- **Private modules stay private**: Modules prefixed with `_` are
  implementation details, not public API
- **Reexports serve aggregation**: `__init__.py` should aggregate related
  exports, not create redundant paths
- **Type-only imports don't violate layers**: `TYPE_CHECKING` blocks allow type
  hints without runtime dependencies
- **Cross-cutting concerns need justification**: Some private module imports
  are acceptable when public exports would cause circular dependencies

## Architecture Layers

The library is organized into four layers with strict dependency flow:

```
┌─────────────────────────────────────┐
│       HIGH-LEVEL (Layer 4)          │  User-facing features
│  contrib, evals, cli                │
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
│  filesystem, serde, skills,         │
│  formal, optimizers                 │
└─────────────────────────────────────┘
            ↓ depends on
┌─────────────────────────────────────┐
│       FOUNDATION (Layer 1)          │  Base types & utilities
│  types, errors, dataclasses,        │
│  dbc, deadlines, budget             │
└─────────────────────────────────────┘
```

### Foundation Layer (Layer 1)

**Purpose**: Provide foundational types, errors, and utilities with zero
dependencies on higher layers.

**Packages**:
- `types` - JSON types, adapter names, dataclass protocols
- `errors` - Base exception hierarchy
- `dataclasses` - `FrozenDataclass` utilities and serialization helpers
- `dbc` - Design-by-contract decorators (`@require`, `@ensure`, `@pure`)
- `deadlines` - Time-based deadline tracking
- `budget` - Resource envelope (time + token limits)

**Rules**:
- MUST NOT import from core, adapters, or high-level layers
- SHOULD have minimal external dependencies (standard library preferred)
- Public API should be stable (these types appear in many signatures)

**Examples**:
```python
# ✅ Good - foundation imports only from foundation or stdlib
from weakincentives.types import JSONDict
from weakincentives.errors import WinkError

# ❌ Bad - foundation importing from core
from weakincentives.runtime import Session  # Layer violation
```

### Core Layer (Layer 2)

**Purpose**: Provide library primitives for sessions, prompts, resources, and
tooling.

**Packages**:
- `runtime` - Session state, events, lifecycle, mailbox, transactions
- `prompt` - Section/Prompt composition, overrides, tool protocols
- `resources` - Dependency injection with scoped lifecycles
- `filesystem` - Filesystem protocol and host implementation
- `serde` - Dataclass serialization (no Pydantic)
- `skills` - Agent Skills spec support
- `formal` - TLA+ formal specification embedding
- `optimizers` - Optimizer framework and protocols

**Rules**:
- CAN import from foundation layer
- CAN import between core packages (e.g., `runtime` ↔ `prompt`)
- MUST NOT import from adapters or high-level layers (except via
  `TYPE_CHECKING`)
- Private modules (`_foo.py`) should not be imported outside their package

**Examples**:
```python
# ✅ Good - core imports from foundation and other core
from weakincentives.types import JSONDict
from weakincentives.runtime import Session
from weakincentives.prompt import Prompt

# ✅ Good - type-only import from higher layer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from weakincentives.adapters.core import ProviderAdapter

# ❌ Bad - runtime import from adapters at module level
from weakincentives.adapters.core import ProviderAdapter
```

### Adapters Layer (Layer 3)

**Purpose**: Bridge the prompt abstraction and external LLM providers.

**Packages**:
- `adapters` - Provider adapters (OpenAI, LiteLLM, Claude Agent SDK)

**Rules**:
- CAN import from foundation and core layers
- MUST NOT import from high-level layer
- Provider-specific logic stays within adapter modules

**Examples**:
```python
# ✅ Good - adapter imports from core
from weakincentives.prompt import Prompt
from weakincentives.runtime import Session

# ❌ Bad - adapter imports from high-level
from weakincentives.contrib.tools import PlanningToolsSection
```

### High-Level Layer (Layer 4)

**Purpose**: Provide domain-specific tools, evaluation framework, and CLI.

**Packages**:
- `contrib` - Batteries for specific agent styles (planning, VFS, Podman)
- `evals` - Evaluation framework (datasets, evaluators, loops)
- `cli` - Command-line interface (`wink` entrypoints)

**Rules**:
- CAN import from any lower layer
- Provides convenience APIs and domain-specific compositions
- Allowed to have deeper dependency trees (this is the integration layer)

**Examples**:
```python
# ✅ Good - contrib imports from all lower layers
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import MarkdownSection
from weakincentives.adapters.openai import OpenAIAdapter
```

## Import Rules

### 1. Layer Violations

**Rule**: Lower layers MUST NOT import from higher layers at runtime.

**Detection**: The validator checks that imports respect layer ordering:
`foundation` → `core` → `adapters` → `high_level`.

**Examples**:
```python
# ❌ Bad - foundation importing from core
# In: weakincentives/budget.py
from weakincentives.runtime.events import TokenUsage  # LAYER_VIOLATION

# ✅ Good - use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from weakincentives.runtime.events import TokenUsage  # OK: type-only
```

### 2. Private Module Leaks

**Rule**: Private modules (prefixed with `_`) MUST NOT be imported outside
their package.

**Rationale**: Private modules are implementation details that may change
without notice.

**Examples**:
```python
# ❌ Bad - importing from private module outside package
from weakincentives.prompt._rendering import render_section

# ✅ Good - import from public module
from weakincentives.prompt import render_section

# ✅ Acceptable - cross-cutting concern (see exceptions below)
from weakincentives.prompt._visibility import SectionVisibility  # Justified
```

### 3. Circular Dependencies

**Rule**: Packages MUST NOT have circular import dependencies.

**Detection**: The validator detects import cycles like `A → B → A`.

**Examples**:
```python
# ❌ Bad - creates circular dependency
# In: weakincentives/runtime/session.py
from weakincentives.prompt import Prompt  # runtime → prompt

# In: weakincentives/prompt/section.py
from weakincentives.runtime import Session  # prompt → runtime
# Circular: runtime → prompt → runtime

# ✅ Good - use protocols or TYPE_CHECKING
from typing import TYPE_CHECKING, Protocol

class SessionProtocol(Protocol):
    """Minimal session interface."""
    ...

if TYPE_CHECKING:
    from weakincentives.runtime import Session
```

### 4. Redundant Reexports

**Rule**: `__init__.py` should not both import a submodule AND reexport items
from that submodule.

**Rationale**: Creates multiple import paths for the same symbol.

**Examples**:
```python
# ❌ Bad - redundant reexport
from . import tools  # Imports submodule
from .tools import PlanningTool  # Also imports item from submodule
# Now accessible as both:
# - weakincentives.contrib.tools.PlanningTool
# - weakincentives.contrib.PlanningTool

# ✅ Good - choose one pattern
# Option 1: Import items only
from .tools import PlanningTool, VfsTool

# Option 2: Import submodule only
from . import tools
# Usage: tools.PlanningTool
```

## Acceptable Exceptions

Some patterns appear as violations but are acceptable with justification:

### 1. TYPE_CHECKING Imports

**Pattern**: Import from higher layer inside `TYPE_CHECKING` block.

**Justification**: Type hints don't create runtime dependencies.

**Example**:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.adapters.core import ProviderAdapter  # OK

def create_adapter_config(
    adapter: ProviderAdapter,  # Type hint only
) -> dict:
    ...
```

**Note**: The validation script currently flags these as violations but they
are acceptable patterns.

### 2. Cross-Cutting Private Imports

**Pattern**: Import from private module for cross-cutting concerns.

**Justification**: Making the module public would cause circular imports or
expose too much surface area.

**Current Examples**:
- `weakincentives.contrib.tools.digests` imports from
  `weakincentives.prompt._visibility` - needed for workspace digest
  optimization
- `weakincentives.runtime.session.visibility_overrides` imports from
  `weakincentives.prompt._visibility` - session needs to manipulate visibility

**Guidelines**:
- Document why the import is necessary
- Keep the import minimal (don't import the whole module)
- Consider if the abstraction boundary is correct

### 3. Lazy Imports

**Pattern**: Import inside function/method rather than at module level.

**Justification**: Break import cycles or defer expensive imports.

**Example**:
```python
def get_podman_workspace():
    # Import only when needed (Podman is expensive)
    from weakincentives.contrib.tools.podman import PodmanWorkspace
    return PodmanWorkspace()
```

### 4. Protocol Imports

**Pattern**: Import protocol from lower layer to satisfy type checker.

**Justification**: Protocols define interfaces without creating dependencies.

**Example**:
```python
from weakincentives.filesystem import Filesystem  # Protocol

class MyTool:
    def __init__(self, fs: Filesystem):  # Any implementation works
        self.fs = fs
```

## Validation

### Running the Validator

```bash
# Run validation manually
python scripts/validate_module_boundaries.py

# Add to CI/development workflow
make validate-modules
```

### Exit Codes

- `0` - No violations found
- `1` - Violations detected

### Output Format

```
❌ Found 3 module boundary violation(s):

  LAYER_VIOLATION: 1
  PRIVATE_MODULE_LEAK: 2

Details:

================================================================================
LAYER_VIOLATION (1 violation(s))
================================================================================

  📍 /path/to/file.py:42
     weakincentives.runtime (core layer) imports from
     weakincentives.adapters (adapters layer). Lower layers cannot import
     from higher layers.

...
```

### Violation Types

- **LAYER_VIOLATION** - Lower layer imports from higher layer (check for
  `TYPE_CHECKING`)
- **PRIVATE_MODULE_LEAK** - Private module imported outside its package (check
  if justified)
- **CIRCULAR_DEPENDENCY** - Import cycle detected (use protocols or lazy
  imports)
- **REDUNDANT_REEXPORT** - Module and its items both reexported (remove one)
- **SYNTAX_ERROR** - Unable to parse file (usually false positive with Python
  3.12+ syntax)

### Known Limitations

The validator has some false positives:

1. **Syntax errors**: May flag Python 3.12+ syntax features as errors (the code
   still works)
2. **TYPE_CHECKING imports**: Currently flagged as layer violations but are
   acceptable
3. **String-based imports**: `importlib.import_module("foo")` not detected

Always verify violations with the actual import and consider the context before
fixing.

## Migration Guide

### Fixing Layer Violations

```python
# Before: Runtime dependency on adapter
from weakincentives.adapters.core import PromptResponse

class Optimizer:
    def optimize(self) -> PromptResponse:
        ...

# After: Type-only import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.adapters.core import PromptResponse

class Optimizer:
    def optimize(self) -> "PromptResponse":  # String annotation
        ...
```

### Fixing Private Module Leaks

```python
# Before: Import from private module
from weakincentives.runtime.events._types import EventHandler

# After: Import from public module
from weakincentives.runtime.events.types import EventHandler

# If no public export exists, add it to __init__.py:
# In weakincentives/runtime/events/__init__.py:
from .types import EventHandler  # Promoted from _types to types

__all__ = ["EventHandler", ...]
```

### Fixing Circular Dependencies

```python
# Before: Circular import
# module_a.py
from module_b import B

class A:
    def use_b(self, b: B): ...

# module_b.py
from module_a import A  # Circular!

class B:
    def use_a(self, a: A): ...

# After: Use TYPE_CHECKING
# module_a.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_b import B

class A:
    def use_b(self, b: "B"): ...

# module_b.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_a import A

class B:
    def use_a(self, a: "A"): ...
```

## Examples

### Good Module Organization

```python
# weakincentives/foundation/types.py (Layer 1)
from typing import Any

JSONDict = dict[str, Any]  # No dependencies

# weakincentives/core/runtime/session.py (Layer 2)
from weakincentives.types import JSONDict  # Foundation import OK
from weakincentives.runtime.events import Dispatcher  # Core-to-core OK

class Session:
    def get_state(self) -> JSONDict: ...

# weakincentives/adapters/openai.py (Layer 3)
from weakincentives.prompt import Prompt  # Core import OK
from weakincentives.runtime import Session  # Core import OK

class OpenAIAdapter:
    def evaluate(self, prompt: Prompt, session: Session): ...

# weakincentives/contrib/tools/planning.py (Layer 4)
from weakincentives.prompt import MarkdownSection  # Core import OK
from weakincentives.adapters.openai import OpenAIAdapter  # Adapter import OK

class PlanningToolsSection:
    def render(self): ...
```

### Bad Module Organization

```python
# ❌ Foundation importing from core
# weakincentives/budget.py (Layer 1)
from weakincentives.runtime.events import TokenUsage  # LAYER_VIOLATION

# ❌ Core importing from adapters
# weakincentives/runtime/session.py (Layer 2)
from weakincentives.adapters.core import ProviderAdapter  # LAYER_VIOLATION

# ❌ Private module leak
# weakincentives/contrib/tools/vfs.py
from weakincentives.filesystem._host import validate_path  # PRIVATE_LEAK

# ❌ Circular dependency
# weakincentives/runtime/__init__.py
from weakincentives.runtime.session import Session
# weakincentives/runtime/session.py
from weakincentives.runtime import InProcessDispatcher  # Circular
```

## Integration with Development Workflow

### Pre-commit Hooks

The validation script can be added to pre-commit hooks:

```bash
# In .git/hooks/pre-commit
#!/bin/bash
python scripts/validate_module_boundaries.py || exit 1
```

### CI Pipeline

Add to GitHub Actions or CI config:

```yaml
- name: Validate Module Boundaries
  run: python scripts/validate_module_boundaries.py
```

### IDE Integration

Configure IDE to highlight violations:

```python
# .vscode/settings.json
{
  "python.linting.pylintArgs": [
    "--load-plugins=scripts.validate_module_boundaries"
  ]
}
```

## Future Improvements

Potential enhancements to the validation system:

1. **Whitelist for TYPE_CHECKING**: Don't flag type-only imports as violations
2. **Justification comments**: Allow `# boundary-exception: reason` comments
3. **Dependency graph visualization**: Generate layer dependency diagrams
4. **Import cost tracking**: Measure and report import load times by layer
5. **Auto-fix suggestions**: Propose corrections for common violations

## Related Specifications

- `ADAPTERS.md` - Adapter layer architecture and provider integrations
- `PROMPTS.md` - Prompt system in core layer
- `SESSIONS.md` - Runtime session management
- `TOOLS.md` - Tool protocols and handlers
- `WORKSPACE.md` - Filesystem and contrib tools architecture
