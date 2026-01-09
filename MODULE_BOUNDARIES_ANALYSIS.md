# Module Boundaries Analysis and Recommendations

**Date:** 2026-01-09
**Status:** Initial assessment completed
**Validation Script:** `scripts/validate_module_boundaries.py`

## Executive Summary

The weakincentives library demonstrates **excellent modularization** (Grade: A-) with clear separation of concerns, appropriate layering, and minimal coupling between subsystems. However, there are **26 module boundary violations** that should be addressed to improve architecture hygiene and enforce stricter boundaries.

### Key Strengths

- ✅ Clear 4-layer architecture (foundation → core → adapters → high-level)
- ✅ Flat hierarchy (max depth of 4 levels)
- ✅ Minimal cross-subsystem coupling
- ✅ Thoughtful public API curation (28 items at top level)
- ✅ Explicit circular dependency handling with lazy loading
- ✅ Well-organized subsystems with clear responsibilities

### Issues Found

| Category | Count | Severity |
|----------|-------|----------|
| Circular Dependencies | 4 | Medium |
| Layer Violations | 6 | High |
| Private Module Leaks | 16 | Medium |

## Detailed Findings

### 1. Layer Violations (6 violations) 🔴 HIGH PRIORITY

Layer violations indicate that lower-level modules are importing from higher-level ones, breaking the intended architecture.

#### Issue 1.1: `budget.py` (foundation) imports from `runtime.events` (core)

**Location:** `src/weakincentives/budget.py:27, :110`

**Problem:**
```python
from weakincentives.runtime.events import TokenUsage  # Line 27
```

The `budget` module is in the foundation layer and should not depend on `runtime` (core layer).

**Impact:** Breaks layer separation; budget becomes coupled to runtime implementation.

**Recommendation:**
- **Option A (Preferred):** Move `TokenUsage` to a more foundational location (e.g., `types.py` or `budget.py` itself)
- **Option B:** Accept that budget needs runtime and move it to core layer
- **Option C:** Use lazy imports with TYPE_CHECKING to avoid runtime dependency

**Implementation:**
```python
# In budget.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.runtime.events import TokenUsage

# Use forward references in type annotations
def track_usage(self, usage: "TokenUsage") -> None:
    ...
```

#### Issue 1.2-1.4: `optimizers` (core) imports from `adapters` (adapters layer)

**Locations:**
- `src/weakincentives/optimizers/_results.py:20`
- `src/weakincentives/optimizers/_context.py:26`

**Problem:**
```python
from weakincentives.adapters.core import PromptResponse
```

Optimizers are in the core layer but depend on adapters layer.

**Impact:** Creates tight coupling; optimizers cannot be used without adapters.

**Recommendation:**
- **Option A (Preferred):** Move optimizers to adapters layer or higher (they're optimization tools for adapters)
- **Option B:** Define a protocol in core that adapters implement, avoid concrete imports
- **Option C:** Accept the coupling and move optimizers to same layer as adapters

**Implementation (Option B):**
```python
# In optimizers/_protocols.py
from typing import Protocol

class OptimizableResponse(Protocol):
    output: str
    ...

# In optimizers/_context.py
from ._protocols import OptimizableResponse
```

#### Issue 1.5: `runtime.main_loop` (core) imports from `adapters.core` (adapters layer)

**Location:** `src/weakincentives/runtime/main_loop.py:66`

**Problem:** MainLoop directly imports from adapters.

**Impact:** Runtime becomes coupled to specific adapter implementations.

**Recommendation:** Use protocol/interface from prompt or runtime instead.

#### Issue 1.6: `runtime.events._types` (core) imports from `adapters._names` (adapters layer)

**Location:** `src/weakincentives/runtime/events/_types.py:23`

**Problem:** Event types depend on adapter names.

**Impact:** Events cannot be defined without knowledge of adapters.

**Recommendation:** Move adapter names to types or runtime, or use string literals.

---

### 2. Private Module Leaks (16 violations) 🟡 MEDIUM PRIORITY

Private modules (starting with `_`) are being imported outside their immediate package, breaking encapsulation.

#### Category 2.1: Imports from `runtime.events._types`

**Violations:** 5 occurrences
- `optimizers/_context.py:22`
- `adapters/claude_agent_sdk/adapter.py:31`
- `adapters/claude_agent_sdk/_hooks.py:29`
- `runtime/session/protocols.py:23`
- `runtime/session/session_view.py:23`

**Problem:** `runtime.events._types` is private but widely used across the codebase.

**Recommendation:** **Promote `_types` to public module** → `runtime/events/types.py`

**Implementation:**
```bash
# 1. Rename the file
mv src/weakincentives/runtime/events/_types.py src/weakincentives/runtime/events/types.py

# 2. Update runtime/events/__init__.py to export types
# from .types import TokenUsage, PromptExecuted, ...

# 3. Update all imports across codebase
```

#### Category 2.2: Imports from `prompt._visibility`

**Violations:** 3 occurrences
- `contrib/tools/digests.py:21`
- `contrib/tools/planning.py:24`
- `runtime/session/visibility_overrides.py:27`

**Problem:** Visibility logic is private but needed by multiple modules.

**Recommendation:** **Promote visibility utilities to public API** or create a dedicated public module.

**Implementation:**
```python
# In prompt/__init__.py, add:
from .visibility import (
    SectionVisibility,
    expand_visibility,
    # ... other visibility utilities
)
```

#### Category 2.3: Imports from `serde._utils`

**Violations:** 2 occurrences
- `runtime/transactions.py:50`
- `runtime/session/snapshots.py:28`

**Problem:** Serde utilities are private but needed for transactions and snapshots.

**Recommendation:** **Export necessary utilities from `serde/__init__.py`** or move to public module.

#### Category 2.4: Imports from `optimizers._base`, `._context`, `._results`

**Violations:** 3 occurrences in `contrib/optimizers/workspace_digest.py`

**Problem:** Contrib optimizer needs to extend core optimizer framework but all modules are private.

**Recommendation:**
- **Option A:** Make optimizer framework public (remove `_` prefix)
- **Option B:** Move workspace digest optimizer into core `optimizers/` package
- **Option C:** Create public base classes/protocols in `optimizers/__init__.py`

#### Category 2.5: Other private module leaks

**Violations:**
- `prompt._normalization` imported by `contrib/tools/digests.py:44`
- `adapters._names` imported by `runtime/events/_types.py:23`
- `runtime/mailbox/_types` imported by `contrib/mailbox/_redis.py:61`

**Recommendation:** Case-by-case review:
- Export types through public `__init__.py` files
- Or accept that private modules within same subsystem can cross-import

---

### 3. Circular Dependencies (4 violations) 🟡 MEDIUM PRIORITY

**Note:** The circular dependency detector is showing some duplicates/false positives. The core issue is:

**Primary Cycle:** `adapters ↔ runtime ↔ prompt ↔ resources`

**Paths detected:**
1. `adapters → runtime → runtime` (likely false positive)
2. `adapters → budget → runtime → runtime` (budget couples layers)
3. `adapters → prompt → runtime → runtime`
4. `adapters → prompt → resources → runtime → runtime`

**Analysis:**
- Some cycles are acceptable within a subsystem (e.g., `runtime.session` modules cross-referencing)
- Cross-subsystem cycles are mitigated by lazy imports and protocols
- The `contrib/__init__.py` uses `__getattr__` for lazy loading to break cycles

**Current Mitigation:**
```python
# In contrib/__init__.py
def __getattr__(name: str) -> ModuleType:
    """Lazy import submodules to avoid circular dependency issues."""
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(...)
```

**Recommendation:**
- ✅ Keep existing lazy loading patterns
- ✅ Continue using `TYPE_CHECKING` for type-only imports
- 🔧 Fix layer violations (Issue #1) which contribute to cycles
- 📊 Refine circular dependency detection to reduce false positives

---

## Reexport Analysis

### Current Reexport Patterns

The library uses three main reexport patterns:

#### Pattern 1: Subsystem Aggregation (Appropriate)

Example: `runtime/__init__.py`
```python
from . import events, lifecycle, mailbox, main_loop, session, watchdog
from .events import (PromptExecuted, TokenUsage, ...)
from .session import (Session, SessionProtocol, ...)
```

**Status:** ✅ Appropriate - provides both fine-grained and coarse-grained imports

**Note:** Some argue this creates redundancy (users can import both `runtime.events` and `runtime.PromptExecuted`), but this is intentional for API flexibility.

#### Pattern 2: Public API Gateway (Excellent)

Example: `weakincentives/__init__.py`
```python
from .prompt import Prompt, Tool, ToolContext
from .runtime import configure_logging, get_logger
from .skills import Skill, SkillConfig
# ... curated 28 items total
```

**Status:** ✅ Excellent - minimal, curated public surface

#### Pattern 3: Lazy Loading (Good)

Example: `contrib/__init__.py` and `contrib/tools/__init__.py`
```python
def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        return module
```

**Status:** ✅ Good - breaks circular dependencies

### Problematic Reexport

#### Issue: `contrib/tools/__init__.py` reexports from core

**Lines 37-50:**
```python
from weakincentives.filesystem import (
    READ_ENTIRE_FILE,
    FileEncoding,
    FileEntry,
    # ... 13 items total
)
```

**Problem:** Contrib should not reexport core modules. This creates confusion about where types are defined.

**Impact:** Users may import `FileEntry` from `contrib.tools` when it belongs in `filesystem`.

**Recommendation:** Remove these reexports. Users should import from `weakincentives.filesystem` directly.

**Implementation:**
```python
# Remove from contrib/tools/__init__.py:
# - All imports from weakincentives.filesystem
# - All imports from weakincentives.prompt.protocols

# Update __all__ to only include contrib/tools items:
__all__ = [
    "AddStep",
    "AstevalConfig",
    "AstevalSection",
    # ... only items defined in contrib/tools
]

# Users should import as:
from weakincentives.filesystem import FileEntry
from weakincentives.contrib.tools import VfsToolsSection
```

---

## Recommendations Summary

### Immediate Actions (High Priority)

1. **Fix Layer Violations**
   - [ ] Move `TokenUsage` out of `runtime.events` or move `budget` to core layer
   - [ ] Resolve `optimizers` ↔ `adapters` coupling
   - [ ] Add protocols to avoid concrete imports across layers

2. **Promote Widely-Used Private Modules**
   - [ ] Rename `runtime/events/_types.py` → `runtime/events/types.py`
   - [ ] Export visibility utilities from `prompt/__init__.py`
   - [ ] Export serde utilities needed by transactions/snapshots

3. **Remove Core Reexports from Contrib**
   - [ ] Remove `filesystem` imports from `contrib/tools/__init__.py`
   - [ ] Update documentation about correct import paths

### Medium Priority

4. **Resolve Private Module Leaks**
   - [ ] Make optimizer framework public or move workspace digest to core
   - [ ] Export adapter names from a public module
   - [ ] Export mailbox types through public `__init__.py`

5. **Refine Validation Script**
   - [ ] Improve circular dependency detection (reduce false positives)
   - [ ] Add allowlist for acceptable within-subsystem private imports
   - [ ] Add configuration file for architecture rules

### Long-Term Improvements

6. **Architecture Documentation**
   - [ ] Add dependency diagram to `CLAUDE.md`
   - [ ] Document layering rules explicitly
   - [ ] Create architecture decision records (ADRs)

7. **Monitor Growth**
   - [ ] Set size thresholds for subsystems (contrib/tools, runtime/session, adapters)
   - [ ] Establish metrics for when to split packages
   - [ ] Run validation in CI/CD pipeline

---

## Validation Script Usage

### Running Validation

```bash
# Run validation manually
make validate-modules

# Or directly with uv
uv run python scripts/validate_module_boundaries.py
```

### Adding to CI

Uncomment in `Makefile` line 217:
```makefile
check: format-check lint typecheck type-coverage bandit vulture deptry check-core-imports validate-modules pip-audit markdown-check verify-doc-examples validate-integration-tests test
```

**Note:** Do NOT enable in `make check` until violations are fixed, or CI will fail.

### Validation Rules

The script enforces:

1. **Layer Architecture** - Foundation ← Core ← Adapters ← High-level (no reverse imports)
2. **Private Module Encapsulation** - Modules starting with `_` cannot be imported outside their package
3. **Circular Dependencies** - No cycles between top-level packages
4. **Contrib/Core Separation** - Contrib should not reexport core modules

### Allowlist Configuration

For acceptable violations (e.g., internal cross-imports), add configuration to the script:

```python
# In validate_module_boundaries.py
ALLOWED_PRIVATE_IMPORTS = {
    ("runtime.transactions", "serde._utils"),  # Allowed: transactions need serde internals
    # Add more as needed
}
```

---

## Migration Path

### Phase 1: Non-Breaking Fixes (Week 1)

- Promote private modules to public (rename `_types.py` → `types.py`)
- Add public exports to `__init__.py` files
- Remove core reexports from contrib
- Update internal imports to use public APIs

### Phase 2: Layer Refactoring (Week 2)

- Move `TokenUsage` to appropriate layer
- Refactor optimizers/adapters coupling
- Add protocols to break tight coupling
- Fix mainloop/adapter dependency

### Phase 3: Enforcement (Week 3)

- Enable `validate-modules` in `make check`
- Add validation to pre-commit hooks
- Update documentation
- Train team on architecture rules

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  HIGH-LEVEL LAYER                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   contrib   │  │    evals    │  │     cli     │         │
│  │  (tools,    │  │ (evaluation │  │ (commands)  │         │
│  │  mailbox,   │  │  framework) │  └─────────────┘         │
│  │ optimizers) │  └─────────────┘                           │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓ imports from
┌─────────────────────────────────────────────────────────────┐
│  ADAPTERS LAYER                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             adapters (OpenAI, LiteLLM, SDK)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↓ imports from
┌─────────────────────────────────────────────────────────────┐
│  CORE LAYER                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ runtime  │ │  prompt  │ │resources │ │filesystem│      │
│  │(sessions,│ │(sections,│ │   (DI)   │ │ (VFS)    │      │
│  │ mailbox, │ │  tools,  │ └──────────┘ └──────────┘      │
│  │lifecycle)│ │overrides)│                                 │
│  └──────────┘ └──────────┘                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │  serde   │ │  skills  │ │optimizers│                   │
│  │(serializ)│ │ (agent   │ │(prompt   │                   │
│  └──────────┘ │  skills) │ │optimize) │                   │
│               └──────────┘ └──────────┘                   │
│  ┌──────────┐                                              │
│  │  formal  │                                              │
│  │  (TLA+)  │                                              │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
                           ↓ imports from
┌─────────────────────────────────────────────────────────────┐
│  FOUNDATION LAYER                                           │
│  ┌──────┐ ┌───────┐ ┌───────────┐ ┌─────┐ ┌──────────┐    │
│  │types │ │errors │ │dataclasses│ │ dbc │ │deadlines │    │
│  └──────┘ └───────┘ └───────────┘ └─────┘ └──────────┘    │
│                     ┌──────────┐                            │
│                     │  budget  │ ← VIOLATION: imports       │
│                     └──────────┘   from runtime (core)      │
└─────────────────────────────────────────────────────────────┘
```

**Legend:**
- ✅ Solid arrows = Allowed imports (lower → higher layers OK)
- ❌ Dashed arrows = Violations (higher → lower layers NOT allowed)
- 🔄 Cycles within layer = Acceptable (use protocols/lazy loading)

---

## Conclusion

The weakincentives library has **strong foundational architecture** with clear separation of concerns. The 26 violations found are **fixable** and mostly involve:

1. Making appropriate private modules public
2. Moving a few misplaced modules to correct layers
3. Removing unnecessary reexports

With these fixes, the library will have **A+ modularization** with enforced boundaries and excellent maintainability.

**Next Steps:**
1. Review this analysis with the team
2. Prioritize fixes (start with layer violations)
3. Enable validation in CI after fixes
4. Document architecture decisions

---

**Validation Command:**
```bash
make validate-modules
```

**Status:** 26 violations pending fixes
**Target:** 0 violations (enable in CI)
