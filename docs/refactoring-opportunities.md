# Refactoring Opportunities for weakincentives

This document identifies opportunities to improve the internal structure of the library, making it easier to understand, change, and maintain.

## Executive Summary

The codebase has solid foundations with clean protocol-based abstractions and a good separation between core and contrib modules. However, several areas have accumulated complexity that could benefit from refactoring:

| Priority | Issue | Files Affected | Estimated Impact |
|----------|-------|----------------|------------------|
| High | God module in adapters | `adapters/shared.py` (1,895 lines) | High maintainability gain |
| High | Session class responsibilities | `session.py` (890 lines) + 12 related files | Improved testability |
| Medium | Tool section boilerplate | 4 contrib tool modules | Reduced duplication |
| Medium | Path logic duplication | `vfs.py`, `filesystem.py` | Code reuse |
| Medium | Circular import suppressions | 8 files | Cleaner architecture |
| Low | Serde coercion patterns | `parse.py`, `dump.py` | Minor DRY improvement |

---

## High Priority Refactorings

### 1. Decompose `adapters/shared.py` (1,895 lines)

**Problem**: This module is a "god module" containing multiple unrelated concerns:
- Throttle policy and retry logic (~200 lines)
- Provider protocol interfaces
- `InnerLoop` class for conversation orchestration (~380 lines)
- `ToolExecutor` for tool invocation
- `ResponseParser` for output extraction
- Schema/JSON utilities
- Error handling

**Impact**: Changes to throttling logic require understanding the entire file. Testing individual components is difficult.

**Recommended Split**:

```
adapters/
├── shared/
│   ├── __init__.py          # Re-exports for backward compatibility
│   ├── throttle.py          # ThrottlePolicy, ThrottleError, retry logic
│   ├── inner_loop.py        # InnerLoop class
│   ├── tool_executor.py     # ToolExecutor class
│   ├── response_parser.py   # ResponseParser class
│   └── utilities.py         # Schema helpers, message utilities
└── ...
```

**Specific concerns to extract**:

1. **`throttle.py`** - Lines 124-320 approximately:
   - `ThrottleKind`, `ThrottlePolicy`, `ThrottleError`
   - `new_throttle_policy()`, `should_retry_throttle()`
   - Retry calculation logic

2. **`inner_loop.py`** - Lines 1465-1845:
   - `InnerLoop` class
   - `InnerLoopInputs`, `InnerLoopConfig` dataclasses
   - Related helper methods

3. **`tool_executor.py`** - Lines 1299-1395:
   - `ToolExecutor` class
   - Tool invocation logic

4. **`response_parser.py`** - Lines 1398-1463:
   - `ResponseParser` class
   - Output extraction and structured output parsing

5. **`utilities.py`** - Scattered throughout:
   - `message_text_content()`, `extract_payload()`
   - `token_usage_from_payload()`, `deadline_provider_payload()`
   - Schema generation helpers

---

### 2. Reduce Session Class Responsibilities

**Problem**: The `Session` class (`runtime/session/session.py`, 890 lines) handles:
- State storage and slice management
- Reducer registration and dispatch
- Observer/subscription pattern
- Snapshot/restore operations
- Child session hierarchy
- Thread locking
- Event broadcasting

**Location**: `src/weakincentives/runtime/session/session.py`

**Impact**: The class is difficult to test in isolation. Changes to snapshot logic can affect slice management.

**Recommended Decomposition**:

```python
# Option A: Composition-based refactoring
@dataclass(slots=True)
class Session:
    """Facade over specialized managers."""
    _state: StateStore
    _observers: ObserverManager
    _snapshots: SnapshotManager
    _children: ChildSessionManager

# Option B: Extract mixin behaviors into protocols
class SliceQueryable(Protocol):
    """Protocol for querying session slices."""
    def __getitem__(self, slice_type: type[T]) -> SliceAccessor[T]: ...

class SnapshotRestoreable(Protocol):
    """Protocol for snapshot operations."""
    def snapshot(self) -> Snapshot: ...
    def restore(self, snapshot: Snapshot) -> None: ...
```

**Specific extraction candidates**:

1. **Observer management** (lines 350-450):
   - `observe()`, `unobserve()`, `_notify_observers()`
   - Move to `ObserverManager` class

2. **Snapshot operations** (lines 650-750):
   - `snapshot()`, `restore()`, serialization logic
   - Already partially in `snapshots.py` - consolidate fully

3. **Child session management** (lines 500-600):
   - `spawn_child()`, `children` property
   - Move to `ChildSessionManager`

---

### 3. Break Circular Import Dependencies

**Problem**: 8 files suppress import cycle warnings with `# pyright: reportImportCycles=false`, indicating architectural tension:

```
runtime/session/session.py
runtime/session/state_slice.py
runtime/session/protocols.py
runtime/session/_types.py
runtime/session/slice_accessor.py
contrib/tools/__init__.py
prompt/visibility_overrides.py
cli/__init__.py
```

**Root cause**: `adapters/shared.py` imports `contrib.tools.filesystem.Filesystem` directly:
```python
# adapters/shared.py line 29
from ..contrib.tools.filesystem import Filesystem
```

This creates tight coupling between the adapter layer and contrib tools.

**Solutions**:

1. **Define `Filesystem` protocol in core**:
   ```python
   # types/filesystem.py or runtime/protocols.py
   class FilesystemProtocol(Protocol):
       def read_text(self, path: str) -> str: ...
       def write_text(self, path: str, content: str) -> None: ...
       # ...
   ```

2. **Inject filesystem via `ToolContext`**: The filesystem should be passed in, not imported.

3. **Move commonly-used protocols to a shared location** that doesn't depend on contrib.

---

## Medium Priority Refactorings

### 4. Extract Tool Section Factory Pattern

**Problem**: Four contrib tool modules implement nearly identical section patterns:

| Module | Section Class | Params Class |
|--------|---------------|--------------|
| `planning.py` | `PlanningToolsSection` | `_PlanningSectionParams` |
| `asteval.py` | `AstevalSection` | `_AstevalSectionParams` |
| `podman.py` | `PodmanSandboxSection` | `_PodmanSectionParams` |
| `vfs.py` | `VfsToolsSection` | `_VfsSectionParams` |

Each follows the same pattern:
1. Define a `_*SectionParams` frozen dataclass
2. Define a `*Section(MarkdownSection[_*SectionParams])` class
3. Implement `__init__` with config resolution logic
4. Build tools from a `_build_tools()` function
5. Initialize session state

**Recommended abstraction**:

```python
# prompt/contrib_section.py
class ContribToolsSectionFactory[P, C]:
    """Factory for building tool sections with consistent patterns."""

    def __init__(
        self,
        title: str,
        key: str,
        params_type: type[P],
        config_type: type[C],
        template_builder: Callable[[C], str],
        tools_builder: Callable[[Self, C], tuple[Tool, ...]],
    ) -> None: ...

    def create_section(
        self,
        *,
        session: Session,
        config: C | None = None,
        **overrides: Any,
    ) -> MarkdownSection[P]: ...
```

This would reduce boilerplate across all four modules.

---

### 5. Consolidate Filesystem Path Logic

**Problem**: Path normalization and validation logic is duplicated between:
- `contrib/tools/vfs.py` (1,782 lines)
- `contrib/tools/filesystem.py` (1,502 lines)

Both implement:
- Path normalization (`_normalize_path()`)
- Glob matching
- Path validation
- Similar result type definitions

**Recommended consolidation**:

```python
# contrib/tools/_path_utils.py
def normalize_path(path: str, *, allow_absolute: bool = False) -> str: ...
def validate_path(path: str, *, existing: bool = False) -> None: ...
def match_glob(pattern: str, paths: Iterable[str]) -> list[str]: ...

# Result types shared between vfs.py and filesystem.py
@FrozenDataclass()
class PathOperationResult:
    path: str
    success: bool
    message: str | None = None
```

---

### 6. Simplify Prompt Module Boundaries

**Problem**: The prompt module has 20+ files with unclear boundaries:

```
prompt/
├── prompt.py (415 lines)          # Core prompt class
├── section.py (358 lines)         # Section base class
├── tool.py (919 lines)            # Tool definitions
├── rendering.py (444 lines)       # Rendering logic
├── registry.py (741 lines)        # Registry management
├── progressive_disclosure.py (543 lines)
├── visibility_overrides.py
├── overrides/
│   ├── validation.py (602 lines)
│   └── ...
└── 10+ other helper modules
```

**Concerns that could be consolidated**:

1. **Rendering pipeline**: `rendering.py` + parts of `registry.py` + `progressive_disclosure.py` form a cohesive rendering concern

2. **Override system**: `visibility_overrides.py` + `overrides/` directory + related parts of `progressive_disclosure.py`

3. **Section composition**: `section.py` + `markdown.py` + `_normalization.py`

**Recommended structure**:
```
prompt/
├── core/
│   ├── prompt.py
│   ├── section.py
│   └── tool.py
├── rendering/
│   ├── __init__.py
│   ├── engine.py
│   └── registry.py
├── overrides/
│   └── ... (consolidate override system)
└── structured_output.py
```

---

## Lower Priority Refactorings

### 7. Consolidate Serde Coercion Logic

**Problem**: `serde/parse.py` (720 lines) and `serde/dump.py` (303 lines) have parallel type-handling logic:

```python
# parse.py
def _bool_from_str(value: str) -> bool: ...
def _decimal_from_any(value: object) -> object: ...
def _uuid_from_any(value: object) -> object: ...
def _path_from_any(value: object) -> object: ...
# ... 20+ coercion functions

# dump.py
def _serialize_primitive(value: object, ...) -> JSONValue: ...
# Handles same types: Enum, datetime, date, time, UUID, Decimal, Path
```

**Recommended consolidation**:

```python
# serde/coercers.py
@dataclass(frozen=True)
class TypeCoercer[T]:
    """Bidirectional coercer for a specific type."""
    type_: type[T]
    to_json: Callable[[T], JSONValue]
    from_json: Callable[[JSONValue], T]

# Registry of coercers
STANDARD_COERCERS: tuple[TypeCoercer, ...] = (
    TypeCoercer(UUID, str, UUID),
    TypeCoercer(Decimal, str, Decimal),
    TypeCoercer(Path, str, Path),
    # ...
)
```

---

### 8. Decouple DbC Module

**Problem**: The `dbc/__init__.py` (511 lines) is imported by many core modules and adds overhead:

```python
# Used in session.py, prompt.py, adapters, etc.
from ...dbc import invariant, require, ensure
```

**Considerations**:
- DbC is a cross-cutting concern that benefits from being always-on in development
- However, the extensive reflection and patching machinery adds complexity

**Options**:
1. **Keep as-is** - The DbC pattern is well-integrated and valuable
2. **Make contracts optional** - Environment variable to disable for production
3. **Simplify contract evaluation** - Reduce reflection overhead

---

## Architecture Observations

### Strengths
- Clean protocol-based abstractions (`Filesystem`, `Session`, `Snapshotable`)
- Good separation between core and contrib modules
- Dataclass-centric design enables serde flexibility
- DbC decorators catch invariant violations early

### Areas for Improvement
- Large modules need decomposition for testability
- Circular imports indicate layer violations
- Tool sections share boilerplate that could be abstracted
- Path logic is duplicated across filesystem modules

---

## Implementation Order

If addressing these refactorings, the recommended order is:

1. **Break circular imports** - This unblocks other refactorings
2. **Decompose `shared.py`** - Highest impact for maintainability
3. **Consolidate path logic** - Quick win, reduces duplication
4. **Extract tool section factory** - Reduces boilerplate in new tools
5. **Reduce Session responsibilities** - Improves testability
6. **Simplify prompt module** - Larger undertaking, do incrementally

Each refactoring should be done incrementally with passing tests at each step.
