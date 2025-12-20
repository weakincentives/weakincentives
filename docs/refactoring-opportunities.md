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

---

## Additional Findings (Deep Scan)

### 9. Function Parameter Bloat

**Problem**: Several functions have been marked with `# noqa: PLR0913` (too many arguments), indicating they need parameter object refactoring:

| File | Function | Args |
|------|----------|------|
| `adapters/shared.py:762` | `_execute_tool_handler()` | 7 |
| `adapters/shared.py:796` | `_handle_tool_exception()` | 8 |
| `adapters/shared.py:830` | `_execute_tool_with_snapshot()` | 9 |
| `serde/dump.py:163` | `dump()` | 6+ |
| `prompt/registry.py:668` | `_validate_step_output_type()` | 7+ |
| `adapters/litellm.py:220` | `__init__()` | 6+ |

**Solution**: Create context objects to group related parameters:

```python
@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    tool: Tool
    handler: ToolHandler
    tool_name: str
    params: dict[str, Any]
    context: ToolContext
    # ...
```

---

### 10. Complex Deserialization in `execution_state.py`

**Problem**: `CompositeSnapshot.from_json()` at line 208 is marked with four noqa suppressions:
- `C901` (too complex)
- `PLR0912` (too many branches)
- `PLR0914` (too many local variables)
- `PLR0915` (too many statements)

This 100+ line method manually parses JSON with extensive error handling.

**Solution**: Break into smaller parsing functions:
```python
def _parse_snapshot_id(payload: Mapping) -> UUID: ...
def _parse_created_at(payload: Mapping) -> datetime: ...
def _parse_session_snapshot(payload: Mapping) -> Snapshot: ...
def _parse_resource_snapshots(payload: Mapping) -> dict[type, object]: ...
```

---

### 11. Protocol Proliferation

**Finding**: 45+ Protocol classes across the codebase, with some modules defining multiple related protocols:

| Module | Protocols |
|--------|-----------|
| `prompt/protocols.py` | 5 protocols |
| `prompt/overrides/versioning.py` | 5 protocols |
| `adapters/_provider_protocols.py` | 6 protocols |
| `runtime/session/*.py` | 4 protocols |

**Observation**: While protocols are good for decoupling, some may be overly fine-grained. Consider consolidating related protocols into fewer, more cohesive interfaces.

---

### 12. Error Message Construction Pattern

**Problem**: 100+ occurrences of inline `msg = ` variable assignments for error messages throughout the codebase. This creates:
- No central source of truth for error messages
- Inconsistent message formatting
- Difficulty maintaining consistent error messaging

**Example locations**:
- `prompt/overrides/validation.py:112-128` - Multiple error message constructions
- `adapters/shared.py` - Scattered error strings
- `runtime/execution_state.py:233-237` - Inline message formatting

**Solution**: Create an `_errors.py` or `messages.py` module per package with message templates:

```python
# adapters/_messages.py
class AdapterMessages:
    TOOL_VALIDATION_FAILED = "Tool validation failed: {error}"
    DEADLINE_EXCEEDED = "Tool '{tool_name}' exceeded the deadline."
    # ...
```

---

### 13. Excessive Type Casting

**Finding**: 269 occurrences of `cast()` across the codebase, heavily concentrated in:
- `serde/parse.py` - Complex type coercion
- `adapters/shared.py` - Provider response handling
- `prompt/overrides/validation.py` - Override processing

**High cast() usage suggests**:
1. Type annotations could be more precise
2. Some `cast()` calls could be replaced with `TypeGuard` functions
3. Protocol definitions might need refinement

**Recommendation**: Audit cast() usage and consider:
- Using `TypeGuard` for type narrowing
- Improving generic type parameters
- Simplifying type hierarchies where possible

---

### 14. Pragma No Cover Accumulation

**Finding**: 79 `# pragma: no cover` directives, primarily in:
- `dbc/__init__.py` - Defensive guards for pure patches
- `adapters/shared.py` - Defensive fallbacks
- `prompt/overrides/inspection.py` - OS error handling

**Concern**: While some are legitimate, the quantity suggests:
- Potentially untestable code paths
- Overly defensive programming
- Opportunities for simplification

---

### 15. DBC Pure Patching Global State

**Problem**: `dbc/__init__.py` lines 334-407 manage patching state using:
- 4 global variables (`_pure_patch_depth`, `_pure_patch_original_*`)
- RLock for synchronization
- Complex activation/deactivation lifecycle

**Issue**: The `_activate_pure_patches()` and `_deactivate_pure_patches()` functions form a fragile state machine with global mutable state.

**Solution**: Extract into a `PurePatchManager` class:
```python
class PurePatchManager:
    def __init__(self) -> None:
        self._depth = 0
        self._lock = RLock()
        self._originals: dict[str, Any] = {}

    @contextmanager
    def activate(self) -> Iterator[None]:
        with self._lock:
            # ...
```

---

### 16. Logging Component Name Strings

**Problem**: Component names in structured logging are hardcoded strings scattered across modules:
- `"adapters.shared"`
- `"sdk_hooks"`
- `"event_bus"`
- `"prompt_overrides"`
- `"session"`

**Solution**: Create a `LoggingComponents` enum:
```python
class LoggingComponents(str, Enum):
    ADAPTERS_SHARED = "adapters.shared"
    SDK_HOOKS = "sdk_hooks"
    # ...
```

---

### 17. `Any` Type Overuse in Claude SDK Hooks

**Problem**: `adapters/claude_agent_sdk/_hooks.py` has 20+ functions using `Any` type with `# noqa: ANN401`:

```python
async def pre_tool_use_hook(
    input_data: Any,  # noqa: ANN401
    sdk_context: Any,  # noqa: ANN401
) -> ...:
```

**Issue**: The Claude SDK types are not well-typed, forcing `Any` usage. This reduces type safety.

**Partial solution**: Create type stubs or wrapper types:
```python
@dataclass
class SDKToolUseData:
    """Typed wrapper for SDK tool use data."""
    name: str
    arguments: dict[str, Any]
    # ...

def parse_tool_use_data(input_data: Any) -> SDKToolUseData: ...
```

---

### 18. Validation Config Dataclass Proliferation

**Problem**: `prompt/overrides/validation.py` creates config dataclasses just to pass error message templates:
- `SectionValidationConfig` (lines 94-98)
- `ToolValidationConfig` (lines 147-152)

These are instantiated multiple times per validation call.

**Solution**: Replace with simpler approach:
```python
def _validate_section(
    override: dict,
    *,
    unknown_items_msg: str = "Unknown sections: {items}",
    # ...
) -> SectionOverride: ...
```

---

### 19. Duplicate Index-Building Functions

**Problem**: `prompt/overrides/validation.py` has parallel functions:
- `_section_descriptor_index()` (line 62)
- `_tool_descriptor_index()` (line 67)

Similarly for serialization:
- `serialize_sections()` (line 501)
- `serialize_tools()` (line 524)

**Solution**: Create generic utilities:
```python
def build_descriptor_index[T](
    descriptors: Iterable[T],
    key_fn: Callable[[T], str],
) -> dict[str, T]: ...
```

---

### 20. Rendering Complexity

**Problem**: `prompt/rendering.py:166` `render()` method is marked with `# noqa: PLR0914, C901` (too complex, too many local variables).

The method handles:
- Section traversal
- Tool collection
- Visibility resolution
- Parameter binding
- Structured output configuration

**Solution**: Break into pipeline stages:
```python
class RenderingPipeline:
    def collect_sections(self, prompt: Prompt) -> list[Section]: ...
    def resolve_visibility(self, sections: list[Section]) -> list[Section]: ...
    def bind_parameters(self, sections: list[Section]) -> list[Section]: ...
    def generate_output(self, sections: list[Section]) -> RenderedPrompt: ...
```

---

## Summary of All Findings

| Category | Count | Priority |
|----------|-------|----------|
| God modules (>1000 lines) | 5 | High |
| Functions needing param objects | 6+ | High |
| Circular import suppressions | 8 | High |
| Complex methods (multi-noqa) | 3 | Medium |
| Duplicate patterns | 6+ | Medium |
| Type casting overuse | 269 | Medium |
| Pragma no cover | 79 | Low |
| Error message scatter | 100+ | Low |
| Protocol proliferation | 45+ | Low |

## Recommended Refactoring Order

1. **Create parameter context objects** - Quick win, improves readability
2. **Break circular imports** - Unblocks architectural improvements
3. **Decompose `adapters/shared.py`** - Highest maintainability gain
4. **Extract error messages** - Enables consistent messaging
5. **Reduce Session responsibilities** - Improves testability
6. **Consolidate path utilities** - Reduces duplication
7. **Simplify complex methods** - Break down multi-noqa functions
8. **Address type casting** - Improve type safety incrementally
