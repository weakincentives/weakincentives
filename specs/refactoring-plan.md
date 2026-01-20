# Refactoring Plan: Improving Abstractions and Reducing Bug Risk

## Executive Summary

This plan addresses complexity hotspots identified in the weakincentives codebase that could harbor critical bugs. The focus is on introducing reusable abstractions, reducing cyclomatic complexity, and improving testability through better composition patterns.

---

## Priority 1: High-Complexity Method Decomposition

### 1.1 Extract SDK Query Builder Pattern (`adapter.py`)

**Problem**: `_run_sdk_query()` (lines 696-900+) has cyclomatic complexity suppressed via `# noqa: C901, PLR0912, PLR0915`. It handles:
- Options building (15+ conditional assignments)
- Hook creation (8 hook types)
- Error handling across multiple failure modes
- Logging at every step

**Solution**: Introduce a **Builder pattern** for SDK query options.

```python
# New file: src/weakincentives/adapters/claude_agent_sdk/_query_builder.py

@dataclass(slots=True, frozen=True)
class SdkQueryOptions:
    """Immutable SDK query configuration."""
    model: str
    cwd: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    betas: tuple[str, ...] = ()
    output_format: dict[str, Any] | None = None
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    setting_sources: tuple[str, ...] = ()
    max_thinking_tokens: int | None = None
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    stderr_handler: Callable[[str], None] | None = None


class SdkQueryBuilder:
    """Builder for SDK query options with validation."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._options: dict[str, Any] = {}

    def with_client_config(self, config: ClientConfig) -> Self:
        """Apply all client configuration options."""
        if config.permission_mode:
            self._options["permission_mode"] = config.permission_mode
        # ... etc
        return self

    def with_model_config(self, config: ModelConfig) -> Self:
        """Apply model configuration."""
        # ...
        return self

    def with_ephemeral_home(self, home: EphemeralHome) -> Self:
        """Apply isolation configuration."""
        # ...
        return self

    def with_mcp_tools(self, tools: tuple[Any, ...]) -> Self:
        """Register bridged tools via MCP."""
        # ...
        return self

    def build(self) -> SdkQueryOptions:
        """Build immutable options, validating all constraints."""
        return SdkQueryOptions(model=self._model, **self._options)
```

**Benefits**:
- Each `with_*` method is independently testable
- Options building logic is extracted from execution logic
- Immutable result prevents accidental mutation
- Builder validates constraints at build time

### 1.2 Extract Hook Registry Pattern (`adapter.py`)

**Problem**: Hook creation is inline with 8 different hook types manually assembled.

**Solution**: Create a `HookRegistry` that encapsulates hook composition.

```python
# New file: src/weakincentives/adapters/claude_agent_sdk/_hook_registry.py

@dataclass(slots=True, frozen=True)
class HookSet:
    """Complete set of hooks for an SDK query."""
    pre_tool_use: Callable[..., Any]
    post_tool_use: Callable[..., Any]
    stop: Callable[..., Any]
    user_prompt_submit: Callable[..., Any]
    subagent_start: Callable[..., Any]
    subagent_stop: Callable[..., Any]
    pre_compact: Callable[..., Any]
    notification: Callable[..., Any]


class HookRegistry:
    """Factory for creating coordinated hook sets."""

    def __init__(self, context: HookContext) -> None:
        self._context = context

    def create_hook_set(
        self,
        *,
        stop_on_structured_output: bool = False,
        task_completion_checker: TaskCompletionChecker | None = None,
    ) -> HookSet:
        """Create a complete, coordinated hook set."""
        # All hook creation logic centralized here
        return HookSet(
            pre_tool_use=create_pre_tool_use_hook(self._context),
            post_tool_use=create_post_tool_use_hook(
                self._context,
                stop_on_structured_output=stop_on_structured_output,
                task_completion_checker=task_completion_checker,
            ),
            # ... etc
        )

    def to_sdk_hooks(self, hook_set: HookSet) -> dict[str, list[Any]]:
        """Convert HookSet to SDK-compatible hooks dict."""
        from claude_agent_sdk.types import HookMatcher
        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[hook_set.pre_tool_use])],
            # ... etc
        }
```

**Benefits**:
- Hook creation is unit testable without SDK dependency
- Hook coordination logic is explicit and verifiable
- Easier to add new hook types or modify existing ones

---

## Priority 2: Tool Validation Abstraction

### 2.1 Create Reusable Validators (`prompt/tool.py`)

**Problem**: Lines 282-447 contain repeated validation patterns:
- `_validate_name()` - strip, check length, check pattern
- `_validate_description()` - strip, check length, check ASCII encoding
- `_validate_example_description()` - same as above
- Multiple example validators with similar structure

**Solution**: Create a composable validation framework.

```python
# New file: src/weakincentives/prompt/_validation.py

@dataclass(slots=True, frozen=True)
class ValidationError:
    """Structured validation failure."""
    field: str
    message: str
    value: object = None


class Validator(Protocol[T]):
    """Protocol for composable validators."""

    def validate(self, value: T) -> ValidationError | None:
        """Return None if valid, ValidationError if invalid."""
        ...


@dataclass(slots=True, frozen=True)
class StringValidator:
    """Validates string fields with multiple constraints."""
    field_name: str
    min_length: int = 1
    max_length: int = 200
    pattern: re.Pattern[str] | None = None
    require_ascii: bool = False
    strip_whitespace: bool = True
    allow_surrounding_whitespace: bool = True

    def validate(self, value: str) -> ValidationError | None:
        clean = value.strip() if self.strip_whitespace else value

        if not self.allow_surrounding_whitespace and value != clean:
            return ValidationError(
                self.field_name,
                f"{self.field_name} must not have surrounding whitespace",
                value,
            )

        if len(clean) < self.min_length or len(clean) > self.max_length:
            return ValidationError(
                self.field_name,
                f"{self.field_name} must be {self.min_length}-{self.max_length} characters",
                clean,
            )

        if self.pattern and not self.pattern.fullmatch(clean):
            return ValidationError(
                self.field_name,
                f"{self.field_name} must match pattern {self.pattern.pattern}",
                clean,
            )

        if self.require_ascii:
            try:
                clean.encode("ascii")
            except UnicodeEncodeError:
                return ValidationError(
                    self.field_name,
                    f"{self.field_name} must be ASCII",
                    clean,
                )

        return None


# Pre-configured validators for common cases
TOOL_NAME_VALIDATOR = StringValidator(
    field_name="name",
    min_length=1,
    max_length=64,
    pattern=re.compile(r"^[a-z0-9_-]+$"),
    allow_surrounding_whitespace=False,
)

TOOL_DESCRIPTION_VALIDATOR = StringValidator(
    field_name="description",
    min_length=1,
    max_length=200,
    require_ascii=True,
)
```

**Usage in `tool.py`**:
```python
def _validate_name(self, params_type: ParamsType) -> str:
    error = TOOL_NAME_VALIDATOR.validate(self.name)
    if error:
        raise PromptValidationError(
            error.message,
            dataclass_type=params_type,
            placeholder=error.value or self.name,
        )
    return self.name.strip()
```

**Benefits**:
- Validators are independently testable
- Constraints are declarative and inspectable
- Easy to add new validation rules
- Reduces ~150 lines to ~30 lines in tool.py

---

## Priority 3: Exception Handling Consolidation

### 3.1 Create Exception Handler Chain (`tool_executor.py`)

**Problem**: `_handle_tool_exception()` uses cascading isinstance checks that are hard to extend and test.

**Solution**: Implement a **Chain of Responsibility** pattern.

```python
# New file: src/weakincentives/adapters/_exception_handlers.py

@dataclass(slots=True, frozen=True)
class ExceptionContext:
    """Context for exception handling."""
    error: Exception
    tool_name: str
    prompt_name: str
    deadline: datetime | None
    provider_payload: Mapping[str, Any]
    log: StructuredLogger
    snapshot: CompositeSnapshot
    tool_params: SupportsDataclass | None
    arguments_mapping: Mapping[str, Any]


class ExceptionHandler(Protocol):
    """Handler for a specific exception type."""

    def can_handle(self, error: Exception) -> bool:
        """Return True if this handler can process the error."""
        ...

    def handle(
        self, ctx: ExceptionContext
    ) -> ToolResult[SupportsToolResult] | Exception:
        """Handle the error. Return ToolResult or raise/return Exception to propagate."""
        ...


@dataclass(slots=True, frozen=True)
class ValidationErrorHandler:
    """Handles ToolValidationError - no snapshot restore needed."""

    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, ToolValidationError)

    def handle(self, ctx: ExceptionContext) -> ToolResult[SupportsToolResult]:
        error = cast(ToolValidationError, ctx.error)
        ctx.log.warning(
            f"Tool validation failed: {error}",
            event="tool.validation_error",
        )
        return ToolResult.error(str(error))


@dataclass(slots=True, frozen=True)
class DeadlineErrorHandler:
    """Handles DeadlineExceededError - re-raises for context manager."""

    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, DeadlineExceededError)

    def handle(self, ctx: ExceptionContext) -> Exception:
        # Return exception to signal it should be raised
        return ToolDeadlineExceededError(
            prompt_name=ctx.prompt_name,
            tool_name=ctx.tool_name,
            deadline=ctx.deadline,
        )


class ExceptionHandlerChain:
    """Chain of exception handlers with fallback."""

    def __init__(self, handlers: Sequence[ExceptionHandler]) -> None:
        self._handlers = tuple(handlers)

    def handle(
        self,
        ctx: ExceptionContext,
        restore_fn: Callable[[str], None],
    ) -> ToolResult[SupportsToolResult]:
        for handler in self._handlers:
            if handler.can_handle(ctx.error):
                result = handler.handle(ctx)
                if isinstance(result, Exception):
                    raise result from ctx.error
                return result

        # Fallback: restore snapshot and return generic error
        restore_fn("exception")
        return ToolResult.error(f"Unexpected error: {ctx.error}")


# Default chain used by tool executor
DEFAULT_EXCEPTION_CHAIN = ExceptionHandlerChain([
    ValidationErrorHandler(),
    DeadlineErrorHandler(),
    TypeErrorHandler(),  # Handles signature mismatches
    # Fallback is built into chain
])
```

**Benefits**:
- Each handler is independently unit testable
- Easy to add new exception types
- Handler order is explicit and configurable
- Reduces cognitive load when reading tool_executor.py

---

## Priority 4: Session Module Circular Import Resolution

### 4.1 Extract Protocol Definitions

**Problem**: 8 files in session/ disable import cycle checks. Root cause: Session, SliceAccessor, and TypedReducer have mutual dependencies.

**Solution**: Create a dedicated protocols module that all others import from.

```python
# New file: src/weakincentives/runtime/session/_protocols.py
"""Session protocols - import this for type hints, not implementations."""

from typing import TYPE_CHECKING, Protocol, TypeVar

from ...types import SupportsDataclass

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

S = TypeVar("S", bound=SupportsDataclass)


class SliceAccessorProtocol(Protocol[S]):
    """Protocol for slice access operations."""

    def latest(self) -> S | None: ...
    def all(self) -> tuple[S, ...]: ...
    def where(self, predicate: Callable[[S], bool]) -> tuple[S, ...]: ...
    def append(self, item: S) -> None: ...
    def replace(self, item: S) -> None: ...
    def clear(self) -> None: ...


class SessionProtocol(Protocol):
    """Protocol for session operations."""

    @property
    def session_id(self) -> UUID: ...

    @property
    def created_at(self) -> datetime: ...

    def __getitem__(self, slice_type: type[S]) -> SliceAccessorProtocol[S]: ...

    def dispatch(self, event: SupportsDataclass) -> None: ...


class TypedReducerProtocol(Protocol[S]):
    """Protocol for typed reducers."""

    def reduce(
        self, state: tuple[S, ...], event: SupportsDataclass
    ) -> SliceOp[S]: ...
```

**Refactoring steps**:
1. Create `_protocols.py` with protocol definitions
2. Update `session.py` to inherit from `SessionProtocol`
3. Update `slice_accessor.py` to inherit from `SliceAccessorProtocol`
4. Update imports throughout to use protocols for type hints

**Benefits**:
- Eliminates circular import disabling
- Protocols serve as documentation
- Easier to mock for testing

---

## Priority 5: DBC Global State Encapsulation

### 5.1 Replace Global State with Context Variables

**Problem**: `dbc/__init__.py` uses module-level globals (`_forced_state`, `_pure_patch_depth`, `_PURE_PATCH_LOCK`) which are hard to test and can leak between tests.

**Solution**: Encapsulate state in a `DbcContext` class with context variables.

```python
# Refactored src/weakincentives/dbc/__init__.py

from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import RLock


@dataclass(slots=True)
class DbcState:
    """Mutable DBC state - only accessed via context."""
    forced: bool | None = None
    pure_patch_depth: int = 0


_dbc_context: ContextVar[DbcState] = ContextVar("dbc_context", default=DbcState())
_pure_patch_lock = RLock()


def _get_state() -> DbcState:
    """Get current DBC state from context."""
    return _dbc_context.get()


def dbc_active() -> bool:
    """Return True when DbC checks should run."""
    state = _get_state()
    if state.forced is not None:
        return state.forced
    return _coerce_flag(os.getenv(_ENV_FLAG))


@contextmanager
def dbc_enabled(active: bool = True) -> Iterator[None]:
    """Temporarily set DBC state in a context-safe way."""
    state = _get_state()
    previous = state.forced
    state.forced = active
    try:
        yield
    finally:
        state.forced = previous


@contextmanager
def fresh_dbc_context() -> Iterator[None]:
    """Create isolated DBC context for testing."""
    token = _dbc_context.set(DbcState())
    try:
        yield
    finally:
        _dbc_context.reset(token)
```

**Benefits**:
- Test isolation via `fresh_dbc_context()`
- No global state leakage
- Thread-safe via context variables
- Async-safe (context vars work with asyncio)

---

## Priority 6: Type Casting Reduction

### 6.1 Create Type-Safe Wrappers

**Problem**: 265+ `cast()` calls throughout codebase, many indicating type system workarounds.

**Solution**: Create type-safe wrapper functions for common patterns.

```python
# New file: src/weakincentives/types/_guards.py
"""Type guards and safe casting utilities."""

from typing import TypeGuard, TypeVar, overload

T = TypeVar("T")


def is_instance_of(value: object, type_: type[T]) -> TypeGuard[T]:
    """Type guard for isinstance checks."""
    return isinstance(value, type_)


def ensure_type(value: object, type_: type[T], context: str = "") -> T:
    """Assert value is of type and return with correct typing.

    Unlike cast(), this validates at runtime in debug mode.
    """
    if not isinstance(value, type_):
        msg = f"Expected {type_.__name__}, got {type(value).__name__}"
        if context:
            msg = f"{context}: {msg}"
        raise TypeError(msg)
    return value


def narrow_optional(value: T | None, context: str = "") -> T:
    """Narrow Optional[T] to T, raising if None."""
    if value is None:
        msg = f"Expected non-None value"
        if context:
            msg = f"{context}: {msg}"
        raise ValueError(msg)
    return value
```

**Benefits**:
- Runtime validation in debug mode catches bugs early
- Type checker understands the narrowing
- Clearer intent than `cast()`

---

## Implementation Phases

### Phase 1: Foundation (Low Risk, High Impact)
1. Create `_validation.py` with `StringValidator`
2. Create `_protocols.py` in session module
3. Create `_guards.py` type utilities
4. Add tests for all new modules

### Phase 2: SDK Adapter Refactoring
1. Extract `SdkQueryBuilder`
2. Extract `HookRegistry`
3. Refactor `_run_sdk_query()` to use new abstractions
4. Maintain backward compatibility during transition

### Phase 3: Tool Executor Refactoring
1. Create `ExceptionHandlerChain`
2. Refactor `_handle_tool_exception()` to use chain
3. Update tool.py to use validators

### Phase 4: DBC and Session Cleanup
1. Refactor DBC to use context variables
2. Update session imports to use protocols
3. Remove `# ruff: noqa: I001` directives where possible

---

## Testing Strategy

### Unit Tests for New Abstractions
Each new module must have 100% coverage:
- `test_validation.py` - all validator edge cases
- `test_query_builder.py` - builder state transitions
- `test_hook_registry.py` - hook creation and coordination
- `test_exception_handlers.py` - each handler type
- `test_type_guards.py` - guard functions

### Integration Tests
- Verify refactored components work together
- Test backward compatibility of public APIs
- Test error paths with real exceptions

### Property-Based Tests
Consider adding hypothesis tests for:
- Validator edge cases (unicode, empty strings, boundary lengths)
- Builder state invariants
- Exception handler ordering

---

## Risk Mitigation

1. **Feature flags**: New abstractions can be enabled gradually
2. **Parallel implementation**: Keep old code paths during transition
3. **Incremental commits**: One refactoring per commit for easy rollback
4. **CI gates**: All phases must pass `make check`

---

## Success Metrics

After completing all phases:
- [ ] Remove all `# noqa: C901` (cyclomatic complexity)
- [ ] Remove all `# noqa: PLR0912, PLR0915` (too many branches/statements)
- [ ] Reduce `cast()` usage by 50%+
- [ ] Remove import cycle disabling from session module
- [ ] All new modules have 100% test coverage
- [ ] `make check` passes

---

## Appendix: Files to Create

| File | Purpose |
|------|---------|
| `src/weakincentives/prompt/_validation.py` | Composable validators |
| `src/weakincentives/adapters/claude_agent_sdk/_query_builder.py` | SDK query builder |
| `src/weakincentives/adapters/claude_agent_sdk/_hook_registry.py` | Hook management |
| `src/weakincentives/adapters/_exception_handlers.py` | Exception chain |
| `src/weakincentives/runtime/session/_protocols.py` | Session protocols |
| `src/weakincentives/types/_guards.py` | Type guard utilities |

## Appendix: Files to Modify

| File | Changes |
|------|---------|
| `adapters/claude_agent_sdk/adapter.py` | Use builder and hook registry |
| `adapters/tool_executor.py` | Use exception chain |
| `prompt/tool.py` | Use validators |
| `dbc/__init__.py` | Use context variables |
| `runtime/session/*.py` | Use protocols |
