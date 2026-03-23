# Architectural Review: weakincentives Library

**Date:** 2026-03-23
**Scope:** Comprehensive quality-centric review focused on architectural
shortcomings, bugs, inconsistencies, and type safety.

---

## 1. Backward-Compatibility Properties Exposing Raw Internals

**Location:** `src/weakincentives/runtime/session/session.py:595-642`
**Severity:** High (architectural)

Session exposes `_slices`, `_slice_policies`, and `_reducers` as properties
that return **mutable references** to internal subsystem state, bypassing the
facade pattern entirely:

```python
@property
def _slices(self) -> dict[SessionSliceType, Slice[Any]]:
    return self._store._slices  # Double-private access
```

CLAUDE.md says *"Delete unused code completely; no backward-compatibility
shims"* — yet this is an alpha library with explicit backward-compatibility
properties reaching through two encapsulation layers (`Session → SliceStore →
_slices`). The `session_cloning` module is the only consumer, and it should use
proper internal APIs on the subsystems directly.

**Recommendation:** Add explicit internal methods to `SliceStore` and
`ReducerRegistry` (e.g., `_clone_state()`, `_apply_cloned_state()`) and remove
these bridging properties.

---

## 2. Blanket pyright Suppressions in serde Module

**Location:** `src/weakincentives/serde/parse.py:15`,
`src/weakincentives/serde/_coercers.py:16`,
`src/weakincentives/serde/_generics.py:15`
**Severity:** High (type safety)

```python
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false,
# reportUnknownMemberType=false, reportUnknownParameterType=false,
# reportUnnecessaryIsInstance=false, reportCallIssue=false,
# reportArgumentType=false, reportPrivateUsage=false
```

This blanket suppresses **8 categories of type errors** across the entire serde
module — the very module whose purpose is type-safe serialization. These
suppressions make it impossible for pyright to catch genuine type errors
introduced during maintenance.

**Recommendation:** Replace blanket file-level suppressions with targeted
per-line `# pyright: ignore[specificRule]` annotations. Consider introducing a
`CoercionResult` type or using overloads to make the coercion chain type-safe.

---

## 3. The `_NOT_HANDLED` Sentinel Pattern in Coercers

**Location:** `src/weakincentives/serde/_coercers.py:60, 469-514`
**Severity:** Medium (type safety)

```python
_NOT_HANDLED = object()
```

A chain-of-responsibility using a sentinel object that is indistinguishable
from a valid return value at the type level. Every coercer returns `object`, and
the sentinel check is purely a runtime convention.

**Recommendation:** Use an explicit discriminated union:

```python
@dataclass(frozen=True, slots=True)
class Handled[T]:
    value: T

type CoercionResult[T] = Handled[T] | None  # None = not handled
```

---

## 4. TOCTOU Race in Session Dispatch

**Location:** `src/weakincentives/runtime/session/session_dispatch.py:129-177`
**Severity:** High (correctness)

The lock is acquired and released **three separate times** per reducer
invocation. Between the first lock release (after getting registrations) and the
subsequent per-reducer locks:

- New reducers could be registered
- The slice could be replaced or cleared
- The `slice_instance` reference could be stale

The gap between "read slice view" and "apply op" means the reducer sees a
potentially stale view and applies its op to potentially changed state.

**Recommendation:** Hold the lock for the entire dispatch-to-apply sequence per
reducer. The reducer itself is pure and quick (it only computes a `SliceOp`), so
holding the lock during reducer execution should not cause contention issues.

---

## 5. `Tool` is Mutable Despite Being Central to an Immutable System

**Location:** `src/weakincentives/prompt/tool.py:212`
**Severity:** Medium (consistency)

```python
@dataclass(slots=True)  # NOT frozen!
class Tool[ParamsT, ResultT]:
```

`Tool` is `@dataclass(slots=True)` **without** `frozen=True`, even though
`PromptTemplate`, `ToolContext`, `ToolResult`, and `ToolExample` are all frozen.
The `__post_init__` mutates fields that could instead be computed via the
library's own `FrozenDataclass()`'s `__pre_init__` hook.

**Recommendation:** Convert `Tool` to `@FrozenDataclass()` using `__pre_init__`
for type resolution.

---

## 6. `Prompt.bind()` Mutates In Place

**Location:** `src/weakincentives/prompt/prompt.py:388-444`
**Severity:** Medium (consistency)

```python
def bind(self, *params, resources=None) -> Prompt[OutputT]:
    self._params = tuple(current)  # Mutates in place
    return self  # Returns self for chaining illusion
```

In a system built around immutability and frozen dataclasses, having the primary
user-facing API mutate state is an inconsistency.

**Recommendation:** Either make `bind()` return a new `Prompt` instance
(preferred), or rename to signal mutation (e.g., `configure()`).

---

## 7. Public APIs Using `Any` to Avoid Import Cycles

**Location:** `src/weakincentives/runtime/session/slice_accessor.py:185`,
`src/weakincentives/serde/_utils.py:53`
**Severity:** Medium (type safety)

```python
# slice_accessor.py
reducer: Any,  # TypedReducer[T] - avoiding import cycle

# _utils.py
scope: object = None  # SerdeScope | None, using object to avoid circular import
```

Public APIs using `Any` or `object` to work around import cycles means the type
checker cannot catch misuse. Use `TYPE_CHECKING` blocks instead:

```python
if TYPE_CHECKING:
    from ._types import TypedReducer
```

---

## 8. `PromptResources` Created on Every `.resources` Access

**Location:** `src/weakincentives/prompt/prompt.py:568-586`
**Severity:** Low (performance/fragility)

```python
@property
def resources(self) -> PromptResources:
    return PromptResources(self)  # New instance every time
```

This means `with prompt.resources:` and `prompt.resources.get()` operate on
different wrapper instances. Works only because the wrapper is stateless, but is
fragile and wasteful.

**Recommendation:** Use `@cached_property` instead.

---

## 9. `_collected_resources()` Called Without Caching

**Location:** `src/weakincentives/prompt/_prompt_resources.py:145, 166`
**Severity:** Low (performance)

Both `get()` and `get_optional()` call `self._prompt._collected_resources()` on
every invocation, performing a full section tree walk and registry merge each
time.

**Recommendation:** Cache the collected registry on `Prompt` after first bind.

---

## 10. Reducer Exceptions Silently Swallowed

**Location:** `src/weakincentives/runtime/session/session_dispatch.py:178-188`
**Severity:** High (correctness)

```python
except Exception:  # log and continue
    logger.exception("Reducer application failed.", ...)
    continue
```

A failing reducer silently continues, leaving the session in a
partially-updated state (previous reducers' ops were applied, this one's
wasn't). For a system that promises "failed tools don't leave partial state,"
silently swallowing reducer failures without rolling back previous ops is a
correctness issue.

**Recommendation:** Either roll back all ops from this dispatch on any reducer
failure, or make this behavior explicit in the API contract.

---

## 11. Inconsistent Error Hierarchies

**Location:** `src/weakincentives/errors.py`, `src/weakincentives/prompt/errors.py`,
`src/weakincentives/resources/errors.py`
**Severity:** Medium (consistency)

The error hierarchy mixes in stdlib exceptions inconsistently:

- `ToolValidationError(WinkError, ValueError)` — mixes stdlib
- `PromptEvaluationError(WinkError, RuntimeError)` — mixes stdlib
- `ResourceError` — only `WinkError`

Catching `ValueError` or `RuntimeError` accidentally catches WINK errors.

**Recommendation:** Establish a clear policy for stdlib exception mixing and
apply it consistently.

---

## 12. `ContractResult` Type Provides Zero Type Safety

**Location:** `src/weakincentives/types/__init__.py`,
`src/weakincentives/dbc/__init__.py:291-306`
**Severity:** Medium (type safety)

```python
type ContractResult = bool | tuple[bool, *tuple[object, ...]] | None
# Parameter type: ContractResult | object = object
```

The Design by Contract system's own contract types accept any object.
`ContractCallable` = `Callable[..., object]` provides no type safety.

**Recommendation:** Tighten to `type ContractResult = bool | tuple[bool, str]`.

---

## 13. `Section.__init__` Uses `Sequence[object]` for Typed Parameters

**Location:** `src/weakincentives/prompt/section.py:68-69`
**Severity:** Medium (type safety)

```python
tools: Sequence[object] | None = None,  # Should be Sequence[Tool[...]]
skills: Sequence[object] | None = None,  # Should be Sequence[SkillMount]
```

The type checker cannot catch passing non-Tool objects. Use `TYPE_CHECKING`
blocks for proper annotations.

---

## 14. `ProviderAdapter.__class_getitem__` Returns Same Class

**Location:** `src/weakincentives/adapters/core.py:52-53`
**Severity:** Low (misleading API)

```python
@classmethod
def __class_getitem__(cls, _: object) -> type[ProviderAdapter[Any]]:
    return cls
```

`ProviderAdapter[int]` is `ProviderAdapter[str]` at runtime. If the generic is
meaningless, remove it; if it's meaningful, implement specialization.

---

## 15. `PromptTemplate.sections` Union Type Changes After Init

**Location:** `src/weakincentives/prompt/prompt.py:114-117`
**Severity:** Medium (readability)

```python
sections: (
    Sequence[Section[SupportsDataclass]]
    | tuple[SectionNode[SupportsDataclass], ...]
) = ()
```

Accepts sections as input but stores section nodes. After construction, it's
always `tuple[SectionNode[...], ...]`. This dual-type annotation makes it hard
to reason about what `template.sections` actually contains.

**Recommendation:** Use separate types for input and stored value via
`__pre_init__`.

---

## 16. 274 `cast()` Calls Across 67 Files

**Severity:** Medium (systemic type safety)

The `cast()` count is high for a "strict type-centric" library. Each `cast()`
tells the type checker to trust the programmer. Notable patterns:

- `cast(type[T], origin if origin is not None else cls)` in serde
- `cast(Mapping[str, object], data)` — could use type narrowing
- `cast(type[ParamsT], params_type)` in Tool initialization

**Recommendation:** Audit high-frequency cast patterns and replace with type
guards, overloads, or proper `TypeVar` bounds.

---

## Summary

The library demonstrates strong architectural thinking in its core patterns
(event-sourced sessions, pure reducers, immutable templates, protocol-based DI).
The areas where it falls short:

1. **Type safety compromises** — blanket pyright suppressions, `Any`/`object`
   escape hatches, 274 casts
2. **Inconsistent immutability** — `Tool` mutable, `Prompt.bind()` mutates,
   while everything else is frozen
3. **Encapsulation leaks** — backward-compat properties, double-private access
4. **Silent failure modes** — reducer exceptions swallowed, TOCTOU in dispatch

These are all fixable without changing the fundamental architecture. The core
design — separating definition from harness, policies over workflows,
transactional tools — is sound.
