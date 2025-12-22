# DbC Coverage Gaps Analysis

This document tracks Design-by-Contract (DbC) annotation gaps in the
weakincentives codebase and defines the necessary work to address them.

## Important: Test-Time Enforcement Only

Per `specs/DBC.md`, DbC decorators are **no-ops in production**. They only
activate during test runs (via pytest plugin or `WEAKINCENTIVES_DBC=1`).

This means:
- **Runtime validation must remain** in production code (e.g., `if not isinstance(...)`)
- DbC annotations **supplement** runtime checks, they don't replace them
- The value is **catching contract violations during development** with clear diagnostics
- Annotations document invariants and contracts for contributors

## Current Coverage Summary

DbC annotations are currently used in **6 files**:

| File | Decorators | What's Covered |
|------|------------|----------------|
| `prompt/registry.py` | `@invariant` | `PromptRegistry` class consistency |
| `runtime/session/session.py` | `@invariant` | `Session` class (id, timestamps) |
| `runtime/session/slice_accessor.py` | `@pure` | Query methods |
| `runtime/session/reducers.py` | `@pure` | Built-in reducer functions |
| `runtime/session/state_slice.py` | `@pure` | Method reducer wrapper |

## Identified Gaps

### Phase 1: Serde Module (High Priority)

The serde module performs dataclass serialization/deserialization. It already
has runtime checks but lacks DbC coverage for test-time contract enforcement.

#### `serde/parse.py`

```python
# Existing runtime checks remain (lines 681-684)
# Add DbC for test-time enforcement:
@require(
    lambda cls, data: isinstance(data, Mapping),
    lambda cls, data, extra: extra in {"ignore", "forbid", "allow"},
)
def parse(cls, data, *, extra="ignore", ...): ...
```

#### `serde/dump.py`

```python
# Existing runtime check remains (line 175-176)
# Add DbC for test-time enforcement:
@require(lambda obj: dataclasses.is_dataclass(obj) and not isinstance(obj, type))
def dump(obj, *, by_alias=True, ...): ...

# Existing runtime check remains (line 284-285)
# Add DbC for test-time enforcement:
@require(lambda obj: dataclasses.is_dataclass(obj) and not isinstance(obj, type))
def clone(obj, **updates): ...
```

### Phase 2: Budget and Deadline Classes (Medium Priority)

These classes have `__post_init__` validation (runtime checks that must remain).
Adding DbC `@invariant` provides test-time enforcement that catches violations
earlier in the development cycle with better diagnostics.

#### `budget.py`

```python
# Recommended invariants for Budget:
@invariant(
    lambda self: any([
        self.deadline is not None,
        self.max_total_tokens is not None,
        self.max_input_tokens is not None,
        self.max_output_tokens is not None,
    ]),
    lambda self: self.max_total_tokens is None or self.max_total_tokens > 0,
    lambda self: self.max_input_tokens is None or self.max_input_tokens > 0,
    lambda self: self.max_output_tokens is None or self.max_output_tokens > 0,
)
class Budget: ...

# Recommended for BudgetTracker:
@require(lambda self, evaluation_id: evaluation_id)
def record_cumulative(self, evaluation_id, usage): ...
```

#### `deadlines.py`

```python
# Recommended invariants for Deadline:
@invariant(
    lambda self: self.expires_at.tzinfo is not None,
)
class Deadline: ...

# Recommended precondition:
@require(lambda self, now: now is None or now.tzinfo is not None)
def remaining(self, *, now=None): ...
```

### Phase 3: Prompt Module (Medium Priority)

#### `prompt/tool.py`

```python
# ToolResult could have invariant ensuring message is set:
@invariant(lambda self: self.message is not None)
class ToolResult: ...
```

#### `prompt/prompt.py`

```python
# Prompt.render could have postcondition:
@ensure(lambda result: result.body is not None)
def render(self, *, session=None): ...
```

### Phase 4: Contrib Tools (Low Priority)

#### `contrib/tools/planning.py`

```python
# Plan could have invariants for step consistency:
@invariant(
    lambda self: all(step.id for step in self.steps),
)
class Plan: ...
```

## Implementation Notes

Per `specs/DBC.md`:

- DbC decorators are **internal-only** (not exported to users)
- Decorators are **no-ops** by default; tests enable enforcement
- Focus on **common footguns**: argument validation, invariants, purity
- Contract violations raise `AssertionError` with clear diagnostics

## Work Items

| # | Module | Item | Priority | Effort |
|---|--------|------|----------|--------|
| 1 | serde | Add `@require` to `parse()` | High | Small |
| 2 | serde | Add `@require` to `dump()` and `clone()` | High | Small |
| 3 | serde | Add `@pure` where applicable | High | Small |
| 4 | budget | Add `@invariant` to `Budget` | Medium | Small |
| 5 | deadlines | Add `@invariant` to `Deadline` | Medium | Small |
| 6 | budget | Add `@require` to `BudgetTracker` methods | Medium | Small |
| 7 | prompt | Add `@invariant` to `ToolResult` | Medium | Small |
| 8 | prompt | Add `@ensure` to `Prompt.render()` | Medium | Medium |
| 9 | contrib | Add `@invariant` to `Plan` | Low | Small |
| 10 | contrib | Add `@require` to filesystem utilities | Low | Small |

## Testing Considerations

When adding DbC annotations:

1. Ensure existing tests pass with `WEAKINCENTIVES_DBC=1`
2. Add negative tests that verify contracts catch invalid usage
3. Verify contracts don't break coverage requirements (100%)
4. Run `make check` to validate all quality gates
