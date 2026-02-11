# Design by Contract Specification

## Purpose

Internal-only DbC framework for `weakincentives`. Decorators describe
preconditions, postconditions, invariants, and purity expectations. Always
enabled by default; use `dbc_suspended()` for performance-critical paths.
Implementation at `src/weakincentives/dbc/__init__.py`.

## Principles

- **Internal safety net**: Reinforces contributor discipline; not public API
- **Always-on by default**: Contracts enforced in all contexts (tests and production)
- **Pragmatic coverage**: Catches common footguns, not exhaustive verification
- **Clear diagnostics**: Names decorator, callable, and offending predicate
- **Local opt-out only**: `dbc_suspended()` for scoped performance-critical code

## Contract Decorators

### @require

At `src/weakincentives/dbc/__init__.py`. Preconditions on function entry.

Callables receive `*args`, `**kwargs`, optionally bound `self`/`cls`.
All must return truthy; falsy or exception fails test with descriptive assertion.

```python
@require(lambda amount: amount >= 0)
@require(lambda account, amount: account.can_withdraw(amount))
def withdraw(account: Account, amount: int) -> None: ...
```

### @ensure

At `src/weakincentives/dbc/__init__.py`. Postconditions after return or raise.

Callables receive original arguments plus `result` or `exception` keyword.
Supports `(bool, message)` tuples for custom diagnostics.

```python
@ensure(lambda amount, result: result.balance >= 0)
def withdraw(account: Account, amount: int) -> Account: ...
```

### @invariant

At `src/weakincentives/dbc/__init__.py`. Class invariants checked before/after public methods.

Wraps `__init__` and public methods (excludes `_`-prefixed, static, class methods).
Use `skip_invariant()` to exclude specific methods.

```python
@invariant(lambda self: self.balance >= 0)
class Account:
    def deposit(self, amount: int) -> None:
        self.balance += amount
```

Implementation details:

- `_wrap_init_with_invariants()` at `src/weakincentives/dbc/__init__.py`
- `_wrap_methods_with_invariants()` at `src/weakincentives/dbc/__init__.py`

## Runtime Behavior

| Context | Behavior |
| --- | --- |
| Default | Contracts always enforced |
| `with dbc_suspended():` | Temporarily disable for performance-critical code |

Contract violations raise `AssertionError` with decorator type, callable name,
and formatted argument dump.

### Control Functions

Defined in `src/weakincentives/dbc/__init__.py`:

| Function | Description |
| --- | --- |
| `dbc_active()` | Check if DbC currently active (always `True` unless suspended) |
| `dbc_suspended()` | Context manager to temporarily suspend enforcement |

### Design Rationale

DbC cannot be globally disabled. This ensures:

- Contracts are checked in production, catching bugs early
- No accidental deployment with contracts disabled
- Performance-sensitive code can explicitly opt-out via `dbc_suspended()`

### Pytest Integration

Plugin at `tests/plugins/dbc.py` exists for compatibility but no longer needs
to toggle state since DbC is always enabled.

## Example Use Cases

| Domain | Pattern |
| --- | --- |
| Prompt builders | `@require(lambda ctx: "user" in ctx)` |
| Session reducers | `@invariant` for monotonic state |
| Serialization | `@ensure` to validate output format |
| Budget counters | `@require` for non-negative, `@invariant` for non-negative balance |

## Exhaustiveness Checking

### assert_never Pattern

End match statements on union types with `assert_never` sentinel:

```python
from typing import assert_never

match op:
    case Append(item=item):
        ...
    case _ as unreachable:  # pragma: no cover
        assert_never(unreachable)  # pyright: ignore[reportUnreachable]
```

Adding new variant to union immediately surfaces as pyright error.

### When to Use

| Use | Skip |
| --- | --- |
| Match on union types | isinstance chains (pyright handles) |
| `SliceOp[T]` in `apply_slice_op()` (`session_dispatch.py`) | After exhaustive isinstance narrowing |

## Testing

- Unit: Successful and failing contracts for each decorator
- Integration: Pytest plugin toggles correctly
- Regression: Decorator stacking, inheritance, async compatibility

## Limitations

- **Internal only**: Not exported; not in public docs
- **Synchronous only**: No async support
- **Best-effort purity**: Checks common side effects, not exhaustive
- **Runtime overhead**: Contracts are always enforced; use `dbc_suspended()` for hot paths
