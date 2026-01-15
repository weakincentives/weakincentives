# Design by Contract Specification

## Purpose

Internal-only DbC framework for `weakincentives`. Decorators describe
preconditions, postconditions, invariants, and purity expectations. Active in
tests only; zero-cost in production. Implementation at `dbc/__init__.py`.

## Principles

- **Internal safety net**: Reinforces contributor discipline; not public API
- **Zero-cost default**: No-ops unless explicitly enabled
- **Pragmatic coverage**: Catches common footguns, not exhaustive verification
- **Clear diagnostics**: Names decorator, callable, and offending predicate
- **Composable opt-ins**: `dbc_enabled()` and env flags for scoped enforcement

## Contract Decorators

### @require

At `dbc/__init__.py:157-182`. Preconditions on function entry.

Callables receive `*args`, `**kwargs`, optionally bound `self`/`cls`.
All must return truthy; falsy or exception fails test with descriptive assertion.

```python
@require(lambda amount: amount >= 0)
@require(lambda account, amount: account.can_withdraw(amount))
def withdraw(account: Account, amount: int) -> None: ...
```

### @ensure

At `dbc/__init__.py:185-225`. Postconditions after return or raise.

Callables receive original arguments plus `result` or `exception` keyword.
Supports `(bool, message)` tuples for custom diagnostics.

```python
@ensure(lambda amount, result: result.balance >= 0)
def withdraw(account: Account, amount: int) -> Account: ...
```

### @invariant

At `dbc/__init__.py:323-333`. Class invariants checked before/after public methods.

Wraps `__init__` and public methods (excludes `_`-prefixed, static, class methods).
Use `skip_invariant()` to exclude specific methods.

```python
@invariant(lambda self: self.balance >= 0)
class Account:
    def deposit(self, amount: int) -> None:
        self.balance += amount
```

Implementation details:

- `_wrap_init_with_invariants()` at `dbc/__init__.py:264-277`
- `_wrap_methods_with_invariants()` at `dbc/__init__.py:308-320`

### @pure

At `dbc/__init__.py:475-500`. Documents side-effect-free functions.

Enforcement (best-effort via `deepcopy` + equality):

- No mutation of arguments
- No calls to `builtins.open`, `Path.write_text/bytes`, `logging`

Patching at `_activate_pure_patches()` (`dbc/__init__.py:390-409`).

## Runtime Behavior

| Context | Behavior |
| --- | --- |
| Production (`WEAKINCENTIVES_DBC=0`) | Decorators return original callable |
| Tests | Pytest plugin activates enforcement |
| Manual | `enable_dbc()`, `disable_dbc()`, `dbc_enabled()` context manager |

Contract violations raise `AssertionError` with decorator type, callable name,
and formatted argument dump.

### Control Functions

At `dbc/__init__.py`:

| Function | Line | Description |
| --- | --- | --- |
| `dbc_active()` | 50-55 | Check if DbC enabled |
| `enable_dbc()` | 62-66 | Force enable |
| `disable_dbc()` | 69-73 | Force disable |
| `dbc_enabled()` | 76-86 | Context manager for temporary override |

### Pytest Integration

Plugin at `tests/plugins/dbc.py` toggles flag via `pytest_configure` and
`pytest_unconfigure`.

## Example Use Cases

| Domain | Pattern |
| --- | --- |
| Prompt builders | `@require(lambda ctx: "user" in ctx)` |
| Session reducers | `@invariant` for monotonic state |
| Serialization | `@pure` to catch accidental writes |
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
| `SliceOp[T]` in `Session._apply_slice_op()` | After exhaustive isinstance narrowing |
| `DataEvent` in subscription routing | |

## Testing

- Unit: Successful and failing contracts for each decorator
- Integration: Pytest plugin toggles correctly
- Regression: Decorator stacking, inheritance, async compatibility

## Limitations

- **Internal only**: Not exported; not in public docs
- **Synchronous only**: No async support
- **Best-effort purity**: Checks common side effects, not exhaustive
- **Runtime overhead in tests**: Active enforcement during test runs only
