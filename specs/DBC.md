# Design by Contract Specification

Internal-only DbC helpers for preconditions, postconditions, invariants, and
purity expectations. Zero-cost in production; active during tests.

**Source:** `src/weakincentives/dbc/`

## Principles

- **Internal safety net**: Not part of public API
- **Zero-cost default**: No-op unless explicitly enabled
- **Clear diagnostics**: Failures name decorator, callable, predicate
- **Test-time enforcement**: Activated via pytest plugin

## Decorators

### @require

Preconditions checked before execution:

```python
@require(lambda amount: amount >= 0)
def withdraw(account: Account, amount: int) -> None: ...
```

### @ensure

Postconditions checked after execution (receives `result` or `exception`):

```python
@ensure(lambda amount, result: result.balance >= 0)
def withdraw(account: Account, amount: int) -> Account: ...
```

### @invariant

Class-level invariants checked before/after public method execution:

```python
@invariant(lambda self: self.balance >= 0)
class Account:
    def deposit(self, amount: int) -> None: ...
```

### @pure

Documents side-effect-free functions. Checks: no argument mutation, no file I/O,
no logging.

```python
@pure
def calculate_total(items: list[Item]) -> int: ...
```

## Activation

| Method | Scope |
|--------|-------|
| `WEAKINCENTIVES_DBC=1` | Environment variable |
| pytest plugin | Test runs |
| `enable_dbc()` / `disable_dbc()` | Programmatic |
| `dbc_enabled()` | Context manager |

## Exhaustiveness Checking

Use `assert_never` for match statements on union types:

```python
match op:
    case Append(item=item): ...
    case Extend(items=items): ...
    case _ as unreachable:  # pragma: no cover
        assert_never(unreachable)  # pyright: ignore[reportUnreachable]
```

**When to use:** Match statements on union types. Skip for isinstance chains.

## Limitations

- Synchronous callables only
- No static verification or sandboxing
- Purity checks are best-effort (common side effects only)
