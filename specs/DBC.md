# Design by Contract Specification

## Purpose

This document specifies a tiny design-by-contract (DbC) helper framework for
`weakincentives`. The framework exposes annotation-style decorators that let
library authors describe preconditions, postconditions, invariants, and purity
expectations on Python callables. The decorators are intentionally inert during
normal runtime. When the test suite runs, they activate and enforce the declared
contracts so regressions surface early without shipping runtime overhead.

## Goals

- Provide four decorators, `@require`, `@ensure`, `@invariant`, and `@pure`, that
  look and feel like standard function decorators.
- Keep production code free from extra runtime cost by defaulting the
  decorators to no-ops outside the test harness.
- Make the decorators easy to adopt incrementally across modules without
  invasive refactors.
- Integrate with the existing pytest-based test runner so DbC checks run as part
  of `make test` / `make check`.
- Offer helpful failure diagnostics that explain which contract failed, which
  arguments were involved, and any additional context captured by the contract.
- Document example use cases to guide adopters and keep contracts focused on
  correctness guarantees rather than general validation.

## Non-Goals

- This framework will not attempt full static verification, symbolic execution,
  or runtime sandboxing. The contracts execute Python callables provided by the
  author and rely on pytest failures to signal problems.
- There is no plan to inject runtime enforcement into production builds. Any
  consumer who wants runtime DbC must opt-in explicitly through a documented
  hook.
- The scope is limited to synchronous Python callables. Asynchronous or
  generator-specific helpers can be added later if needed.

## Contract Decorators

### `@require`

`@require` expresses preconditions. It accepts one or more callables that take
`*args`, `**kwargs`, and optionally the bound `self` or `cls`. The contract
passes when all callables return truthy values. If any callable returns falsy or
raises an exception, pytest should fail with a descriptive assertion.

```python
@require(lambda amount: amount >= 0, lambda account, amount: account.can_withdraw(amount))
def withdraw(account: Account, amount: int) -> None:
    ...
```

### `@ensure`

`@ensure` declares postconditions. Callables receive the original arguments and
keyword arguments plus the return value (or raised exception object). Decorators
should support both simple boolean functions and more advanced callables that
return `(bool, message)` tuples for custom diagnostics.

```python
@ensure(lambda amount, result: result.balance >= 0)
def withdraw(account: Account, amount: int) -> Account:
    ...
```

### `@invariant`

`@invariant` targets class definitions. When applied, it wraps `__init__` and
public methods to validate the invariant callable(s) before and after method
execution. Invariants run in tests whenever the instance mutates observable
state. The decorator should expose hooks to skip enforcement for designated
private helpers where invariants would be redundant.

```python
@invariant(lambda self: self.balance >= 0)
class Account:
    def deposit(self, amount: int) -> None:
        self.balance += amount
```

### `@pure`

`@pure` documents that a function is side-effect free. During tests, the
framework wraps the target callable and instruments it to ensure:

- It reads no global mutable state (best-effort via snapshot comparison of
  provided globals the function touches).
- It writes no attributes on arguments or global singletons.
- It performs no I/O through tracked modules (e.g., `os`, `pathlib`, network
  sockets).

Instrumentation can start simple: monkeypatch common side-effect primitives
(`open`, `Path.write_text`, `logging.Logger.*`) to raise if invoked from a pure
function context. The focus is to catch accidental side effects in helper
utilities, not to provide complete effect tracking.

## Runtime Behavior

- In production (`PYTEST_CURRENT_TEST` absent and `WEAKINCENTIVES_DBC=0` by
  default) the decorators should return the original callable untouched. Any
  helper context manager or pytest fixture must check a shared `dbc_active`
  flag before running expensive validations.
- During tests, a pytest plugin activates the flag. The plugin hooks into
  `pytest_configure` to toggle DbC checks and registers fixtures that expose
  helper utilities (e.g., capturing return values for `@ensure`).
- Contract violations raise `AssertionError` with clear messaging. Include the
  decorator type, the callable name, and a formatted argument dump to ease
  debugging.

## Implementation Sketch

1. Add `src/weakincentives/dbc/__init__.py` exposing the four decorators and a
   `dbc_active()` helper.
2. Implement `require`, `ensure`, and `pure` as decorator factories that consult
   the activation flag and wrap the target with validation logic only when
   active.
3. For `invariant`, create a class decorator that proxies attribute access and
   wraps relevant callables via a metaclass or `__init_subclass__` hook.
4. Add `tests/test_dbc_contracts.py` exercising happy paths and failure modes.
5. Ship a pytest plugin (`tests/conftest.py` or `src/weakincentives/testing/dbc.py`)
   that flips the activation flag and provides helpful fixtures.
6. Document opt-in knobs for other environments (e.g., `WEAKINCENTIVES_DBC=1`
   environment variable) in `README.md` if needed later.

## Example Use Cases

### Prompt Template Builders

Prompt assembly helpers often assume that template variables exist and follow a
specific structure. Decorating builders with `@require(lambda ctx: "user" in ctx)`
prevents regressions that drop required keys before reaching the LLM adapter.
`@ensure` can assert that generated prompts stay within a token budget or embed
necessary metadata (e.g., `"safety": "high"`).

### Session Reducers

Reducers that mutate session state should maintain invariants like
"history length never decreases". Marking reducers with `@invariant` ensures
that state transitions remain monotonic and that refactors do not accidentally
trim history buffers or break thread-safety assumptions.

### Tool Result Serialization

Serialization helpers can claim purity using `@pure`, catching accidental file
writes or logging noise. Preconditions can verify supported schema versions,
while postconditions confirm the resulting JSON includes expected fields.

### Financial or Quota Counters

Even though the project does not handle real money, internal counters (e.g.,
model usage budgets) can require non-negative balances. `@require` guards
against invalid debit attempts, and `@invariant` confirms counter objects never
fall below zero after adjustments.

## Testing Strategy

- Unit tests should cover successful and failing contracts for each decorator.
- Integration tests verify the pytest plugin toggles enforcement correctly and
  that runtime execution without pytest remains unaffected.
- Snapshot or log-based tests ensure diagnostic messages are actionable.
- Include regression tests for decorator stacking, inheritance with invariants,
  and async compatibility where relevant.

## Future Extensions

- Add async-aware variants (`async_require`, etc.) if adoption grows.
- Provide context managers for temporarily disabling contract checks during
  specific test flows (e.g., fuzzing that intentionally violates contracts).
- Integrate with static analyzers by emitting metadata describing declared
  preconditions and postconditions.

