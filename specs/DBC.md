# Design by Contract Specification

## Purpose

This document specifies a tiny, internal-only design-by-contract (DbC) helper
framework for `weakincentives`. The framework exposes annotation-style
decorators that let library authors describe preconditions, postconditions,
invariants, and purity expectations on Python callables. The decorators are
intentionally inert during normal runtime; they activate inside the test suite
and other explicitly opted-in flows so regressions surface early without
shipping runtime overhead. The DbC helpers are a mini-framework for maintainers
and contributors—they are **not** part of the public API and must not be
exported or promoted to external consumers.

## Guiding Principles

- **Internal safety net, not a public contract**: DbC reinforces contributor
  discipline without promising compatibility to external users. Do not export
  `weakincentives.dbc` outside the package-level namespace or reference it in
  public docs.
- **Zero-cost default**: The decorators should be no-ops unless explicitly
  enabled. Tests toggle enforcement; production paths must remain unaffected.
- **Pragmatic coverage over total verification**: Enforcement focuses on common
  footguns (argument validation, invariants, shallow purity checks) rather than
  exhaustive program analysis.
- **Clear diagnostics**: Failures should name the decorator, the callable, and
  the offending predicate to reduce debugging time.
- **Composable opt-ins**: Helpers like `dbc_enabled()` and environment flags let
  developers scope DbC to specific runs without global side effects.

## Goals

- Provide four decorators, `@require`, `@ensure`, `@invariant`, and `@pure`, that
  look and feel like standard function decorators and remain internal-facing.
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
  hook (`WEAKINCENTIVES_DBC`, `dbc_enabled()`, or `enable_dbc()`).
- The scope is limited to synchronous Python callables. Asynchronous or
  generator-specific helpers can be added later if needed.

## Contract Decorators

### `@require`

`@require` expresses preconditions. It accepts one or more callables that take
`*args`, `**kwargs`, and optionally the bound `self` or `cls`. The contract
passes when all callables return truthy values. If any callable returns falsy or
raises an exception, pytest fails with a descriptive assertion. The decorator is
implemented in `src/weakincentives/dbc/__init__.py` and raises immediately if no
predicates are supplied, ensuring internal callers opt into meaningful guards.

```python
@require(lambda amount: amount >= 0, lambda account, amount: account.can_withdraw(amount))
def withdraw(account: Account, amount: int) -> None:
    ...
```

### `@ensure`

`@ensure` declares postconditions. Callables receive the original arguments and
keyword arguments plus the return value (or raised exception object) via the
`result` or `exception` keyword. Decorators support both simple boolean
functions and more advanced callables that return `(bool, message)` tuples for
custom diagnostics. When DbC is inactive, `@ensure` is a no-op; when active, it
evaluates predicates whether the wrapped callable returns or raises, passing the
exception object through for inspection.

```python
@ensure(lambda amount, result: result.balance >= 0)
def withdraw(account: Account, amount: int) -> Account:
    ...
```

### `@invariant`

`@invariant` targets class definitions. When applied, it wraps `__init__` and
public methods to validate the invariant callable(s) before and after method
execution. Invariants run in tests whenever the instance mutates observable
state. The decorator exposes `skip_invariant()` to mark helper methods that
should not trigger checks. Methods beginning with `_`, static methods, class
methods, and non-callable attributes are intentionally excluded from wrapping
to avoid altering private or meta-level behavior.

```python
@invariant(lambda self: self.balance >= 0)
class Account:
    def deposit(self, amount: int) -> None:
        self.balance += amount
```

### `@pure`

`@pure` documents that a function is side-effect free. During tests, the
framework wraps the target callable and instruments it to ensure:

- It does not mutate positional or keyword arguments (best-effort via
  `deepcopy()` + equality comparisons; non-copyable values are skipped).
- It does not perform a small set of common side effects by calling
  `builtins.open`, `Path.write_text`, `Path.write_bytes`, or `logging` (enforced
  by monkeypatching these call sites to raise `AssertionError`).

Instrumentation monkeypatches common side-effect primitives (`open`,
`Path.write_text`, `Path.write_bytes`, `logging.Logger._log`) to raise when
invoked from a pure function context. The focus is to catch accidental side
effects in helper utilities, not to provide complete effect tracking. Callers
should keep purity contracts narrow and local to internal helpers rather than
external entry points.

## Runtime Behavior

- In production (`WEAKINCENTIVES_DBC=0` by default) the decorators return the
  original callable untouched. The shared `dbc_active()` flag gates every
  enforcement path so internal code can safely import decorators without
  impacting runtime.
- During tests, a pytest plugin activates the flag. The plugin hooks into
  `pytest_configure` and `pytest_unconfigure` to toggle DbC checks.
- Contract violations raise `AssertionError` with clear messaging. Include the
  decorator type, the callable name, and a formatted argument dump to ease
  debugging.
- Manual overrides exist for focused debugging or local experimentation:
  `enable_dbc()`, `disable_dbc()`, and the `dbc_enabled()` context manager force
  the flag on/off inside a scope, while `WEAKINCENTIVES_DBC=1` flips enforcement
  globally. These switches are for maintainers only and should not be surfaced
  to users.

## Implementation Sketch

1. `src/weakincentives/dbc/__init__.py` exposes the four decorators, enforcement
   helpers (`dbc_active`, `enable_dbc`, `disable_dbc`, `dbc_enabled`), and
   shared predicate handling (normalizing `(bool, message)` tuples, formatting
   failures).
1. `require`, `ensure`, and `pure` are decorator factories that consult
   `dbc_active()` and only wrap the target with validation logic when active;
   otherwise they return the original callable untouched.
1. `invariant` wraps public instance methods and `__init__` only. The wrapper
   checks invariants before and after calls when active and respects
   `skip_invariant()` markers to avoid redundant checks.
1. `tests/plugins/dbc.py` toggles the flag during pytest runs and integrates
   diagnostics for contract failures.
1. Additional opt-in knobs (`WEAKINCENTIVES_DBC=1` or the context manager
   helpers) can enable enforcement in ad hoc environments, but they remain
   internal debugging tools rather than user-facing configuration.

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
