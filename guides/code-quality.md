# Approach to Code Quality

WINK applies strict quality gates that go beyond typical Python projects. These
gates exist because agent code has unusual failure modes: type mismatches
surface mid-conversation, subtle bugs can cause cascading failures across tool
calls, and security vulnerabilities in tool handlers can have serious
consequences.

The gates aren't bureaucracy—they're aligned with the "weak incentives"
philosophy. Just as we design prompts to make correct model behavior natural, we
design the codebase to make correct code natural.

## Strict Type Checking

WINK enforces pyright strict mode. Type annotations are the source of truth:

```python nocheck
# Pyright catches this at edit time, not runtime
def handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    return ToolResult(message="ok", value=None)  # Error: expected MyResult
```

**Why this matters for agents:**

- Tool params and results are serialized/deserialized automatically. Type
  mismatches that would cause runtime failures are caught at construction.
- Session slices are keyed by type. A typo in a type annotation silently creates
  a separate slice.
- Adapters use type information to generate JSON schemas. Wrong types mean wrong
  schemas sent to the model.

**Practical implications:**

- Every function has type annotations
- Use `slots=True, frozen=True` dataclasses for immutable data
- Avoid `Any` except where truly necessary
- Run `make typecheck` frequently

## Design-by-Contract

Public APIs use decorators from `weakincentives.dbc`:

```python nocheck
from weakincentives.dbc import require, ensure, invariant

@require(lambda x: x > 0)  # x must be positive
@ensure(lambda result: result >= 0)  # result must be non-negative
def compute(x: int) -> int:
    ...
```

**What the decorators do:**

- `@require`: precondition checked on entry
- `@ensure`: postcondition checked on exit
- `@invariant`: class invariant checked after each method

**Why this matters for agents:**

- Contracts document expectations that types can't express ("non-empty list",
  "valid path", "positive budget")
- Violations fail fast with clear messages, not silently corrupted state

**When to use contracts:**

- Public API boundaries
- Tool handlers (validate params beyond type checking)
- Reducers (invariants on state transitions)
- Anywhere a comment would say "assumes X" or "requires Y"

Read `specs/DBC.md` before modifying DbC-decorated modules.

## Coverage Requirements

WINK requires 100% line coverage for `src/weakincentives/`. This is enforced by
pytest-cov in CI and blocks merges if coverage drops.

```bash
make test           # Coverage-gated unit tests
```

The 100% requirement isn't about the number. It's about the habit: every line of
code should have a reason to exist, and that reason should be testable. If a
line can't be tested, it probably shouldn't exist.

## Security Scanning

Agent code often handles untrusted input (user requests, model outputs) and
performs privileged operations (file access, command execution). Security
scanning is not optional.

**Tools:**

- **Bandit**: static analysis for common Python security issues
- **Deptry**: finds unused, missing, or misplaced dependencies
- **pip-audit**: checks dependencies for known vulnerabilities

**These run automatically in CI.** You can also run them locally:

```bash
make bandit      # Security analysis
make deptry      # Dependency hygiene
make pip-audit   # Vulnerability scan
```

**Security considerations for tool handlers:**

- Never pass unsanitized model output to shell commands
- Validate file paths against allowed roots
- Use sandboxed adapters for file operations when possible
- Avoid pickle, eval, or exec on untrusted data

## Custom Checkers

Beyond standard linting and type checking, WINK runs project-specific checkers
as part of `make check`. These enforce architectural and import conventions that
generic tools cannot express.

### Banned Time Imports

**Checker:** `BannedTimeImportsChecker` in `toolchain/checkers/banned_time_imports.py`

Production code must not use `import time` or `from time import ...` directly.
All time-dependent code must go through the clock protocols (`MonotonicClock`,
`WallClock`, `Sleeper`) from `weakincentives.clock`. The only exception is
`clock.py` itself, which wraps the stdlib `time` module behind the protocol
abstraction.

```python nocheck
# Bad: direct time usage
import time
elapsed = time.monotonic()

# Good: inject clock protocol
from weakincentives.clock import MonotonicClock, SYSTEM_CLOCK

def measure(clock: MonotonicClock = SYSTEM_CLOCK) -> float:
    return clock.monotonic()
```

This enables deterministic testing via `FakeClock` (see `specs/TESTING.md`).

### Private Import Checker

**Checker:** `PrivateImportChecker` in `toolchain/checkers/private_imports.py`

Modules with a `_` prefix (e.g., `_scope.py`, `_internal/`) are private to
their owning package. Importing them from outside that package is an error.

For example, `weakincentives.serde._scope` is owned by `weakincentives.serde`.
Only modules under `weakincentives.serde` may import from it. Code outside that
package must use the public API instead.

### Architecture Checker (4-Layer Model)

**Checker:** `ArchitectureChecker` in `toolchain/checkers/architecture.py`

Enforces a four-layer module boundary model:

| Layer | Name | Packages |
|-------|------|----------|
| 4 | High-level | `contrib`, `evals`, `cli`, `docs` |
| 3 | Adapters | `adapters` |
| 2 | Core | `runtime`, `prompt`, `resources`, `filesystem`, `serde`, `skills`, `formal`, `debug`, `optimizers` |
| 1 | Foundation | `types`, `errors`, `dataclasses`, `dbc`, `deadlines`, `budget`, `clock`, `experiment` |

A module in layer *N* may import from layers 1..*N* (same or lower). Imports
inside `if TYPE_CHECKING:` are always allowed. See `specs/MODULE_BOUNDARIES.md`.

### Pydantic Ban (TID251)

Ruff rule `TID251` bans direct pydantic imports. All serialization must use
`weakincentives.serde` instead. This keeps the codebase free of pydantic's
runtime overhead and ensures consistent serialization behavior across the
project.

## Quality Gates in Practice

All gates are combined in `make check`:

```bash
make check  # Runs: format, lint, typecheck, test, bandit, deptry, pip-audit
```

**Before every commit:**

1. Run `make check`
1. Fix any failures
1. Commit only when clean

**Pre-commit hooks enforce this.** After running `./install-hooks.sh`, commits
are blocked unless `make check` passes.

## Exhaustiveness Checking

*Canonical spec: [specs/DBC.md](../specs/DBC.md)*

When you have a union type like `SliceOp = Append | Extend | Replace | Clear`,
you need to handle all cases. If someone adds a new variant later, code that
doesn't handle it should fail loudly, not silently skip.

WINK uses the `assert_never` pattern:

```python nocheck
from typing import assert_never
from weakincentives.runtime.session import SliceOp, Append, Extend, Replace, Clear

def apply_op(op: SliceOp) -> None:
    match op:
        case Append(value):
            # handle append
            pass
        case Extend(values):
            # handle extend
            pass
        case Replace(values):
            # handle replace
            pass
        case Clear():
            # handle clear
            pass
        case _:
            assert_never(op)  # Type error if any case is missing
```

**Why this matters:**

- Adding a new `SliceOp` variant is a breaking change. Code that doesn't handle
  it will fail at the type level (pyright) or at runtime (`assert_never` raises).
- Without exhaustiveness checking, new variants get silently ignored.
- The pattern makes the code self-documenting: you can see all cases in one place.

**When to use `assert_never`:**

- Match statements over union types (like `SliceOp`, `DataEvent`)
- Any branching logic where new variants might be added later

**Coverage note:** The `assert_never(op)` line is unreachable in correct code.
Add `# pragma: no cover` to exclude it from coverage requirements:

```text
case _:
    assert_never(op)  # pragma: no cover
```

## Why the Gates Are Strict

Agent systems have compounding failure modes. A type mismatch in a tool param
causes a serialization error, which causes a tool failure, which causes the
model to retry with bad assumptions, which causes a cascade of wasted tokens and
incorrect behavior.

Catching errors early—at the type level, at the contract level, at the test
level—prevents these cascades. The strictness isn't bureaucracy; it's
protection.

## Next Steps

- [Testing](testing.md): Write effective tests
- [Formal Verification](formal-verification.md): TLA+ for critical code
- [Tools](tools.md): Security considerations for tool handlers
