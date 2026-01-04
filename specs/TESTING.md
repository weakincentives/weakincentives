# TESTING.md

Testing standards for weakincentives.

## Coverage Requirements

### Line + Branch Coverage

**100% line and branch coverage is strictly enforced.** Every conditional branch
must be tested - no exceptions.

```toml
# pyproject.toml
[tool.coverage.run]
branch = true
source = ["src/weakincentives"]

[tool.coverage.report]
fail_under = 100
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "@overload",
    "@abstractmethod",
    "^\\s*\\.\\.\\.\\s*$",
    ":\\s*\\.\\.\\.\\s*$",
]
```

#### Strict Requirements

1. **No `pragma: no branch`**: Do not use branch exclusion pragmas. If a branch
   cannot be tested, simplify the code to eliminate the branch.

1. **Test all conditionals**: Every `if`, `elif`, `else`, `for`, `while`, and
   ternary expression must have both branches exercised by tests.

1. **Simplify over exclude**: If a branch is defensive code that "should never
   happen", either:

   - Remove it (trust type annotations and upstream validation)
   - Add a test that proves the branch can be reached
   - Simplify the code to eliminate the branch

#### Excluded Patterns

The following patterns are automatically excluded from coverage:

- `pragma: no cover` - For genuine impossibilities (e.g., `assert False`)
- `if TYPE_CHECKING:` - Type-only imports
- `@overload` - Typing overloads
- `@abstractmethod` - Abstract method declarations
- `...` - Protocol/abstract method stubs (both inline and standalone)

### Mutation Testing

Mutation testing protects correctness-critical modules. Expand scope beyond current hotspots:

| Module | Minimum Score | Rationale |
| ---------------------- | ------------- | ----------------------------------------- |
| `runtime/session/*.py` | 90% | State management core |
| `serde/*.py` | 85% | Snapshot integrity depends on round-trips |
| `dbc/decorators.py` | 85% | Contract enforcement must be watertight |

```toml
# pyproject.toml
[tool.mutmut]
paths_to_mutate = [
    "src/weakincentives/runtime/session/",
    "src/weakincentives/serde/",
    "src/weakincentives/dbc/decorators.py",
]
```

## Regression Test Policy

Every bug fix requires a regression test:

1. Test MUST fail before the fix, pass after
1. Test MUST be named `test_regression_<issue>_<description>`
1. Test MUST include docstring linking to the issue

```python
# tests/regression/test_issue_42.py
def test_regression_42_session_deadlock():
    """
    Regression for #42: Session deadlocks on concurrent broadcast.
    https://github.com/org/weakincentives/issues/42
    """
    # Reproduce exact conditions from bug report
    ...
```

## Snapshot Integrity Tests

Snapshots are critical for rollback correctness. Test:

1. **Round-trip integrity**: `snapshot() → rollback() → snapshot()` produces identical state
1. **Corruption detection**: Tampered snapshots raise `SnapshotCorruptionError`
1. **Partial rollback**: Interrupted rollback leaves session in recoverable state

```python
def test_snapshot_roundtrip_integrity(session_factory):
    """Snapshot and rollback preserve exact state."""
    session, _ = session_factory()
    session.dispatch(SomeEvent(...))

    snapshot1 = session.snapshot()
    session.dispatch(AnotherEvent(...))
    session.restore(snapshot1)
    snapshot2 = session.snapshot()

    assert snapshot1 == snapshot2
```

## CI Pipeline Integration

### Makefile Targets

```makefile
# Fast checks (pre-commit, local dev)
check: format-check lint typecheck bandit vulture deptry pip-audit markdown-check test

# Slow checks (CI only)
mutation-check:
	uv run python build/run_mutmut.py --enforce-gates
```

### `make check` Sequence

Fast gates run in `make check`. Failures block commits via pre-commit hook:

```
format-check    → Code style (ruff format --check)
lint            → Static analysis (ruff --preview)
typecheck       → Type correctness (pyright strict + ty)
bandit          → Security scanning
vulture         → Dead code detection
deptry          → Dependency hygiene
pip-audit       → Vulnerability scanning
markdown-check  → Doc formatting
test            → 100% line+branch coverage
```

### CI-Only Gates

`mutation-check` runs separately in CI (too slow for local dev):

```yaml
# .github/workflows/ci.yml
jobs:
  check:
    steps:
      - run: make check
  mutation:
    steps:
      - run: make mutation-check
```

### Gate Enforcement

| Gate | Threshold | Where |
| ------------------ | --------------- | ------------ |
| Line coverage | 100% | `make check` |
| Branch coverage | 100% | `make check` |
| Type errors | 0 | `make check` |
| Lint errors | 0 | `make check` |
| Security issues | 0 high/critical | `make check` |
| Mutation (session) | 90% | CI only |
| Mutation (serde) | 85% | CI only |
| Mutation (dbc) | 85% | CI only |

## Test Organization

```
tests/
├── unit/           # Fast isolated tests (existing)
├── property/       # Hypothesis tests (existing, expand)
├── regression/     # Bug reproduction tests (new)
├── helpers/        # Shared fixtures
└── plugins/        # pytest plugins
```

No separate fault injection, fuzz, or stress directories needed - these concerns fold into existing structure:

- **Fuzz testing**: Already covered by Hypothesis in `tests/property/`
- **Stress testing**: Already covered by threadstress plugin
- **Fault injection**: Add specific fixtures to `tests/helpers/` as needed

## Implementation

### Step 1: Enable Branch Coverage (COMPLETED)

Branch coverage is now enabled in `pyproject.toml`:

```toml
[tool.coverage.run]
branch = true
source = ["src/weakincentives"]

[tool.coverage.report]
fail_under = 100
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "@overload",
    "@abstractmethod",
    "^\\s*\\.\\.\\.\\s*$",
    ":\\s*\\.\\.\\.\\s*$",
]
```

Both line and branch coverage are enforced at 100%. Tests will fail if any
branch is not covered.

### Step 2: Expand Mutation Scope

```diff
# pyproject.toml
[tool.mutmut]
paths_to_mutate = [
-    "src/weakincentives/runtime/session/reducers.py",
-    "src/weakincentives/runtime/session/session.py",
+    "src/weakincentives/runtime/session/",
+    "src/weakincentives/serde/",
+    "src/weakincentives/dbc/decorators.py",
]
```

Add score gates to `build/run_mutmut.py`:

```python
SCORE_GATES = {
    "runtime/session/": 90,
    "serde/": 85,
    "dbc/decorators.py": 85,
}
```

### Step 3: Create Regression Test Directory

```bash
mkdir -p tests/regression
touch tests/regression/__init__.py
```

Add to CONTRIBUTING.md: "Every bug fix requires a regression test in `tests/regression/`."

## What This Spec Does NOT Include

Explicitly excluded as over-engineering for this library:

- OOM simulation (Python's GC handles memory; not a database)
- I/O error injection (adapters handle their own transport errors)
- Crash recovery testing beyond snapshot integrity
- Separate fuzz corpus management (Hypothesis handles this)
- Release checklists (CI gates are sufficient)
- Test-to-code ratio tracking (100% coverage + mutation is enough)
