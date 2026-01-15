# Testing Specification

Testing standards for weakincentives.

## Coverage Requirements

### Line + Branch Coverage

**100% line and branch coverage strictly enforced.** Every conditional branch
must be tested.

#### Strict Requirements

1. **No `pragma: no branch`**: Simplify code to eliminate untestable branches
2. **Test all conditionals**: `if`, `elif`, `else`, `for`, `while`, ternary
3. **Simplify over exclude**: Remove defensive code or prove branch reachable

#### Excluded Patterns

- `pragma: no cover` - Genuine impossibilities
- `if TYPE_CHECKING:` - Type-only imports
- `@overload` - Typing overloads
- `@abstractmethod` - Abstract declarations
- `...` - Protocol stubs

## Regression Test Policy

Every bug fix requires a regression test:

1. MUST fail before fix, pass after
2. MUST be named `test_regression_<issue>_<description>`
3. MUST include docstring linking to issue

## Snapshot Integrity Tests

Test critical rollback correctness:

1. **Round-trip**: `snapshot() → rollback() → snapshot()` identical
2. **Corruption detection**: Tampered snapshots raise error
3. **Partial rollback**: Interrupted leaves recoverable state

## CI Pipeline

### `make check` Sequence

```
format-check    → ruff format --check
lint            → ruff --preview
typecheck       → pyright strict + ty
bandit          → Security scanning
vulture         → Dead code detection
deptry          → Dependency hygiene
pip-audit       → Vulnerability scanning
markdown-check  → Doc formatting
test            → 100% line+branch coverage
```

### Gate Enforcement

| Gate | Threshold | Where |
|------|-----------|-------|
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
├── unit/           # Fast isolated tests
├── property/       # Hypothesis tests
├── regression/     # Bug reproduction tests
├── helpers/        # Shared fixtures
└── plugins/        # pytest plugins
```

## What This Spec Does NOT Include

Explicitly excluded as over-engineering:

- OOM simulation
- I/O error injection
- Crash recovery beyond snapshot integrity
- Separate fuzz corpus management
- Release checklists
- Test-to-code ratio tracking
