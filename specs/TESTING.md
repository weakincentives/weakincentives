# Testing Specification

Testing standards and coverage requirements.

**Source:** `pyproject.toml`, `tests/`

## Coverage Requirements

**100% line and branch coverage strictly enforced.**

```toml
[tool.coverage.run]
branch = true
fail_under = 100
```

### Rules

- No `pragma: no branch` - simplify code instead
- Test all conditionals (`if`, `elif`, `else`, ternary)
- Trust type annotations over defensive branches

### Excluded Patterns

- `pragma: no cover` - Genuine impossibilities
- `if TYPE_CHECKING:` - Type-only imports
- `@overload`, `@abstractmethod`, `...` stubs

## Regression Test Policy

Every bug fix requires a test:

```python
def test_regression_42_session_deadlock():
    """
    Regression for #42: Session deadlocks on concurrent broadcast.
    https://github.com/org/weakincentives/issues/42
    """
```

- Test MUST fail before fix, pass after
- Name: `test_regression_<issue>_<description>`
- Docstring links to issue

## CI Pipeline

### `make check` Sequence

```
format-check → lint → typecheck → bandit → vulture → deptry → pip-audit → markdown-check → test
```

### Gate Thresholds

| Gate | Threshold |
|------|-----------|
| Line/branch coverage | 100% |
| Type errors | 0 |
| Lint errors | 0 |
| Security issues | 0 high/critical |
| Mutation (CI only) | 85-90% |

## Test Organization

```
tests/
├── unit/           # Fast isolated tests
├── property/       # Hypothesis tests
├── regression/     # Bug reproduction
├── helpers/        # Shared fixtures
└── plugins/        # pytest plugins (dbc, threadstress)
```
