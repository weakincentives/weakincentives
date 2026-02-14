# Testing Specification

Testing standards for weakincentives.

## Coverage Requirements

### Line + Branch Coverage

**100% line and branch coverage strictly enforced.** Every conditional branch
must be tested.

#### Strict Requirements

1. **Minimize `pragma: no branch`**: Prefer simplifying code; use sparingly for
   framework boundary code where branches depend on external state
1. **Test all conditionals**: `if`, `elif`, `else`, `for`, `while`, ternary
1. **Simplify over exclude**: Remove defensive code or prove branch reachable

#### Excluded Patterns

- `pragma: no cover` - Genuine impossibilities
- `if TYPE_CHECKING:` - Type-only imports
- `@overload` - Typing overloads
- `@abstractmethod` - Abstract declarations
- `...` - Protocol stubs

## Regression Test Policy

Every bug fix requires a regression test:

1. MUST fail before fix, pass after
1. MUST be named `test_regression_<issue>_<description>`
1. MUST include docstring linking to issue

## Snapshot Integrity Tests

Test critical rollback correctness:

1. **Round-trip**: `snapshot() → rollback() → snapshot()` identical
1. **Corruption detection**: Tampered snapshots raise error
1. **Partial rollback**: Interrupted leaves recoverable state

## CI Pipeline

### `make check` Sequence

```
format-check    → ruff format --check
lint            → ruff --preview
typecheck       → pyright strict + ty
bandit          → Security scanning
deptry          → Dependency hygiene
pip-audit       → Vulnerability scanning
architecture    → Core/contrib separation
code-length     → Function/file length limits
docs            → Documentation verification
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

## Test Organization

Tests mirror the `src/weakincentives/` package structure:

```
tests/
├── adapters/       # Adapter tests (claude_agent_sdk, codex_app_server, acp, etc.)
├── cli/            # CLI command tests
├── contrib/        # Contrib module tests
├── debug/          # Debug bundle tests
├── evals/          # Evaluation framework tests
├── filesystem/     # Filesystem abstraction tests
├── prompt/         # Prompt system tests
├── runtime/        # Runtime (session, agent_loop, lifecycle) tests
├── serde/          # Serialization tests
├── skills/         # Skill tests
├── helpers/        # Shared fixtures
├── plugins/        # pytest plugins
└── toolchain/      # Verification toolchain tests
```

Integration tests live in a separate `integration-tests/` directory with
their own timeout defaults (120s per test via `ACK_TIMEOUT` environment
variable).

## What This Spec Does NOT Include

Explicitly excluded as over-engineering:

- OOM simulation
- I/O error injection
- Crash recovery beyond snapshot integrity
- Separate fuzz corpus management
- Release checklists
- Test-to-code ratio tracking
