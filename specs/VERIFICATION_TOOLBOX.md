# Verification Toolchain Specification

A minimal, extensible verification framework for running checks on the codebase.

**Implementation:**

- Framework: `toolchain/` (excluded from package)
- Checkers: `toolchain/checkers/`
- Entry point: `check.py`
- Makefile targets: `Makefile`
- Pre-commit hook: `hooks/pre-commit`
- Hook installer: `install-hooks.sh`

## Overview

The verification toolchain provides a single entry point (`check.py`) for running
all verification checks. It is designed for:

- **Immediate debugging** - Failures include file:line locations you can click
- **Extensibility** - Add checkers by implementing a simple protocol
- **Concise output** - Quiet on success, detailed on failure
- **Dual-mode operation** - Auto-fixes locally, check-only in CI
- **Not packaged** - Development tooling only, lives outside `src/`

## Architecture

```
check.py                  # Entry point at repository root
toolchain/                # Framework (excluded from package)
├── __init__.py           # Public exports
├── result.py             # Location, Diagnostic, CheckResult, Report
├── checker.py            # Checker protocol, SubprocessChecker base
├── runner.py             # Orchestrates checker execution
├── output.py             # Formatters (Console, JSON, Quiet)
├── parsers.py            # Tool output parsers
├── utils.py              # Git, AST, markdown utilities
└── checkers/             # Checker implementations
    ├── __init__.py       # Factory functions for all checkers
    ├── architecture.py   # Four-layer module boundary model
    ├── private_imports.py # Cross-package private module import check
    ├── banned_time_imports.py # Direct time module usage ban
    ├── code_length.py    # Function/file length limits
    ├── code_length_baseline.txt # Baseline for grandfathered lengths
    └── docs.py           # Documentation verification
```

## Result Types

See `toolchain/result.py` for the full dataclass definitions.

| Type | Fields | Description |
|------|--------|-------------|
| `Location` | `file`, `line`, `column` | Clickable `file:line:col` format |
| `Diagnostic` | `message`, `location`, `severity` | Single issue with context |
| `CheckResult` | `name`, `status`, `duration_ms`, `diagnostics`, `output` | Result from one checker |
| `Report` | `results`, `total_duration_ms` | Aggregated results from all checkers |

Severity levels: `"error"`, `"warning"`, `"info"`.
Status values: `"passed"`, `"failed"`, `"skipped"`.

## Checker Protocol

See `toolchain/checker.py` for the protocol and base implementations.

Every checker implements the `Checker` protocol: `name` (property), `description`
(property), `run() -> CheckResult`.

### SubprocessChecker

Base class for checkers that wrap external tools. Runs a command, captures output,
invokes a parser to extract diagnostics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Checker identifier |
| `description` | `str` | (required) | Human-readable description |
| `command` | `list[str]` | (required) | Command to run |
| `parser` | `DiagnosticParser` | `_no_parse` | Extracts diagnostics from output |
| `timeout` | `int` | `300` | Timeout in seconds |
| `env` | `dict[str, str]` | `{}` | Extra environment variables |

### AutoFormatChecker

Dual-mode checker for formatting tools. Detects CI vs local environment via
`is_ci_environment()`.

| Environment | Behavior |
|-------------|----------|
| **CI** (`CI=true` or `GITHUB_ACTIONS=true`) | Runs check command only; fails if changes needed |
| **Local** | Runs formatter to auto-fix; reports changed files as info diagnostics |

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Checker identifier |
| `description` | `str` | (required) | Human-readable description |
| `check_command` | `list[str]` | (required) | Command for check-only mode |
| `fix_command` | `list[str]` | (required) | Command for auto-fix mode |
| `json_check_command` | `list[str] \| None` | `None` | JSON output variant for file list |
| `file_list_parser` | `FileListParser` | `_no_file_list_parse` | Text-based file list parser |
| `parser` | `DiagnosticParser` | `_no_parse` | Diagnostic parser |
| `timeout` | `int` | `300` | Timeout in seconds |

Used by the `format` checker (ruff format) and `markdown` checker (mdformat).

## Available Checkers

See `toolchain/checkers/__init__.py` for factory functions and execution order.

| Checker | Type | Description | What It Checks |
|---------|------|-------------|----------------|
| `format` | `AutoFormatChecker` | Code formatting | `ruff format` (auto-fix locally, check in CI) |
| `lint` | `SubprocessChecker` | Code style | `ruff check --preview` |
| `typecheck` | `SubprocessChecker` | Type safety | `ty check src && pyright` (diagnostics prefixed with tool name) |
| `bandit` | `SubprocessChecker` | Security | Bandit security scanner |
| `deptry` | `SubprocessChecker` | Dependencies | Unused/missing dependencies |
| `pip-audit` | `SubprocessChecker` | Vulnerabilities | Known CVEs in dependencies |
| `architecture` | `ArchitectureChecker` | Code structure | Four-layer module boundary model (Foundation → Core → Adapters → High-level) |
| `private-imports` | `PrivateImportChecker` | Import boundaries | Cross-package `_`-prefixed module import check |
| `banned-time-imports` | `BannedTimeImportsChecker` | Import hygiene | Direct `import time` / `from time import` ban (`clock.py` exempt) |
| `code-length` | `CodeLengthChecker` | Code size | Function/method and file length limits with baseline |
| `docs` | `DocsChecker` | Documentation | Examples, links, references |
| `markdown` | `AutoFormatChecker` | Markdown format | `mdformat` (auto-fix locally, check in CI) |
| `bun-test` | `SubprocessChecker` | JavaScript tests | `bun test --coverage` (skips if bun not installed) |
| `test` | `SubprocessChecker` | Python tests | pytest with 100% coverage, 10s timeout per test |

## Failure Reporting

The toolchain is designed to make debugging immediate. Every failure includes:

1. **Checker name** - Which check failed
1. **Duration** - How long it took
1. **Diagnostics** - Structured issues with locations
1. **File:line** - Clickable in most terminals

### Example Output

**All passing:**

```
✓ format               (1.2s)
✓ lint                 (2.3s)
✓ typecheck            (12.5s)
✓ test                 (45.2s)

✓ All checks passed (61.2s)
```

**With failures:**

```
✓ format               (1.2s)
✗ lint                 (2.3s)
  src/foo.py:42:10: [E501] Line too long (120 > 88)
  src/bar.py:17:1: [F401] 'os' imported but unused
  src/baz.py:8:5: [E721] Do not compare types, use isinstance()
✓ typecheck            (12.5s)
✗ test                 (45.2s)
  tests/test_foo.py: Test failed: test_addition
  tests/test_bar.py: Test failed: test_serialization
  Coverage below threshold: 98.5%

✗ 2 failed, 2 passed (61.2s)
```

### Diagnostic Format

Each diagnostic follows the pattern:

```
<location>: <message>
```

Where location is one of:

- `file:line:col` - Full position (e.g., `src/foo.py:42:10`)
- `file:line` - Line only (e.g., `src/foo.py:42`)
- `file` - File only (e.g., `README.md`)

This format is recognized by most terminals and IDEs, enabling click-to-navigate.

### Test Failures

Tests run to completion, reporting ALL failures (not just the first). This ensures
you see the full scope of issues in a single run:

```
✗ test                 (45.2s)
  tests/test_session.py: Test failed: test_dispatch_event
  tests/test_session.py: Test failed: test_restore_snapshot
  tests/test_prompt.py: Test failed: test_render_with_tools
  tests/test_adapter.py: Test failed: test_structured_output
```

### Verbose Mode

Use `-v` to see full tool output for failed checks:

```bash
python check.py -v
```

This shows the complete stdout/stderr from the underlying tool, useful when
the parsed diagnostics don't provide enough context.

## CLI Usage

**Note**: Use `uv run` to ensure Python 3.12+ (required for PEP 695 syntax).

```bash
# Run all checks
uv run python check.py

# Run specific checks
uv run python check.py lint test

# List available checks
uv run python check.py --list

# Verbose output (full tool output on failure)
uv run python check.py -v

# Quiet mode (only show failures)
uv run python check.py -q

# JSON output for CI integration
uv run python check.py --json

# Disable colors
uv run python check.py --no-color
```

Or use the Makefile which handles this automatically:

```bash
make check    # Run all checks
make lint     # Just lint
make test     # Just tests
make bun-test # Just JavaScript tests
```

## Efficient Testing with pytest-testmon

`make check` and `make test` automatically detect local vs CI execution:

| Environment | Behavior |
|-------------|----------|
| **CI** (`CI=true`) | Full test suite with 100% coverage enforcement |
| **Local** | Only tests affected by changes (uses testmon coverage database) |

The first local run builds a coverage database (`.testmondata`). Subsequent
runs use this database to identify which tests cover changed code and skip
the rest. This dramatically reduces iteration time when working on focused
changes.

The test checker in CI mode uses:

```bash
pytest --strict-config --strict-markers --cov-fail-under=100 \
  --timeout=10 --timeout-method=thread --tb=short tests
```

Local mode uses:

```bash
pytest -p no:cov -o addopts= --testmon --strict-config --strict-markers \
  --timeout=10 --timeout-method=thread --tb=short --reruns=2 tests
```

See `Makefile` for the `test` target implementation.

## Pre-commit Hooks

Git hooks enforce the full CI test suite before every commit, preventing
commits that would fail in CI.

**Installation:** `./install-hooks.sh` (mandatory after cloning)

The pre-commit hook runs `CI=true make check`, which:

1. Runs the **full** test suite (not the testmon subset)
1. Enforces 100% coverage on all code paths
1. Runs all linting, type checking, security scanning, etc.
1. Exactly emulates CI verification

See `hooks/pre-commit` for the hook implementation and `install-hooks.sh` for
the installer.

## Bun Test Integration

JavaScript tests are integrated into the unified toolchain via the `bun-test`
checker. The checker gracefully handles environments without bun installed by
exiting 0 with a skip message.

```bash
make bun-test                    # Run via Makefile
uv run python check.py bun-test # Run via toolchain
```

Coverage is enabled via `--coverage`. The `parse_bun_test` parser in
`toolchain/parsers.py` extracts diagnostics from bun test output.

## Adding a New Checker

1. **For subprocess-based checks**, add a factory function in
   `toolchain/checkers/__init__.py` returning `SubprocessChecker` or
   `AutoFormatChecker`

1. **For custom logic**, create a checker class in `toolchain/checkers/`
   implementing the `Checker` protocol (`name`, `description`, `run()`)

1. **Add a parser** in `toolchain/parsers.py` if the tool output needs
   structured diagnostic extraction

1. **Register the checker** in `create_all_checkers()` in
   `toolchain/checkers/__init__.py`

See existing checkers for patterns:

- `create_format_checker()` for `AutoFormatChecker` with JSON output parsing
- `create_markdown_checker()` for `AutoFormatChecker` with text file list parsing
- `create_bun_test_checker()` for `SubprocessChecker` with graceful skip
- `ArchitectureChecker` for custom logic checker
- `CodeLengthChecker` for AST-based analysis with baseline support

## Output Formatters

See `toolchain/output.py` for formatter implementations.

### ConsoleFormatter

Default formatter with colored output:

- `✓` Green checkmark for passed
- `✗` Red X for failed
- `○` Yellow circle for skipped
- Diagnostics shown in gray for readability

### JSONFormatter

Machine-readable output for CI:

```json
{
  "passed": false,
  "summary": {
    "passed": 8,
    "failed": 2,
    "skipped": 0,
    "duration_ms": 61234
  },
  "results": [
    {
      "name": "lint",
      "status": "failed",
      "duration_ms": 2345,
      "diagnostics": [
        {
          "message": "[E501] Line too long",
          "location": {"file": "src/foo.py", "line": 42, "column": 10},
          "severity": "error"
        }
      ]
    }
  ]
}
```

### QuietFormatter

Minimal output - only shows failures:

```
✗ lint
  src/foo.py:42:10: [E501] Line too long
✗ test
  tests/test_foo.py: Test failed: test_addition
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |

## Configuration

Checkers use configuration from `pyproject.toml` where applicable:

- `[tool.ruff]` - Ruff linter/formatter settings
- `[tool.pyright]` - Type checker settings
- `[tool.deptry]` - Dependency checker settings
- `[tool.bandit]` - Security scanner settings

The toolchain itself requires no configuration - it discovers and runs all
registered checkers.

## Agent Debugging Guide

This section provides guidance for AI coding agents debugging failures.

### Reading Failure Output

When a check fails, the output provides everything needed to fix the issue:

```
✗ lint                 (2.3s)
  src/foo.py:42:10: [E501] Line too long (120 > 88)
```

From this you know:

1. **Which check**: `lint`
1. **Which file**: `src/foo.py`
1. **Which line**: `42`
1. **What's wrong**: Line exceeds 88 characters

**Action**: Read `src/foo.py:42` and fix the line length.

### Test Failure Workflow

```
✗ test                 (45.2s)
  tests/test_session.py:87: test_dispatch_event: AssertionError: assert 1 == 2
```

1. **Read the test**: `tests/test_session.py:87`
1. **Understand the assertion**: What was expected vs actual?
1. **Read the implementation**: What code is being tested?
1. **Fix the bug** or **fix the test** depending on which is wrong

### Type Error Workflow

```
✗ typecheck            (12.5s)
  src/prompt/tool.py:145:23: Argument of type "str" cannot be assigned to "int"
```

1. **Read the location**: `src/prompt/tool.py:145`
1. **Check the function signature**: What type is expected?
1. **Trace the value**: Where does the wrong-typed value come from?
1. **Fix the type mismatch**

### Running Specific Checks

When debugging, run only the relevant check to save time:

```bash
python check.py lint          # Just lint
python check.py typecheck     # Just types
python check.py test          # Just tests
```

### Verbose Mode for Context

If the diagnostic isn't enough, use verbose mode:

```bash
python check.py test -v
```

This shows the full pytest output including stack traces.

### JSON for Programmatic Access

For structured parsing:

```bash
python check.py --json
```

Returns machine-readable results with all diagnostics.

## Design Principles

1. **Failures are clickable** - Always include file:line locations
1. **Run all, report all** - Don't stop at first failure
1. **Quiet on success** - No noise when things work
1. **Verbose on failure** - Full context for debugging
1. **No emojis by default** - Consistent with production standards
1. **Simple protocol** - Easy to add new checkers
1. **Not packaged** - Development tooling stays out of releases

## Related Specifications

- `specs/VERIFICATION.md` - Redis mailbox formal verification
- `specs/TESTING.md` - Testing conventions and coverage requirements

## Future Additions

Potential enhancements for later:

### Planned

- **`--fail-fast`** - Stop at first failure (useful for quick iteration)
- **`--errors-only`** - Filter to only show errors, not warnings
- **Parallel execution** - Run independent checkers concurrently

### Considered

- **Suggested fixes** - Add `fix: str | None` field to Diagnostic
- **Watch mode** - Re-run on file changes
- **Custom checker plugins** - Load checkers from user config

### Not Planned

- **IDE integration** - Use existing LSP tools instead
- **Historical tracking** - Keep it simple, no database
- **Notifications** - stdout only, external tools can wrap
