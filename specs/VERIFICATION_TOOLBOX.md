# Verification Toolchain Specification

A minimal, extensible verification framework for running checks on the codebase.

## Overview

The verification toolchain provides a single entry point (`check.py`) for running
all verification checks. It is designed for:

- **Immediate debugging** - Failures include file:line locations you can click
- **Extensibility** - Add checkers by implementing a simple protocol
- **Concise output** - Quiet on success, detailed on failure
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
    ├── architecture.py   # Core/contrib separation
    └── docs.py           # Documentation verification
```

## Result Types

### Location

Pinpoints exactly where an issue occurred:

```python
@dataclass(frozen=True, slots=True)
class Location:
    file: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        # Returns "file:line:col" format for easy navigation
        # Examples: "src/foo.py:42:10", "src/bar.py:17", "README.md"
```

### Diagnostic

A single issue with full context for debugging:

```python
@dataclass(frozen=True, slots=True)
class Diagnostic:
    message: str                              # What went wrong
    location: Location | None = None          # Where (clickable)
    severity: Literal["error", "warning", "info"] = "error"

    def __str__(self) -> str:
        # "src/foo.py:42: Type error: expected int, got str"
```

### CheckResult

Complete result from a single checker:

```python
@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str                                 # Checker name (e.g., "lint")
    status: Literal["passed", "failed", "skipped"]
    duration_ms: int                          # For performance tracking
    diagnostics: tuple[Diagnostic, ...] = ()  # All issues found
    output: str = ""                          # Raw output for verbose mode
```

### Report

Aggregated results from all checkers:

```python
@dataclass(frozen=True, slots=True)
class Report:
    results: tuple[CheckResult, ...]
    total_duration_ms: int

    @property
    def passed(self) -> bool:
        return all(r.status != "failed" for r in self.results)
```

## Checker Protocol

Every checker implements this minimal interface:

```python
class Checker(Protocol):
    @property
    def name(self) -> str:
        """Short identifier (e.g., 'lint', 'test')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def run(self) -> CheckResult:
        """Execute the check and return the result."""
        ...
```

### SubprocessChecker

Base class for checkers that wrap external tools:

```python
@dataclass
class SubprocessChecker:
    name: str
    description: str
    command: list[str]                        # Command to run
    parser: DiagnosticParser = _no_parse      # Extracts diagnostics from output
    timeout: int = 300                        # 5 minutes default
```

The `parser` function extracts structured `Diagnostic` objects from tool output,
enabling clickable file:line locations in the terminal.

## Available Checkers

| Checker | Description | What It Checks |
|---------|-------------|----------------|
| `format` | Code formatting | `ruff format --check` |
| `lint` | Code style | `ruff check --preview` |
| `typecheck` | Type safety | `ty check src && pyright` |
| `bandit` | Security | Bandit security scanner |
| `deptry` | Dependencies | Unused/missing dependencies |
| `pip-audit` | Vulnerabilities | Known CVEs in dependencies |
| `architecture` | Code structure | Core/contrib separation |
| `docs` | Documentation | Examples, links, references |
| `markdown` | Markdown format | `mdformat --check` |
| `test` | Tests | pytest with 100% coverage |

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
```

## Adding a New Checker

1. **For subprocess-based checks**, add a factory function:

```python
# In toolchain/checkers/__init__.py

def create_mycheck_checker() -> SubprocessChecker:
    return SubprocessChecker(
        name="mycheck",
        description="Description of what it checks",
        command=["uv", "run", "mytool", "--args"],
        parser=parse_mytool,  # Optional: extract diagnostics
    )
```

2. **For custom logic**, create a new checker class:

```python
# In toolchain/checkers/mycheck.py

@dataclass
class MyChecker:
    @property
    def name(self) -> str:
        return "mycheck"

    @property
    def description(self) -> str:
        return "Description of what it checks"

    def run(self) -> CheckResult:
        start = time.monotonic()
        diagnostics = []

        # Your verification logic here
        # Append Diagnostic objects for each issue found

        return CheckResult(
            name=self.name,
            status="passed" if not diagnostics else "failed",
            duration_ms=int((time.monotonic() - start) * 1000),
            diagnostics=tuple(diagnostics),
        )
```

3. **Register the checker** in `create_all_checkers()`.

## Output Formatters

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

## Future Additions

Potential enhancements for later:

### Planned

- **`--fail-fast`** - Stop at first failure (useful for quick iteration)
- **`--errors-only`** - Filter to only show errors, not warnings
- **Parallel execution** - Run independent checkers concurrently

### Considered

- **Suggested fixes** - Add `fix: str | None` field to Diagnostic
- **Watch mode** - Re-run on file changes
- **Caching** - Skip unchanged files (like ruff already does)
- **Custom checker plugins** - Load checkers from user config

### Not Planned

- **IDE integration** - Use existing LSP tools instead
- **Historical tracking** - Keep it simple, no database
- **Notifications** - stdout only, external tools can wrap
