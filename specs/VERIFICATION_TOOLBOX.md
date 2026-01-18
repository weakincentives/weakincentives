# Unified Verification Toolbox Specification

Design specification for consolidating `scripts/` and `build/` into a unified,
production-quality verification framework.

## Problem Statement

The current verification infrastructure has grown organically into two
directories (`scripts/`, `build/`) containing 12 scripts with no shared
infrastructure:

**Current Pain Points:**

1. **No shared patterns** - Each script re-implements path resolution, output
   formatting, subprocess execution, exit code handling, and error reporting
2. **Unclear organization** - The split between `scripts/` and `build/` has no
   principled basis (both contain validators invoked by Makefile)
3. **Inconsistent UX** - Some scripts use emojis, others don't; some are quiet
   on success, others verbose; error formats vary
4. **Duplicated AST analysis** - Three scripts parse Python ASTs independently
   (`validate_module_boundaries.py`, `check_core_imports.py`,
   `verify_doc_examples.py`)
5. **Duplicated markdown processing** - Three scripts enumerate and parse
   markdown files independently
6. **Not testable** - Scripts lack unit tests despite enforcing 100% coverage
   on production code

## Goals

1. **Unified CLI** - Single `wink verify` entrypoint with subcommands
2. **Shared infrastructure** - Common patterns for output, subprocess, paths
3. **Production quality** - Same standards as `src/weakincentives/` (typed,
   tested, DbC-decorated)
4. **Composable checkers** - Mix and match verification passes
5. **Parallel execution** - Independent checks run concurrently
6. **Clear categories** - Checks grouped by what they verify

## Architecture

```
src/weakincentives/
├── verify/                    # New package
│   ├── __init__.py           # Public API exports
│   ├── _runner.py            # Orchestration, parallel execution
│   ├── _output.py            # Unified output formatting
│   ├── _subprocess.py        # Subprocess execution with retries
│   ├── _paths.py             # Project path discovery
│   ├── _git.py               # Git operations (tracked files, status)
│   ├── _ast.py               # Shared AST analysis utilities
│   ├── _markdown.py          # Shared markdown parsing
│   ├── checkers/             # Individual verification passes
│   │   ├── __init__.py
│   │   ├── architecture.py   # Module boundaries, layers, core/contrib
│   │   ├── documentation.py  # Spec refs, doc examples, md links, md format
│   │   ├── security.py       # Bandit, pip-audit
│   │   ├── dependencies.py   # Deptry
│   │   ├── types.py          # Type coverage, integration test validation
│   │   └── tests.py          # Pytest runner
│   └── cli.py                # CLI entrypoint
```

## Checker Protocol

Every checker implements a common interface:

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()

@dataclass(frozen=True, slots=True)
class Finding:
    """A single verification finding."""
    checker: str          # e.g., "architecture.module_boundaries"
    severity: Severity
    message: str
    file: Path | None = None
    line: int | None = None
    suggestion: str | None = None

@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of running a checker."""
    checker: str
    findings: tuple[Finding, ...]
    duration_ms: int

    @property
    def passed(self) -> bool:
        return not any(f.severity == Severity.ERROR for f in self.findings)

class Checker(Protocol):
    """Protocol for verification checkers."""

    @property
    def name(self) -> str:
        """Short identifier (e.g., 'module_boundaries')."""
        ...

    @property
    def category(self) -> str:
        """Category (e.g., 'architecture', 'documentation')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def check(self, ctx: CheckContext) -> CheckResult:
        """Run the check and return findings."""
        ...

@dataclass(frozen=True, slots=True)
class CheckContext:
    """Context passed to all checkers."""
    project_root: Path
    src_dir: Path
    quiet: bool = False
    fix: bool = False  # For checkers that support auto-fix
```

## Checker Categories

### Architecture Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `layer_violations` | `validate_module_boundaries.py` | Foundation → core → adapters → high-level flow |
| `private_module_leaks` | `validate_module_boundaries.py` | No imports from `_private` modules outside package |
| `circular_dependencies` | `validate_module_boundaries.py` | No package-level cycles |
| `redundant_reexports` | `validate_module_boundaries.py` | No submodule + items reexports |
| `core_contrib_separation` | `check_core_imports.py` | Core never imports contrib |

### Documentation Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `spec_references` | `validate_spec_references.py` | File paths in specs exist |
| `doc_examples` | `verify_doc_examples.py` | Python code blocks type-check |
| `markdown_links` | `check_md_links.py` | Local markdown links resolve |
| `markdown_format` | `run_mdformat.py` | Markdown formatting consistent |

### Security Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `bandit` | `run_bandit.py` | No security anti-patterns |
| `vulnerabilities` | `run_pip_audit.py` | No known CVEs in dependencies |

### Dependency Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `dependency_hygiene` | `run_deptry.py` | No unused/missing dependencies |

### Type Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `type_coverage` | `run_type_coverage.py` | 100% type completeness |
| `integration_types` | `validate_integration_tests.py` | Integration tests type-check |

### Test Checkers

| Checker | Source | What It Verifies |
|---------|--------|------------------|
| `unit_tests` | `run_pytest.py` | Tests pass with 100% coverage |

## Shared Infrastructure

### Output Formatting (`_output.py`)

Unified output with consistent formatting:

```python
@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output configuration."""
    quiet: bool = False      # Suppress success messages
    color: bool = True       # Use ANSI colors
    emoji: bool = False      # Never use emojis by default
    json: bool = False       # Machine-readable output

class Output:
    """Unified output handler."""

    def __init__(self, config: OutputConfig) -> None:
        self._config = config

    def success(self, checker: str, message: str) -> None:
        """Report checker success (suppressed in quiet mode)."""
        ...

    def finding(self, finding: Finding) -> None:
        """Report a single finding."""
        ...

    def summary(self, results: Sequence[CheckResult]) -> None:
        """Report final summary."""
        ...
```

**Principles:**

1. **Quiet on success** - Default behavior shows nothing for passing checks
2. **Verbose on failure** - Errors always shown with full context
3. **No emojis** - Consistent with production code standards
4. **Structured option** - `--json` for CI integration

### Subprocess Execution (`_subprocess.py`)

Unified subprocess handling:

```python
@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """Result of subprocess execution."""
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

def run_tool(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 120.0,
    capture: bool = True,
) -> SubprocessResult:
    """Run an external tool with consistent handling."""
    ...
```

**Features:**

- Timeout with clear error messages
- Environment variable merging
- Output capture with streaming option
- Duration tracking for performance analysis

### AST Utilities (`_ast.py`)

Shared Python analysis:

```python
@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about an import statement."""
    module: str
    imported_from: str
    items: tuple[str, ...]
    lineno: int
    is_relative: bool

def extract_imports(source: str, module_name: str) -> tuple[ImportInfo, ...]:
    """Extract all imports from Python source."""
    ...

def module_to_path(module: str, src_dir: Path) -> Path | None:
    """Convert module name to file path."""
    ...

def path_to_module(path: Path, src_dir: Path) -> str:
    """Convert file path to module name."""
    ...
```

### Markdown Utilities (`_markdown.py`)

Shared markdown processing:

```python
@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced code block from markdown."""
    file: Path
    start_line: int
    end_line: int
    language: str
    code: str
    meta: str  # e.g., "nocheck" in ```python nocheck

@dataclass(frozen=True, slots=True)
class Link:
    """A markdown link."""
    file: Path
    line: int
    text: str
    target: str
    is_local: bool

def extract_code_blocks(
    file: Path,
    *,
    languages: frozenset[str] = frozenset({"python", "py"}),
) -> tuple[CodeBlock, ...]:
    """Extract fenced code blocks from markdown."""
    ...

def extract_links(file: Path) -> tuple[Link, ...]:
    """Extract all links from markdown."""
    ...
```

### Git Utilities (`_git.py`)

```python
def tracked_files(
    root: Path,
    *,
    pattern: str = "*",
    exclude_parts: frozenset[str] = frozenset(),
) -> tuple[Path, ...]:
    """Get git-tracked files matching pattern."""
    ...

def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    ...
```

## CLI Design

```
wink verify [OPTIONS] [CHECKERS...]

Options:
  --all, -a          Run all checkers (default if none specified)
  --category, -c     Run all checkers in category
  --quiet, -q        Suppress success output
  --json             Output as JSON
  --fix              Apply auto-fixes where supported
  --parallel, -j N   Max parallel checkers (default: CPU count)
  --list             List available checkers

Categories:
  architecture       Module boundaries and layering
  documentation      Docs, specs, examples
  security           Security scanning
  dependencies       Dependency analysis
  types              Type checking
  tests              Test execution

Examples:
  wink verify                    # Run all checkers
  wink verify -c architecture    # Run architecture checkers only
  wink verify bandit vulns       # Run specific checkers
  wink verify --list             # Show available checkers
```

## Execution Model

### Parallel Execution

Independent checkers run concurrently:

```python
async def run_checkers(
    checkers: Sequence[Checker],
    ctx: CheckContext,
    *,
    max_parallel: int | None = None,
) -> tuple[CheckResult, ...]:
    """Run checkers with bounded parallelism."""
    semaphore = asyncio.Semaphore(max_parallel or os.cpu_count() or 4)

    async def run_one(checker: Checker) -> CheckResult:
        async with semaphore:
            return await asyncio.to_thread(checker.check, ctx)

    return tuple(await asyncio.gather(*(run_one(c) for c in checkers)))
```

### Dependency Ordering

Some checkers depend on others:

```
format-check → lint → typecheck → tests
```

The runner topologically sorts checkers and runs independent groups in parallel.

### Early Exit

With `--maxfail=N`, stop after N failures:

```python
@dataclass(frozen=True, slots=True)
class RunConfig:
    max_failures: int | None = None  # None = no limit
```

## Migration Path

### Phase 1: Infrastructure

1. Create `src/weakincentives/verify/` package
2. Implement shared utilities (`_output.py`, `_subprocess.py`, etc.)
3. Add tests for shared utilities (100% coverage)

### Phase 2: Port Checkers

Port checkers one at a time, maintaining backward compatibility:

1. Port checker to new framework
2. Add comprehensive tests
3. Update Makefile target to use new implementation
4. Remove old script

**Order:**

1. Simple wrappers first: `bandit`, `deptry`, `pip-audit`, `mdformat`
2. Markdown processors: `md_links`, `spec_references`
3. AST analyzers: `core_imports`, `module_boundaries`
4. Complex checkers: `doc_examples`, `type_coverage`, `pytest_runner`

### Phase 3: CLI Integration

1. Add `wink verify` CLI command
2. Deprecate direct script invocation
3. Update Makefile to use `wink verify`

### Phase 4: Cleanup

1. Remove `scripts/` directory
2. Remove `build/` directory (or repurpose for CI-only scripts)
3. Update documentation

## Testing Requirements

The verification toolbox must meet the same standards as production code:

1. **100% coverage** - All code paths tested
2. **Property tests** - Hypothesis tests for parsers
3. **Integration tests** - End-to-end verification of each checker
4. **Mock external tools** - Tests don't require bandit/pyright installed

```python
# Example test structure
def test_module_boundaries_detects_layer_violation(tmp_path: Path) -> None:
    """Importing from higher layer should report violation."""
    # Arrange: Create minimal package with layer violation
    (tmp_path / "src" / "weakincentives" / "types").mkdir(parents=True)
    (tmp_path / "src" / "weakincentives" / "types" / "__init__.py").write_text(
        "from weakincentives.contrib.tools import Plan  # layer violation"
    )

    # Act
    checker = LayerViolationsChecker()
    result = checker.check(CheckContext(project_root=tmp_path, ...))

    # Assert
    assert not result.passed
    assert any("LAYER_VIOLATION" in f.message for f in result.findings)
```

## Configuration

Checkers can be configured via `pyproject.toml`:

```toml
[tool.wink.verify]
# Global settings
quiet = false
parallel = true

# Per-checker settings
[tool.wink.verify.checkers.type_coverage]
threshold = 100.0

[tool.wink.verify.checkers.bandit]
skip = ["B101"]  # assert used in tests

[tool.wink.verify.checkers.doc_examples]
skip_files = ["CHANGELOG.md"]
```

## Success Criteria

The migration is complete when:

1. All current checks pass via `wink verify`
2. `scripts/` and `build/` directories are removed
3. Verification code has 100% test coverage
4. `make check` uses `wink verify` internally
5. CI runs in same time or faster (parallel execution)
6. All checkers have typed, documented APIs

## What This Spec Does NOT Include

Explicitly out of scope:

- **Remote CI integration** - This is local tooling; CI config is separate
- **IDE plugins** - Focus on CLI; IDE integration via LSP is future work
- **Custom checker plugins** - Keep it simple; extensibility later if needed
- **Historical trend tracking** - No metrics database; plain pass/fail
- **Notification systems** - No Slack/email; stdout only

## Implementation Notes

### Python 3.14 Compatibility

The bandit shim for deprecated AST nodes should move to `_ast.py`:

```python
def patch_ast_for_legacy_tools() -> None:
    """Restore AST nodes removed in Python 3.14."""
    # Used by bandit which hasn't updated yet
    ...
```

### Exit Codes

Standard exit codes for CI:

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |
| 2 | Configuration/usage error |
| 3 | Internal error (bug in toolbox) |

### Performance Targets

| Checker | Target | Current |
|---------|--------|---------|
| Architecture | <2s | ~1s |
| Documentation | <5s | ~3s |
| Security (bandit) | <10s | ~5s |
| Type coverage | <15s | ~10s |
| Full suite | <60s parallel | ~90s sequential |
