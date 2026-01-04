# WINK_DOCS Specification

This document specifies the `wink docs` CLI subcommand, which provides access
to WINK documentation from the command line.

## Overview

The `docs` subcommand exposes bundled documentation for users who install WINK
as a package. This enables LLM-assisted development workflows where
documentation can be piped directly to tools or copied into prompts.

```bash
wink docs --reference   # Print llms.md (API reference)
wink docs --guide       # Print WINK_GUIDE.md (usage guide)
wink docs --specs       # Print all specs concatenated
```

## Motivation

When users install WINK via `pip install weakincentives`, they don't have
access to the repository's documentation files. The `docs` subcommand solves
this by:

1. **Bundling documentation** inside the package at build time
1. **Exposing documentation** via CLI for easy access
1. **Supporting LLM workflows** by outputting plain text to stdout

Common use cases:

```bash
# Copy API reference to clipboard
wink docs --reference | pbcopy

# Pipe to an LLM context
wink docs --guide | llm "Summarize the key concepts"

# Export all specs to a file
wink docs --specs > all-specs.md
```

## Package Structure

Documentation files must be bundled inside the package to be available after
installation. The following structure is required:

```
src/weakincentives/
└── docs/
    ├── __init__.py          # Empty, marks as package
    ├── llms.md              # API reference (copy of root llms.md)
    ├── WINK_GUIDE.md        # Usage guide (copy of root WINK_GUIDE.md)
    └── specs/
        ├── ADAPTERS.md
        ├── CLAUDE_AGENT_SDK.md
        ├── DATACLASSES.md
        ├── DBC.md
        ├── EVALS.md
        ├── EXAMPLES.md
        ├── EXECUTION_STATE.md
        ├── FILESYSTEM.md
        ├── LANGSMITH.md
        ├── LOGGING.md
        ├── MAILBOX.md
        ├── MAIN_LOOP.md
        ├── PROMPT_OPTIMIZATION.md
        ├── PROMPTS.md
        ├── SESSIONS.md
        ├── TESTING.md
        ├── THREAD_SAFETY.md
        ├── TOOLS.md
        ├── WINK_DEBUG.md
        ├── WINK_DOCS.md
        └── WORKSPACE.md
```

### Build-Time Synchronization

A custom Hatch build hook synchronizes documentation from the repository root
into `src/weakincentives/docs/` automatically during packaging:

```python
# hatch_build.py
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from pathlib import Path
import shutil


class DocsSyncHook(BuildHookInterface):
    """Synchronize documentation files into package before build."""

    PLUGIN_NAME = "docs-sync"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        docs_dir = root / "src" / "weakincentives" / "docs"
        specs_dir = docs_dir / "specs"

        # Create directories
        docs_dir.mkdir(parents=True, exist_ok=True)
        specs_dir.mkdir(exist_ok=True)

        # Copy documentation files
        shutil.copy(root / "llms.md", docs_dir / "llms.md")
        shutil.copy(root / "WINK_GUIDE.md", docs_dir / "WINK_GUIDE.md")

        # Copy all spec files
        for spec_file in (root / "specs").glob("*.md"):
            shutil.copy(spec_file, specs_dir / spec_file.name)

        # Ensure __init__.py exists
        (docs_dir / "__init__.py").touch()
```

Configure in `pyproject.toml`:

```toml
[tool.hatch.build.hooks.custom]
path = "hatch_build.py"
```

This approach is preferred because:

1. **Automatic**: Documentation is always synchronized on `hatch build`
1. **No manual steps**: Developers cannot forget to run a sync command
1. **CI-friendly**: Works seamlessly in automated build pipelines

### Manual Synchronization Alternative

For local development or debugging, a Makefile target can synchronize docs
manually:

```makefile
sync-docs:
	@mkdir -p src/weakincentives/docs/specs
	@cp llms.md src/weakincentives/docs/
	@cp WINK_GUIDE.md src/weakincentives/docs/
	@cp specs/*.md src/weakincentives/docs/specs/
	@touch src/weakincentives/docs/__init__.py
```

This is useful for testing `wink docs` locally without a full build cycle.

## CLI Interface

### Command Signature

```
wink docs [--reference] [--guide] [--specs]
```

### Arguments

| Argument      | Type | Description                                |
| ------------- | ---- | ------------------------------------------ |
| `--reference` | flag | Print the API reference (`llms.md`)        |
| `--guide`     | flag | Print the usage guide (`WINK_GUIDE.md`)    |
| `--specs`     | flag | Print all specification files concatenated |

### Behavior

1. **At least one flag required**: If no flags are provided, print usage help
   and exit with code 1.

1. **Multiple flags allowed**: Flags can be combined. Output is printed in
   order: reference, guide, specs.

1. **Separator between sections**: When multiple flags are used, print a
   separator line between sections:

   ```
   ---
   ```

1. **Spec ordering**: Specs are printed in alphabetical order by filename.

1. **Spec headers**: Each spec file is prefixed with a header comment:

   ```
   <!-- specs/ADAPTERS.md -->
   ```

### Exit Codes

| Code | Meaning                                         |
| ---- | ----------------------------------------------- |
| 0    | Success                                         |
| 1    | No flags provided or invalid usage              |
| 2    | Documentation files not found (packaging error) |

### Examples

```bash
# Print API reference
$ wink docs --reference
# WINK (Weak Incentives)
...

# Print all documentation
$ wink docs --reference --guide --specs
# WINK (Weak Incentives)
...
---
# WINK Guide
...
---
<!-- specs/ADAPTERS.md -->
# Adapters Specification
...

# Combine with other tools
$ wink docs --specs | grep -A5 "## Invariants"
```

## Implementation

### Resource Loading

Use `importlib.resources` for reliable access to bundled files:

```python
from importlib.resources import files, as_file
from pathlib import Path


def _read_doc(name: str) -> str:
    """Read a documentation file from the package."""
    doc_files = files("weakincentives.docs")
    return doc_files.joinpath(name).read_text(encoding="utf-8")


def _read_specs() -> str:
    """Read all spec files, concatenated with headers."""
    specs_dir = files("weakincentives.docs.specs")
    parts: list[str] = []

    # Get all .md files, sorted alphabetically by name
    # Note: iterdir() yields Traversable objects, use .name for the filename
    spec_entries = sorted(
        (entry for entry in specs_dir.iterdir() if entry.name.endswith(".md")),
        key=lambda entry: entry.name,
    )

    for entry in spec_entries:
        header = f"<!-- specs/{entry.name} -->"
        content = entry.read_text(encoding="utf-8")
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)
```

### Subcommand Registration

Add to `src/weakincentives/cli/wink.py`:

```python
def register_docs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the docs subcommand."""
    docs_parser = subparsers.add_parser(
        "docs",
        help="Print bundled documentation",
        description="Access WINK documentation from the command line.",
    )
    docs_parser.add_argument(
        "--reference",
        action="store_true",
        help="Print API reference (llms.md)",
    )
    docs_parser.add_argument(
        "--guide",
        action="store_true",
        help="Print usage guide (WINK_GUIDE.md)",
    )
    docs_parser.add_argument(
        "--specs",
        action="store_true",
        help="Print all specification files",
    )
    docs_parser.set_defaults(func=handle_docs)


def handle_docs(args: argparse.Namespace) -> int:
    """Handle the docs subcommand."""
    if not (args.reference or args.guide or args.specs):
        print("Error: At least one of --reference, --guide, or --specs required")
        print("Usage: wink docs [--reference] [--guide] [--specs]")
        return 1

    parts: list[str] = []

    try:
        if args.reference:
            parts.append(_read_doc("llms.md"))
        if args.guide:
            parts.append(_read_doc("WINK_GUIDE.md"))
        if args.specs:
            parts.append(_read_specs())
    except FileNotFoundError as e:
        print(f"Error: Documentation not found: {e}", file=sys.stderr)
        print("This may indicate a packaging error.", file=sys.stderr)
        return 2

    print("\n---\n".join(parts))
    return 0
```

## Testing

### Unit Tests

```python
# tests/cli/test_docs.py
import subprocess
import sys


def test_docs_reference_outputs_llms():
    """--reference prints llms.md content."""
    result = subprocess.run(
        [sys.executable, "-m", "weakincentives.cli.wink", "docs", "--reference"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "# WINK" in result.stdout or "weakincentives" in result.stdout.lower()


def test_docs_guide_outputs_guide():
    """--guide prints WINK_GUIDE.md content."""
    result = subprocess.run(
        [sys.executable, "-m", "weakincentives.cli.wink", "docs", "--guide"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert len(result.stdout) > 1000  # Guide is substantial


def test_docs_specs_outputs_all_specs():
    """--specs prints all spec files with headers."""
    result = subprocess.run(
        [sys.executable, "-m", "weakincentives.cli.wink", "docs", "--specs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "<!-- specs/ADAPTERS.md -->" in result.stdout
    assert "<!-- specs/TOOLS.md -->" in result.stdout


def test_docs_no_flags_returns_error():
    """No flags prints usage and returns error."""
    result = subprocess.run(
        [sys.executable, "-m", "weakincentives.cli.wink", "docs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--reference" in result.stdout or "--reference" in result.stderr


def test_docs_multiple_flags_combines_output():
    """Multiple flags combine output with separators."""
    result = subprocess.run(
        [sys.executable, "-m", "weakincentives.cli.wink", "docs", "--reference", "--guide"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "---" in result.stdout
```

### Integration Tests

Verify documentation is correctly packaged:

```python
def test_docs_available_in_installed_package(tmp_path):
    """Documentation is accessible after pip install."""
    # Build wheel
    subprocess.run(["make", "sync-docs"], check=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "-o", str(tmp_path)],
        check=True,
    )

    # Install in isolated environment
    venv_path = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    pip = venv_path / "bin" / "pip"
    wheel = next(tmp_path.glob("*.whl"))
    subprocess.run([str(pip), "install", str(wheel)], check=True)

    # Verify docs command works
    wink = venv_path / "bin" / "wink"
    result = subprocess.run(
        [str(wink), "docs", "--reference"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

## Versioning Considerations

Documentation files are versioned with the package. When documentation changes:

1. Update the source files (`llms.md`, `WINK_GUIDE.md`, `specs/*.md`)
1. Run `make sync-docs` (or let the build hook handle it)
1. Bump version and release

The bundled documentation always matches the installed package version.

## Alternatives Considered

### 1. External Documentation URL

Instead of bundling, print URLs to online documentation:

```bash
wink docs --reference
# Visit: https://weakincentives.readthedocs.io/en/latest/reference.html
```

**Rejected**: Requires internet access and doesn't support offline/LLM
workflows.

### 2. Fetch Documentation at Runtime

Download documentation from a CDN on first access:

```python
def get_docs():
    cache_path = Path.home() / ".cache" / "wink" / "docs"
    if not cache_path.exists():
        download_docs(cache_path)
    return cache_path.read_text()
```

**Rejected**: Adds complexity, network dependency, and version mismatch risks.

### 3. Symlinks in Package

Symlink documentation files into the package directory:

```
src/weakincentives/docs/llms.md -> ../../llms.md
```

**Rejected**: Symlinks don't work reliably across platforms and are often
excluded from wheels.

## Security Considerations

- Documentation files are static text bundled at build time
- No user input is processed by the `docs` command
- No file paths are accepted as arguments (preventing path traversal)
- Output is raw text to stdout (no interpretation or execution)

## Future Extensions

Potential enhancements (not in scope for initial implementation):

1. **`--list`**: List available documentation files without printing content
1. **`--spec NAME`**: Print a single spec by name
1. **`--format json`**: Output as JSON with metadata
1. **`--search QUERY`**: Search documentation content
