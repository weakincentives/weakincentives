# WINK Docs Specification

CLI subcommand for accessing bundled documentation.

**Source:** `src/weakincentives/cli/wink.py`

## Purpose

Exposes bundled documentation for LLM-assisted development workflows:

```bash
wink docs --reference   # Print llms.md (API reference)
wink docs --guide       # Print WINK_GUIDE.md
wink docs --specs       # Print all specs concatenated
wink docs --changelog   # Print CHANGELOG.md
```

## Usage Examples

```bash
wink docs --reference | pbcopy           # Copy to clipboard
wink docs --guide | llm "Summarize"      # Pipe to LLM
wink docs --specs > all-specs.md         # Export to file
```

## CLI Interface

```
wink docs [--reference] [--guide] [--specs] [--changelog]
```

| Argument | Description |
|----------|-------------|
| `--reference` | Print API reference (`llms.md`) |
| `--guide` | Print usage guide (`WINK_GUIDE.md`) |
| `--specs` | Print all spec files concatenated |
| `--changelog` | Print changelog |

At least one flag required. Multiple flags combine output with `---` separators.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No flags provided |
| 2 | Documentation not found (packaging error) |

## Package Structure

```
src/weakincentives/docs/
├── __init__.py
├── llms.md
├── WINK_GUIDE.md
├── CHANGELOG.md
└── specs/
    ├── ADAPTERS.md
    └── ...
```

## Build-Time Synchronization

Hatch build hook copies docs from repo root into package:

```python
class DocsSyncHook(BuildHookInterface):
    def initialize(self, version, build_data):
        shutil.copy(root / "llms.md", docs_dir / "llms.md")
        for spec_file in (root / "specs").glob("*.md"):
            shutil.copy(spec_file, specs_dir / spec_file.name)
```

Configure in `pyproject.toml`:

```toml
[tool.hatch.build.hooks.custom]
path = "hatch_build.py"
```

## Implementation

```python
from importlib.resources import files

def _read_doc(name: str) -> str:
    return files("weakincentives.docs").joinpath(name).read_text()

def _read_specs() -> str:
    specs_dir = files("weakincentives.docs.specs")
    parts = []
    for entry in sorted(specs_dir.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".md"):
            parts.append(f"<!-- specs/{entry.name} -->\n{entry.read_text()}")
    return "\n\n".join(parts)
```

## Limitations

- Documentation versioned with package (matches installed version)
- No search or single-spec selection (future extensions)
