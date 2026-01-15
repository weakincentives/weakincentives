# WINK_DOCS Specification

## Purpose

The `wink docs` CLI subcommand provides access to bundled documentation for
users who install WINK as a package, enabling LLM-assisted workflows.

**Implementation:** `src/weakincentives/cli/wink.py` (`_read_doc`, `_read_example`,
`_read_specs`, `_handle_docs`)

## CLI Interface

```bash
wink docs --reference      # Print llms.md (API reference)
wink docs --guide          # Print WINK_GUIDE.md (usage guide)
wink docs --spec ADAPTERS  # Print a single spec by name
wink docs --specs          # Print all specs concatenated
wink docs --changelog      # Print CHANGELOG.md (release history)
wink docs --example        # Print code review example as markdown
```

### Behavior

- At least one flag required
- Multiple flags combine output with `---` separator
- Specs printed alphabetically with header comments

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No flags or invalid usage |
| 2 | Documentation not found |

## Package Structure

```
src/weakincentives/docs/
├── __init__.py
├── llms.md
├── WINK_GUIDE.md
├── CHANGELOG.md
├── code_reviewer_example.py
└── specs/
    ├── ADAPTERS.md
    ├── ...
```

## Build-Time Synchronization

Hatch build hook synchronizes docs from repository root into package:

```python
class DocsSyncHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Copy llms.md, WINK_GUIDE.md, CHANGELOG.md, code_reviewer_example.py, specs/*.md
```

## Use Cases

```bash
# Copy to clipboard
wink docs --reference | pbcopy

# Pipe to LLM
wink docs --guide | llm "Summarize key concepts"

# Export
wink docs --specs > all-specs.md

# View example
wink docs --example | less
```

## Resource Loading

Uses `importlib.resources` for reliable bundled file access:

```python
from importlib.resources import files

def _read_doc(name: str) -> str:
    return files("weakincentives.docs").joinpath(name).read_text()
```

## Security

- Static text files bundled at build time
- No user input processed
- No file path arguments
- Output is raw text to stdout
