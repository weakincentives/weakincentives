# WINK_DOCS Specification

## Purpose

The `wink docs` CLI subcommand provides access to bundled documentation for
users who install WINK as a package. The interface is optimized for AI coding
agents that need to explore documentation efficiently without overwhelming
their context window.

**Design Principles:**

1. **Progressive disclosure** - List available documents before loading content
1. **Targeted retrieval** - Read individual documents, not bulk dumps
1. **Search-first discovery** - Find relevant content without reading everything
1. **Minimal context cost** - Return only what's needed for the current task

**Implementation:** `src/weakincentives/cli/wink.py`

## CLI Interface

### List Available Documents

```bash
wink docs list                    # List all document categories
wink docs list specs              # List all specs with descriptions
wink docs list guides             # List all guides with descriptions
```

**Output format** (machine-readable for AI parsing):

```
SPECS (34 documents)
────────────────────
ADAPTERS          Provider integrations, structured output, throttling
CLAUDE_AGENT_SDK  Claude Agent SDK adapter, MCP tool bridging, skill mounting
DATACLASSES       Serde utilities, frozen dataclass patterns
DBC               Design-by-contract decorators, exhaustiveness checking
...

GUIDES (25 documents)
─────────────────────
quickstart        Get a working agent running quickly
philosophy        The "weak incentives" approach and why WINK exists
prompts           Build typed, testable prompts
...
```

Descriptions are sourced from:

- `CLAUDE.md` spec table for specs
- `guides/README.md` tables for guides

### Search Documentation

```bash
wink docs search PATTERN                    # Search all docs
wink docs search PATTERN --specs            # Search only specs
wink docs search PATTERN --guides           # Search only guides
wink docs search PATTERN --context 3        # Show N lines of context (default: 2)
wink docs search PATTERN --max-results 10   # Limit results (default: 20)
```

**Output format:**

```
Found 5 matches for "reducer"

specs/SESSIONS.md:42
  Reducers receive `SliceView[S]` and return `SliceOp[S]` operations:

specs/SESSIONS.md:58
  The reducer pattern ensures all state changes are:
  - Deterministic (same input → same output)
  - Auditable (every change traced to an event)

guides/sessions.md:127
  ### Writing Your First Reducer

  A reducer is a pure function that takes the current state and an event,
  returning the new state...
```

Search uses case-insensitive substring matching by default. Regex patterns are
supported with `--regex` flag.

### Read Individual Documents

```bash
wink docs read reference              # llms.md (API reference)
wink docs read changelog              # CHANGELOG.md
wink docs read example                # Code review example

wink docs read spec ADAPTERS          # Single spec by name
wink docs read guide quickstart       # Single guide by name
```

Document names are case-insensitive and `.md` extension is optional.

### Show Table of Contents

```bash
wink docs toc spec SESSIONS           # Show headings from a spec
wink docs toc guide sessions          # Show headings from a guide
```

**Output format:**

```
specs/SESSIONS.md - Table of Contents
─────────────────────────────────────
# SESSIONS Specification
## Purpose
## Core Concepts
### Session
### Events
### Reducers
## Session API
### Creating a Session
### Dispatching Events
### Querying State
## Slice Operations
### Append
### Replace
### Clear
## Transaction Support
## Snapshots
## Thread Safety
```

This allows agents to understand document structure before reading the full
content, enabling targeted reads of specific sections.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid usage (missing required argument, unknown flag) |
| 2 | Document not found |

## Package Structure

```
src/weakincentives/docs/
├── __init__.py
├── llms.md
├── CHANGELOG.md
├── code_reviewer_example.py
├── metadata.py           # Document descriptions for list command
├── guides/
│   ├── README.md
│   ├── quickstart.md
│   └── ...
└── specs/
    ├── ADAPTERS.md
    └── ...
```

**Note:** The `src/weakincentives/docs/` directory is generated during builds by `hatch_build.py` and is intentionally not committed (listed in `.gitignore`). When browsing the repository, you will not see this directory unless you run the build.

### Document Metadata

The `metadata.py` module provides document descriptions:

```python
SPEC_DESCRIPTIONS: dict[str, str] = {
    "ADAPTERS": "Provider integrations, structured output, throttling",
    "CLAUDE_AGENT_SDK": "Claude Agent SDK adapter, MCP tool bridging",
    # ... sourced from CLAUDE.md spec table
}

GUIDE_DESCRIPTIONS: dict[str, str] = {
    "quickstart": "Get a working agent running quickly",
    "philosophy": "The 'weak incentives' approach and why WINK exists",
    # ... sourced from guides/README.md tables
}
```

This metadata is maintained separately from the documents themselves to allow
easy updates and ensure consistent descriptions across the CLI and
documentation.

## Build-Time Synchronization

Hatch build hook synchronizes docs from repository root into package:

```python
class DocsSyncHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Copy llms.md, CHANGELOG.md, code_reviewer_example.py
        # Copy guides/*.md, specs/*.md
        # Generate metadata.py from CLAUDE.md and guides/README.md
```

## AI Agent Workflow

Recommended exploration pattern for AI coding agents:

```bash
# 1. Discover what's available
wink docs list specs

# 2. Search for relevant topics
wink docs search "session state"

# 3. Preview document structure
wink docs toc spec SESSIONS

# 4. Read specific document
wink docs read spec SESSIONS

# 5. Read related guide for usage patterns
wink docs read guide sessions
```

This workflow minimizes context usage while providing comprehensive coverage.

## Security

- Static text files bundled at build time
- Search patterns are sanitized (no shell injection)
- No user-controlled file paths (only predefined document names)
- Output is raw text to stdout
