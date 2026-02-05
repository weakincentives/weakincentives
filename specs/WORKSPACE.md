# Workspace Tools Specification

## Purpose

Workspace tools provide surfaces for caching workspace summaries and testing
with in-memory filesystems. Tool sections for file operations and shell
execution are provided by the execution harness (e.g., Claude Agent SDK).

**Implementation:** `src/weakincentives/contrib/tools/`

## Guiding Principles

- **Definition vs Harness**: Agent definitions specify what; harness provides how
- **Predictable paths**: Normalized POSIX-style, ASCII-only, relative to session root
- **Single source of state**: Reducers own mutations; handlers remain pure

## Workspace Digest

Task-agnostic repository summaries cached in session state.

**Implementation:** `src/weakincentives/contrib/tools/digests.py`

### Data Model

| Type | Fields | Description |
|------|--------|-------------|
| `WorkspaceDigest` | `section_key`, `summary`, `body` | Cached digest entry |

### Section

`WorkspaceDigestSection` renders cached workspace digests from session state.

```python
from weakincentives.contrib.tools import (
    WorkspaceDigestSection,
    set_workspace_digest,
    latest_workspace_digest,
)
from weakincentives.runtime import Session

session = Session()
section = WorkspaceDigestSection(session=session)

# Populate digest (typically done by exploration agent)
set_workspace_digest(
    session,
    section_key="workspace-digest",
    body="Full project analysis with dependencies, structure...",
    summary="Python web app with FastAPI backend.",
)

# Query digest
digest = latest_workspace_digest(session, "workspace-digest")
```

### Resolution Order

1. **Session snapshot** - `latest_workspace_digest(session, key)`
1. **Override fallback** - From `PromptOverridesStore`
1. **Placeholder** - Default text with warning log

### Data Captured

- Repository layout and notable directories
- Tooling commands (tests, linting, formatting)
- Known caveats and recurring pitfalls

## In-Memory Filesystem

Session-scoped filesystem for testing and evaluation scenarios.

**Implementation:** `src/weakincentives/contrib/tools/filesystem_memory.py`

### Data Model

| Type | Description |
|------|-------------|
| `InMemoryFilesystem` | In-memory implementation of `Filesystem` protocol |
| `ReadResult` | Result of read operations with content and metadata |
| `WriteResult` | Result of write operations with path and size |

### Usage

```python
from weakincentives.contrib.tools import InMemoryFilesystem

fs = InMemoryFilesystem()
fs.write("test.txt", "Hello, world!")
result = fs.read("test.txt")
print(result.content)  # "Hello, world!"
```

### Limits

| Limit | Value |
|-------|-------|
| Content per write | 48,000 characters |
| Path depth | 16 segments |
| Segment length | 80 characters |
| Encoding | UTF-8 text only |

## Cloning

Workspace sections support `clone(session=..., dispatcher=...)`:

- Re-registers reducers on new session
- Binds telemetry to new event dispatcher
- Fully decoupled from original section

## Execution Harness Tools

Tool sections for filesystem operations, planning, and shell execution are
provided by the execution harness rather than defined in WINK. This keeps
agent definitions portable across runtimes.

### Claude Agent SDK

When using `ClaudeAgentSDKAdapter`, the harness provides:

- **Native file tools** - Built-in Claude Code file operations
- **Shell execution** - Command execution in sandboxed environment
- **Planning tools** - Native task tracking

See `specs/CLAUDE_AGENT_SDK.md` for details on workspace configuration.

## Limitations

- **Ephemeral state**: All workspace data dies with session
- **Text-only**: Binary content rejected for in-memory filesystem
- **Synchronized clocks**: Timestamps require UTC synchronization
