# Workspace Tools Specification

## Purpose

Workspace tools provide surfaces for caching workspace summaries, testing
with in-memory filesystems, and managing temporary workspace directories
with host file mounts. Tool sections for file operations and shell execution
are provided by the execution harness (e.g., Claude Agent SDK, Codex App Server).

**Implementation:**

- Workspace digests: `src/weakincentives/contrib/tools/digests.py`
- In-memory filesystem: `src/weakincentives/contrib/tools/filesystem_memory.py`
- Public API: `src/weakincentives/contrib/tools/__init__.py`
- Claude Agent SDK workspace: `src/weakincentives/adapters/claude_agent_sdk/workspace.py`
- Codex App Server workspace: `src/weakincentives/adapters/codex_app_server/workspace.py`

## Guiding Principles

- **Definition vs Harness**: Agent definitions specify what; harness provides how
- **Predictable paths**: Normalized POSIX-style, ASCII-only, relative to session root
- **Single source of state**: Reducers own mutations; handlers remain pure

## Workspace Sections

Two adapter-specific workspace sections manage temporary directories with host
file mounts:

| Section | Adapter | Key | Source |
|---------|---------|-----|--------|
| `ClaudeAgentWorkspaceSection` | Claude Agent SDK | `claude-agent-workspace` | `src/weakincentives/adapters/claude_agent_sdk/workspace.py` |
| `CodexWorkspaceSection` | Codex App Server | `codex-workspace` | `src/weakincentives/adapters/codex_app_server/workspace.py` |

Both sections share the same host mount model and workspace lifecycle but render
provider-specific section descriptions.

### Host Mounts

Host mounts copy files from the host filesystem into the workspace temp directory.
See `HostMount` and `HostMountPreview` dataclasses in the workspace modules.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host_path` | `str` | (required) | Absolute or relative path to host file/directory |
| `mount_path` | `str \| None` | `None` | Relative path in temp directory (defaults to basename) |
| `include_glob` | `tuple[str, ...]` | `()` | Glob patterns to include (empty = all) |
| `exclude_glob` | `tuple[str, ...]` | `()` | Glob patterns to exclude |
| `max_bytes` | `int \| None` | `None` | Maximum total bytes to copy |
| `follow_symlinks` | `bool` | `False` | Whether to follow symbolic links |

### Security

- **Allowed roots**: Mounts are validated against `allowed_host_roots` to
  prevent path traversal outside security boundaries
- **Symlink handling**: Symlinks are rejected by default (`follow_symlinks=False`)
- **Mount path validation**: Codex workspace validates mount paths cannot escape
  the temp directory

### Workspace Lifecycle

1. **Creation**: Temp directory created, host files copied per mount config
1. **Usage**: Workspace section renders mount previews for the prompt
1. **Cloning**: `clone(session=...)` shares the temp directory with reference counting
1. **Cleanup**: `cleanup()` decrements reference count; last reference deletes temp dir

### Resources

Both workspace sections contribute a `Filesystem` resource (backed by
`HostFilesystem`) to the prompt's resource registry.

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

- Shares temp directory with original via reference counting
- Shares filesystem instance for consistency
- Re-registers reducers on new session
- Fully decoupled from original section lifecycle

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

### Codex App Server

When using `CodexAppServerAdapter`, the harness provides:

- **Shared module tools** - File operations via JSON-RPC
- **Workspace fingerprinting** - Deterministic fingerprint for reuse detection

See `specs/CODEX_APP_SERVER.md` for details on workspace configuration.

## Limitations

- **Ephemeral state**: All workspace data dies with session
- **Text-only**: Binary content rejected for in-memory filesystem
- **Synchronized clocks**: Timestamps require UTC synchronization

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` - Claude Agent SDK adapter and workspace
- `specs/CODEX_APP_SERVER.md` - Codex App Server adapter and workspace
- `specs/FILESYSTEM.md` - Filesystem protocol
- `specs/SESSIONS.md` - Session lifecycle
