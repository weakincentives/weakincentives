# Workspace Specification

## Purpose

The workspace is the sandbox the agent acts on. Prompt templates declare
environment intent via `WorkspaceConfig` (see `specs/SANDBOX.md`); the
evaluating adapter materializes one sandbox per evaluation, points the
harness `cwd` at `sandbox.root`, and renders a **workspace preview** from
the *opened* sandbox into the prompt. This spec covers the prompt-side
preview plus the digest and in-memory-filesystem surfaces.

**Implementation:**

- Workspace preview: `src/weakincentives/prompt/workspace.py`
- Workspace digests: `src/weakincentives/contrib/tools/digests.py`
- In-memory filesystem: `src/weakincentives/filesystem/_memory.py`
- Public API: `src/weakincentives/contrib/tools/__init__.py`

## Guiding Principles

- **Definition vs Harness**: Agent definitions specify what; harness provides how
- **One environment model**: Mounts and posture live in `WorkspaceConfig`;
  there is no separate workspace lifecycle to manage
- **Render from reality**: The preview lists what the opened sandbox
  actually contains, not copy-time bookkeeping

## Workspace Preview

`PromptTemplate.create(..., workspace=WorkspaceConfig(...))` appends a preview
section (key `workspace`) to the template. Its params resolve at render
time: `prompt.render(session=..., sandbox=sandbox)` builds a fresh listing
from the open sandbox, so no run state is ever stored on the prompt:

| Symbol | Role |
|--------|------|
| `WorkspacePreviewParams` | `listing` rendered into the section body |
| `workspace_preview_section()` | Section factory (used by `PromptTemplate.create`) |
| `workspace_preview_params(filesystem)` | Builds the listing via `filesystem.list(".")` |
| `WORKSPACE_PREVIEW_KEY` | The auto-appended section key (`"workspace"`) |

Without a `sandbox` argument the section renders a "not yet
materialized" placeholder, so direct `prompt.render()` calls still
succeed.

Mount declarations (`HostMount`), allowed-root validation, byte budgets,
and symlink handling are part of `WorkspaceConfig` and are specified in
`specs/SANDBOX.md`.

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

**Implementation:** `src/weakincentives/filesystem/_memory.py` (facade:
`Filesystem.in_memory()`)

### Data Model

| Type | Description |
|------|-------------|
| `Filesystem.in_memory()` | Filesystem facade over the in-memory backend |
| `ReadResult` | Result of read operations with content and metadata |
| `WriteResult` | Result of write operations with path and size |

### Usage

```python
from weakincentives.filesystem import Filesystem

fs = Filesystem.in_memory()
fs.write("test.txt", "Hello, world!")
result = fs.read("test.txt")
print(result.content)  # "Hello, world!"
```

### Limits

| Limit | Value |
|-------|-------|
| Convenience read/write | 32MB |
| Path depth | 16 segments |
| Segment length | 80 characters |
| Default encoding | UTF-8 |

**Note:** Streaming operations (`open_read`, `open_write`) have no inherent size
limits. See `FILESYSTEM.md` for streaming API details.

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

See `specs/CODEX_APP_SERVER.md` for details on workspace configuration.

## Limitations

- **Ephemeral state**: All workspace data dies with session
- **No network**: Podman containers have no network access
- **Cooperative timeout**: ASTEval interrupts cooperatively only
- **Synchronized clocks**: Timestamps require UTC synchronization

## Related Specifications

- `specs/SANDBOX.md` - WorkspaceConfig, providers, and the sandbox aggregate
- `specs/CLAUDE_AGENT_SDK.md` - Claude Agent SDK adapter and workspace
- `specs/CODEX_APP_SERVER.md` - Codex App Server adapter and workspace
- `specs/FILESYSTEM.md` - Filesystem protocol
- `specs/SESSIONS.md` - Session lifecycle
