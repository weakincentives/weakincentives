# Workspaces and Filesystem

*Canonical specs: [specs/WORKSPACE.md](../specs/WORKSPACE.md),
[specs/FILESYSTEM.md](../specs/FILESYSTEM.md)*

This guide explains how WINK manages file access for agents. Agents need
to read and write files, but giving them unrestricted host access is
dangerous. Workspaces and the filesystem protocol solve this by providing
controlled, sandboxed, and portable file operations.

## Why a Filesystem Abstraction

Agents operate on files: reading source code, writing analysis results,
modifying configuration. The naive approach is to give tools direct host
filesystem access. This creates three problems:

1. **Security**: An agent with unrestricted access can read credentials,
   overwrite system files, or traverse outside its intended scope.

1. **Portability**: Tool handlers that use `open()` and `os.path` are
   coupled to the host. They cannot run in-memory for tests, inside
   containers, or across different backends.

1. **Rollback**: When a tool fails partway through writing files, you
   need a way to undo the damage. Raw filesystem calls leave partial
   state that is hard to clean up.

The `Filesystem` protocol solves all three. Tools call
`context.filesystem.read()` and `context.filesystem.write()` without
knowing whether the backend is an in-memory store, a sandboxed temp
directory, or a Podman container. The protocol handles path validation,
size limits, and snapshot/restore.

## The Workspace Mental Model

A workspace is a temporary directory on the host that contains a
controlled subset of files the agent needs. Think of it as a sandbox:

```
Host filesystem             Workspace (temp dir)
/repos/project/             /tmp/wink-abc123/
  src/                        code/
  tests/                        src/
  .git/           -- mount -->    tests/
  .env                          (no .git, no .env)
  node_modules/
```

The workspace copies files from the host into a temp directory according
to mount rules you define. The agent sees only what you explicitly mount.
Everything else is invisible.

This is the core security boundary. The agent cannot read `/etc/passwd`
or your `.env` file because those paths were never mounted. Even if the
model tries to access paths outside the workspace root, the filesystem
protocol rejects them.

## Host Mounts

A `HostMount` defines what gets copied from the host into the workspace:

```python nocheck
from weakincentives.prompt import HostMount

mount = HostMount(
    host_path="/repos/project",
    mount_path="code",
    exclude_glob=(".git/*", "*.pyc", "__pycache__/*"),
    max_bytes=5_000_000,
)
```

| Field | Purpose |
|-------|---------|
| `host_path` | What to copy from the host |
| `mount_path` | Where it appears in the workspace |
| `include_glob` | Whitelist patterns (empty means all) |
| `exclude_glob` | Blacklist patterns |
| `max_bytes` | Size cap to prevent copying large repos |

**Security**: Mounts are validated against `allowed_host_roots`. If a
mount's `host_path` does not fall under an allowed root, creation fails.
This prevents path traversal attacks, especially important when mount
paths come from user input.

```python nocheck
workspace = WorkspaceSection(
    session=session,
    mounts=(mount,),
    allowed_host_roots=("/repos",),  # Only /repos/** allowed
)
```

Symlinks are rejected by default. This prevents symlink-based escapes
where a link inside the mounted directory points outside the allowed
roots.

## Workspace Lifecycle

Workspaces follow a predictable lifecycle:

1. **Create**: Temp directory created, files copied per mount config
1. **Use**: Agent reads and writes files within the workspace
1. **Clone**: Sub-agents can share the workspace via reference counting
1. **Cleanup**: Last reference deletes the temp directory

The workspace section implements context manager protocol. When used
with `Prompt.resources`, cleanup is automatic:

```python nocheck
with prompt.resources:
    response = adapter.evaluate(prompt, session=session)
# Workspace temp dir cleaned up here
```

For full adapter integration patterns, see the
[Claude Agent SDK guide](claude-agent-sdk.md).

## Workspace Digests

Agents often need repository context: project structure, build commands,
known caveats. Loading the full repo into the prompt wastes tokens.
Workspace digests solve this by caching a structured summary of the
repository in session state.

```python nocheck
from weakincentives.contrib.tools import (
    WorkspaceDigestSection,
    set_workspace_digest,
    latest_workspace_digest,
)

# An exploration agent populates the digest once
set_workspace_digest(
    session,
    section_key="workspace-digest",
    body="Full project analysis...",
    summary="Python web app with FastAPI backend.",
)

# Later agents read the cached digest
digest = latest_workspace_digest(session, "workspace-digest")
```

**Why digests matter for context efficiency:**

- A full repo tree might consume thousands of tokens
- A digest captures the same information in a compact summary
- The digest is computed once and reused across agent turns
- Resolution falls back through session state, prompt overrides, and
  finally a placeholder with a warning log

Digests typically capture: repository layout, notable directories,
tooling commands (test, lint, format), and known caveats.

## Filesystem Backends

The `Filesystem` protocol has two backend implementations:

| Backend | Storage | Use Case |
|---------|---------|----------|
| `InMemoryFilesystem` | Python dicts | Tests, evals, lightweight agents |
| `HostFilesystem` | Sandboxed host directory | Production workspace access |

### When to Use InMemoryFilesystem

Use `InMemoryFilesystem` when you do not need real files on disk:

- **Unit tests**: Fast, no cleanup, deterministic
- **Evaluations**: Reproducible file state without host dependencies
- **Lightweight agents**: Agents that produce output files without
  needing host access

```python nocheck
from weakincentives.contrib.tools import InMemoryFilesystem

fs = InMemoryFilesystem()
fs.write("analysis.md", "# Results\n...")
result = fs.read("analysis.md")
```

In-memory filesystems enforce the same limits as host filesystems
(32MB per convenience write, 16-segment path depth, UTF-8 text only). This
means tests exercise the same validation paths as production.

### When to Use HostFilesystem

Use `HostFilesystem` when the agent needs access to real files:

- **Code review**: Reading actual source code
- **Code generation**: Writing files that will be committed
- **Build tasks**: Running commands against real project files

`HostFilesystem` validates all paths relative to a root directory and
rejects any attempt to escape. It also supports git-based snapshots for
transactional rollback.

## Snapshots and Transactional File Operations

When a tool modifies files and then fails, you need to undo those
changes. WINK handles this through filesystem snapshots.

### How Snapshots Work

The `SnapshotableFilesystem` protocol adds two methods to the base
`Filesystem`:

- `snapshot(tag)`: Capture the current state
- `restore(snapshot)`: Roll back to a previous state

Each backend implements snapshots differently:

| Backend | Snapshot Strategy |
|---------|-------------------|
| `InMemoryFilesystem` | Structural sharing via frozen dicts |
| `HostFilesystem` | Git commits with external `--git-dir` |

### Why Snapshots Matter

WINK's transactional tool execution relies on snapshots. Before a tool
runs, WINK snapshots the filesystem. If the tool fails, the snapshot is
restored automatically. This is the filesystem half of the transaction
(the session half uses slice snapshots).

```
Tool call starts
  1. Snapshot filesystem + session state
  2. Execute tool handler
  3a. Success -> keep changes
  3b. Failure -> restore snapshot (files + state reverted)
```

You do not write rollback logic in your tool handlers. The framework
handles it. This is covered in more detail in the
[tools guide](tools.md#transactional-tool-execution).

### Snapshot Integration with Sessions

Snapshots are recorded in session state for auditability:

```python nocheck
from weakincentives.filesystem import FilesystemSnapshot

fs_snapshot = filesystem.snapshot(tag="before-refactor")
session[FilesystemSnapshot].append(fs_snapshot)

# Later, if needed:
filesystem.restore(fs_snapshot)
```

Each snapshot records: a unique ID, creation timestamp, git commit ref
(for host filesystems), root path, and an optional human-readable tag.

## Streaming API

For large files, the streaming layer provides memory-bounded operations:

```python nocheck
# Read a file in chunks (O(chunk_size) memory)
with filesystem.open_read("large-dataset.csv") as reader:
    for chunk in reader:
        process(chunk)

# Write large output incrementally
with filesystem.open_write("results.bin") as writer:
    for batch in generate_batches():
        writer.write(batch)

# Line-by-line text reading with lazy UTF-8 decoding
with filesystem.open_text("logs.txt") as text:
    for line in text.lines(strip=True):
        if "ERROR" in line:
            handle_error(line)
```

The convenience methods (`read()`, `write()`) are built on top of streaming
and remain the simplest choice for typical agent operations. Use streaming
when file sizes may exceed reasonable memory limits.

## Limits

All filesystem backends enforce consistent limits:

| Limit | Value |
|-------|-------|
| Convenience write limit | 32MB |
| Default chunk size (streaming) | 64KB |
| Path depth | 16 segments |
| Segment length | 80 characters |
| Default read limit | 2,000 lines |
| Max grep matches | 1,000 |
| Encoding | UTF-8 text (binary via `read_bytes`/`write_bytes` or streaming) |

These limits exist to prevent agents from producing unbounded output
or creating deeply nested directory structures that are hard to inspect.
For files larger than the convenience limits, use the streaming API.

## Limitations

- **Ephemeral**: All workspace data dies with the session
- **Text-focused**: `InMemoryFilesystem` is text-only; use
  `read_bytes`/`write_bytes` on host backends for binary
- **Single-threaded**: Filesystem instances are not thread-safe
- **Git dependency**: Host filesystem snapshots require git
- **All-or-nothing restore**: No partial rollback of individual files

## Next Steps

- [Claude Agent SDK](claude-agent-sdk.md): Production workspace setup
  with isolation, mounts, and security
- [Tools](tools.md): How tool handlers access the filesystem via
  `ToolContext`
- [Sessions](sessions.md): How snapshots integrate with session state
- [Testing](testing.md): Using `InMemoryFilesystem` in tests
