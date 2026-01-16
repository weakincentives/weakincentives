# Workspace Tools

*Canonical spec: [specs/WORKSPACE.md](../specs/WORKSPACE.md)*

WINK includes several tool suites aimed at background agents that need to
inspect and manipulate a repository safely. They live in
`weakincentives.contrib.tools`.

## PlanningToolsSection

**Tools:**

- `planning_setup_plan`
- `planning_add_step`
- `planning_update_step`
- `planning_read_plan`

The plan is stored in session state and updated via reducers. Each step has an
ID, title, details, and status.

Use it when you want the model to externalize its plan without inventing its own
format. Many models plan better when they have explicit tools for planning.

```python
from weakincentives.contrib.tools import PlanningToolsSection, PlanningStrategy

planning = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
    accepts_overrides=True,
)
```

**Strategies:**

- `REACT`: Plan, act, observe, repeat
- `PLAN_ACT_REFLECT`: Plan upfront, act, reflect on results

## The Filesystem Protocol

*Canonical spec: [specs/FILESYSTEM.md](../specs/FILESYSTEM.md)*

All workspace tools use a unified `Filesystem` protocol. This abstraction lets
tool handlers perform file operations without coupling to a specific storage
backend (in-memory VFS, Podman containers, or host filesystem).

**Core operations:**

| Operation | Description |
| ------------- | ----------------------------------------- |
| `read(path)` | Read file content as text with pagination |
| `write(path)` | Write text content to a file |
| `read_bytes()` | Read raw bytes (for copying files) |
| `write_bytes()` | Write raw bytes (for binary content) |
| `exists(path)` | Check if a path exists |
| `stat(path)` | Get file metadata |
| `list(path)` | List directory contents |
| `glob(pattern)` | Match files by glob pattern |
| `grep(pattern)` | Search file contents by regex |
| `delete(path)` | Delete a file or directory |
| `mkdir(path)` | Create a directory |

**Accessing the filesystem from tool handlers:**

```python
def my_handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    fs = context.filesystem
    if fs is None:
        return ToolResult.error("No filesystem available")

    try:
        result = fs.read(params.path, offset=0, limit=100)
        return ToolResult.ok(MyResult(answer=result.content))
    except FileNotFoundError:
        return ToolResult.error(f"File not found: {params.path}")
```

The filesystem is propagated from the workspace section through the prompt to
the tool context automatically.

## VfsToolsSection

A copy-on-write virtual filesystem with tools:

- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `rm`

You can mount host directories into the VFS snapshot via `HostMount`. The VFS
copies files into memory; writes go to the copy, not the host. This is the
default "repo agent" workspace because it avoids accidental host writes.

```python
from weakincentives.contrib.tools import VfsToolsSection, VfsConfig, HostMount

vfs = VfsToolsSection(
    session=session,
    config=VfsConfig(
        mounts=(
            HostMount(
                host_path="src",
                mount_path=None,  # mount at /src
                include_glob=("*.py",),
                exclude_glob=("__pycache__/*",),
            ),
        ),
        allowed_host_roots=(".",),
    ),
)
```

**Key behaviors:**

- Files are copied into memory at mount time
- Writes modify the in-memory copy, not the host
- The VFS supports all standard filesystem operations
- `ReadBeforeWritePolicy` is applied by default

## WorkspaceDigestSection

Renders a cached repo digest stored in session state. The digest is a structured
summary of the repository: file tree, key files, detected patterns.

It works well with progressive disclosure: default to `SUMMARY`, expand on
demand. The model gets an overview without the full file contents.

```python
from weakincentives.contrib.tools import WorkspaceDigestSection

digest = WorkspaceDigestSection(session=session)
```

The digest is typically populated by an optimizer at startup, then cached in the
session for subsequent requests.

## AstevalSection

Exposes `evaluate_python` (safe-ish expression evaluation) with captured
stdout/stderr.

`asteval` restricts what Python code can do: no imports, no file access, no
network. Useful for small transformations (string formatting, arithmetic)
without granting shell access.

**Install:** `pip install "weakincentives[asteval]"`

```python
from weakincentives.contrib.tools import AstevalSection

asteval = AstevalSection(session=session, accepts_overrides=False)
```

## PodmanSandboxSection

Runs shell commands and Python evaluation inside a Podman container.

Use it when you need strict isolation and reproducible execution (tests,
linters). The container provides a clean environment; writes don't affect the
host.

**Install:** `pip install "weakincentives[podman]"`

```python
from weakincentives.contrib.tools import PodmanSandboxSection, PodmanSandboxConfig

sandbox = PodmanSandboxSection(
    session=session,
    config=PodmanSandboxConfig(
        image="python:3.12-slim",
        work_dir="/workspace",
    ),
)
```

## Wiring a Workspace into a Prompt

A practical pattern (also used by `code_reviewer_example.py`):

```python
from typing import Any
from weakincentives.contrib.tools import (
    PlanningToolsSection,
    PlanningStrategy,
    VfsToolsSection,
    VfsConfig,
    WorkspaceDigestSection,
)
from weakincentives.contrib.tools.vfs_types import HostMount as VfsHostMount
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


def build_repo_agent_template(*, session: Session) -> PromptTemplate[Any]:
    vfs_mounts: tuple[VfsHostMount, ...] = (
        VfsHostMount(host_path="src"),
        VfsHostMount(host_path="README.md"),
    )
    vfs = VfsToolsSection(
        session=session,
        config=VfsConfig(
            mounts=vfs_mounts,
            allowed_host_roots=(".",),
        ),
        accepts_overrides=True,
    )

    return PromptTemplate(
        ns="examples",
        key="repo-agent",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="Answer questions about the repo.",
            ),
            WorkspaceDigestSection(session=session),
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.REACT,
            ),
            vfs,
        ),
    )
```

**The important idea**: the workspace sections are built with the active
session. Each run gets its own session with its own tool sections.

## Choosing Between VFS and Podman

**Use VFS when:**

- You need fast, in-memory file operations
- The agent only needs to read/write files
- You want minimal setup

**Use Podman when:**

- You need to run arbitrary shell commands
- You need strict process isolation
- You need reproducible execution environments
- Security is critical and you don't trust the model

Many agents use both: VFS for file operations, Podman for running tests or
linters.

## Filesystem Snapshots

Filesystem backends that implement `SnapshotableFilesystem` support capturing
and restoring workspace state. This enables rollback after failed tool
invocations or exploratory changes.

```python
# Create a snapshot before risky operations
snapshot = filesystem.snapshot(tag="before-refactor")

# Make changes...
filesystem.write("config.py", "NEW_CONFIG = True")

# Rollback if something goes wrong
filesystem.restore(snapshot)
```

**How snapshots work:**

- **InMemoryFilesystem**: Uses Python's structural sharing—snapshot is O(n) dict
  copy but values are shared references
- **HostFilesystem**: Uses git's content-addressed storage for copy-on-write
  semantics. The git repository is stored outside the workspace root to prevent
  agents from accessing snapshot internals.

**Git-based snapshots** are used by both `HostFilesystem` and
`PodmanSandboxSection`. When files are unchanged between snapshots, they share
the same git blob objects, making snapshots space-efficient.

**Session integration:**

```python
# Store snapshot in session
session[FilesystemSnapshot].append(fs_snapshot)

# Later: coordinated rollback
def rollback_to(session: Session, snapshot: Snapshot, fs: SnapshotableFilesystem):
    fs_snapshots = snapshot.slices.get(FilesystemSnapshot, ())
    if fs_snapshots:
        fs.restore(fs_snapshots[-1])
    session.restore(snapshot)
```

## Next Steps

- [Code Review Agent](code-review-agent.md): See workspace tools in action
- [Tools](tools.md): Learn about tool contracts and handlers
- [Sessions](sessions.md): Understand how workspace state is managed
