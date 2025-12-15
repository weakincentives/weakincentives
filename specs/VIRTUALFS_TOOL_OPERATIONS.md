# VirtualFilesystem Tool Operations Specification

## Status

**Draft** — Proposed refactoring for unified filesystem abstraction.

## Non-Goals

**Backward compatibility is explicitly not a goal.** This spec describes the
end state. Existing VFS, Podman, and Claude Agent SDK workspace implementations
will be replaced, not wrapped or deprecated incrementally.

## Problem

Three backends provide overlapping file operations with no shared interface:

| Backend | State Model | Duplication |
|---------|-------------|-------------|
| VirtualFileSystem | Session slice + reducers | 7 tools + handlers |
| PodmanSandboxSection | Container exec | Same 7 tools reimplemented |
| ClaudeAgentWorkspaceSection | Temp directory | None (uses SDK native tools) |

Tool handlers cannot operate on "the filesystem" without knowing which backend
is active. This forces duplication and prevents composition.

## First Principles

A filesystem is a key-value store:
- **Keys** are paths (strings)
- **Values** are file contents (strings, UTF-8 only)
- **Operations**: read, write, delete, list, search

Everything else—pagination, path normalization, security boundaries—is
implementation detail handled by backends.

## Filesystem Protocol

```python
from typing import Protocol, Literal
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class FileEntry:
    path: str
    kind: Literal["file", "directory"]
    size_bytes: int | None = None  # None for directories

@dataclass(slots=True, frozen=True)
class FileContent:
    path: str
    content: str
    total_lines: int
    offset: int
    limit: int

@dataclass(slots=True, frozen=True)
class GrepMatch:
    path: str
    line_number: int
    line_content: str


class Filesystem(Protocol):
    """Minimal filesystem interface for tool handlers."""

    # ─── Core Operations ───────────────────────────────────────────

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        """Read file content. Raises FileNotFoundError if missing."""
        ...

    def write(self, path: str, content: str) -> int:
        """Write content to file. Creates parent directories. Returns bytes written."""
        ...

    def delete(self, path: str) -> int:
        """Delete file or directory recursively. Returns count of deleted entries."""
        ...

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        """List directory entries. Empty path means root."""
        ...

    # ─── Search Operations ─────────────────────────────────────────

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        """Find files matching glob pattern."""
        ...

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        """Search file contents with regex. Optional file glob filter."""
        ...

    # ─── Snapshot Operations ───────────────────────────────────────

    def snapshot(self, path: Path) -> None:
        """Write current state to file at path. Creates parent directories."""
        ...

    def restore(self, path: Path) -> None:
        """Restore state from file at path. Raises FileNotFoundError if missing."""
        ...
```

Eight methods total. `snapshot()` and `restore()` write to/read from files
(default: temp directory) rather than returning bytes, avoiding large base64
blobs in session JSON.

## Session and ToolContext Integration

### The Problem

`Session.install()` only accepts frozen dataclass slices with `@reducer` methods.
`Filesystem` is a Protocol with mutable implementations. These don't compose.

### Solution: ToolContext Carries Filesystem

Extend `ToolContext` with an optional filesystem field:

```python
@dataclass(slots=True, frozen=True)
class ToolContext:
    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapterProtocol[Any]
    session: SessionProtocol
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None
    filesystem: Filesystem | None = None  # NEW
```

Sections that provide a filesystem set this field when building tool context.
The adapter's tool dispatch logic checks if the section provides a filesystem
and injects it into context.

### Tool Handlers

Tools access filesystem via context, not session:

```python
def read_file_handler(params: ReadParams, *, context: ToolContext) -> ToolResult[FileContent]:
    if context.filesystem is None:
        return ToolResult(message="No filesystem available", value=None, success=False)
    content = context.filesystem.read(params.path, offset=params.offset, limit=params.limit)
    return ToolResult(message=_format(content), value=content)

def write_file_handler(params: WriteParams, *, context: ToolContext) -> ToolResult[int]:
    if context.filesystem is None:
        return ToolResult(message="No filesystem available", value=None, success=False)
    bytes_written = context.filesystem.write(params.path, params.content)
    return ToolResult(message=f"Wrote {bytes_written} bytes to {params.path}", value=bytes_written)

# Single tool definition used by all sections
FILESYSTEM_TOOLS = (
    Tool(name="ls", handler=ls_handler, ...),
    Tool(name="read_file", handler=read_file_handler, ...),
    Tool(name="write_file", handler=write_file_handler, ...),
    Tool(name="edit_file", handler=edit_file_handler, ...),
    Tool(name="glob", handler=glob_handler, ...),
    Tool(name="grep", handler=grep_handler, ...),
    Tool(name="rm", handler=rm_handler, ...),
)
```

## Session State: InMemoryBackend

For `InMemoryBackend`, state must integrate with session for snapshots and
rollback. The backend is backed by a frozen dataclass slice:

```python
@dataclass(slots=True, frozen=True)
class _FileRecord:
    path: str
    content: str
    size_bytes: int

@dataclass(slots=True, frozen=True)
class InMemoryState:
    """Session slice holding in-memory filesystem state."""
    files: tuple[_FileRecord, ...] = ()


class InMemoryBackend:
    """Filesystem backed by session state."""

    def __init__(self, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._session = session
        session.install(InMemoryState, initial=lambda: InMemoryState())
        self._hydrate_mounts(mounts)

    def _state(self) -> InMemoryState:
        return self._session.query(InMemoryState).latest() or InMemoryState()

    def _set_state(self, state: InMemoryState) -> None:
        self._session.mutate(InMemoryState).seed(state)

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        state = self._state()
        record = next((f for f in state.files if f.path == _normalize(path)), None)
        if record is None:
            raise FileNotFoundError(path)
        return _paginate(record.content, offset, limit)

    def write(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        size = len(content.encode())
        record = _FileRecord(path=normalized, content=content, size_bytes=size)

        state = self._state()
        files = [f for f in state.files if f.path != normalized]
        files.append(record)
        files.sort(key=lambda f: f.path)
        self._set_state(InMemoryState(files=tuple(files)))
        return size

    def delete(self, path: str) -> int:
        normalized = _normalize(path)
        state = self._state()
        remaining = []
        deleted = 0
        for f in state.files:
            if f.path == normalized or f.path.startswith(normalized + "/"):
                deleted += 1
            else:
                remaining.append(f)
        self._set_state(InMemoryState(files=tuple(remaining)))
        return deleted

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        # ... same logic as before, reading from self._state() ...

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        # ... same logic ...

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        # ... same logic ...

    def snapshot(self) -> bytes:
        state = self._state()
        return _serialize_state(state)

    def restore(self, data: bytes) -> None:
        state = _deserialize_state(data)
        self._set_state(state)
```

## Snapshot Architecture

Session snapshots and filesystem snapshots are **coupled**. The section that
provides the filesystem is responsible for coordinating both.

### Snapshot Protocol

```python
class Filesystem(Protocol):
    # ... read, write, delete, list, glob, grep ...

    def snapshot(self, path: Path) -> None:
        """Write current state to file. Creates parent directories."""
        ...

    def restore(self, path: Path) -> None:
        """Restore state from file. Raises FileNotFoundError if missing."""
        ...
```

Snapshots write to files (not bytes). Default location is a temp file.

### Snapshot Slice

All filesystem sections use a common slice to track snapshot file paths:

```python
@dataclass(slots=True, frozen=True)
class FilesystemSnapshot:
    """Session slice tracking filesystem snapshot location."""
    path: str | None = None  # Path to snapshot file, None if not snapshotted
```

### InMemoryBackend Snapshots

For `InMemoryBackend`, state lives in `InMemoryState` slice. Snapshots serialize
this to a file:

```python
class InMemoryBackend:
    def snapshot(self, path: Path) -> None:
        state = self._state()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serde.dump(state))

    def restore(self, path: Path) -> None:
        data = path.read_text()
        state = serde.parse(InMemoryState, json.loads(data))
        self._set_state(state)
```

### ContainerBackend Snapshots

```python
class ContainerBackend:
    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stream tarball directly to file
        with open(path, "wb") as f:
            self._exec(["tar", "-czf", "-", "-C", self._workdir, "."], stdout=f)

    def restore(self, path: Path) -> None:
        self._exec(["rm", "-rf", f"{self._workdir}/*"])
        with open(path, "rb") as f:
            self._exec(["tar", "-xzf", "-", "-C", self._workdir], stdin=f)
```

### HostBackend Snapshots

```python
class HostBackend:
    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(self._root, arcname=".")

    def restore(self, path: Path) -> None:
        shutil.rmtree(self._root)
        self._root.mkdir(parents=True)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(self._root)
```

## Session-Filesystem Snapshot Coupling

Session snapshots and filesystem snapshots must happen together. This coupling
is implemented via **session lifecycle hooks** registered during section
initialization.

### Session Hook Protocol

Session exposes hooks for snapshot lifecycle events:

```python
class Session:
    def register_snapshot_hook(
        self,
        on_snapshot: Callable[[str, Path], None],  # (snapshot_id, snapshot_dir) -> None
        on_rollback: Callable[[str], None],        # (snapshot_id) -> None
    ) -> None:
        """Register callbacks for snapshot lifecycle events."""
        ...
```

When `session.snapshot(id)` is called:
1. Session serializes all slices
2. Session calls each registered `on_snapshot(id, snapshot_dir)` hook
3. Hooks write additional state to `snapshot_dir / id / <hook_name>`

When `session.rollback(id)` is called:
1. Session restores all slices from snapshot
2. Session calls each registered `on_rollback(id)` hook
3. Hooks restore additional state from `snapshot_dir / id / <hook_name>`

### FilesystemSection Registers Hooks

During initialization, the section registers hooks to couple filesystem state:

```python
class FilesystemSection(MarkdownSection):
    """Base for sections that provide a Filesystem."""

    _HOOK_NAME: ClassVar[str] = "filesystem"

    def __init__(self, *, session: Session, snapshot_dir: Path | None = None) -> None:
        self._session = session
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"

        # Register hooks during initialization
        session.register_snapshot_hook(
            on_snapshot=self._on_snapshot,
            on_rollback=self._on_rollback,
        )

    @property
    def filesystem(self) -> Filesystem:
        raise NotImplementedError

    def _on_snapshot(self, snapshot_id: str, snapshot_dir: Path) -> None:
        """Hook called by session.snapshot(). Writes filesystem state to file."""
        fs_snapshot_path = snapshot_dir / snapshot_id / self._HOOK_NAME
        fs_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.filesystem.snapshot(fs_snapshot_path)

    def _on_rollback(self, snapshot_id: str) -> None:
        """Hook called by session.rollback(). Restores filesystem state from file."""
        fs_snapshot_path = self._snapshot_dir / snapshot_id / self._HOOK_NAME
        if fs_snapshot_path.exists():
            self.filesystem.restore(fs_snapshot_path)
```

### How It Works

```
User calls session.snapshot("abc123")
         │
         ▼
    Session.snapshot()
         │
         ├─► 1. Serialize all slices (InMemoryState, Plan, etc.)
         │
         ├─► 2. Write slices to snapshot_dir/abc123/session.json
         │
         └─► 3. Call registered hooks:
                  │
                  └─► FilesystemSection._on_snapshot("abc123", snapshot_dir)
                           │
                           └─► self.filesystem.snapshot(snapshot_dir/abc123/filesystem)
                                    │
                                    └─► Write tarball or JSON to file


User calls session.rollback("abc123")
         │
         ▼
    Session.rollback()
         │
         ├─► 1. Read slices from snapshot_dir/abc123/session.json
         │
         ├─► 2. Restore all slices (InMemoryState now has old value)
         │
         └─► 3. Call registered hooks:
                  │
                  └─► FilesystemSection._on_rollback("abc123")
                           │
                           └─► self.filesystem.restore(snapshot_dir/abc123/filesystem)
                                    │
                                    └─► Read tarball or JSON from file
```

### Snapshot Directory Structure

```
snapshot_dir/
├── abc123/                     # snapshot_id
│   ├── session.json            # Session slices (InMemoryState, Plan, etc.)
│   └── filesystem              # Filesystem snapshot (tarball or JSON)
├── def456/
│   ├── session.json
│   └── filesystem
└── ...
```

### Key Points

1. **Registration happens at init**: `FilesystemSection.__init__` registers hooks
   with the session. No separate registration step needed.

2. **Session drives the lifecycle**: Users call `session.snapshot()` and
   `session.rollback()` directly. The hooks ensure filesystem is included.

3. **Hooks receive snapshot context**: `on_snapshot` receives the snapshot
   directory so hooks can write adjacent files.

4. **Section holds the backend reference**: The hook methods access
   `self.filesystem` to snapshot/restore. The session doesn't know about
   the filesystem directly—it just calls the registered hooks.

5. **Multiple sections supported**: If a prompt has multiple filesystem sections
   (unusual but possible), each registers its own hooks with distinct names.

### InMemoryBackend Special Case

For `InMemoryBackend`, the `InMemoryState` slice is already captured by
`session.snapshot()`. The hook can be a no-op:

```python
class VfsSection(FilesystemSection):
    def _on_snapshot(self, snapshot_id: str, snapshot_dir: Path) -> None:
        # InMemoryState slice already captured by session
        # Optionally write redundant backup for consistency
        pass

    def _on_rollback(self, snapshot_id: str) -> None:
        # InMemoryState slice already restored by session
        # Backend reads from restored slice automatically
        pass
```

For external backends (Container, Host), the hooks are required:

```python
class PodmanSection(FilesystemSection):
    def _on_snapshot(self, snapshot_id: str, snapshot_dir: Path) -> None:
        # Must write tarball—container state not in session
        fs_snapshot_path = snapshot_dir / snapshot_id / self._HOOK_NAME
        fs_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.filesystem.snapshot(fs_snapshot_path)

    def _on_rollback(self, snapshot_id: str) -> None:
        # Must restore from tarball
        fs_snapshot_path = self._snapshot_dir / snapshot_id / self._HOOK_NAME
        if fs_snapshot_path.exists():
            self.filesystem.restore(fs_snapshot_path)
```

### Persistence (Save/Load)

When saving a session to disk:

```python
# session.save() includes hook data
session.save("/path/to/session")

# Creates:
# /path/to/session.json          (slices)
# /path/to/session.snapshots/    (hook data, if any active snapshots)
```

When loading:

```python
# Load session (slices restored)
session = Session.load("/path/to/session.json", bus=bus)

# Create section (registers hooks, reads InMemoryState if present)
section = VfsSection(session=session, mounts=mounts)

# For external backends, manually restore latest state:
if isinstance(section, PodmanSection):
    latest_snapshot = session.latest_snapshot_id()
    if latest_snapshot:
        section._on_rollback(latest_snapshot)
```

### Summary Table

| Event | Session Action | Hook Action |
|-------|----------------|-------------|
| `session.snapshot(id)` | Serialize slices | `on_snapshot(id, dir)` → write fs to file |
| `session.rollback(id)` | Restore slices | `on_rollback(id)` → restore fs from file |
| Section init | — | Register hooks with session |
| `session.save()` | Write slices + snapshot dir | — |
| `session.load()` | Read slices | Section re-registers hooks on creation |

## Section Definitions

Sections create backend and expose it via a property:

```python
class FilesystemSection(MarkdownSection):
    """Base for sections that provide a Filesystem."""

    @property
    def filesystem(self) -> Filesystem:
        raise NotImplementedError


class VfsSection(FilesystemSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._backend = InMemoryBackend(session, mounts)
        super().__init__(
            title="Virtual Filesystem",
            key="vfs",
            template=_VFS_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
        )

    @property
    def filesystem(self) -> Filesystem:
        return self._backend


class PodmanSection(FilesystemSection):
    def __init__(self, *, session: Session, config: PodmanConfig) -> None:
        self._container = _create_container(config)
        self._backend = ContainerBackend(self._container)
        super().__init__(
            title="Sandbox",
            key="podman",
            template=_PODMAN_TEMPLATE,
            tools=(*FILESYSTEM_TOOLS, shell_tool, eval_tool),
        )

    @property
    def filesystem(self) -> Filesystem:
        return self._backend


class WorkspaceSection(FilesystemSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._temp_dir = _create_workspace(mounts)
        self._backend = HostBackend(self._temp_dir)
        super().__init__(
            title="Workspace",
            key="workspace",
            template=_WORKSPACE_TEMPLATE,
            tools=(),  # SDK uses native tools
        )

    @property
    def filesystem(self) -> Filesystem:
        return self._backend
```

## Adapter Integration

The adapter's tool dispatch finds the filesystem from the section that provides
the tool and injects it into `ToolContext`:

```python
def _build_tool_context(
    tool: Tool,
    rendered_prompt: RenderedPrompt,
    session: Session,
    adapter: ProviderAdapter,
    ...
) -> ToolContext:
    # Find section that owns this tool
    section = _find_section_for_tool(rendered_prompt, tool)

    # Check if section provides a filesystem
    filesystem = None
    if isinstance(section, FilesystemSection):
        filesystem = section.filesystem

    return ToolContext(
        prompt=rendered_prompt.prompt,
        rendered_prompt=rendered_prompt,
        adapter=adapter,
        session=session,
        filesystem=filesystem,
        ...
    )
```

## edit_file Implementation

`edit_file` is a tool-layer operation combining read + write:

```python
def edit_file_handler(params: EditParams, *, context: ToolContext) -> ToolResult[str]:
    if context.filesystem is None:
        return ToolResult(message="No filesystem available", value=None, success=False)

    fs = context.filesystem

    try:
        content = fs.read(params.path)
    except FileNotFoundError:
        return ToolResult(message=f"File not found: {params.path}", value=None, success=False)

    old = params.old_string
    new = params.new_string
    if old not in content.content:
        return ToolResult(message=f"String not found: {old!r}", value=None, success=False)

    if params.replace_all:
        updated = content.content.replace(old, new)
    else:
        updated = content.content.replace(old, new, 1)

    fs.write(params.path, updated)
    return ToolResult(message=f"Edited {params.path}", value=updated)
```

## Testing

Parameterized tests cover all backends:

```python
@pytest.fixture
def memory_fs(session: Session) -> Filesystem:
    return InMemoryBackend(session)

@pytest.fixture
def host_fs(tmp_path: Path) -> Filesystem:
    return HostBackend(tmp_path)

@pytest.fixture(params=["memory", "host"])
def fs(request, memory_fs, host_fs) -> Filesystem:
    return {"memory": memory_fs, "host": host_fs}[request.param]

def test_write_read_roundtrip(fs: Filesystem) -> None:
    fs.write("test.txt", "hello")
    content = fs.read("test.txt")
    assert content.content == "hello"

def test_delete(fs: Filesystem) -> None:
    fs.write("a/b.txt", "x")
    assert fs.delete("a") == 1
    with pytest.raises(FileNotFoundError):
        fs.read("a/b.txt")

def test_snapshot_restore(fs: Filesystem, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.tar.gz"
    fs.write("file.txt", "original")
    fs.snapshot(snapshot_path)
    fs.write("file.txt", "modified")
    fs.restore(snapshot_path)
    assert fs.read("file.txt").content == "original"

def test_section_coupled_snapshot(session: Session, tmp_path: Path) -> None:
    """Section.snapshot() couples session and filesystem snapshots."""
    section = VfsSection(session=session, snapshot_dir=tmp_path)
    section.filesystem.write("data.txt", "v1")

    snapshot_id = section.snapshot()

    section.filesystem.write("data.txt", "v2")
    assert section.filesystem.read("data.txt").content == "v2"

    section.rollback(snapshot_id)
    assert section.filesystem.read("data.txt").content == "v1"
```

Container backend tests are integration tests requiring Podman runtime.

## Summary

| Aspect | InMemoryBackend | ContainerBackend | HostBackend |
|--------|-----------------|------------------|-------------|
| State location | Session slice (`InMemoryState`) | Container filesystem | Host temp directory |
| Snapshot file | Optional (state in slice) | Required (tarball) | Required (tarball) |
| Session coupling | Automatic via slice | Via `FilesystemSnapshot` path | Via `FilesystemSnapshot` path |
| Rollback | `section.rollback()` | `section.rollback()` | `section.rollback()` |

**Key insight**: Session and filesystem snapshots are coupled. The section
coordinates both via `section.snapshot()` and `section.rollback()`. Snapshot
files are stored in temp directory by default; paths are recorded in the
`FilesystemSnapshot` session slice.

The `Filesystem` protocol provides eight methods: `read`, `write`, `delete`,
`list`, `glob`, `grep`, `snapshot`, `restore`. Tool handlers access the
filesystem via `context.filesystem`. Sections expose backends via a
`filesystem` property and coordinate snapshot coupling.
