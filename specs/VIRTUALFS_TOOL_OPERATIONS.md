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

**Critical**: When a session snapshot is created, the section must also snapshot
the filesystem. These operations are coupled and must happen together.

### FilesystemSection Coordinates Snapshots

```python
class FilesystemSection(MarkdownSection):
    """Base for sections that provide a Filesystem."""

    def __init__(self, *, session: Session, snapshot_dir: Path | None = None) -> None:
        self._session = session
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"
        session.install(FilesystemSnapshot, initial=lambda: FilesystemSnapshot())

    @property
    def filesystem(self) -> Filesystem:
        raise NotImplementedError

    def snapshot(self) -> str:
        """Snapshot both session and filesystem. Returns snapshot ID."""
        # 1. Generate snapshot path
        snapshot_id = f"{uuid.uuid4()}"
        snapshot_path = self._snapshot_dir / f"{snapshot_id}.snapshot"

        # 2. Snapshot filesystem to file
        self.filesystem.snapshot(snapshot_path)

        # 3. Record path in session slice
        self._session.mutate(FilesystemSnapshot).seed(
            FilesystemSnapshot(path=str(snapshot_path))
        )

        # 4. Snapshot session (includes FilesystemSnapshot slice)
        self._session.snapshot(snapshot_id)

        return snapshot_id

    def rollback(self, snapshot_id: str) -> None:
        """Rollback both session and filesystem."""
        # 1. Rollback session (restores FilesystemSnapshot slice)
        self._session.rollback(snapshot_id)

        # 2. Read restored snapshot path from slice
        fs_snapshot = self._session.query(FilesystemSnapshot).latest()
        if fs_snapshot and fs_snapshot.path:
            # 3. Restore filesystem from file
            self.filesystem.restore(Path(fs_snapshot.path))
```

### Usage

```python
section = VfsSection(session=session, mounts=mounts)

# Create coupled snapshot
snapshot_id = section.snapshot()

# ... operations modify filesystem ...
section.filesystem.write("config.json", new_config)

# Rollback restores both session state AND filesystem
section.rollback(snapshot_id)

# Filesystem reflects rolled-back state
section.filesystem.read("config.json")  # Original content
```

### Persistence

When saving a session to disk, the filesystem snapshot file must also be preserved:

```python
def save_session_with_filesystem(session: Session, section: FilesystemSection, path: Path) -> None:
    """Save session and filesystem snapshot together."""
    # 1. Snapshot filesystem to file alongside session
    fs_snapshot_path = path.with_suffix(".fs.tar.gz")
    section.filesystem.snapshot(fs_snapshot_path)

    # 2. Record path in session
    session.mutate(FilesystemSnapshot).seed(
        FilesystemSnapshot(path=str(fs_snapshot_path))
    )

    # 3. Save session
    session.save(path)


def load_session_with_filesystem(
    path: Path,
    bus: EventBus,
    section_factory: Callable[[Session], FilesystemSection],
) -> tuple[Session, FilesystemSection]:
    """Load session and restore filesystem snapshot."""
    # 1. Load session
    session = Session.load(path, bus=bus)

    # 2. Create section (backend reads existing InMemoryState if applicable)
    section = section_factory(session)

    # 3. Restore filesystem from snapshot file if present
    fs_snapshot = session.query(FilesystemSnapshot).latest()
    if fs_snapshot and fs_snapshot.path:
        snapshot_path = Path(fs_snapshot.path)
        if snapshot_path.exists():
            section.filesystem.restore(snapshot_path)

    return session, section
```

### Snapshot File Lifecycle

| Event | Action |
|-------|--------|
| `section.snapshot()` | Write to temp file, record path in slice |
| `section.rollback(id)` | Restore from file at recorded path |
| `save_session_with_filesystem()` | Write to persistent file alongside session |
| `load_session_with_filesystem()` | Restore from file recorded in loaded session |
| Session garbage collection | Caller responsible for cleaning up snapshot files |

### InMemoryBackend: Dual State

For `InMemoryBackend`, state exists in two places:

1. **`InMemoryState` slice** — Always in session, automatically included in session snapshots
2. **Snapshot file** — Written on explicit `snapshot()` call

The snapshot file is redundant for `InMemoryBackend` but provides a consistent
interface. Alternatively, `InMemoryBackend.snapshot()` can be a no-op since state
is already in the session:

```python
class InMemoryBackend:
    def snapshot(self, path: Path) -> None:
        # State already in InMemoryState slice; file is optional backup
        pass

    def restore(self, path: Path) -> None:
        # State restored via session.rollback(); file not needed
        pass
```

This simplifies the coupling: for `InMemoryBackend`, session snapshot alone
suffices. For external backends, the file is required.

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
