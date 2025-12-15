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

    def snapshot(self) -> bytes:
        """Serialize current state for persistence. Backend-specific format."""
        ...

    def restore(self, data: bytes) -> None:
        """Restore state from serialized snapshot."""
        ...
```

Eight methods total. `snapshot()` and `restore()` enable persistence and rollback.

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

### Snapshots and Rollback

The session manages state history. When a session snapshot is taken:

```python
# Session stores all slice states including InMemoryState
snapshot_id = session.snapshot()

# ... operations modify filesystem ...
backend.write("foo.txt", "modified")

# Rollback restores InMemoryState to previous value
session.rollback(snapshot_id)

# Filesystem reflects rolled-back state
backend.read("foo.txt")  # Returns original content or raises FileNotFoundError
```

### Persistence

Saving session to disk includes `InMemoryState`:

```python
# Save session (includes all slices)
session.save("/path/to/session.json")

# Load session
session = Session.load("/path/to/session.json", bus=bus)

# Recreate backend pointing at loaded session
backend = InMemoryBackend(session)  # Reads existing InMemoryState
```

The `InMemoryState` slice is serialized as JSON via the standard serde module.

## External State: ContainerBackend and HostBackend

These backends have state external to the session. They implement `snapshot()`
and `restore()` differently:

### ContainerBackend

```python
class ContainerBackend:
    def __init__(self, container: Container, workdir: str = "/workspace") -> None:
        self._container = container
        self._workdir = workdir

    def snapshot(self) -> bytes:
        # Create tarball of workspace
        result = self._exec(["tar", "-czf", "-", "-C", self._workdir, "."])
        return result.stdout_bytes

    def restore(self, data: bytes) -> None:
        # Clear workspace and extract tarball
        self._exec(["rm", "-rf", f"{self._workdir}/*"])
        self._exec(["tar", "-xzf", "-", "-C", self._workdir], stdin_bytes=data)

    # ... read, write, delete, list, glob, grep same as before ...
```

Container snapshots are expensive (tar/untar). Use sparingly.

### HostBackend

```python
class HostBackend:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def snapshot(self) -> bytes:
        # Serialize directory tree to tarball
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(self._root, arcname=".")
        return buffer.getvalue()

    def restore(self, data: bytes) -> None:
        # Clear and restore from tarball
        shutil.rmtree(self._root)
        self._root.mkdir(parents=True)
        buffer = io.BytesIO(data)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(self._root)

    # ... read, write, delete, list, glob, grep same as before ...
```

### Session Integration for External Backends

External backends don't automatically participate in session snapshots. For
explicit snapshot/rollback:

```python
# Manual snapshot before risky operation
fs_snapshot = backend.snapshot()

try:
    backend.write("config.json", new_config)
    run_tests()
except TestFailure:
    backend.restore(fs_snapshot)
```

To persist external state with session, store snapshot in a session slice:

```python
@dataclass(slots=True, frozen=True)
class ExternalFsSnapshot:
    data: bytes

# Before saving session
session.mutate(ExternalFsSnapshot).seed(ExternalFsSnapshot(data=backend.snapshot()))
session.save("/path/to/session.json")

# After loading session
session = Session.load("/path/to/session.json", bus=bus)
snapshot = session.query(ExternalFsSnapshot).latest()
if snapshot:
    backend.restore(snapshot.data)
```

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

def test_snapshot_restore(fs: Filesystem) -> None:
    fs.write("file.txt", "original")
    snap = fs.snapshot()
    fs.write("file.txt", "modified")
    fs.restore(snap)
    assert fs.read("file.txt").content == "original"
```

Container backend tests are integration tests requiring Podman runtime.

## Summary

| Aspect | InMemoryBackend | ContainerBackend | HostBackend |
|--------|-----------------|------------------|-------------|
| State location | Session slice (`InMemoryState`) | Container filesystem | Host temp directory |
| Session snapshots | Automatic | Manual via `snapshot()`/`restore()` | Manual |
| Persistence | Via session save/load | Via `ExternalFsSnapshot` slice | Via `ExternalFsSnapshot` slice |
| Rollback | Automatic with session | Manual | Manual |

The `Filesystem` protocol provides eight methods: `read`, `write`, `delete`,
`list`, `glob`, `grep`, `snapshot`, `restore`. Tool handlers access the
filesystem via `context.filesystem`. Sections expose backends via a
`filesystem` property. The adapter injects the filesystem into tool context.

This eliminates tool duplication while properly addressing state management
across different backend types.
