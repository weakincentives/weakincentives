# VirtualFilesystem Tool Operations Specification

## Status

**Draft** — Unified filesystem abstraction for tool operations.

## Non-Goals

**Backward compatibility is explicitly not a goal.** This spec describes the
end state.

## Problem

Three backends provide overlapping file operations with no shared interface:

| Backend | State Model | Duplication |
|---------|-------------|-------------|
| VirtualFileSystem | Session slice + reducers | 7 tools + handlers |
| PodmanSandboxSection | Container exec | Same 7 tools reimplemented |
| ClaudeAgentWorkspaceSection | Temp directory | None (uses SDK native tools) |

## First Principles

A filesystem is a key-value store:
- **Keys** are paths (strings)
- **Values** are file contents (strings, UTF-8 only)
- **Core operations**: read, write, delete, list, search
- **Persistence**: snapshot to ZIP, restore from ZIP

## Data Types

```python
@dataclass(slots=True, frozen=True)
class FileEntry:
    path: str
    kind: Literal["file", "directory"]
    size_bytes: int | None = None

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
```

## Filesystem Protocol

```python
class Filesystem(Protocol):
    """Minimal filesystem interface for tool handlers."""

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

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        """Find files matching glob pattern."""
        ...

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        """Search file contents with regex."""
        ...

    def snapshot(self, path: Path) -> FilesystemSnapshotRef:
        """Write current state to ZIP file. Returns metadata."""
        ...

    def restore(self, path: Path) -> None:
        """Restore state from ZIP file."""
        ...
```

Eight methods. The `snapshot()` method returns metadata that callers store in
session state.

## Session State

### FilesystemSnapshotRef

Stores reference to the latest filesystem snapshot:

```python
@dataclass(slots=True, frozen=True)
class FilesystemSnapshotRef:
    """Session slice tracking filesystem snapshot location."""
    archive_path: str | None = None      # Relative path to ZIP
    backend_type: str | None = None      # "InMemoryBackend", etc.
    file_count: int = 0
    total_bytes: int = 0
```

This slice is updated when `export_snapshot()` is called on the section. The
session serialization naturally includes this reference.

### InMemoryState (InMemoryBackend only)

```python
@dataclass(slots=True, frozen=True)
class FileRecord:
    path: str
    content: str
    size_bytes: int

@dataclass(slots=True, frozen=True)
class InMemoryState:
    """Session slice holding in-memory filesystem state."""
    files: tuple[FileRecord, ...] = ()
```

For `InMemoryBackend`, the actual file data lives in this session slice.
Session snapshot/rollback automatically handles state. The ZIP export is
redundant but provided for consistency with external backends.

## ToolContext Integration

Tools access filesystem via `ToolContext`, not `session.query()`:

```python
@dataclass(slots=True, frozen=True)
class ToolContext:
    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapterProtocol[Any]
    session: SessionProtocol
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None
    filesystem: Filesystem | None = None  # Injected by adapter
```

Tool handlers:

```python
def read_file_handler(params: ReadParams, *, context: ToolContext) -> ToolResult[FileContent]:
    if context.filesystem is None:
        return ToolResult(message="No filesystem available", value=None, success=False)
    content = context.filesystem.read(params.path, offset=params.offset, limit=params.limit)
    return ToolResult(message=_format(content), value=content)

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

## ZIP Archive Format

All backends produce identical ZIP structure:

```
snapshot.fs.zip
├── manifest.json
└── files/
    ├── src/main.py
    ├── tests/test_main.py
    └── README.md
```

Manifest:

```json
{
  "version": "1",
  "backend": "InMemoryBackend",
  "created_at": "2024-01-15T10:30:00+00:00",
  "file_count": 42,
  "total_bytes": 123456
}
```

## Backend Implementations

### InMemoryBackend

State lives in `InMemoryState` session slice:

```python
class InMemoryBackend:
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
        record = FileRecord(path=normalized, content=content, size_bytes=size)
        state = self._state()
        files = [f for f in state.files if f.path != normalized]
        files.append(record)
        files.sort(key=lambda f: f.path)
        self._set_state(InMemoryState(files=tuple(files)))
        return size

    def delete(self, path: str) -> int:
        normalized = _normalize(path)
        state = self._state()
        remaining, deleted = [], 0
        for f in state.files:
            if f.path == normalized or f.path.startswith(normalized + "/"):
                deleted += 1
            else:
                remaining.append(f)
        self._set_state(InMemoryState(files=tuple(remaining)))
        return deleted

    def snapshot(self, path: Path) -> FilesystemSnapshotRef:
        state = self._state()
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "version": "1",
                "backend": "InMemoryBackend",
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": len(state.files),
                "total_bytes": sum(f.size_bytes for f in state.files),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for file in state.files:
                zf.writestr(f"files/{file.path}", file.content)
        return FilesystemSnapshotRef(
            archive_path=str(path),
            backend_type="InMemoryBackend",
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
        )

    def restore(self, path: Path) -> None:
        with zipfile.ZipFile(path, "r") as zf:
            files = []
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]
                    content = zf.read(name).decode("utf-8")
                    files.append(FileRecord(
                        path=rel_path,
                        content=content,
                        size_bytes=len(content.encode()),
                    ))
            self._set_state(InMemoryState(files=tuple(sorted(files, key=lambda f: f.path))))
```

### ContainerBackend

State lives in container filesystem:

```python
class ContainerBackend:
    def __init__(self, container: Container, workdir: str = "/workspace") -> None:
        self._container = container
        self._workdir = workdir

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        result = self._exec(["cat", self._resolve(path)])
        return _paginate(result.stdout, offset, limit)

    def write(self, path: str, content: str) -> int:
        full = self._resolve(path)
        self._exec(["mkdir", "-p", str(Path(full).parent)])
        self._exec(["tee", full], stdin=content)
        return len(content.encode())

    def delete(self, path: str) -> int:
        self._exec(["rm", "-rf", self._resolve(path)])
        return 1

    def snapshot(self, path: Path) -> FilesystemSnapshotRef:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create ZIP in container
        self._exec(["python3", "-c", _SNAPSHOT_SCRIPT.format(workdir=self._workdir)])
        self._copy_from_container("/tmp/snapshot.zip", path)
        # Read manifest for metadata
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        return FilesystemSnapshotRef(
            archive_path=str(path),
            backend_type="ContainerBackend",
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
        )

    def restore(self, path: Path) -> None:
        self._copy_to_container(path, "/tmp/snapshot.zip")
        self._exec(["rm", "-rf", f"{self._workdir}/*"])
        self._exec(["python3", "-c", _RESTORE_SCRIPT.format(workdir=self._workdir)])

    def _resolve(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._workdir}/{path}"
```

### HostBackend

State lives in host temp directory:

```python
class HostBackend:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        content = self._resolve(path).read_text()
        return _paginate(content, offset, limit)

    def write(self, path: str, content: str) -> int:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return len(content.encode())

    def delete(self, path: str) -> int:
        target = self._resolve(path)
        if target.is_dir():
            count = sum(1 for _ in target.rglob("*"))
            shutil.rmtree(target)
            return count
        elif target.exists():
            target.unlink()
            return 1
        return 0

    def snapshot(self, path: Path) -> FilesystemSnapshotRef:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_count, total_bytes = 0, 0
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in self._root.rglob("*"):
                if fpath.is_file():
                    rel = fpath.relative_to(self._root)
                    content = fpath.read_bytes()
                    zf.writestr(f"files/{rel}", content)
                    file_count += 1
                    total_bytes += len(content)
            manifest = {
                "version": "1",
                "backend": "HostBackend",
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return FilesystemSnapshotRef(
            archive_path=str(path),
            backend_type="HostBackend",
            file_count=file_count,
            total_bytes=total_bytes,
        )

    def restore(self, path: Path) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    target = self._root / name[6:]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes root: {path}")
        return resolved
```

## FilesystemSection

Base class for sections that provide a filesystem:

```python
class FilesystemSection(MarkdownSection):
    """Base for sections that provide a Filesystem."""

    def __init__(self, *, session: Session, snapshot_dir: Path | None = None) -> None:
        self._session = session
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"
        self._backend: Filesystem | None = None
        self._backend_initialized = False

        # Install snapshot ref slice
        session.install(FilesystemSnapshotRef, initial=lambda: FilesystemSnapshotRef())

    @property
    def filesystem(self) -> Filesystem:
        """Lazily initialize and return the filesystem backend."""
        if not self._backend_initialized:
            self._initialize_backend()
            self._backend_initialized = True
        assert self._backend is not None
        return self._backend

    def _initialize_backend(self) -> None:
        """Create backend, restoring from snapshot if available."""
        raise NotImplementedError

    def export_snapshot(self, archive_path: Path | None = None) -> FilesystemSnapshotRef:
        """Write filesystem to ZIP and update session slice.

        Call this BEFORE session.snapshot() to ensure the ref is included.
        """
        if archive_path is None:
            archive_path = self._snapshot_dir / f"{uuid.uuid4()}.fs.zip"

        ref = self.filesystem.snapshot(archive_path)
        self._session.mutate(FilesystemSnapshotRef).seed(ref)
        return ref

    def restore_from_snapshot(self, archive_path: Path | None = None) -> None:
        """Restore filesystem from ZIP.

        If archive_path is None, reads from FilesystemSnapshotRef slice.
        """
        if archive_path is None:
            ref = self._session.query(FilesystemSnapshotRef).latest()
            if ref is None or ref.archive_path is None:
                return
            archive_path = Path(ref.archive_path)

        if archive_path.exists():
            self.filesystem.restore(archive_path)
```

### Concrete Sections

```python
class VfsSection(FilesystemSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = (), **kwargs) -> None:
        super().__init__(session=session, **kwargs)
        self._mounts = mounts
        super(MarkdownSection, self).__init__(
            title="Virtual Filesystem",
            key="vfs",
            template=_VFS_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
        )

    def _initialize_backend(self) -> None:
        self._backend = InMemoryBackend(self._session, self._mounts)
        # For InMemoryBackend, state is already in session slice
        # No need to restore from ZIP on normal operation


class PodmanSection(FilesystemSection):
    def __init__(self, *, session: Session, config: PodmanConfig, **kwargs) -> None:
        super().__init__(session=session, **kwargs)
        self._config = config
        self._container: Container | None = None
        super(MarkdownSection, self).__init__(
            title="Sandbox",
            key="podman",
            template=_PODMAN_TEMPLATE,
            tools=(*FILESYSTEM_TOOLS, shell_tool, eval_tool),
        )

    def _initialize_backend(self) -> None:
        self._container = _create_container(self._config)
        self._backend = ContainerBackend(self._container)
        # Restore from snapshot if available
        self.restore_from_snapshot()


class WorkspaceSection(FilesystemSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = (), **kwargs) -> None:
        super().__init__(session=session, **kwargs)
        self._mounts = mounts
        self._temp_dir: Path | None = None
        super(MarkdownSection, self).__init__(
            title="Workspace",
            key="workspace",
            template=_WORKSPACE_TEMPLATE,
            tools=(),
        )

    def _initialize_backend(self) -> None:
        self._temp_dir = _create_workspace(self._mounts)
        self._backend = HostBackend(self._temp_dir)
        # Restore from snapshot if available
        self.restore_from_snapshot()
```

## Snapshot and Rollback Workflow

### Snapshot

```python
# 1. Export filesystem to ZIP (updates FilesystemSnapshotRef slice)
ref = section.export_snapshot(snapshot_dir / f"{snapshot_id}.fs.zip")

# 2. Snapshot session (includes the updated FilesystemSnapshotRef)
session.snapshot(snapshot_id)
```

The key insight: `export_snapshot()` runs BEFORE `session.snapshot()`, so the
session naturally includes the archive reference.

### Rollback

```python
# 1. Rollback session (restores FilesystemSnapshotRef slice)
session.rollback(snapshot_id)

# 2. For InMemoryBackend: state already restored via InMemoryState slice
# 3. For external backends: explicitly restore from archive
if isinstance(section, (PodmanSection, WorkspaceSection)):
    section.restore_from_snapshot()
```

For `InMemoryBackend`, the `InMemoryState` slice is restored by session
rollback, so the filesystem automatically reflects the old state.

For external backends, an explicit `restore_from_snapshot()` call is needed
because their state lives outside the session.

## Integration with dump_session_tree

`dump_session_tree` persists session state to JSONL. Filesystem snapshots are
stored as adjacent ZIP files.

### Extended dump_session_tree

```python
def dump_session_tree(
    session: Session,
    target: Path,
    *,
    filesystem_section: FilesystemSection | None = None,
) -> Path | None:
    """Persist session tree to JSONL with optional filesystem snapshot.

    Args:
        session: Root session to persist.
        target: Target directory or file path.
        filesystem_section: If provided, exports filesystem before dumping.

    Returns:
        Path to generated JSONL file, or None if no slices to persist.
    """
    target_dir = target if target.is_dir() else target.parent
    session_id = session.tags.get("session_id", str(uuid.uuid4()))

    # Export filesystem if section provided
    if filesystem_section is not None:
        archive_path = target_dir / f"{session_id}.fs.zip"
        filesystem_section.export_snapshot(archive_path)

    # Standard session dump (now includes FilesystemSnapshotRef)
    return _dump_session_to_jsonl(session, target_dir / f"{session_id}.jsonl")
```

### Output Structure

```
/snapshots/
├── abc123.jsonl              # Session state (includes FilesystemSnapshotRef)
└── abc123.fs.zip             # Filesystem archive
```

### JSONL Entry

```json
{
  "version": "1",
  "created_at": "2024-01-15T10:30:00+00:00",
  "slices": [
    {
      "slice_type": "...FilesystemSnapshotRef",
      "items": [{
        "archive_path": "abc123.fs.zip",
        "backend_type": "InMemoryBackend",
        "file_count": 42,
        "total_bytes": 123456
      }]
    }
  ],
  "tags": {"session_id": "abc123"}
}
```

## Integration with wink debug

The `wink debug` command displays filesystem contents from adjacent ZIP archives.

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/filesystem/meta` | GET | Filesystem manifest and summary |
| `/api/filesystem/tree` | GET | Directory tree structure |
| `/api/filesystem/file` | GET | Single file content (`?path=src/main.py`) |

### Implementation

```python
class SnapshotStore:
    def get_filesystem_archive(self) -> Path | None:
        """Find adjacent filesystem ZIP for current snapshot."""
        ref = self._get_slice(FilesystemSnapshotRef)
        if ref and ref.archive_path:
            # Handle relative paths
            archive = self._current_path.parent / Path(ref.archive_path).name
            if archive.exists():
                return archive
        return None

    def get_filesystem_manifest(self) -> dict | None:
        archive = self.get_filesystem_archive()
        if archive is None:
            return None
        with zipfile.ZipFile(archive, "r") as zf:
            return json.loads(zf.read("manifest.json"))

    def get_filesystem_tree(self) -> list[dict]:
        archive = self.get_filesystem_archive()
        if archive is None:
            return []
        with zipfile.ZipFile(archive, "r") as zf:
            return [
                {"path": name[6:], "size_bytes": zf.getinfo(name).file_size}
                for name in sorted(zf.namelist())
                if name.startswith("files/") and not name.endswith("/")
            ]

    def get_filesystem_file(self, path: str) -> str | None:
        archive = self.get_filesystem_archive()
        if archive is None:
            return None
        with zipfile.ZipFile(archive, "r") as zf:
            try:
                return zf.read(f"files/{path}").decode("utf-8")
            except KeyError:
                return None
```

### Web UI

Filesystem tab shows file tree and content viewer when `FilesystemSnapshotRef`
is present in the snapshot.

## Adapter Integration

The adapter finds the filesystem from the section and injects it into context:

```python
def _build_tool_context(
    tool: Tool,
    rendered_prompt: RenderedPrompt,
    session: Session,
    adapter: ProviderAdapter,
) -> ToolContext:
    section = _find_section_for_tool(rendered_prompt, tool)

    filesystem = None
    if isinstance(section, FilesystemSection):
        filesystem = section.filesystem

    return ToolContext(
        prompt=rendered_prompt.prompt,
        rendered_prompt=rendered_prompt,
        adapter=adapter,
        session=session,
        filesystem=filesystem,
    )
```

## Testing

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

def test_write_read(fs: Filesystem) -> None:
    fs.write("test.txt", "hello")
    assert fs.read("test.txt").content == "hello"

def test_snapshot_restore(fs: Filesystem, tmp_path: Path) -> None:
    fs.write("file.txt", "original")
    ref = fs.snapshot(tmp_path / "snapshot.zip")
    assert ref.file_count == 1
    fs.write("file.txt", "modified")
    fs.restore(tmp_path / "snapshot.zip")
    assert fs.read("file.txt").content == "original"
```

## Summary

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Tool Layer                              │
│    ls, read_file, write_file, edit_file, glob, grep, rm        │
│                              │                                   │
│                   context.filesystem                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                      Filesystem Protocol                         │
│    read, write, delete, list, glob, grep, snapshot, restore     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ InMemoryBackend│   │ ContainerBackend│   │   HostBackend   │
│                │   │                 │   │                 │
│ InMemoryState  │   │  podman exec    │   │  temp dir I/O   │
│ session slice  │   │                 │   │                 │
└────────────────┘   └─────────────────┘   └─────────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                      ZIP Archive Format
                  (manifest.json + files/*)
```

### Key Design Decisions

1. **Filesystem Protocol** — 8 methods: `read`, `write`, `delete`, `list`,
   `glob`, `grep`, `snapshot`, `restore`

2. **ToolContext carries filesystem** — Injected by adapter, not retrieved via
   `session.query()`

3. **Two session slices**:
   - `InMemoryState` — File data for InMemoryBackend (auto-restored on rollback)
   - `FilesystemSnapshotRef` — Archive path for all backends (enables lazy restore)

4. **Explicit snapshot workflow** — `section.export_snapshot()` before
   `session.snapshot()` ensures ref is included

5. **Lazy initialization** — Backend created on first `filesystem` property access

6. **ZIP archive format** — Consistent structure across all backends; enables
   inspection in wink debug and standard tools

7. **dump_session_tree integration** — Takes optional section parameter; exports
   ZIP before serializing session

### State Management Comparison

| Backend | Live State | Session Slice | Snapshot | Rollback |
|---------|------------|---------------|----------|----------|
| InMemoryBackend | `InMemoryState` | `InMemoryState` + `FilesystemSnapshotRef` | Auto (via slice) | Auto (via slice) |
| ContainerBackend | Container FS | `FilesystemSnapshotRef` | Manual export | Manual restore |
| HostBackend | Temp directory | `FilesystemSnapshotRef` | Manual export | Manual restore |
