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

Snapshots are first-class:
- `FilesystemSnapshot` mirrors `Session` snapshot semantics
- Any backend can snapshot; any backend can restore
- ZIP archive is the universal exchange format

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

    def export_archive(self, path: Path) -> int:
        """Write current state to ZIP file. Returns file count."""
        ...

    def import_archive(self, path: Path) -> int:
        """Replace state from ZIP file. Returns file count."""
        ...
```

Eight methods. The `export_archive()` and `import_archive()` methods use a
backend-agnostic ZIP format. The Filesystem protocol has no knowledge of
snapshots—that's handled by `FilesystemSnapshot`.

## FilesystemSnapshot

Manages filesystem snapshots analogous to `Session` snapshots:

```python
@dataclass(slots=True, frozen=True)
class SnapshotInfo:
    """Metadata about a filesystem snapshot."""
    snapshot_id: str
    archive_path: Path
    file_count: int
    total_bytes: int
    created_at: datetime


class FilesystemSnapshot:
    """Manages filesystem snapshots. Counterpart to Session snapshots."""

    def __init__(
        self,
        filesystem: Filesystem,
        snapshot_dir: Path | None = None,
    ) -> None:
        self._filesystem = filesystem
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"
        self._snapshots: dict[str, SnapshotInfo] = {}

    @property
    def filesystem(self) -> Filesystem:
        """The managed filesystem."""
        return self._filesystem

    def snapshot(self, snapshot_id: str) -> SnapshotInfo:
        """Create a snapshot of current filesystem state.

        Args:
            snapshot_id: Unique identifier for this snapshot.

        Returns:
            Metadata about the created snapshot.
        """
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._snapshot_dir / f"{snapshot_id}.fs.zip"

        file_count = self._filesystem.export_archive(archive_path)
        total_bytes = archive_path.stat().st_size

        info = SnapshotInfo(
            snapshot_id=snapshot_id,
            archive_path=archive_path,
            file_count=file_count,
            total_bytes=total_bytes,
            created_at=datetime.now(UTC),
        )
        self._snapshots[snapshot_id] = info
        return info

    def rollback(self, snapshot_id: str) -> int:
        """Restore filesystem to a previous snapshot.

        Args:
            snapshot_id: The snapshot to restore.

        Returns:
            Number of files restored.

        Raises:
            KeyError: If snapshot_id not found.
        """
        info = self._snapshots.get(snapshot_id)
        if info is None:
            raise KeyError(f"Snapshot not found: {snapshot_id}")
        return self._filesystem.import_archive(info.archive_path)

    def restore(self, archive_path: Path) -> int:
        """Restore filesystem from an external archive.

        Use this to restore from a previously saved session.

        Args:
            archive_path: Path to ZIP archive.

        Returns:
            Number of files restored.
        """
        return self._filesystem.import_archive(archive_path)

    def get(self, snapshot_id: str) -> SnapshotInfo | None:
        """Get snapshot metadata by ID."""
        return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> tuple[SnapshotInfo, ...]:
        """List all snapshots in creation order."""
        return tuple(sorted(self._snapshots.values(), key=lambda s: s.created_at))

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its archive.

        Returns:
            True if deleted, False if not found.
        """
        info = self._snapshots.pop(snapshot_id, None)
        if info is None:
            return False
        info.archive_path.unlink(missing_ok=True)
        return True
```

### Usage Pattern

```python
# Create filesystem and snapshot manager
fs = InMemoryBackend()
snapshots = FilesystemSnapshot(fs)

# Work with filesystem
fs.write("src/main.py", "print('hello')")

# Create snapshot before risky operation
snapshots.snapshot("before-refactor")

# Make changes
fs.write("src/main.py", "print('broken')")
fs.delete("tests/")

# Rollback if needed
snapshots.rollback("before-refactor")
assert fs.read("src/main.py").content == "print('hello')"
```

### Coordinating with Session Snapshots

Filesystem and session snapshots use the same ID:

```python
# Snapshot both with same ID
snapshot_id = "turn-5"
fs_info = fs_snapshots.snapshot(snapshot_id)
session.snapshot(snapshot_id)

# Rollback both
session.rollback(snapshot_id)
fs_snapshots.rollback(snapshot_id)
```

## ZIP Archive Format

All backends produce and consume identical ZIP structure:

```
{snapshot_id}.fs.zip
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
  "created_at": "2024-01-15T10:30:00+00:00",
  "file_count": 42,
  "total_bytes": 123456
}
```

No backend information. The format is pure data.

## Backend Implementations

All backends maintain internal state and implement export/import uniformly.

### InMemoryBackend

```python
class InMemoryBackend:
    def __init__(self, mounts: Sequence[HostMount] = ()) -> None:
        self._files: dict[str, str] = {}
        self._hydrate_mounts(mounts)

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        normalized = _normalize(path)
        if normalized not in self._files:
            raise FileNotFoundError(path)
        return _paginate(self._files[normalized], offset, limit)

    def write(self, path: str, content: str) -> int:
        self._files[_normalize(path)] = content
        return len(content.encode())

    def delete(self, path: str) -> int:
        normalized = _normalize(path)
        to_delete = [
            p for p in self._files
            if p == normalized or p.startswith(normalized + "/")
        ]
        for p in to_delete:
            del self._files[p]
        return len(to_delete)

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        normalized = _normalize(path) if path else ""
        prefix = normalized + "/" if normalized else ""
        entries = set()
        for p in self._files:
            if p.startswith(prefix):
                rest = p[len(prefix):]
                first = rest.split("/")[0]
                is_dir = "/" in rest
                entries.add((prefix + first, "directory" if is_dir else "file"))
        return tuple(
            FileEntry(path=p, kind=k, size_bytes=len(self._files.get(p, "").encode()) if k == "file" else None)
            for p, k in sorted(entries)
        )

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        import fnmatch
        matches = [p for p in self._files if fnmatch.fnmatch(p, pattern)]
        return tuple(
            FileEntry(path=p, kind="file", size_bytes=len(self._files[p].encode()))
            for p in sorted(matches)
        )

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        import fnmatch, re
        regex = re.compile(pattern)
        matches = []
        for path, content in sorted(self._files.items()):
            if file_glob and not fnmatch.fnmatch(path, file_glob):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append(GrepMatch(path=path, line_number=i, line_content=line))
        return tuple(matches)

    def export_archive(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath, content in sorted(self._files.items()):
                data = content.encode()
                zf.writestr(f"files/{fpath}", data)
                total_bytes += len(data)
            manifest = {
                "version": "1",
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": len(self._files),
                "total_bytes": total_bytes,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return len(self._files)

    def import_archive(self, path: Path) -> int:
        self._files.clear()
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]
                    self._files[rel_path] = zf.read(name).decode("utf-8")
        return len(self._files)
```

### ContainerBackend

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

    def export_archive(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._exec(["python3", "-c", _EXPORT_SCRIPT.format(workdir=self._workdir)])
        self._copy_from_container("/tmp/snapshot.zip", path)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        return manifest["file_count"]

    def import_archive(self, path: Path) -> int:
        self._copy_to_container(path, "/tmp/snapshot.zip")
        self._exec(["rm", "-rf", f"{self._workdir}/*"])
        self._exec(["python3", "-c", _IMPORT_SCRIPT.format(workdir=self._workdir)])
        with zipfile.ZipFile(path, "r") as zf:
            return sum(1 for n in zf.namelist() if n.startswith("files/") and not n.endswith("/"))

    def _resolve(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._workdir}/{path}"
```

### HostBackend

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

    def export_archive(self, path: Path) -> int:
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
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return file_count

    def import_archive(self, path: Path) -> int:
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    target = self._root / name[6:]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))
                    count += 1
        return count

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes root: {path}")
        return resolved
```

## ToolContext Integration

Tools access filesystem via `ToolContext`:

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

## FilesystemSection

Section that provides a filesystem and snapshot manager:

```python
class FilesystemSection(MarkdownSection):
    """Section that provides a Filesystem with snapshot support."""

    def __init__(
        self,
        *,
        snapshot_dir: Path | None = None,
        **section_kwargs,
    ) -> None:
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"
        self._backend: Filesystem | None = None
        self._snapshots: FilesystemSnapshot | None = None
        super().__init__(**section_kwargs)

    @property
    def filesystem(self) -> Filesystem:
        """Lazily initialize and return the filesystem backend."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    @property
    def snapshots(self) -> FilesystemSnapshot:
        """Lazily initialize and return the snapshot manager."""
        if self._snapshots is None:
            self._snapshots = FilesystemSnapshot(self.filesystem, self._snapshot_dir)
        return self._snapshots

    def _create_backend(self) -> Filesystem:
        """Create a fresh backend instance. Override in subclasses."""
        raise NotImplementedError


class VfsSection(FilesystemSection):
    def __init__(self, *, mounts: Sequence[HostMount] = (), **kwargs) -> None:
        self._mounts = mounts
        super().__init__(
            title="Virtual Filesystem",
            key="vfs",
            template=_VFS_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
            **kwargs,
        )

    def _create_backend(self) -> Filesystem:
        return InMemoryBackend(self._mounts)


class PodmanSection(FilesystemSection):
    def __init__(self, *, config: PodmanConfig, **kwargs) -> None:
        self._config = config
        self._container: Container | None = None
        super().__init__(
            title="Sandbox",
            key="podman",
            template=_PODMAN_TEMPLATE,
            tools=(*FILESYSTEM_TOOLS, shell_tool, eval_tool),
            **kwargs,
        )

    def _create_backend(self) -> Filesystem:
        self._container = _create_container(self._config)
        return ContainerBackend(self._container)


class WorkspaceSection(FilesystemSection):
    def __init__(self, *, root: Path | None = None, **kwargs) -> None:
        self._root = root
        super().__init__(
            title="Workspace",
            key="workspace",
            template=_WORKSPACE_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
            **kwargs,
        )

    def _create_backend(self) -> Filesystem:
        root = self._root or Path(tempfile.mkdtemp(prefix="wink-workspace-"))
        return HostBackend(root)
```

## Coordinated Snapshot Workflow

### Taking Snapshots

```python
snapshot_id = f"turn-{turn_number}"

# Snapshot filesystem first
fs_info = section.snapshots.snapshot(snapshot_id)

# Then snapshot session
session.snapshot(snapshot_id)
```

### Rollback

```python
# Rollback session first (restores conversation state)
session.rollback(snapshot_id)

# Then rollback filesystem
section.snapshots.rollback(snapshot_id)
```

### Resuming with Different Backend

```python
# Load session from disk
session = load_session(jsonl_path)

# Find the filesystem archive from adjacent file
archive_path = jsonl_path.with_suffix(".fs.zip")

# Create section with any backend type
section = VfsSection()  # or PodmanSection, WorkspaceSection

# Restore from archive (works with any backend)
section.snapshots.restore(archive_path)
```

## Integration with dump_session_tree

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
        filesystem_section: If provided, exports filesystem alongside session.

    Returns:
        Path to generated JSONL file, or None if no slices to persist.
    """
    target_dir = target if target.is_dir() else target.parent
    session_id = session.tags.get("session_id", str(uuid.uuid4()))

    # Export filesystem if section provided
    if filesystem_section is not None:
        archive_path = target_dir / f"{session_id}.fs.zip"
        filesystem_section.filesystem.export_archive(archive_path)

    return _dump_session_to_jsonl(session, target_dir / f"{session_id}.jsonl")
```

### Output Structure

```
/snapshots/
├── abc123.jsonl              # Session state
└── abc123.fs.zip             # Filesystem archive (if section provided)
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
        archive = self._current_path.with_suffix(".fs.zip")
        return archive if archive.exists() else None

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
def memory_fs() -> Filesystem:
    return InMemoryBackend()

@pytest.fixture
def host_fs(tmp_path: Path) -> Filesystem:
    return HostBackend(tmp_path)

@pytest.fixture(params=["memory", "host"])
def fs(request, memory_fs, host_fs) -> Filesystem:
    return {"memory": memory_fs, "host": host_fs}[request.param]


def test_write_read(fs: Filesystem) -> None:
    fs.write("test.txt", "hello")
    assert fs.read("test.txt").content == "hello"


def test_export_import(fs: Filesystem, tmp_path: Path) -> None:
    fs.write("file.txt", "original")
    archive = tmp_path / "snapshot.zip"
    count = fs.export_archive(archive)
    assert count == 1
    fs.write("file.txt", "modified")
    fs.import_archive(archive)
    assert fs.read("file.txt").content == "original"


def test_cross_backend_restore(tmp_path: Path) -> None:
    """Export from one backend, import to another."""
    memory = InMemoryBackend()
    memory.write("src/main.py", "print('hello')")
    memory.write("README.md", "# Project")

    archive = tmp_path / "snapshot.zip"
    memory.export_archive(archive)

    host_root = tmp_path / "workspace"
    host_root.mkdir()
    host = HostBackend(host_root)
    host.import_archive(archive)

    assert host.read("src/main.py").content == "print('hello')"
    assert host.read("README.md").content == "# Project"


def test_filesystem_snapshot(tmp_path: Path) -> None:
    """Test FilesystemSnapshot rollback."""
    fs = InMemoryBackend()
    snapshots = FilesystemSnapshot(fs, tmp_path)

    fs.write("file.txt", "v1")
    snapshots.snapshot("s1")

    fs.write("file.txt", "v2")
    snapshots.snapshot("s2")

    fs.write("file.txt", "v3")

    # Rollback to s1
    snapshots.rollback("s1")
    assert fs.read("file.txt").content == "v1"

    # Rollback to s2
    snapshots.rollback("s2")
    assert fs.read("file.txt").content == "v2"


def test_coordinated_snapshots(tmp_path: Path) -> None:
    """Test filesystem and session snapshots together."""
    session = Session(bus=InProcessEventBus())
    section = VfsSection(snapshot_dir=tmp_path)

    # Work
    section.filesystem.write("code.py", "v1")
    session.mutate(SomeSlice).dispatch(SomeEvent())

    # Coordinated snapshot
    snapshot_id = "checkpoint"
    section.snapshots.snapshot(snapshot_id)
    session.snapshot(snapshot_id)

    # More work
    section.filesystem.write("code.py", "v2")
    session.mutate(SomeSlice).dispatch(AnotherEvent())

    # Coordinated rollback
    session.rollback(snapshot_id)
    section.snapshots.rollback(snapshot_id)

    assert section.filesystem.read("code.py").content == "v1"
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
│    read, write, delete, list, glob, grep, export, import        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ InMemoryBackend│   │ ContainerBackend│   │   HostBackend   │
│  internal dict │   │  podman exec    │   │  directory I/O  │
└────────────────┘   └─────────────────┘   └─────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                     FilesystemSnapshot                           │
│              snapshot(id), rollback(id), restore(path)          │
└─────────────────────────────────────────────────────────────────┘
                               │
                      ZIP Archive Format
                  (manifest.json + files/*)
```

### Key Design Decisions

1. **Filesystem Protocol** — 8 methods: `read`, `write`, `delete`, `list`,
   `glob`, `grep`, `export_archive`, `import_archive`

2. **FilesystemSnapshot** — Manages snapshots analogous to Session:
   - `snapshot(id)` — Create named snapshot
   - `rollback(id)` — Restore to named snapshot
   - `restore(path)` — Restore from external archive

3. **ToolContext carries filesystem** — Injected by adapter

4. **Backend-agnostic ZIP format** — Any backend can export/import

5. **Coordinated snapshots** — Use same ID for filesystem and session snapshots

6. **Cross-backend restore** — Export from any backend, import to any other

7. **No session slices for filesystem** — FilesystemSnapshot manages its own
   state; session snapshots and filesystem snapshots are parallel but separate
