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

The ZIP archive is the universal exchange format. Any backend can produce a
snapshot, and any backend can restore from it. This enables:
- Snapshot from Podman, restore to in-memory
- Snapshot from in-memory, restore to host directory
- Resume a session with a different workspace and adapter

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
        """Restore state from ZIP file. Clears existing state first."""
        ...
```

Eight methods. All backends implement the same interface. The `snapshot()` and
`restore()` methods use a backend-agnostic ZIP format.

## Session State

### FilesystemSnapshotRef

Single session slice tracking the archive location:

```python
@dataclass(slots=True, frozen=True)
class FilesystemSnapshotRef:
    """Session slice tracking filesystem snapshot location."""
    archive_path: str | None = None
    file_count: int = 0
    total_bytes: int = 0
```

No `backend_type` field. The ZIP format is self-describing and any Filesystem
implementation can restore from it.

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

## ZIP Archive Format

All backends produce and consume identical ZIP structure:

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
  "created_at": "2024-01-15T10:30:00+00:00",
  "file_count": 42,
  "total_bytes": 123456
}
```

No backend information in manifest. The format is pure data.

## Backend Implementations

All backends maintain their own internal state and implement snapshot/restore
uniformly through the ZIP format.

### InMemoryBackend

State lives in internal dict:

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
        deleted = 0
        to_delete = [
            p for p in self._files
            if p == normalized or p.startswith(normalized + "/")
        ]
        for p in to_delete:
            del self._files[p]
            deleted += 1
        return deleted

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

    def snapshot(self, path: Path) -> FilesystemSnapshotRef:
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
        return FilesystemSnapshotRef(
            archive_path=str(path),
            file_count=len(self._files),
            total_bytes=total_bytes,
        )

    def restore(self, path: Path) -> None:
        self._files.clear()
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]
                    self._files[rel_path] = zf.read(name).decode("utf-8")
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
        self._exec(["python3", "-c", _SNAPSHOT_SCRIPT.format(workdir=self._workdir)])
        self._copy_from_container("/tmp/snapshot.zip", path)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        return FilesystemSnapshotRef(
            archive_path=str(path),
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

State lives in host directory:

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
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return FilesystemSnapshotRef(
            archive_path=str(path),
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

    def _create_backend(self) -> Filesystem:
        """Create a fresh backend instance. Override in subclasses."""
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

    def _create_backend(self) -> Filesystem:
        return InMemoryBackend(self._mounts)

    def _initialize_backend(self) -> None:
        self._backend = self._create_backend()
        self.restore_from_snapshot()


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

    def _create_backend(self) -> Filesystem:
        self._container = _create_container(self._config)
        return ContainerBackend(self._container)

    def _initialize_backend(self) -> None:
        self._backend = self._create_backend()
        self.restore_from_snapshot()


class WorkspaceSection(FilesystemSection):
    def __init__(self, *, session: Session, root: Path | None = None, **kwargs) -> None:
        super().__init__(session=session, **kwargs)
        self._root = root
        super(MarkdownSection, self).__init__(
            title="Workspace",
            key="workspace",
            template=_WORKSPACE_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
        )

    def _create_backend(self) -> Filesystem:
        root = self._root or Path(tempfile.mkdtemp(prefix="wink-workspace-"))
        return HostBackend(root)

    def _initialize_backend(self) -> None:
        self._backend = self._create_backend()
        self.restore_from_snapshot()
```

## Snapshot and Restore Workflow

### Creating a Snapshot

```python
# 1. Export filesystem to ZIP (updates FilesystemSnapshotRef slice)
ref = section.export_snapshot(snapshot_dir / f"{snapshot_id}.fs.zip")

# 2. Snapshot session (includes the updated FilesystemSnapshotRef)
session.snapshot(snapshot_id)
```

### Restoring (Same Backend)

```python
# 1. Rollback session (restores FilesystemSnapshotRef slice)
session.rollback(snapshot_id)

# 2. Restore filesystem from archive
section.restore_from_snapshot()
```

### Restoring with Different Backend

The ZIP format is backend-agnostic. Restore to any implementation:

```python
# Load session from disk
session = load_session(jsonl_path)

# Create section with different backend type
# Original was PodmanSection, now using VfsSection
section = VfsSection(session=session)

# Restore from the same archive - works regardless of original backend
section.restore_from_snapshot()
```

This enables:
- Development with in-memory backend, production with Podman
- Debugging a container session locally with HostBackend
- Migrating between workspace types without data loss

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
        filesystem_section: If provided, exports filesystem before dumping.

    Returns:
        Path to generated JSONL file, or None if no slices to persist.
    """
    target_dir = target if target.is_dir() else target.parent
    session_id = session.tags.get("session_id", str(uuid.uuid4()))

    if filesystem_section is not None:
        archive_path = target_dir / f"{session_id}.fs.zip"
        filesystem_section.export_snapshot(archive_path)

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

def test_snapshot_restore(fs: Filesystem, tmp_path: Path) -> None:
    fs.write("file.txt", "original")
    ref = fs.snapshot(tmp_path / "snapshot.zip")
    assert ref.file_count == 1
    fs.write("file.txt", "modified")
    fs.restore(tmp_path / "snapshot.zip")
    assert fs.read("file.txt").content == "original"

def test_cross_backend_restore(tmp_path: Path) -> None:
    """Snapshot from one backend, restore to another."""
    # Create and populate in-memory backend
    memory = InMemoryBackend()
    memory.write("src/main.py", "print('hello')")
    memory.write("README.md", "# Project")

    # Snapshot
    archive = tmp_path / "snapshot.zip"
    ref = memory.snapshot(archive)
    assert ref.file_count == 2

    # Restore to host backend
    host_root = tmp_path / "workspace"
    host_root.mkdir()
    host = HostBackend(host_root)
    host.restore(archive)

    # Verify contents match
    assert host.read("src/main.py").content == "print('hello')"
    assert host.read("README.md").content == "# Project"
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
│  internal dict │   │  podman exec    │   │  directory I/O  │
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

3. **Single session slice** — `FilesystemSnapshotRef` with archive path only;
   no backend type encoding

4. **Backend-agnostic ZIP format** — Any backend can produce or consume the
   same archive format, enabling cross-backend restore

5. **Internal state management** — Each backend manages its own state
   (dict, container FS, or directory); no special session slice for any backend

6. **Uniform snapshot/restore** — All backends work identically: export to ZIP
   before session snapshot, restore from ZIP after session rollback

7. **Cross-backend restore** — Snapshot from Podman, restore to in-memory;
   enables different workspace/adapter combinations when resuming
