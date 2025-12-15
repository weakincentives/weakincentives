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

## Filesystem Archive Format

All backends use a **common ZIP archive format** for filesystem snapshots. This
provides:

1. **Consistency** — Same format for InMemory, Container, and Host backends
2. **Inspectability** — ZIP is a standard format; can be opened with any tool
3. **Efficiency** — Compression reduces disk usage
4. **Integration** — Works with `dump_session_tree` and `wink debug`

### Archive Structure

```
filesystem.zip
├── manifest.json           # Metadata about the snapshot
├── files/                  # Directory containing all files
│   ├── src/
│   │   └── main.py
│   ├── tests/
│   │   └── test_main.py
│   └── README.md
└── (no directories stored explicitly—only files)
```

### Manifest Format

```json
{
  "version": "1",
  "backend": "InMemoryBackend",
  "created_at": "2024-01-15T10:30:00+00:00",
  "file_count": 42,
  "total_bytes": 123456,
  "root": ""
}
```

### Backend Snapshot Implementation

All backends produce the same ZIP format:

```python
class InMemoryBackend:
    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "version": "1",
                "backend": "InMemoryBackend",
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": len(state.files),
                "total_bytes": sum(f.size_bytes for f in state.files),
                "root": "",
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for file in state.files:
                zf.writestr(f"files/{file.path}", file.content)

    def restore(self, path: Path) -> None:
        with zipfile.ZipFile(path, "r") as zf:
            files = []
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]  # Strip "files/" prefix
                    content = zf.read(name).decode("utf-8")
                    files.append(_FileRecord(
                        path=rel_path,
                        content=content,
                        size_bytes=len(content.encode()),
                    ))
            self._set_state(InMemoryState(files=tuple(sorted(files, key=lambda f: f.path))))


class ContainerBackend:
    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create ZIP in container, stream to host
        self._exec([
            "python3", "-c", f'''
import zipfile, os, json
from datetime import datetime, timezone
with zipfile.ZipFile("/tmp/snapshot.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    file_count = 0
    total_bytes = 0
    for root, dirs, files in os.walk("{self._workdir}"):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, "{self._workdir}")
            content = open(fpath, "rb").read()
            zf.writestr(f"files/{{rel}}", content)
            file_count += 1
            total_bytes += len(content)
    manifest = {{"version": "1", "backend": "ContainerBackend",
                 "created_at": datetime.now(timezone.utc).isoformat(),
                 "file_count": file_count, "total_bytes": total_bytes,
                 "root": "{self._workdir}"}}
    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
'''
        ])
        # Copy ZIP from container to host
        self._copy_from_container("/tmp/snapshot.zip", path)

    def restore(self, path: Path) -> None:
        self._copy_to_container(path, "/tmp/snapshot.zip")
        self._exec(["rm", "-rf", f"{self._workdir}/*"])
        self._exec([
            "python3", "-c", f'''
import zipfile
with zipfile.ZipFile("/tmp/snapshot.zip", "r") as zf:
    for name in zf.namelist():
        if name.startswith("files/") and not name.endswith("/"):
            rel = name[6:]
            target = f"{self._workdir}/{{rel}}"
            import os
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(zf.read(name))
'''
        ])


class HostBackend:
    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            file_count = 0
            total_bytes = 0
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
                "root": str(self._root),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    def restore(self, path: Path) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]
                    target = self._root / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))
```

## Integration with dump_session_tree

`dump_session_tree` persists session state to JSONL. Filesystem snapshots are
stored as **adjacent ZIP files** referenced by a session slice.

### FilesystemSnapshotRef Slice

```python
@dataclass(slots=True, frozen=True)
class FilesystemSnapshotRef:
    """Session slice storing reference to filesystem archive."""
    archive_path: str | None = None  # Relative path to ZIP file
    backend_type: str | None = None  # "InMemoryBackend", "ContainerBackend", etc.
    file_count: int = 0
    total_bytes: int = 0
```

### Dump Flow

When `dump_session_tree(session, target_dir)` is called:

```
dump_session_tree(session, "/snapshots")
         │
         ├─► 1. Session calls snapshot hooks
         │        └─► FilesystemSection._on_snapshot() writes ZIP
         │
         ├─► 2. Hook updates FilesystemSnapshotRef slice with relative path
         │
         ├─► 3. dump_session_tree serializes session to JSONL
         │
         └─► Creates:
              /snapshots/
              ├── abc123.jsonl              # Session state (includes FilesystemSnapshotRef)
              └── abc123.fs.zip             # Filesystem archive
```

### Hook Implementation for dump_session_tree

```python
class FilesystemSection(MarkdownSection):
    def _on_snapshot(self, snapshot_id: str, snapshot_dir: Path) -> None:
        # Write ZIP adjacent to session JSONL
        archive_path = snapshot_dir / f"{snapshot_id}.fs.zip"
        self.filesystem.snapshot(archive_path)

        # Read manifest for metadata
        with zipfile.ZipFile(archive_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))

        # Update session slice with reference (relative path)
        ref = FilesystemSnapshotRef(
            archive_path=f"{snapshot_id}.fs.zip",
            backend_type=manifest["backend"],
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
        )
        self._session.mutate(FilesystemSnapshotRef).seed(ref)
```

### JSONL Entry with Filesystem Reference

```json
{
  "version": "1",
  "created_at": "2024-01-15T10:30:00+00:00",
  "slices": [
    {
      "slice_type": "weakincentives.contrib.tools.filesystem.FilesystemSnapshotRef",
      "items": [{
        "archive_path": "abc123.fs.zip",
        "backend_type": "InMemoryBackend",
        "file_count": 42,
        "total_bytes": 123456
      }]
    },
    {
      "slice_type": "weakincentives.contrib.tools.planning.Plan",
      "items": [...]
    }
  ],
  "tags": {"session_id": "abc123"}
}
```

## Integration with wink debug

The `wink debug` command is enhanced to display filesystem contents from the
adjacent ZIP archive.

### API Routes for Filesystem Inspection

| Route | Method | Description |
|-------|--------|-------------|
| `/api/filesystem/meta` | GET | Filesystem manifest and summary |
| `/api/filesystem/tree` | GET | Directory tree structure |
| `/api/filesystem/file` | GET | Single file content (query: `?path=src/main.py`) |
| `/api/filesystem/search` | GET | Search files (query: `?pattern=*.py&content=def`) |

### Implementation

```python
# In debug_app.py

class SnapshotStore:
    def get_filesystem_archive(self, snapshot_path: Path) -> Path | None:
        """Find adjacent filesystem ZIP for a snapshot."""
        # Check for FilesystemSnapshotRef in loaded snapshot
        ref = self._get_slice(FilesystemSnapshotRef)
        if ref and ref.archive_path:
            archive = snapshot_path.parent / ref.archive_path
            if archive.exists():
                return archive
        return None

    def get_filesystem_manifest(self) -> dict | None:
        archive = self.get_filesystem_archive(self._current_path)
        if archive is None:
            return None
        with zipfile.ZipFile(archive, "r") as zf:
            return json.loads(zf.read("manifest.json"))

    def get_filesystem_tree(self) -> list[dict]:
        archive = self.get_filesystem_archive(self._current_path)
        if archive is None:
            return []
        with zipfile.ZipFile(archive, "r") as zf:
            entries = []
            for name in sorted(zf.namelist()):
                if name.startswith("files/") and not name.endswith("/"):
                    rel_path = name[6:]
                    info = zf.getinfo(name)
                    entries.append({
                        "path": rel_path,
                        "size_bytes": info.file_size,
                        "compressed_size": info.compress_size,
                    })
            return entries

    def get_filesystem_file(self, path: str) -> str | None:
        archive = self.get_filesystem_archive(self._current_path)
        if archive is None:
            return None
        with zipfile.ZipFile(archive, "r") as zf:
            try:
                return zf.read(f"files/{path}").decode("utf-8")
            except KeyError:
                return None
```

### Web UI Enhancement

The debug UI shows a "Filesystem" tab when `FilesystemSnapshotRef` is present:

```
┌─────────────────────────────────────────────────────────────────┐
│ wink debug - Session abc123                                     │
├─────────────────────────────────────────────────────────────────┤
│ [Slices] [Events] [Filesystem] [Raw]                            │
├─────────────────────────────────────────────────────────────────┤
│ Filesystem: InMemoryBackend                                     │
│ Files: 42 | Total: 120.5 KB | Archive: abc123.fs.zip           │
├─────────────────────────────────────────────────────────────────┤
│ 📁 src/                                                         │
│   📄 main.py (2.3 KB)                                          │
│   📄 utils.py (1.1 KB)                                         │
│ 📁 tests/                                                       │
│   📄 test_main.py (3.2 KB)                                     │
│ 📄 README.md (0.5 KB)                                          │
├─────────────────────────────────────────────────────────────────┤
│ [Selected: src/main.py]                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1  def main():                                              │ │
│ │ 2      print("Hello, world!")                               │ │
│ │ 3                                                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Lazy Loading at Prompt Rendering Time

Filesystem state is loaded **lazily** when the section's `filesystem` property
is first accessed, not at section construction. This enables:

1. **Lightweight session loading** — Session loads without reading large archives
2. **Deferred initialization** — Backend created only when needed
3. **Prompt rendering triggers load** — Filesystem available when tools execute

### Lazy Backend Pattern

```python
class FilesystemSection(MarkdownSection):
    def __init__(self, *, session: Session, snapshot_dir: Path | None = None) -> None:
        self._session = session
        self._snapshot_dir = snapshot_dir or Path(tempfile.gettempdir()) / "wink-fs-snapshots"
        self._backend: Filesystem | None = None
        self._backend_initialized = False

        # Register hooks
        session.register_snapshot_hook(
            on_snapshot=self._on_snapshot,
            on_rollback=self._on_rollback,
        )

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
        raise NotImplementedError  # Subclasses implement


class VfsSection(FilesystemSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        super().__init__(session=session)
        self._mounts = mounts

    def _initialize_backend(self) -> None:
        self._backend = InMemoryBackend(self._session, self._mounts)

        # Check for existing snapshot to restore
        ref = self._session.query(FilesystemSnapshotRef).latest()
        if ref and ref.archive_path:
            archive_path = self._resolve_archive_path(ref.archive_path)
            if archive_path.exists():
                self._backend.restore(archive_path)


class PodmanSection(FilesystemSection):
    def __init__(self, *, session: Session, config: PodmanConfig) -> None:
        super().__init__(session=session)
        self._config = config
        self._container: Container | None = None

    def _initialize_backend(self) -> None:
        # Container created lazily
        self._container = _create_container(self._config)
        self._backend = ContainerBackend(self._container)

        # Restore from snapshot if available
        ref = self._session.query(FilesystemSnapshotRef).latest()
        if ref and ref.archive_path:
            archive_path = self._resolve_archive_path(ref.archive_path)
            if archive_path.exists():
                self._backend.restore(archive_path)
```

### Load Timeline

```
Session.load("/path/to/session.jsonl")
         │
         └─► Slices restored (including FilesystemSnapshotRef)
              But NO filesystem loaded yet


section = VfsSection(session=session, mounts=mounts)
         │
         └─► Hooks registered, but backend NOT created yet


prompt.render(params)
         │
         └─► Sections rendered, but filesystem NOT accessed


adapter.evaluate(prompt, session=session)
         │
         └─► Tool invocation triggers context.filesystem access
                   │
                   └─► VfsSection.filesystem property called
                            │
                            └─► _initialize_backend()
                                     │
                                     ├─► Create InMemoryBackend
                                     │
                                     └─► Restore from abc123.fs.zip
                                              │
                                              └─► Backend now populated
```

### Benefits

1. **Fast session load** — Loading a JSONL doesn't require reading potentially
   large ZIP archives
2. **Memory efficient** — Filesystem not in memory until needed
3. **Resumable sessions** — Load session, inspect slices in `wink debug`,
   then continue execution with filesystem restored on-demand

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
    snapshot_path = tmp_path / "snapshot.zip"
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

### Backend Comparison

| Aspect | InMemoryBackend | ContainerBackend | HostBackend |
|--------|-----------------|------------------|-------------|
| State location | Session slice (`InMemoryState`) | Container filesystem | Host temp directory |
| Archive format | ZIP (same as others) | ZIP | ZIP |
| Session coupling | Via hooks | Via hooks | Via hooks |
| Lazy loading | Yes | Yes | Yes |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Tool Layer                                  │
│   ls, read_file, write_file, edit_file, glob, grep, rm                  │
│                                  │                                       │
│                       context.filesystem                                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────┐
│                         Filesystem Protocol                              │
│   read, write, delete, list, glob, grep, snapshot, restore              │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  InMemoryBackend  │   │  ContainerBackend │   │    HostBackend    │
│                   │   │                   │   │                   │
│ InMemoryState     │   │ podman exec       │   │ temp dir I/O      │
│ session slice     │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                          ZIP Archive Format
                      (manifest.json + files/*)
```

### Key Design Decisions

1. **Filesystem Protocol** — 8 methods: `read`, `write`, `delete`, `list`,
   `glob`, `grep`, `snapshot`, `restore`

2. **ToolContext carries filesystem** — Not `session.query()`, because
   `Session.install()` only accepts frozen dataclass slices

3. **Session hooks for coupling** — `session.register_snapshot_hook()` called
   during section init; hooks fire on `session.snapshot()`/`rollback()`

4. **ZIP archive format** — All backends produce identical ZIP structure with
   `manifest.json` + `files/*`; enables inspection in any tool

5. **FilesystemSnapshotRef slice** — Session stores relative path to ZIP;
   `dump_session_tree` writes JSONL + adjacent ZIP

6. **Lazy loading** — Backend created on first `section.filesystem` access;
   restores from ZIP if `FilesystemSnapshotRef` present

7. **wink debug integration** — New `/api/filesystem/*` routes expose archive
   contents; UI shows file tree and content viewer
