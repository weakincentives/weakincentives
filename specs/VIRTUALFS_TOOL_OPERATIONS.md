# VirtualFilesystem Tool Operations Specification

## Status

**Draft** — Proposed refactoring for unified filesystem abstraction.

## Problem Statement

Three independent workspace backends exist with overlapping functionality but no
common interface:

| Backend | Location | State Model | Tool Surface |
|---------|----------|-------------|--------------|
| VirtualFileSystem | In-memory | Session slice + reducers | 7 tools (ls, read_file, write_file, edit_file, glob, grep, rm) |
| PodmanSandboxSection | Container | Container filesystem | Same 7 tools + shell_execute, evaluate_python |
| ClaudeAgentWorkspaceSection | Temp directory | Direct OS I/O | None (relies on SDK native tools) |

This creates problems:

1. **Tool handlers are duplicated** — Podman re-implements all VFS tool logic for
   container operations
2. **No polymorphism** — A tool cannot operate on "the filesystem" without knowing
   which backend is active
3. **Session integration is inconsistent** — VFS uses reducer-based mutations while
   Podman uses direct I/O
4. **Testing burden** — Each backend requires separate test coverage for identical
   operations

## Design Principles

1. **Protocol over implementation** — Define what a filesystem *does*, not how
2. **Session as registry** — Retrieve the active filesystem from session context
3. **Operations return results, not events** — Tools should work with any backend
4. **Backend chooses mutation strategy** — In-memory uses reducers; Podman uses
   shell; SDK uses native tools

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tool Layer                               │
│  ls, read_file, write_file, edit_file, glob, grep, rm           │
│                              │                                   │
│                    FilesystemProtocol                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ InMemoryFS    │    │ ContainerFS     │    │ HostFS              │
│ (VFS slice)   │    │ (Podman exec)   │    │ (temp dir I/O)      │
│               │    │                 │    │                     │
│ Session-scoped│    │ Container-scope │    │ Temp dir lifecycle  │
│ Reducer-based │    │ Direct exec     │    │ Direct I/O          │
└───────────────┘    └─────────────────┘    └─────────────────────┘
```

## Core Protocol

```python
from typing import Protocol, Iterator
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class FileEntry:
    """Unified file metadata across backends."""
    path: str                    # Normalized POSIX path
    kind: Literal["file", "directory"]
    size_bytes: int | None       # None for directories
    content_hash: str | None     # Optional content fingerprint

@dataclass(slots=True, frozen=True)
class FileContent:
    """File content with optional pagination."""
    path: str
    content: str
    total_lines: int
    offset: int
    limit: int

@dataclass(slots=True, frozen=True)
class WriteResult:
    """Result of a write operation."""
    path: str
    bytes_written: int
    created: bool                # True if file was created, False if modified

@dataclass(slots=True, frozen=True)
class GlobResult:
    """Matches from a glob pattern."""
    pattern: str
    matches: tuple[FileEntry, ...]

@dataclass(slots=True, frozen=True)
class GrepResult:
    """Matches from a regex search."""
    pattern: str
    matches: tuple[GrepMatch, ...]

@dataclass(slots=True, frozen=True)
class GrepMatch:
    """Single grep match with context."""
    path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


class FilesystemProtocol(Protocol):
    """Unified filesystem operations for tool handlers."""

    @property
    def root(self) -> str:
        """Logical root path (e.g., "/", "/workspace")."""
        ...

    # ─── Read Operations ───────────────────────────────────────────

    def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        """List entries in a directory."""
        ...

    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> FileContent:
        """Read file content with optional pagination."""
        ...

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...

    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        ...

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        ...

    # ─── Write Operations ──────────────────────────────────────────

    def write_file(
        self,
        path: str,
        content: str,
        *,
        create_only: bool = False,
    ) -> WriteResult:
        """Write content to a file.

        Args:
            path: Target file path.
            content: UTF-8 content to write.
            create_only: If True, fail if file exists.
        """
        ...

    def delete(self, path: str, *, recursive: bool = False) -> int:
        """Delete file or directory. Returns count of deleted entries."""
        ...

    def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create directory."""
        ...

    # ─── Search Operations ─────────────────────────────────────────

    def glob(self, pattern: str, *, path: str | None = None) -> GlobResult:
        """Find files matching glob pattern."""
        ...

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_filter: str | None = None,
        max_matches: int = 100,
    ) -> GrepResult:
        """Search file contents with regex."""
        ...
```

## Session Integration

### Registration

Each backend registers itself as the active filesystem:

```python
# In section __init__
session.install(
    FilesystemProtocol,
    initial=lambda: self._create_backend(),
)
```

### Retrieval in Tool Handlers

Tools retrieve the filesystem from session, not from section:

```python
def read_file_handler(
    params: ReadFileParams,
    *,
    context: ToolContext,
) -> ToolResult[FileContent]:
    fs = context.session.query(FilesystemProtocol).latest()
    if fs is None:
        return ToolResult(
            message="No filesystem backend configured.",
            value=None,
            success=False,
        )
    content = fs.read_file(params.path, offset=params.offset, limit=params.limit)
    return ToolResult(message=_format_content(content), value=content)
```

### Backend-Specific State

Backends that need session state (like VFS reducers) manage it internally:

```python
class InMemoryFilesystem:
    """In-memory VFS backed by session state."""

    def __init__(self, session: Session) -> None:
        self._session = session
        session.install(VirtualFileSystem, initial=VirtualFileSystem)
        session.mutate(VirtualFileSystem).register(WriteFile, VirtualFileSystem.handle_write)
        session.mutate(VirtualFileSystem).register(DeleteEntry, VirtualFileSystem.handle_delete)

    def write_file(self, path: str, content: str, *, create_only: bool = False) -> WriteResult:
        vfs_path = _normalize_path(path)
        event = WriteFile(path=vfs_path, content=content, mode="create" if create_only else "overwrite")
        self._session.mutate(VirtualFileSystem).dispatch(event)
        return WriteResult(path=path, bytes_written=len(content.encode()), created=create_only)
```

## Tool Suite Simplification

### Before: Section-Bound Tools

```python
# vfs.py - 7 tools defined, each bound to VfsToolsSection
# podman.py - Same 7 tools re-implemented for container

class VfsToolsSection:
    @property
    def tools(self) -> tuple[Tool, ...]:
        return (self._ls, self._read_file, ...)  # Section-specific

class PodmanSandboxSection:
    @property
    def tools(self) -> tuple[Tool, ...]:
        return (self._ls, self._read_file, ...)  # Duplicated implementation
```

### After: Protocol-Based Tools

```python
# filesystem_tools.py - Single set of tools using protocol

def ls_handler(params: LsParams, *, context: ToolContext) -> ToolResult[tuple[FileEntry, ...]]:
    fs = _require_filesystem(context)
    entries = fs.list_dir(params.path)
    return ToolResult(message=_format_entries(entries), value=entries)

ls_tool = Tool(
    name="ls",
    description="List directory contents.",
    handler=ls_handler,
)

# All 7 tools defined once, work with any FilesystemProtocol backend
FILESYSTEM_TOOLS = (ls_tool, read_file_tool, write_file_tool, edit_file_tool, glob_tool, grep_tool, rm_tool)
```

### Section Changes

Sections provide the backend, not the tools:

```python
class VfsToolsSection(MarkdownSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount], ...):
        # Install InMemoryFilesystem as the backend
        self._backend = InMemoryFilesystem(session)
        session.install(FilesystemProtocol, initial=lambda: self._backend)

        super().__init__(
            title="Virtual Filesystem",
            template=_VFS_TEMPLATE,
            tools=FILESYSTEM_TOOLS,  # Shared tool definitions
        )

class PodmanSandboxSection(MarkdownSection):
    def __init__(self, *, session: Session, config: PodmanSandboxConfig, ...):
        self._backend = ContainerFilesystem(session, config)
        session.install(FilesystemProtocol, initial=lambda: self._backend)

        super().__init__(
            title="Podman Sandbox",
            template=_PODMAN_TEMPLATE,
            tools=(*FILESYSTEM_TOOLS, shell_execute_tool, evaluate_python_tool),
        )
```

## Backend Implementations

### InMemoryFilesystem

Wraps existing `VirtualFileSystem` slice with protocol interface:

```python
class InMemoryFilesystem:
    def __init__(self, session: Session, *, mounts: Sequence[HostMount] = ()) -> None:
        self._session = session
        self._initialize_state(mounts)

    def _snapshot(self) -> VirtualFileSystem:
        return self._session.query(VirtualFileSystem).latest() or VirtualFileSystem()

    def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        vfs = self._snapshot()
        file = _find_file(vfs.files, path)
        if file is None:
            raise FileNotFoundError(path)
        lines = file.content.splitlines(keepends=True)
        # Apply pagination...
        return FileContent(...)

    def write_file(self, path: str, content: str, *, create_only: bool = False) -> WriteResult:
        mode = "create" if create_only else "overwrite"
        event = WriteFile(path=_to_vfs_path(path), content=content, mode=mode)
        self._session.mutate(VirtualFileSystem).dispatch(event)
        return WriteResult(path=path, bytes_written=len(content.encode()), created=...)
```

### ContainerFilesystem

Executes operations via podman exec:

```python
class ContainerFilesystem:
    def __init__(self, container: PodmanContainer, workdir: str = "/workspace") -> None:
        self._container = container
        self._workdir = workdir

    @property
    def root(self) -> str:
        return self._workdir

    def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        full_path = self._resolve(path)
        result = self._container.exec(["cat", full_path])
        # Parse and paginate...
        return FileContent(...)

    def write_file(self, path: str, content: str, *, create_only: bool = False) -> WriteResult:
        full_path = self._resolve(path)
        script = _write_script(full_path, content, create_only)
        self._container.exec(["python3", "-c", script])
        return WriteResult(...)

    def _resolve(self, path: str) -> str:
        """Resolve path relative to workspace root."""
        if path.startswith("/"):
            return path
        return f"{self._workdir}/{path}"
```

### HostFilesystem

Direct OS I/O for Claude Agent SDK workspace:

```python
class HostFilesystem:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> str:
        return str(self._root)

    def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        full_path = self._resolve(path)
        content = full_path.read_text()
        # Paginate...
        return FileContent(...)

    def write_file(self, path: str, content: str, *, create_only: bool = False) -> WriteResult:
        full_path = self._resolve(path)
        if create_only and full_path.exists():
            raise FileExistsError(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return WriteResult(...)

    def _resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            resolved = Path(path).resolve()
        else:
            resolved = (self._root / path).resolve()
        # Security check
        if not resolved.is_relative_to(self._root):
            raise SecurityError(f"Path escapes root: {path}")
        return resolved
```

## Claude Agent SDK Integration

The Claude Agent SDK adapter uses SDK-native tools (Read, Write, Edit, etc.) which
bypass this abstraction entirely. However, `HostFilesystem` enables:

1. **Prompt-internal tools** — Custom tools added to prompts can use `HostFilesystem`
2. **State synchronization** — Session can track what files have been modified
3. **Unified testing** — Same test harness works across backends

```python
class ClaudeAgentWorkspaceSection(MarkdownSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount], ...):
        self._temp_dir, self._previews = _create_workspace(mounts, ...)
        self._backend = HostFilesystem(self._temp_dir)
        session.install(FilesystemProtocol, initial=lambda: self._backend)

        # No tools — SDK uses native tools
        super().__init__(
            title="Workspace",
            template=_render_template(self._previews),
            tools=(),
        )

    @property
    def filesystem(self) -> FilesystemProtocol:
        """Expose filesystem for custom prompt tools."""
        return self._backend
```

## Migration Path

### Phase 1: Define Protocol and Results

1. Create `FilesystemProtocol` in `weakincentives.contrib.tools.filesystem`
2. Define unified result dataclasses (`FileEntry`, `FileContent`, `WriteResult`, etc.)
3. Keep existing tool implementations unchanged

### Phase 2: Implement Backends

1. Create `InMemoryFilesystem` wrapping existing `VirtualFileSystem` logic
2. Create `ContainerFilesystem` extracting Podman file operations
3. Create `HostFilesystem` for temp directory I/O
4. Each backend implements `FilesystemProtocol`

### Phase 3: Unify Tool Handlers

1. Create shared tool handlers in `filesystem_tools.py`
2. Tools retrieve backend via `context.session.query(FilesystemProtocol)`
3. Update sections to register backends and use shared tools

### Phase 4: Deprecate Old Tools

1. Mark section-specific tool implementations as deprecated
2. Update documentation and examples
3. Remove deprecated code in next minor release

## Backward Compatibility

- **Tool names unchanged** — `ls`, `read_file`, etc. keep same names
- **Parameter schemas unchanged** — Existing param dataclasses remain compatible
- **Result rendering unchanged** — `render()` methods preserved
- **Session API unchanged** — `query()` and `mutate()` patterns preserved

Sections that previously provided tools continue to work; they just delegate to
the unified implementation.

## Testing Strategy

### Unit Tests

```python
@pytest.fixture
def in_memory_fs(session: Session) -> InMemoryFilesystem:
    return InMemoryFilesystem(session)

@pytest.fixture
def host_fs(tmp_path: Path) -> HostFilesystem:
    return HostFilesystem(tmp_path)

class TestFilesystemProtocol:
    """Parameterized tests run against all backends."""

    @pytest.fixture(params=["in_memory", "host"])
    def fs(self, request, in_memory_fs, host_fs) -> FilesystemProtocol:
        return {"in_memory": in_memory_fs, "host": host_fs}[request.param]

    def test_write_and_read(self, fs: FilesystemProtocol) -> None:
        fs.write_file("test.txt", "hello")
        content = fs.read_file("test.txt")
        assert content.content == "hello"

    def test_list_dir(self, fs: FilesystemProtocol) -> None:
        fs.write_file("a/b.txt", "")
        entries = fs.list_dir("a")
        assert len(entries) == 1
        assert entries[0].path == "a/b.txt"
```

### Integration Tests

Podman tests require container runtime and run in CI with appropriate markers.

## Open Questions

1. **Edit operation** — Should `edit_file` (string replacement) be part of the
   protocol or remain a higher-level tool that combines `read_file` + `write_file`?

   *Recommendation*: Keep `edit_file` as a tool-layer operation that uses
   `read_file` and `write_file` from the protocol. This keeps the protocol
   minimal and avoids duplicating edit logic in each backend.

2. **Binary files** — Current VFS is text-only. Should the protocol support
   `read_bytes` / `write_bytes`?

   *Recommendation*: Defer binary support. All current use cases are text-focused.
   Add `bytes` variants to protocol in future if needed.

3. **Event publishing** — Should write operations publish `FileWritten` events
   to the session event bus?

   *Recommendation*: Yes. Backends publish `FileWritten(path, bytes_written)` after
   successful writes. This enables telemetry and debugging without coupling
   tools to specific backends.

4. **Concurrency** — How should concurrent writes be handled?

   *Recommendation*: Backends are not thread-safe by default. Document that
   concurrent tool invocations require external synchronization. This matches
   current VFS behavior.

## Summary

This refactoring unifies three divergent workspace implementations behind a
single `FilesystemProtocol`. Benefits:

- **Single tool implementation** — 7 tools defined once, not three times
- **Polymorphic handlers** — Tools work with any registered backend
- **Cleaner session integration** — Filesystem retrieved from session context
- **Easier testing** — Parameterized tests cover all backends
- **Future extensibility** — New backends (S3, Git worktree, etc.) just implement
  the protocol

The migration is incremental and maintains backward compatibility throughout.
