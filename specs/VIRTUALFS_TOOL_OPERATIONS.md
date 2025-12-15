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
```

Six methods. No `exists()`, `is_file()`, `mkdir()`—these are derivable from
`read()` (catch exception), `list()` (check kind), and `write()` (implicit).

## Session Registration

Sections create a backend and register it as the active filesystem:

```python
class VfsSection(MarkdownSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._backend = InMemoryBackend(mounts)
        session.install(Filesystem, initial=lambda: self._backend)
        super().__init__(title="Filesystem", template=..., tools=FILESYSTEM_TOOLS)
```

The backend object lives for the session's lifetime. State is internal to the
backend—no separate session slices required.

## Tool Handlers

Tools retrieve the filesystem from context and operate generically:

```python
def read_file_handler(params: ReadParams, *, context: ToolContext) -> ToolResult[FileContent]:
    fs: Filesystem = context.session.query(Filesystem).latest()
    content = fs.read(params.path, offset=params.offset, limit=params.limit)
    return ToolResult(message=_format(content), value=content)

def write_file_handler(params: WriteParams, *, context: ToolContext) -> ToolResult[int]:
    fs: Filesystem = context.session.query(Filesystem).latest()
    bytes_written = fs.write(params.path, params.content)
    return ToolResult(message=f"Wrote {bytes_written} bytes to {params.path}", value=bytes_written)

# Single tool definition used by all sections
FILESYSTEM_TOOLS = (
    Tool(name="ls", handler=ls_handler, ...),
    Tool(name="read_file", handler=read_file_handler, ...),
    Tool(name="write_file", handler=write_file_handler, ...),
    Tool(name="edit_file", handler=edit_file_handler, ...),  # Combines read + write
    Tool(name="glob", handler=glob_handler, ...),
    Tool(name="grep", handler=grep_handler, ...),
    Tool(name="rm", handler=rm_handler, ...),
)
```

## Backend Implementations

### InMemoryBackend

Holds file state directly. No separate session slice—the backend *is* the state:

```python
@dataclass
class _File:
    path: str
    content: str
    size_bytes: int

class InMemoryBackend:
    def __init__(self, mounts: Sequence[HostMount] = ()) -> None:
        self._files: dict[str, _File] = {}
        self._hydrate_mounts(mounts)

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        file = self._files.get(_normalize(path))
        if file is None:
            raise FileNotFoundError(path)
        return _paginate(file.content, offset, limit)

    def write(self, path: str, content: str) -> int:
        normalized = _normalize(path)
        size = len(content.encode())
        self._files[normalized] = _File(path=normalized, content=content, size_bytes=size)
        return size

    def delete(self, path: str) -> int:
        normalized = _normalize(path)
        # Delete exact match or prefix (directory)
        to_delete = [p for p in self._files if p == normalized or p.startswith(normalized + "/")]
        for p in to_delete:
            del self._files[p]
        return len(to_delete)

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        prefix = _normalize(path)
        entries = {}
        for file_path, file in self._files.items():
            if not file_path.startswith(prefix):
                continue
            relative = file_path[len(prefix):].lstrip("/")
            top_segment = relative.split("/")[0]
            full_path = f"{prefix}/{top_segment}".lstrip("/")
            if "/" in relative:
                entries[full_path] = FileEntry(path=full_path, kind="directory")
            else:
                entries[full_path] = FileEntry(path=full_path, kind="file", size_bytes=file.size_bytes)
        return tuple(entries.values())

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        return tuple(
            FileEntry(path=f.path, kind="file", size_bytes=f.size_bytes)
            for f in self._files.values()
            if fnmatch.fnmatch(f.path, pattern)
        )

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        regex = re.compile(pattern)
        matches = []
        for file in self._files.values():
            if file_glob and not fnmatch.fnmatch(file.path, file_glob):
                continue
            for i, line in enumerate(file.content.splitlines(), 1):
                if regex.search(line):
                    matches.append(GrepMatch(path=file.path, line_number=i, line_content=line))
        return tuple(matches)
```

The backend persists for the session's lifetime via `session.install(Filesystem, ...)`.
No reducers, no events, no separate VirtualFileSystem dataclass.

### ContainerBackend

Executes operations via container exec:

```python
class ContainerBackend:
    def __init__(self, container: Container, workdir: str = "/workspace") -> None:
        self._container = container
        self._workdir = workdir

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        full = self._resolve(path)
        result = self._exec(["cat", full])
        return _paginate(result.stdout, offset, limit)

    def write(self, path: str, content: str) -> int:
        full = self._resolve(path)
        self._exec(["mkdir", "-p", str(Path(full).parent)])
        self._exec(["tee", full], stdin=content)
        return len(content.encode())

    def delete(self, path: str) -> int:
        full = self._resolve(path)
        result = self._exec(["rm", "-rf", full])
        return 1 if result.returncode == 0 else 0

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        full = self._resolve(path) if path else self._workdir
        result = self._exec(["find", full, "-maxdepth", "1", "-printf", "%y %s %P\n"])
        return _parse_find_output(result.stdout)

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        result = self._exec(["find", self._workdir, "-name", pattern, "-printf", "%y %s %P\n"])
        return _parse_find_output(result.stdout)

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        cmd = ["grep", "-rn", pattern, self._workdir]
        if file_glob:
            cmd.extend(["--include", file_glob])
        result = self._exec(cmd)
        return _parse_grep_output(result.stdout)

    def _resolve(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self._workdir}/{path}"

    def _exec(self, cmd: list[str], stdin: str | None = None) -> CompletedProcess:
        return self._container.exec(cmd, stdin=stdin)
```

### HostBackend

Direct OS I/O for temp directory workspaces:

```python
class HostBackend:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> FileContent:
        full = self._resolve(path)
        content = full.read_text()
        return _paginate(content, offset, limit)

    def write(self, path: str, content: str) -> int:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return len(content.encode())

    def delete(self, path: str) -> int:
        full = self._resolve(path)
        if full.is_dir():
            count = sum(1 for _ in full.rglob("*"))
            shutil.rmtree(full)
            return count
        elif full.exists():
            full.unlink()
            return 1
        return 0

    def list(self, path: str = "") -> tuple[FileEntry, ...]:
        full = self._resolve(path) if path else self._root
        entries = []
        for item in full.iterdir():
            kind = "directory" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else None
            entries.append(FileEntry(path=str(item.relative_to(self._root)), kind=kind, size_bytes=size))
        return tuple(entries)

    def glob(self, pattern: str) -> tuple[FileEntry, ...]:
        return tuple(
            FileEntry(path=str(p.relative_to(self._root)), kind="file", size_bytes=p.stat().st_size)
            for p in self._root.glob(pattern) if p.is_file()
        )

    def grep(self, pattern: str, *, file_glob: str | None = None) -> tuple[GrepMatch, ...]:
        regex = re.compile(pattern)
        matches = []
        for path in (self._root.glob(file_glob) if file_glob else self._root.rglob("*")):
            if not path.is_file():
                continue
            try:
                for i, line in enumerate(path.read_text().splitlines(), 1):
                    if regex.search(line):
                        matches.append(GrepMatch(
                            path=str(path.relative_to(self._root)),
                            line_number=i,
                            line_content=line,
                        ))
            except UnicodeDecodeError:
                continue
        return tuple(matches)

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes root: {path}")
        return resolved
```

## Section Definitions

Sections create backend, register it, and attach tools:

```python
class VfsSection(MarkdownSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._backend = InMemoryBackend(mounts)
        session.install(Filesystem, initial=lambda: self._backend)
        super().__init__(
            title="Virtual Filesystem",
            key="vfs",
            template=_VFS_TEMPLATE,
            tools=FILESYSTEM_TOOLS,
        )

class PodmanSection(MarkdownSection):
    def __init__(self, *, session: Session, config: PodmanConfig) -> None:
        self._container = _create_container(config)
        self._backend = ContainerBackend(self._container)
        session.install(Filesystem, initial=lambda: self._backend)
        super().__init__(
            title="Sandbox",
            key="podman",
            template=_PODMAN_TEMPLATE,
            tools=(*FILESYSTEM_TOOLS, shell_tool, eval_tool),
        )

class WorkspaceSection(MarkdownSection):
    def __init__(self, *, session: Session, mounts: Sequence[HostMount] = ()) -> None:
        self._temp_dir = _create_workspace(mounts)
        self._backend = HostBackend(self._temp_dir)
        session.install(Filesystem, initial=lambda: self._backend)
        super().__init__(
            title="Workspace",
            key="workspace",
            template=_WORKSPACE_TEMPLATE,
            tools=(),  # SDK uses native tools
        )
```

## edit_file Implementation

`edit_file` is a tool-layer operation combining read + write:

```python
def edit_file_handler(params: EditParams, *, context: ToolContext) -> ToolResult[str]:
    fs: Filesystem = context.session.query(Filesystem).latest()

    # Read current content
    try:
        content = fs.read(params.path)
    except FileNotFoundError:
        return ToolResult(message=f"File not found: {params.path}", value=None, success=False)

    # Apply edit
    old = params.old_string
    new = params.new_string
    if old not in content.content:
        return ToolResult(message=f"String not found: {old!r}", value=None, success=False)

    if params.replace_all:
        updated = content.content.replace(old, new)
    else:
        updated = content.content.replace(old, new, 1)

    # Write back
    fs.write(params.path, updated)
    return ToolResult(message=f"Edited {params.path}", value=updated)
```

## Testing

Parameterized tests cover all backends:

```python
@pytest.fixture(params=["memory", "host"])
def fs(request, tmp_path) -> Filesystem:
    if request.param == "memory":
        return InMemoryBackend()
    return HostBackend(tmp_path)

def test_write_read_roundtrip(fs: Filesystem) -> None:
    fs.write("test.txt", "hello")
    content = fs.read("test.txt")
    assert content.content == "hello"

def test_delete(fs: Filesystem) -> None:
    fs.write("a/b.txt", "x")
    assert fs.delete("a") == 1
    with pytest.raises(FileNotFoundError):
        fs.read("a/b.txt")

def test_glob(fs: Filesystem) -> None:
    fs.write("src/main.py", "")
    fs.write("src/test.py", "")
    fs.write("docs/readme.md", "")
    matches = fs.glob("src/*.py")
    assert len(matches) == 2

def test_grep(fs: Filesystem) -> None:
    fs.write("code.py", "def foo():\n    return 42\n")
    matches = fs.grep(r"return \d+")
    assert len(matches) == 1
    assert matches[0].line_number == 2
```

Container backend tests are integration tests requiring Podman runtime.

## Summary

The `Filesystem` protocol provides six methods: `read`, `write`, `delete`,
`list`, `glob`, `grep`. Three backends implement this protocol for different
execution contexts. Tool handlers operate generically via
`context.session.query(Filesystem)`. Sections register backends and attach
shared tools.

This eliminates tool duplication and enables composition across backends.
