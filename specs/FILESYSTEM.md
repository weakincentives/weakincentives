# Filesystem Protocol Specification

## Purpose

Unified `Filesystem` protocol that tools access through `ToolContext`. Abstracts
over workspace backends (in-memory VFS, Podman containers, host filesystem) so
handlers can perform file operations without coupling to storage implementation.

**Implementation:** `src/weakincentives/filesystem/`

## Guiding Principles

- **Single access pattern**: Tools use one protocol regardless of backend
- **Context-scoped**: Filesystem on `ToolContext` and `Prompt`, not global
- **Immutable snapshots**: Reads return immutable data; writes may be journaled
- **Backend-managed state**: Backends manage persistence; no session slice coupling
- **Backend-agnostic tools**: Handlers call `context.filesystem.*` and remain portable
- **Bytes-first**: Core operations work with raw bytes; text is a higher-level concern
- **Streaming by default**: Operations use iterators for fixed memory footprint
- **Lazy encoding**: UTF-8 conversion deferred until explicitly requested

## Architecture Overview

The filesystem is structured in three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Convenience Layer                        │
│   read(), write() - text-oriented, buffered convenience      │
├─────────────────────────────────────────────────────────────┤
│                     Streaming Layer                          │
│   open_read(), open_write() - chunk-based byte streams       │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                              │
│   exists(), stat(), list(), glob(), grep(), delete(), mkdir()│
└─────────────────────────────────────────────────────────────┘
```

## Protocol Definition

### Stream Types

| Type | Description |
|------|-------------|
| `ByteReader` | Context manager yielding byte chunks via iteration |
| `ByteWriter` | Context manager accepting byte chunks via `write()` |
| `TextReader` | Wrapper over `ByteReader` with lazy UTF-8 decoding |
| `TextWriter` | Wrapper over `ByteWriter` with UTF-8 encoding |

### ByteReader Protocol

```python
@runtime_checkable
class ByteReader(Protocol):
    """Streaming byte reader with fixed memory footprint."""

    @property
    def path(self) -> str:
        """Path being read."""
        ...

    @property
    def size(self) -> int:
        """Total file size in bytes."""
        ...

    @property
    def position(self) -> int:
        """Current read position in bytes."""
        ...

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes. Returns empty bytes at EOF.

        Args:
            size: Maximum bytes to read. -1 means read to EOF.
        """
        ...

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position. Returns new absolute position.

        Args:
            offset: Offset relative to whence.
            whence: 0=start, 1=current, 2=end.
        """
        ...

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks of default size (64KB)."""
        ...

    def chunks(self, size: int) -> Iterator[bytes]:
        """Iterate over chunks of specified size."""
        ...

    def __enter__(self) -> ByteReader: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

### ByteWriter Protocol

```python
@runtime_checkable
class ByteWriter(Protocol):
    """Streaming byte writer with fixed memory footprint."""

    @property
    def path(self) -> str:
        """Path being written."""
        ...

    @property
    def bytes_written(self) -> int:
        """Total bytes written so far."""
        ...

    def write(self, data: bytes) -> int:
        """Write bytes, returns number of bytes written."""
        ...

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks, returns total bytes written."""
        ...

    def __enter__(self) -> ByteWriter: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

### TextReader Protocol

```python
@runtime_checkable
class TextReader(Protocol):
    """Line-oriented text reader with lazy UTF-8 decoding."""

    @property
    def path(self) -> str:
        """Path being read."""
        ...

    @property
    def encoding(self) -> str:
        """Text encoding (always 'utf-8')."""
        ...

    @property
    def line_number(self) -> int:
        """Current 0-indexed line number."""
        ...

    def readline(self) -> str:
        """Read next line including newline. Empty string at EOF."""
        ...

    def read(self, size: int = -1) -> str:
        """Read up to size characters. Empty string at EOF."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over lines (including newlines)."""
        ...

    def lines(self, *, strip: bool = False) -> Iterator[str]:
        """Iterate over lines with optional stripping."""
        ...

    def __enter__(self) -> TextReader: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

### Result Types

| Type | Key Properties |
|------|----------------|
| `FileStat` | `path`, `is_file`, `is_directory`, `size_bytes`, `created_at`, `modified_at` |
| `FileEntry` | `name`, `path`, `is_file`, `is_directory` |
| `GlobMatch` | `path`, `is_file` |
| `GrepMatch` | `path`, `line_number`, `line_content`, `match_start`, `match_end` |
| `WriteResult` | `path`, `bytes_written`, `mode` |

### Filesystem Protocol

**Streaming Operations (Primary API):**

| Method | Returns | Description |
|--------|---------|-------------|
| `open_read(path)` | `ByteReader` | Open file for streaming byte reads |
| `open_write(path, mode, create_parents)` | `ByteWriter` | Open file for streaming byte writes |
| `open_text(path, encoding)` | `TextReader` | Open file for streaming text reads |

**Convenience Operations (Built on Streaming):**

| Method | Parameters | Description |
|--------|------------|-------------|
| `read_bytes(path, offset, limit)` | `path`, `offset`, `limit` | Read bytes with pagination |
| `read(path, offset, limit, encoding)` | `path`, `offset`, `limit`, `encoding` | Read text with line pagination |
| `write_bytes(path, content, mode, create_parents)` | `path`, `content`, `mode`, `create_parents` | Write bytes in one shot |
| `write(path, content, mode, create_parents)` | `path`, `content`, `mode`, `create_parents` | Write text in one shot |

**Metadata Operations:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `exists(path)` | `path` | Check existence |
| `stat(path)` | `path` | Get metadata |
| `list(path)` | `path` | List directory |
| `glob(pattern, path)` | `pattern`, `path` | Match by glob pattern |
| `grep(pattern, path, glob, max_matches)` | `pattern`, `path`, `glob`, `max_matches` | Regex search |

**Mutating Operations:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `delete(path, recursive)` | `path`, `recursive` | Delete file/directory |
| `mkdir(path, parents, exist_ok)` | `path`, `parents`, `exist_ok` | Create directory |

**Write modes:** `"create"`, `"overwrite"`, `"append"`

**Metadata Properties:**

- `root`: Workspace root path
- `read_only`: Whether writes disabled

## Streaming Usage Patterns

### Reading Large Files (Bytes)

```python
# Stream processing with fixed memory
with filesystem.open_read("large_file.bin") as reader:
    for chunk in reader.chunks(size=65536):  # 64KB chunks
        process(chunk)

# Random access
with filesystem.open_read("data.bin") as reader:
    reader.seek(1024)  # Skip first 1KB
    header = reader.read(256)  # Read 256 bytes
```

### Reading Large Files (Text)

```python
# Line-by-line processing (lazy UTF-8 decode)
with filesystem.open_text("log.txt") as reader:
    for line in reader.lines(strip=True):
        if "ERROR" in line:
            print(f"Line {reader.line_number}: {line}")
```

### Writing Large Files

```python
# Stream writing
with filesystem.open_write("output.bin", mode="create") as writer:
    for chunk in data_source:
        writer.write(chunk)

# Write from iterator
with filesystem.open_write("output.bin") as writer:
    writer.write_all(generate_chunks())
```

### Copying Files (Any Size)

```python
# Zero-copy streaming copy
with filesystem.open_read("source.bin") as src:
    with filesystem.open_write("dest.bin") as dst:
        dst.write_all(src)  # ByteReader is iterable
```

## Memory Guarantees

| Operation | Memory Usage |
|-----------|--------------|
| `open_read()` iteration | O(chunk_size), default 64KB |
| `open_text()` iteration | O(line_length), typically < 1KB |
| `open_write()` | O(1), writes immediately |
| `read_bytes(limit=N)` | O(N) |
| `read(limit=N)` | O(N × avg_line_length) |

Backends MUST NOT buffer entire files in memory for streaming operations.

## ToolContext Integration

`ToolContext` includes `filesystem: Filesystem | None`. Handlers access via
`context.filesystem.*`.

## Section Ownership

`WorkspaceSection` protocol exposes `filesystem` property. Sections create and
manage their filesystem instance.

## Prompt Integration

`Prompt.filesystem()` locates workspace section and returns its filesystem.
Adapters construct `ToolContext` with filesystem from prompt.

## Backend Implementations

| Backend | Description | Implementation |
|---------|-------------|----------------|
| `InMemoryFilesystem` | Session-scoped in-memory with structural sharing | `contrib/tools/filesystem_memory.py` |
| `HostFilesystem` | Sandboxed host directory with path validation, git snapshots | `src/weakincentives/filesystem/_host.py` |
| `PodmanSandboxSection` | Containerized workspace using overlay + HostFilesystem | `contrib/tools/podman.py` |

### Design Characteristics

All backends share:

- **Exception mapping**: Standard Python exceptions
- **Path normalization**: Validated relative to root
- **Streaming support**: Chunk-based iteration for large files
- **Read-only mode**: Prevent modifications

### Backend-Specific Streaming

| Backend | Streaming Implementation |
|---------|-------------------------|
| `HostFilesystem` | Wraps native file handles with buffered I/O |
| `InMemoryFilesystem` | `io.BytesIO` over stored bytes with copy-on-read |
| `PodmanSandboxSection` | Delegates to internal HostFilesystem |

## Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Default chunk size | 65,536 bytes (64KB) | Configurable per-read |
| Max path depth | 16 segments | Enforced on all paths |
| Max segment length | 80 chars | Per path segment |
| Max grep matches | 1,000 | Default, configurable |
| Max convenience read | 32MB | For `read_bytes()`/`read()` |
| Max convenience write | 32MB | For `write_bytes()`/`write()` |

**Note:** Streaming operations have no inherent size limits. Convenience methods
have size limits to prevent accidental memory exhaustion.

## Error Handling

| Backend Error | Python Exception |
|---------------|------------------|
| File not found | `FileNotFoundError` |
| Path is directory | `IsADirectoryError` |
| Path is file | `NotADirectoryError` |
| Access denied | `PermissionError` |
| File exists | `FileExistsError` |
| Invalid content | `ValueError` |
| Backend unavailable | `RuntimeError` |
| Decode error | `UnicodeDecodeError` |

### Stream-Specific Errors

| Error Condition | Behavior |
|-----------------|----------|
| Read after close | `ValueError: I/O operation on closed file` |
| Write after close | `ValueError: I/O operation on closed file` |
| Decode failure | `UnicodeDecodeError` with position info |
| Seek on write-only | `OSError: Illegal seek` |

## Filesystem Snapshots

Capture workspace state for rollback after failed tools or exploratory changes.

### FilesystemSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `snapshot_id` | `UUID` | Unique identifier |
| `created_at` | `datetime` | Creation time |
| `commit_ref` | `str` | Git commit or version ID |
| `root_path` | `str` | Root at snapshot time |
| `git_dir` | `str \| None` | External git directory |
| `tag` | `str \| None` | Human-readable label |

### SnapshotableFilesystem Protocol

| Method | Description |
|--------|-------------|
| `snapshot(tag)` | Capture current state |
| `restore(snapshot)` | Restore to previous state |

### Implementation Strategies

- **InMemoryFilesystem**: Structural sharing via frozen dictionaries
- **HostFilesystem**: Git commits with external `--git-dir`
- **PodmanSandboxSection**: Delegates to internal HostFilesystem

### Session Integration

```python
session[FilesystemSnapshot].register(FilesystemSnapshot, append_all)
fs_snapshot = filesystem.snapshot(tag="before-refactor")
session[FilesystemSnapshot].append(fs_snapshot)
filesystem.restore(snapshots[-1])
```

## Binary File Support

- `open_read()` / `open_write()`: Streaming bytes (primary)
- `open_text()`: Streaming text with lazy decode
- `read_bytes()` / `write_bytes()`: Convenience for bounded reads/writes
- `read()` / `write()`: Text convenience (uses UTF-8)

For copying or processing large files, always use streaming operations.

## Testing

### Protocol Compliance

`FilesystemValidationSuite` validates all protocol methods including streaming.
Uses abstract factory for concrete backend testing.

**Test categories:**

- Streaming read/write operations
- Memory usage bounds (no full-file buffering)
- Seek and position tracking
- Text decoding with various encodings
- Error handling on closed streams

**Test suite:** `tests/tools/test_filesystem.py`

### Test Helpers

Test fixtures and utilities for filesystem testing.

**Implementation:** `tests/helpers/filesystem.py`

## Migration Guide

### From Buffered to Streaming

Old (loads entire file):

```python
result = filesystem.read_bytes(path)
process(result.content)
```

New (fixed memory):

```python
with filesystem.open_read(path) as reader:
    for chunk in reader:
        process(chunk)
```

### From Eager Text to Lazy Decode

Old (decodes all at once):

```python
result = filesystem.read(path)
for line in result.content.splitlines():
    process(line)
```

New (decodes lazily per line):

```python
with filesystem.open_text(path) as reader:
    for line in reader.lines(strip=True):
        process(line)
```

## Limitations

- **No symlinks**: Not followed by default
- **No permissions model**: Beyond read-only flag
- **Single-threaded**: Not thread-safe; one per session
- **Path normalization**: Original casing may not be preserved
- **Git dependency**: Disk snapshots require git
- **No partial restore**: All-or-nothing
- **UTF-8 only**: Text operations support only UTF-8 encoding
