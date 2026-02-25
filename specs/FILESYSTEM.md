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

Streaming is the primary API. Convenience methods buffer bounded reads/writes
for callers that don't need chunk-level control.

## Protocol Definition

Streaming type protocols at `src/weakincentives/filesystem/_types.py`:

| Type | Description |
|------|-------------|
| `ByteReader` | Context manager yielding byte chunks via iteration; supports `seek()`, `position`, `size` |
| `ByteWriter` | Context manager accepting byte chunks via `write()`; tracks `bytes_written` |
| `TextReader` | Wrapper over `ByteReader` with lazy UTF-8 decoding; line-oriented iteration |

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

## Memory Guarantees

| Operation | Memory Usage |
|-----------|--------------|
| `open_read()` iteration | O(chunk_size), default 64KB |
| `open_text()` iteration | O(line_length), typically < 1KB |
| `open_write()` | O(1), writes immediately |
| `read_bytes(limit=N)` | O(N) |
| `read(limit=N)` | O(N × avg_line_length) |

Backends MUST NOT buffer entire files in memory for streaming operations.
Streaming operations have no inherent size limits. Convenience methods cap at
32MB to prevent accidental memory exhaustion.

## ToolContext Integration

`ToolContext` includes `filesystem: Filesystem | None`. Handlers access via
`context.filesystem.*`. The filesystem is scoped to the current tool call and
backed by whatever workspace section is active.

`Prompt.filesystem()` locates the workspace section and returns its filesystem.
Adapters construct `ToolContext` with filesystem from prompt.

## Backend Implementations

| Backend | Description | Implementation |
|---------|-------------|----------------|
| `InMemoryFilesystem` | Session-scoped in-memory with structural sharing | `contrib/tools/filesystem_memory.py` |
| `HostFilesystem` | Sandboxed host directory with path validation, git snapshots | `src/weakincentives/filesystem/_host.py` |
| `PodmanSandboxSection` | Containerized workspace using overlay + HostFilesystem | `contrib/tools/podman.py` |

All backends share: standard Python exception mapping, path normalization
relative to root, streaming support, and read-only mode.

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

| Error Condition | Behavior |
|-----------------|----------|
| Read after close | `ValueError: I/O operation on closed file` |
| Write after close | `ValueError: I/O operation on closed file` |
| Decode failure | `UnicodeDecodeError` with position info |
| Seek on write-only | `OSError: Illegal seek` |

## Filesystem Snapshots

Capture workspace state for rollback after failed tools or exploratory changes.
`FilesystemSnapshot` at `src/weakincentives/filesystem/_types.py` records
`snapshot_id`, `created_at`, `commit_ref`, `root_path`, `git_dir`, and `tag`.

`SnapshotableFilesystem` protocol extends `Filesystem` with:

| Method | Description |
|--------|-------------|
| `snapshot(tag)` | Capture current state |
| `restore(snapshot)` | Restore to previous state |

Implementation strategies:

- **InMemoryFilesystem**: Structural sharing via frozen dictionaries
- **HostFilesystem**: Git commits with external `--git-dir`
- **PodmanSandboxSection**: Delegates to internal HostFilesystem

## Testing

`FilesystemValidationSuite` at `tests/filesystem/` validates all protocol
methods including streaming, memory bounds, seek/position tracking, text
decoding, and error handling on closed streams. Test fixtures at
`tests/helpers/filesystem.py`.

## Limitations

- **No symlinks**: Not followed by default
- **No permissions model**: Beyond read-only flag
- **Single-threaded**: Not thread-safe; one per session
- **Path normalization**: Original casing may not be preserved
- **Git dependency**: Disk snapshots require git
- **No partial restore**: All-or-nothing
- **UTF-8 only**: Text operations support only UTF-8 encoding
