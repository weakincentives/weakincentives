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

## Protocol Definition

### Result Types

| Type | Key Properties |
|------|----------------|
| `FileStat` | `path`, `is_file`, `is_directory`, `size_bytes`, `created_at`, `modified_at` |
| `FileEntry` | `name`, `path`, `is_file`, `is_directory` |
| `GlobMatch` | `path`, `is_file` |
| `GrepMatch` | `path`, `line_number`, `line_content`, `match_start`, `match_end` |
| `ReadResult` | `content`, `path`, `total_lines`, `offset`, `limit`, `truncated` |
| `ReadBytesResult` | `content`, `path`, `size_bytes`, `offset`, `limit`, `truncated` |
| `WriteResult` | `path`, `bytes_written`, `mode` |

### Filesystem Protocol

**Read Operations:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `read()` | `path`, `offset`, `limit`, `encoding` | Read text with pagination |
| `read_bytes()` | `path`, `offset`, `limit` | Read binary (for copying) |
| `exists()` | `path` | Check existence |
| `stat()` | `path` | Get metadata |
| `list()` | `path` | List directory |
| `glob()` | `pattern`, `path` | Match by glob pattern |
| `grep()` | `pattern`, `path`, `glob`, `max_matches` | Regex search |

**Write Operations:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `write()` | `path`, `content`, `mode`, `create_parents` | Write text |
| `write_bytes()` | `path`, `content`, `mode`, `create_parents` | Write binary |
| `delete()` | `path`, `recursive` | Delete file/directory |
| `mkdir()` | `path`, `parents`, `exist_ok` | Create directory |

**Write modes:** `"create"`, `"overwrite"`, `"append"`

**Metadata:**
- `root`: Workspace root path
- `read_only`: Whether writes disabled

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
| `InMemoryFilesystem` | Session-scoped in-memory with structural sharing | `filesystem/inmemory.py` |
| `HostFilesystem` | Sandboxed host directory with path validation, git snapshots | `filesystem/host.py` |
| `PodmanSandboxSection` | Containerized workspace using overlay + HostFilesystem | `contrib/tools/podman.py` |

### Design Characteristics

All backends share:
- **Exception mapping**: Standard Python exceptions
- **Path normalization**: Validated relative to root
- **Limit enforcement**: Backend-specific limits
- **Read-only mode**: Prevent modifications

## Limits

| Limit | Value |
|-------|-------|
| Max file size | 48,000 chars |
| Max path depth | 16 segments |
| Max segment length | 80 chars |
| Default read limit | 2,000 lines |
| Max grep matches | 1,000 |

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

- `read()` / `write()`: Text with UTF-8
- `read_bytes()` / `write_bytes()`: Binary without encoding

For copying, always prefer `read_bytes()` / `write_bytes()`.

## Testing

### Protocol Compliance

`FilesystemProtocolTests` validates all protocol methods. Uses abstract factory
for concrete backend testing.

**Test suite:** `tests/test_filesystem/test_protocol_compliance.py`

### Mock Filesystem

`MockFilesystem` records calls and allows pre-configured responses.

**Implementation:** `tests/helpers/mock_filesystem.py`

## Limitations

- **No symlinks**: Not followed by default
- **No permissions model**: Beyond read-only flag
- **Single-threaded**: Not thread-safe; one per session
- **No streaming**: Large files loaded into memory
- **Path normalization**: Original casing may not be preserved
- **Git dependency**: Disk snapshots require git
- **No partial restore**: All-or-nothing
