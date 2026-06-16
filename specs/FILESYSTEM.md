# Filesystem Specification

## Purpose

One concrete `Filesystem` facade that tools access through `ToolContext`,
composed over a **narrow backend protocol** (`FilesystemBackend`, ~10 storage
primitives). Handlers program against the facade and stay portable; backends
implement primitives only, so a remote backend is cheap and a usable fake fits
in under 100 lines.

**Implementation:** `src/weakincentives/filesystem/`

## Guiding Principles

- **Narrow waist**: Backends implement ~10 primitives; everything ergonomic
  (text, pagination, streaming, limits, validation) lives once in the facade
- **Single access pattern**: Tools use one concrete class regardless of backend
- **Context-scoped**: Filesystem on `ToolContext` and `Prompt`, not global
- **Server-side search**: `glob`/`grep` are backend primitives so remote
  backends can run them where the data lives
- **Opaque snapshots**: `SnapshotRef` tokens are backend-private; no git
  details leak into shared types
- **Fail-closed**: Path validation, read-only, and root guards apply uniformly
  to every operation in the facade

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Filesystem (facade, concrete)              │
│  read()/write() text + pagination · read_bytes()/write_bytes()│
│  open_read()/open_write()/open_text() streaming · exists()    │
│  normalization + validation + read-only/root/size guards      │
├──────────────────────────────────────────────────────────────┤
│              FilesystemBackend (protocol, ~11 primitives)     │
│  root · read_only · stat · list · glob · grep · read_range    │
│  write · delete · mkdir · rename · snapshot · restore          │
├──────────────────┬──────────────────┬────────────────────────┤
│   HostBackend    │   MemoryBackend  │   (your backend here)   │
└──────────────────┴──────────────────┴────────────────────────┘
```

## Backend Protocol

`FilesystemBackend` at `src/weakincentives/filesystem/_backend.py`. Paths
arriving at a backend are already normalized by the facade (relative,
`/`-separated, no `.`/`..`; `""` is the root). The facade is the only
intended caller.

| Primitive | Contract |
|-----------|----------|
| `root` | Workspace root path (`"/"` for virtual backends) |
| `read_only` | Whether writes are disabled (enforced by the facade) |
| `stat(path)` | Metadata; raises `FileNotFoundError` |
| `list(path)` | Entries of an existing directory, sorted by name |
| `glob(pattern, path)` | Files under an existing base, sorted by path |
| `grep(pattern, path, glob, max_matches)` | Matches ordered by `(path, line)`; binary files skipped |
| `read_range(path, offset, length)` | Bytes slice; `length=None` reads to EOF; raises `FileNotFoundError`/`IsADirectoryError` |
| `write(path, data, mode)` | Writes bytes, creating parents; raises `FileExistsError` for `mode="create"` |
| `delete(path, recursive)` | Removes file or directory; raises `FileNotFoundError`/`IsADirectoryError` |
| `mkdir(path)` | Creates directory and missing parents (idempotent) |
| `rename(src, dst)` | Moves a file or directory (with contents) to `dst`, creating parents; raises `FileNotFoundError`/`FileExistsError` |
| `snapshot(tag)` | Returns an opaque `SnapshotRef` |
| `restore(ref)` | Restores; raises `SnapshotRestoreError` for unknown refs |

## Facade Surface

`Filesystem` at `src/weakincentives/filesystem/_facade.py`. Construct as
`Filesystem(backend)`, `Filesystem.host(root)`, or `Filesystem.in_memory()`.
The wrapped backend is reachable via the `backend` property (e.g., adapters
check `isinstance(fs.backend, HostBackend)`).

**Streaming Operations** (`_streams.py`; one implementation over any backend):

| Method | Returns | Description |
|--------|---------|-------------|
| `open_read(path)` | `ByteReader` | Chunked reads via backend `read_range`; `seek()`, `position`, `size` |
| `open_write(path, mode, create_parents)` | `ByteWriter` | Buffers and commits via one backend `write` on `close()`; abort discards |
| `open_text(path, encoding)` | `TextReader` | Lazy UTF-8 line decoding over `ByteReader` |

**Convenience Operations:**

| Method | Description |
|--------|-------------|
| `read(path, offset, limit, encoding)` | Text with line pagination (default 2000 lines; `READ_ENTIRE_FILE`) |
| `read_bytes(path, offset, limit)` | Bytes with byte pagination |
| `write(path, content, mode, create_parents)` | Text write, 32 MB cap |
| `write_bytes(path, content, mode, create_parents)` | Byte write, 32 MB cap |

**Metadata and Mutations:** `exists`, `stat`, `list`, `glob`, `grep`,
`delete(recursive)`, `mkdir(parents, exist_ok)`, `rename(src, dst)`, plus
`root` and `read_only` properties.

**Write modes:** `"create"`, `"overwrite"`, `"append"`

### Result Types

| Type | Key Properties |
|------|----------------|
| `FileStat` | `path`, `is_file`, `is_directory`, `size_bytes`, `created_at`, `modified_at` |
| `FileEntry` | `name`, `path`, `is_file`, `is_directory` |
| `GlobMatch` | `path`, `is_file` |
| `GrepMatch` | `path`, `line_number`, `line_content`, `match_start`, `match_end` |
| `ReadResult` / `ReadBytesResult` | content plus pagination metadata |
| `WriteResult` | `path`, `bytes_written`, `mode` |
| `SnapshotRef` | `snapshot_id`, `created_at`, `token` (backend-private), `tag` |

## Backends

| Backend | Description | Implementation |
|---------|-------------|----------------|
| `HostBackend` | Sandboxed host directory; root resolved once at construction; every operation re-validates its symlink-resolved target stays inside the root; git snapshots | `filesystem/_host.py` |
| `MemoryBackend` | Session-scoped dictionaries with structural-sharing snapshots | `filesystem/_memory.py` |

Custom backends implement the protocol and wrap themselves in the facade:
`Filesystem(MyBackend(...))`. The validation suites in
`tests/helpers/filesystem.py` and `tests/helpers/filesystem_streaming.py`
run against any backend through the facade;
`tests/filesystem/fake_backend.py` is a complete example under 100 lines.

## Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Default chunk size | 65,536 bytes (64KB) | Streaming iteration |
| Max path depth | 16 segments | Enforced on every operation |
| Max segment length | 80 chars | Per path segment |
| Max grep matches | 1,000 | Default, configurable |
| Max convenience write | 32MB | For `write_bytes()`/`write()` |
| Default read window | 2,000 lines | For `read()` |

## Error Handling

| Condition | Python Exception |
|-----------|------------------|
| File not found | `FileNotFoundError` |
| Path is directory | `IsADirectoryError` |
| Path is file (list) | `NotADirectoryError` |
| Sandbox escape / read-only write / root delete | `PermissionError` |
| File exists (`mode="create"`) | `FileExistsError` |
| Binary content via `read()`, invalid regex, root write, size cap, path constraints | `ValueError` |
| Read/write after close | `ValueError: I/O operation on closed file` |
| Unknown/foreign snapshot ref | `SnapshotRestoreError` |

## ToolContext Integration

`ToolContext.filesystem` is the sandbox's filesystem facet
(`context.sandbox.filesystem`); it is `None` when no sandbox is attached.
Handlers access it via `context.filesystem.*`. Adapters open one sandbox
per evaluation and thread it into every `ToolContext`.

## Snapshots

`snapshot()` returns an opaque `SnapshotRef`; `restore(ref)` is all-or-nothing.
Refs are only meaningful to the backend kind that created them.

| Backend | Strategy |
|---------|----------|
| `MemoryBackend` | Structural sharing via frozen dictionaries; token is a version key |
| `HostBackend` | Git commits in an external `--git-dir` outside the workspace; token embeds commit and git dir |

### Host restore contract (strict rollback)

Snapshots capture **every** file, including `.gitignore`-matched ones
(`git add --all --force`). Restore hard-resets to the snapshot commit and
removes all untracked files including ignored ones (`git clean -xfd`).
Together: `restore(ref)` returns the workspace to the exact state observed at
`snapshot()` time — ignored files present at snapshot time come back; files
created afterwards are removed regardless of ignore rules.

## Testing

`FilesystemValidationSuite` (`tests/helpers/filesystem.py`) plus streaming,
read-only, and snapshot suites (`tests/helpers/filesystem_streaming.py`) are
backend-agnostic and run over `HostBackend`, `MemoryBackend`, and the toy
`FakeBackend` in `tests/filesystem/test_validation.py`.

## Limitations

- **No symlinks**: Not followed outside the sandbox; host backend rejects
  escaping targets
- **No permissions model**: Beyond read-only flag
- **Single-threaded**: Not thread-safe; one per session
- **Git dependency**: Host snapshots require git
- **No partial restore**: All-or-nothing
- **UTF-8 only**: Text operations support only UTF-8 encoding
