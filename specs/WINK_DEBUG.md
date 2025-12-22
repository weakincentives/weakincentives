# WINK Debug Server Specification

## Purpose

The `wink debug` command launches a local web server for inspecting session
snapshot files and associated filesystem archives. It provides a browser-based
UI for exploring session state, slice contents, snapshot metadata, and
workspace file contents without writing code.

## CLI Contract

```text
wink debug <snapshot_path> [--host HOST] [--port PORT] [--open-browser|--no-open-browser]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `snapshot_path` | Yes | - | Path to a JSONL snapshot file or directory containing snapshots |
| `--host` | No | `127.0.0.1` | Host interface to bind the server |
| `--port` | No | `8000` | Port to bind the server |
| `--open-browser` | No | `True` | Open the default browser automatically |

### Global Options

The CLI inherits global options from the `wink` command:

| Option | Default | Description |
|--------|---------|-------------|
| `--log-level` | `None` | Override log level (CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET) |
| `--json-logs` | `True` | Emit structured JSON logs (disable with `--no-json-logs`) |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Server stopped normally |
| `2` | Snapshot validation failed at startup |
| `3` | Server failed to start |

## Snapshot Loading

### Path Resolution

When `snapshot_path` is a directory:

1. Glob for `*.jsonl` and `*.json` files
1. Sort by modification time (newest first)
1. Load the most recent file
1. Use the directory as the root for snapshot switching

When `snapshot_path` is a file:

1. Load the specified file directly
1. Use the parent directory as the root for snapshot switching

### JSONL Format

Snapshot files contain one JSON object per line. Each line represents a complete
session snapshot. The loader:

1. Skips empty lines
1. Parses each non-empty line as JSON
1. Validates the `SnapshotPayload` structure
1. Attempts full `Snapshot` restoration (validation errors are captured but don't
   block loading)
1. Requires a `session_id` tag on every entry

### Validation Errors

When a snapshot line fails full validation (e.g., missing dataclass types),
the server:

- Logs a warning with event `wink.debug.snapshot_error`
- Stores the validation error message in metadata
- Continues serving the raw payload for inspection

## API Routes

### `GET /`

Returns the HTML index page for the web UI.

### `GET /api/meta`

Returns metadata for the currently selected snapshot entry.

```json
{
  "version": "1",
  "created_at": "2024-01-15T10:30:00+00:00",
  "path": "/path/to/snapshot.jsonl",
  "session_id": "abc123",
  "line_number": 1,
  "tags": {"session_id": "abc123", "custom_tag": "value"},
  "validation_error": null,
  "slices": [
    {"slice_type": "mymodule.Plan", "item_type": "mymodule.Plan", "count": 3}
  ]
}
```

### `GET /api/entries`

Lists all snapshot entries in the current file.

```json
[
  {
    "session_id": "abc123",
    "name": "abc123 (line 1)",
    "path": "/path/to/snapshot.jsonl",
    "line_number": 1,
    "created_at": "2024-01-15T10:30:00+00:00",
    "tags": {"session_id": "abc123"},
    "selected": true
  }
]
```

### `GET /api/slices/{encoded_slice_type}`

Returns items from a specific slice. The `encoded_slice_type` must be
URL-encoded (e.g., `mymodule.Plan` → `mymodule.Plan`).

Query parameters:

- `offset` (int, >= 0): Skip this many items
- `limit` (int, >= 0, optional): Return at most this many items

```json
{
  "slice_type": "mymodule.Plan",
  "item_type": "mymodule.Plan",
  "items": [
    {"field": "value", "__markdown__": {"text": "# Header", "html": "<h1>Header</h1>"}}
  ]
}
```

String values that look like Markdown are automatically wrapped with rendered
HTML. The detection heuristic checks for:

- Headers (`#`, `##`, etc.)
- Lists (`-`, `*`, `+`, `1.`)
- Code spans (`` `code` ``)
- Links (`[text](url)`)
- Bold (`**text**`)
- Paragraph breaks (`\n\n`)

Minimum length for Markdown detection: 16 characters.

### `GET /api/raw`

Returns the raw JSON payload of the current snapshot entry without any
transformation.

### `POST /api/reload`

Reloads the current snapshot file from disk. If the previously selected
`session_id` still exists, it remains selected; otherwise, the first entry
is selected.

Returns the updated metadata (same format as `/api/meta`).

### `GET /api/snapshots`

Lists all snapshot files in the root directory, sorted by modification time
(newest first).

```json
[
  {
    "path": "/path/to/snapshot.jsonl",
    "name": "snapshot.jsonl",
    "created_at": "2024-01-15T10:30:00+00:00"
  }
]
```

### `POST /api/select`

Selects a different entry within the current snapshot file.

Request body (one of):

```json
{"session_id": "abc123"}
{"line_number": 1}
```

Returns the updated metadata.

### `POST /api/switch`

Switches to a different snapshot file. The file must be under the same root
directory established at startup.

Request body:

```json
{
  "path": "/path/to/other-snapshot.jsonl",
  "session_id": "optional-session-id",
  "line_number": null
}
```

Returns the updated metadata.

## Data Types

### SnapshotLoadError

Raised when a snapshot cannot be loaded or validated. Inherits from `WinkError`
and `RuntimeError`.

### LoadedSnapshot

```python
@FrozenDataclass()
class LoadedSnapshot:
    meta: SnapshotMeta
    slices: Mapping[str, SnapshotSlicePayload]
    raw_payload: Mapping[str, JSONValue]
    raw_text: str
    path: Path
```

### SnapshotMeta

```python
@FrozenDataclass()
class SnapshotMeta:
    version: str
    created_at: str
    path: str
    session_id: str
    line_number: int
    slices: tuple[SliceSummary, ...]
    tags: Mapping[str, str]
    validation_error: str | None = None
```

### SliceSummary

```python
@FrozenDataclass()
class SliceSummary:
    slice_type: str
    item_type: str
    count: int
```

### SnapshotStore

Thread-safe in-memory store for loaded snapshots with support for:

- Loading entries from a snapshot file
- Selecting entries by `session_id` or `line_number`
- Reloading the current file from disk
- Switching to a different file within the root directory
- Listing available snapshot files

## Static Assets

The web UI is served from `src/weakincentives/cli/static/`:

| File | Purpose |
|------|---------|
| `index.html` | Main HTML page |
| `style.css` | Stylesheet |
| `app.js` | Client-side JavaScript |

Static files are mounted at `/static/`.

## Logging

| Event | Level | Context |
|-------|-------|---------|
| `wink.debug.snapshot_error` | WARNING | `path`, `line_number`, `error` |
| `debug.server.start` | INFO | `url` |
| `debug.server.reload` | INFO | `path` |
| `debug.server.reload_failed` | WARNING | `path`, `error` |
| `debug.server.switch` | INFO | `path` |
| `debug.server.error` | ERROR | `url`, `error` |
| `debug.server.browser` | WARNING | `url`, `error` |

## Implementation Notes

- Uses FastAPI for the HTTP server
- Uses uvicorn as the ASGI server
- Uses markdown-it for Markdown rendering
- Browser opening uses a 0.2-second timer to avoid blocking server startup
- Snapshot path validation restricts file switching to the initial root directory

## Filesystem Archive Format

When a session is dumped with an associated filesystem, a companion `.zip`
archive is created alongside the JSONL file. The archive uses standard ZIP
format for broad compatibility and tooling support.

### Naming Convention

```text
snapshots/
├── abc123-session.jsonl     # Session snapshot
└── abc123-session.fs.zip    # Filesystem archive (same stem + .fs.zip)
```

The filesystem archive shares the same stem as the session JSONL file with
a `.fs.zip` suffix. This pairing convention allows the debug server to
automatically discover associated archives.

### Archive Structure

The ZIP archive contains all files from the workspace filesystem at the time
of export, preserving the directory structure:

```text
abc123-session.fs.zip
├── sunfish/
│   ├── README.md
│   ├── src/
│   │   ├── main.py
│   │   └── utils.py
│   └── tests/
│       └── test_main.py
└── scratch/
    └── notes.txt
```

Files are stored with UTF-8 encoding. Binary files are stored as-is.

### Archive Metadata

The ZIP archive includes a `_wink_metadata.json` file at the root with
provenance information:

```json
{
  "version": "1",
  "created_at": "2024-01-15T10:30:00+00:00",
  "session_id": "abc123-session",
  "root_path": "/",
  "file_count": 42,
  "total_bytes": 156789
}
```

## Filesystem Explorer

The debug server provides a filesystem explorer for navigating and viewing
files from associated ZIP archives.

### Discovery

When loading a JSONL snapshot, the server checks for a companion `.fs.zip`
archive:

1. Look for `<stem>.fs.zip` next to `<stem>.jsonl`
1. If found, load the archive into memory
1. Build a file tree index for navigation

If no archive exists, the filesystem explorer panel is hidden.

### Filesystem API Routes

#### `GET /api/filesystem/tree`

Returns the hierarchical file tree from the filesystem archive.

```json
{
  "has_archive": true,
  "archive_path": "/path/to/abc123-session.fs.zip",
  "metadata": {
    "version": "1",
    "created_at": "2024-01-15T10:30:00+00:00",
    "session_id": "abc123-session",
    "file_count": 42,
    "total_bytes": 156789
  },
  "tree": {
    "name": "/",
    "type": "directory",
    "children": [
      {
        "name": "sunfish",
        "type": "directory",
        "children": [
          {"name": "README.md", "type": "file", "size": 1234, "path": "sunfish/README.md"},
          {"name": "src", "type": "directory", "children": [...]}
        ]
      }
    ]
  }
}
```

When no archive is available:

```json
{
  "has_archive": false,
  "archive_path": null,
  "metadata": null,
  "tree": null
}
```

#### `GET /api/filesystem/file/{encoded_path}`

Returns the content of a specific file from the archive. The `encoded_path`
must be URL-encoded.

Query parameters:

- `offset` (int, >= 0): Line number to start from (0-indexed)
- `limit` (int, >= 0, optional): Maximum lines to return

```json
{
  "path": "sunfish/src/main.py",
  "content": "#!/usr/bin/env python3\n...",
  "size_bytes": 1234,
  "total_lines": 45,
  "offset": 0,
  "limit": null,
  "truncated": false,
  "encoding": "utf-8"
}
```

For binary files:

```json
{
  "path": "sunfish/data/image.png",
  "content": null,
  "size_bytes": 45678,
  "binary": true,
  "error": "Binary file cannot be displayed as text"
}
```

#### `GET /api/filesystem/download/{encoded_path}`

Returns the raw file content with appropriate Content-Type headers for
download. Supports single files and directory downloads (as ZIP).

### File Detection

The explorer uses file extension and content sniffing to determine file types:

| Pattern | Detection | Display |
|---------|-----------|---------|
| `*.py` | Python source | Syntax highlighting |
| `*.md` | Markdown | Rendered HTML + source toggle |
| `*.json`, `*.yaml` | Structured data | Pretty-printed |
| `*.txt`, no extension | Plain text | Monospace |
| Binary content | Binary | Size info, download link |

### UI Components

The filesystem explorer is displayed as a collapsible panel in the debug UI:

1. **File Tree Panel** (left): Collapsible directory tree with file icons
1. **File Viewer Panel** (right): Content display with:
   - File path breadcrumb
   - Line numbers
   - Syntax highlighting (via Prism.js or similar)
   - Download button
   - Copy button for content
1. **Search**: Quick filter for file paths

### Archive Loading

Archives are loaded into memory on demand:

- Small archives (\<10MB): Fully loaded into memory
- Large archives (>=10MB): Streamed on demand per file access

Memory limit: 50MB total for cached file contents. LRU eviction when exceeded.

### Switching Snapshots

When switching to a different JSONL snapshot via `POST /api/switch`:

1. Check for companion `.fs.zip` for the new snapshot
1. Clear cached archive state
1. Load new archive if present
1. Return updated filesystem availability in metadata

## Debug Module Extensions

The `weakincentives.debug` module provides functions for persisting both
session state and filesystem archives.

### dump_session

Existing function for JSONL session export (unchanged).

```python
def dump_session(root_session: Session, target: str | Path) -> Path | None:
    """Persist a session tree to a JSONL snapshot file."""
    ...
```

### dump_filesystem_snapshot

New function for exporting filesystem state as a ZIP archive.

```python
def dump_filesystem_snapshot(
    filesystem: Filesystem,
    target: str | Path,
    *,
    session_id: str | None = None,
) -> Path | None:
    """Persist filesystem state to a ZIP archive.

    Creates a `.fs.zip` archive containing all files from the filesystem.
    The archive is named to pair with the session JSONL file when both
    exist in the same directory.

    Args:
        filesystem: The filesystem to export.
        target: Target path. If a directory, creates `<session_id>.fs.zip`.
            If a `.jsonl` file path, creates a sibling `.fs.zip` with
            matching stem.
        session_id: Session identifier for metadata and filename generation.
            Required when target is a directory.

    Returns:
        Path to the created archive, or None if the filesystem is empty.

    Raises:
        ValueError: If target is a directory and session_id is not provided.
    """
    ...
```

### dump_session_with_filesystem

Convenience function that exports both session and filesystem together.

```python
def dump_session_with_filesystem(
    root_session: Session,
    target: str | Path,
    *,
    filesystem: Filesystem | None = None,
) -> tuple[Path | None, Path | None]:
    """Persist session tree and associated filesystem as paired archives.

    This is the recommended function for exporting complete debug snapshots.
    Creates:
    - `<session_id>.jsonl` for session state
    - `<session_id>.fs.zip` for filesystem state (if filesystem provided)

    Args:
        root_session: Root session of the tree to export.
        target: Target directory for output files.
        filesystem: Optional filesystem to include. When None, only the
            session JSONL is created.

    Returns:
        Tuple of (session_path, filesystem_path). Either may be None if
        the respective content was empty or not provided.
    """
    ...
```

### Integration Example

```python
from weakincentives.debug import dump_session_with_filesystem

# In application cleanup
def on_shutdown():
    # Get filesystem from prompt or workspace section
    filesystem = loop.prompt.filesystem()

    # Export both session and filesystem together
    session_path, fs_path = dump_session_with_filesystem(
        loop.session,
        SNAPSHOT_DIR,
        filesystem=filesystem,
    )

    if session_path:
        logger.info(f"Session saved: {session_path}")
    if fs_path:
        logger.info(f"Filesystem saved: {fs_path}")
```

## Filesystem Data Types

### FilesystemArchiveMetadata

```python
@FrozenDataclass()
class FilesystemArchiveMetadata:
    version: str
    created_at: str
    session_id: str | None
    root_path: str
    file_count: int
    total_bytes: int
```

### FileTreeNode

```python
@FrozenDataclass()
class FileTreeNode:
    name: str
    type: Literal["file", "directory"]
    path: str | None = None  # Full path for files
    size: int | None = None  # Byte size for files
    children: tuple[FileTreeNode, ...] | None = None  # For directories
```

### FileContent

```python
@FrozenDataclass()
class FileContent:
    path: str
    content: str | None  # None for binary files
    size_bytes: int
    total_lines: int | None = None
    offset: int = 0
    limit: int | None = None
    truncated: bool = False
    encoding: str = "utf-8"
    binary: bool = False
    error: str | None = None
```

## Filesystem Logging

| Event | Level | Context |
|-------|-------|---------|
| `debug.filesystem.archive_created` | INFO | `path`, `file_count`, `total_bytes` |
| `debug.filesystem.archive_loaded` | INFO | `path`, `file_count` |
| `debug.filesystem.archive_not_found` | DEBUG | `expected_path` |
| `debug.filesystem.file_read` | DEBUG | `path`, `size_bytes` |
| `debug.filesystem.file_read_error` | WARNING | `path`, `error` |
| `debug.filesystem.archive_error` | ERROR | `path`, `error` |
