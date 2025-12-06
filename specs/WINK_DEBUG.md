# WINK Debug Server Specification

## Purpose

The `wink debug` command launches a local web server for inspecting session
snapshot files. It provides a browser-based UI for exploring session state,
slice contents, and snapshot metadata without writing code.

## CLI Contract

```
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

- Headers (`# `, `## `, etc.)
- Lists (`- `, `* `, `+ `, `1. `)
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
