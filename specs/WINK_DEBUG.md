# WINK Debug Specification

## Purpose

The `wink debug` command provides tools for inspecting session snapshot files.
It supports two modes:

1. **Static site generation** (default): Produces a set of static files that can
   be published to any web server with an arbitrary path prefix
2. **Development server**: Runs a local server for interactive debugging with
   live reload support

## CLI Contract

```
wink debug <snapshot_path> --output <dir> [--base-path PATH]
wink debug <snapshot_path> --serve [--host HOST] [--port PORT] [--open-browser]
```

### Common Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `snapshot_path` | Yes | - | Path to a JSONL snapshot file or directory containing snapshots |

### Static Generation Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--output` | Yes | - | Output directory for static files |
| `--base-path` | No | `/` | URL path prefix for deployment (e.g., `/reports/` or `/debug/session-123/`) |

### Server Mode Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--serve` | Yes | - | Enable development server mode |
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
| `0` | Success (static generation completed or server stopped normally) |
| `2` | Snapshot validation failed |
| `3` | Server failed to start (server mode only) |
| `4` | Output directory error (static mode only) |

## Static Site Generation

### Usage Examples

```bash
# Generate static site from a snapshot file
wink debug session.jsonl --output ./site

# Generate with custom base path for deployment at /reports/
wink debug session.jsonl --output ./site --base-path /reports/

# Generate from a directory of snapshots
wink debug ./snapshots/ --output ./site
```

### Output Structure

```
output/
├── index.html                          # Main page with embedded base path
├── static/
│   ├── style.css                       # Stylesheet
│   └── app.js                          # Client-side JavaScript
└── data/
    ├── manifest.json                   # Site manifest with snapshot index
    └── snapshots/
        └── <filename>/                 # URL-safe filename
            ├── entries.json            # List of entries in this file
            └── entries/
                └── <line_number>/      # Entry by line number
                    ├── meta.json       # Entry metadata
                    ├── raw.json        # Raw payload
                    └── slices/
                        └── <slice>.json # Slice data with rendered markdown
```

### Manifest Format

`data/manifest.json`:

```json
{
  "version": "1",
  "generated_at": "2024-01-15T10:30:00+00:00",
  "base_path": "/reports/",
  "snapshots": [
    {
      "file": "session.jsonl",
      "path": "snapshots/session.jsonl",
      "entry_count": 3,
      "entries": [1, 2, 3]
    }
  ],
  "default_snapshot": "session.jsonl",
  "default_entry": 1
}
```

### Entry Metadata Format

`data/snapshots/<file>/entries/<line>/meta.json`:

```json
{
  "version": "1",
  "created_at": "2024-01-15T10:30:00+00:00",
  "file": "session.jsonl",
  "session_id": "abc123",
  "line_number": 1,
  "tags": {"session_id": "abc123", "custom_tag": "value"},
  "validation_error": null,
  "slices": [
    {"slice_type": "mymodule.Plan", "item_type": "mymodule.Plan", "count": 3}
  ]
}
```

### Slice Data Format

`data/snapshots/<file>/entries/<line>/slices/<encoded_slice_type>.json`:

```json
{
  "slice_type": "mymodule.Plan",
  "item_type": "mymodule.Plan",
  "items": [
    {"field": "value", "__markdown__": {"text": "# Header", "html": "<h1>Header</h1>"}}
  ]
}
```

### Base Path Handling

The `--base-path` argument controls how URLs are constructed in the generated
site:

1. The `<base href="{base_path}">` tag is injected into `index.html`
2. All data fetches use paths relative to the base
3. Navigation uses URL hash fragments: `#file=session.jsonl&entry=1`

This allows the static site to be deployed at any path prefix without
modifying the generated files.

### Filename Encoding

Snapshot filenames are URL-encoded for use in paths:

- `session.jsonl` → `session.jsonl`
- `my session (1).jsonl` → `my%20session%20%281%29.jsonl`

Slice types are similarly URL-encoded:

- `mymodule.Plan` → `mymodule.Plan`
- `mymodule.Plan<T>` → `mymodule.Plan%3CT%3E`

## Development Server Mode

Server mode provides a live development experience with dynamic data loading
and reload capabilities.

### Usage Examples

```bash
# Start development server
wink debug session.jsonl --serve

# Custom host and port
wink debug session.jsonl --serve --host 0.0.0.0 --port 3000

# Without auto-opening browser
wink debug session.jsonl --serve --no-open-browser
```

### API Routes (Server Mode Only)

These routes are only available in server mode:

#### `GET /`

Returns the HTML index page for the web UI.

#### `GET /api/meta`

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

#### `GET /api/entries`

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

#### `GET /api/slices/{encoded_slice_type}`

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

#### `GET /api/raw`

Returns the raw JSON payload of the current snapshot entry without any
transformation.

#### `POST /api/reload`

Reloads the current snapshot file from disk. If the previously selected
`session_id` still exists, it remains selected; otherwise, the first entry
is selected.

Returns the updated metadata (same format as `/api/meta`).

#### `GET /api/snapshots`

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

#### `POST /api/select`

Selects a different entry within the current snapshot file.

Request body (one of):

```json
{"session_id": "abc123"}
{"line_number": 1}
```

Returns the updated metadata.

#### `POST /api/switch`

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

## Snapshot Loading

### Path Resolution

When `snapshot_path` is a directory:

1. Glob for `*.jsonl` and `*.json` files
1. Sort by modification time (newest first)
1. Load all files (static mode) or the most recent file (server mode)
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

When a snapshot line fails full validation (e.g., missing dataclass types):

- Logs a warning with event `wink.debug.snapshot_error`
- Stores the validation error message in metadata
- Continues processing for inspection

### Markdown Rendering

String values that look like Markdown are automatically wrapped with rendered
HTML. The detection heuristic checks for:

- Headers (`# `, `## `, etc.)
- Lists (`- `, `* `, `+ `, `1. `)
- Code spans (`` `code` ``)
- Links (`[text](url)`)
- Bold (`**text**`)
- Paragraph breaks (`\n\n`)

Minimum length for Markdown detection: 16 characters.

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

### SnapshotStore (Server Mode)

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
| `app.js` | Client-side JavaScript (supports both static and server modes) |

In server mode, static files are mounted at `/static/`.

In static mode, files are copied to the output directory preserving structure.

## Client-Side Navigation (Static Mode)

In static mode, navigation is handled entirely client-side using URL hash
fragments:

```
index.html#file=session.jsonl&entry=1
index.html#file=session.jsonl&entry=2&slice=mymodule.Plan
```

The JavaScript application:

1. Reads the manifest to discover available snapshots and entries
2. Parses the URL hash to determine current selection
3. Fetches the appropriate JSON files from `data/`
4. Updates the hash when the user navigates

This approach ensures:

- All links are shareable and bookmarkable
- Browser back/forward navigation works correctly
- No server-side routing required

## Logging

| Event | Level | Context |
|-------|-------|---------|
| `wink.debug.snapshot_error` | WARNING | `path`, `line_number`, `error` |
| `debug.build.start` | INFO | `output`, `base_path` |
| `debug.build.snapshot` | INFO | `file`, `entry_count` |
| `debug.build.complete` | INFO | `output`, `file_count` |
| `debug.server.start` | INFO | `url` |
| `debug.server.reload` | INFO | `path` |
| `debug.server.reload_failed` | WARNING | `path`, `error` |
| `debug.server.switch` | INFO | `path` |
| `debug.server.error` | ERROR | `url`, `error` |
| `debug.server.browser` | WARNING | `url`, `error` |

## Implementation Notes

### Static Generation

- Uses the same snapshot loading logic as server mode
- Pre-renders all Markdown at generation time
- Writes atomic JSON files with proper encoding
- Generates deterministic output (same input → same output)

### Server Mode

- Uses FastAPI for the HTTP server
- Uses uvicorn as the ASGI server
- Uses markdown-it for Markdown rendering
- Browser opening uses a 0.2-second timer to avoid blocking server startup
- Snapshot path validation restricts file switching to the initial root directory

### JavaScript Mode Detection

The client-side JavaScript detects which mode it's running in:

```javascript
// Check if running in static mode by looking for manifest
const isStaticMode = await fetch('data/manifest.json')
  .then(() => true)
  .catch(() => false);
```

In static mode, it loads data from JSON files. In server mode, it uses the
API endpoints.
