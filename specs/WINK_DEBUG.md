# WINK Debug Specification

## Purpose

The `wink debug` command provides tools for inspecting session snapshot files.
It supports two modes:

1. **Server mode** (default): Launches a local server that generates static
   files to a temporary directory and serves them for interactive debugging
1. **Static export**: Produces a set of static files to a user-specified
   directory that can be published to any web server with an arbitrary path
   prefix

## CLI Contract

```
wink debug <snapshot_path> [--host HOST] [--port PORT] [--open-browser]
wink debug <snapshot_path> --output <dir>
```

### Common Arguments

| Argument | Required | Default | Description |
| --------------- | -------- | ------- | --------------------------------------------------------------- |
| `snapshot_path` | Yes | - | Path to a JSONL snapshot file or directory containing snapshots |

### Server Mode Arguments (Default)

| Argument | Required | Default | Description |
| ---------------- | -------- | ----------- | -------------------------------------- |
| `--host` | No | `127.0.0.1` | Host interface to bind the server |
| `--port` | No | `8000` | Port to bind the server |
| `--open-browser` | No | `True` | Open the default browser automatically |

### Static Export Arguments

| Argument | Required | Default | Description |
| ---------- | -------- | ------- | ---------------------------------------------------------- |
| `--output` | No | - | Output directory for static files (enables static export mode) |

### Global Options

The CLI inherits global options from the `wink` command:

| Option | Default | Description |
| ------------- | ------- | ------------------------------------------------------------------ |
| `--log-level` | `None` | Override log level (CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET) |
| `--json-logs` | `True` | Emit structured JSON logs (disable with `--no-json-logs`) |

### Exit Codes

| Code | Meaning |
| ---- | ---------------------------------------------------------------- |
| `0` | Success (static generation completed or server stopped normally) |
| `2` | Snapshot validation failed |
| `3` | Server failed to start (server mode only) |
| `4` | Output directory error (static mode only) |

## Server Mode (Default)

Server mode generates static files to a temporary directory and serves them
via a local HTTP server. This provides an interactive debugging experience
with the same static file structure used for export.

### Usage Examples

```bash
# Start debug server (default behavior)
wink debug session.jsonl

# Custom host and port
wink debug session.jsonl --host 0.0.0.0 --port 3000

# Without auto-opening browser
wink debug session.jsonl --no-open-browser

# From a directory of snapshots
wink debug ./snapshots/
```

### Server Behavior

1. Generates static files to a temporary directory
1. Starts an HTTP server to serve those files
1. Opens the browser automatically (unless `--no-open-browser`)
1. Watches for changes and regenerates on reload (via UI reload button)
1. Cleans up the temporary directory on exit

The temporary directory uses the same structure as static export, allowing
the JavaScript application to work identically in both modes.

## Static Export

When `--output` is specified, the command generates static files to the
given directory and exits. The output can be published to any static file
server.

### Usage Examples

```bash
# Export static site from a snapshot file
wink debug session.jsonl --output ./site

# Export from a directory of snapshots
wink debug ./snapshots/ --output ./site
```

The generated site uses relative paths, allowing deployment at any URL path
without configuration. Simply copy the output directory to your web server.

### Output Structure

```
output/
├── index.html                          # Main page
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
  "tags": { "session_id": "abc123", "custom_tag": "value" },
  "validation_error": null,
  "slices": [
    { "slice_type": "mymodule.Plan", "item_type": "mymodule.Plan", "count": 3 }
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
    {
      "field": "value",
      "__markdown__": { "text": "# Header", "html": "<h1>Header</h1>" }
    }
  ]
}
```

### Filename Encoding

Snapshot filenames are URL-encoded for use in paths:

- `session.jsonl` → `session.jsonl`
- `my session (1).jsonl` → `my%20session%20%281%29.jsonl`

Slice types are similarly URL-encoded:

- `mymodule.Plan` → `mymodule.Plan`
- `mymodule.Plan<T>` → `mymodule.Plan%3CT%3E`

## API Routes (Server Mode)

In server mode, the server primarily serves static files. One API route
provides reload functionality:

#### `GET /`

Serves `index.html` from the generated static files.

#### `GET /static/*`

Serves static assets (CSS, JavaScript).

#### `GET /data/*`

Serves generated JSON data files.

#### `POST /api/reload`

Regenerates all static files from the source snapshot files. This allows
refreshing the view when snapshot files change on disk.

Returns:

```json
{
  "success": true,
  "generated_at": "2024-01-15T10:30:00+00:00"
}
```

All other navigation (selecting entries, switching snapshots, viewing slices)
is handled client-side by the JavaScript application loading the appropriate
JSON files from `data/`.

## Snapshot Loading

### Path Resolution

When `snapshot_path` is a directory:

1. Glob for `*.jsonl` and `*.json` files
1. Sort by modification time (newest first)
1. Load all files found
1. Use the directory as the root for discovering snapshots

When `snapshot_path` is a file:

1. Load the specified file directly
1. Use the parent directory as the root for discovering other snapshots

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

### SnapshotStore

Manages snapshot loading and static file generation:

- Loading entries from snapshot files
- Iterating over all snapshots in a directory
- Generating static JSON files for each entry
- Re-generating files on reload request

## Static Assets

The web UI source files are in `src/weakincentives/cli/static/`:

| File | Purpose |
| ------------ | ---------------------- |
| `index.html` | Main HTML page |
| `style.css` | Stylesheet |
| `app.js` | Client-side JavaScript |

These files are copied to the output directory (or temporary directory in
server mode) along with the generated `data/` directory.

## Client-Side Navigation

All navigation is handled client-side using URL hash fragments:

```
index.html#file=session.jsonl&entry=1
index.html#file=session.jsonl&entry=2&slice=mymodule.Plan
```

The JavaScript application:

1. Reads the manifest to discover available snapshots and entries
1. Parses the URL hash to determine current selection
1. Fetches the appropriate JSON files from `data/`
1. Updates the hash when the user navigates

This approach ensures:

- All links are shareable and bookmarkable
- Browser back/forward navigation works correctly
- No server-side routing required
- Identical behavior in server mode and static export

## Logging

| Event | Level | Context |
| --------------------------- | ------- | ------------------------------ |
| `wink.debug.snapshot_error` | WARNING | `path`, `line_number`, `error` |
| `debug.generate.start` | INFO | `output`, `base_path` |
| `debug.generate.snapshot` | INFO | `file`, `entry_count` |
| `debug.generate.complete` | INFO | `output`, `file_count` |
| `debug.server.start` | INFO | `url`, `temp_dir` |
| `debug.server.reload` | INFO | `temp_dir` |
| `debug.server.error` | ERROR | `url`, `error` |
| `debug.server.browser` | WARNING | `url`, `error` |

## Implementation Notes

### Static File Generation

- Pre-renders all Markdown at generation time
- Writes atomic JSON files with proper encoding
- Generates deterministic output (same input → same output)
- Uses markdown-it for Markdown rendering

### Server Mode

- Generates static files to a temporary directory on startup
- Uses a simple HTTP server (e.g., `http.server` or similar) to serve files
- Provides `/api/reload` endpoint to regenerate files
- Browser opening uses a 0.2-second timer to avoid blocking server startup
- Temporary directory is cleaned up on exit

### Unified JavaScript

The JavaScript application works identically in both modes:

1. Loads `data/manifest.json` to discover available data
1. Uses URL hash for all navigation state
1. Fetches JSON files from `data/` as needed
1. Calls `/api/reload` for the reload button (server mode only; gracefully
   fails in static export)
