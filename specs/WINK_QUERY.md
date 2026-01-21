# Wink Query Specification

## Purpose

`wink query` enables AI agents to explore debug bundles via SQL. Bundle contents
are loaded into a cached SQLite database with a schema derived from bundle
artifacts.

**Implementation:** `src/weakincentives/cli/query.py`

## CLI

```
wink query <BUNDLE> --schema
wink query <BUNDLE> "<SQL>"
wink query <BUNDLE> "<SQL>" --table
```

| Option | Description |
|--------|-------------|
| `--schema` | Output schema as JSON and exit |
| `--table` | Output as ASCII table (default: JSON) |

## Usage

```bash
# Always start with schema to discover tables
wink query ./bundle.zip --schema

# Query (JSON output)
wink query ./bundle.zip "SELECT * FROM errors"

# Query (table output)
wink query ./bundle.zip "SELECT * FROM errors" --table
```

## Tables

### Core Tables

| Table | Source | Description |
|-------|--------|-------------|
| `manifest` | `manifest.json` | Bundle metadata and execution summary |
| `artifacts` | `manifest.artifacts` | Artifact catalog with sizes, types, and counts |
| `logs` | `logs/app.jsonl` | Log entries |
| `tool_calls` | derived from logs | Tool invocations |
| `errors` | derived | Aggregated errors |
| `session_slices` | `session/after.jsonl` | Session state items |
| `files` | `filesystem/` | Workspace files |
| `config` | `config.json` | Flattened configuration |
| `metrics` | `metrics.json` | Token usage and timing |
| `run_context` | `run_context.json` | Execution IDs |

### Manifest Table Columns

The `manifest` table includes execution summary fields:

| Column | Type | Description |
|--------|------|-------------|
| `bundle_id` | TEXT | Unique bundle identifier |
| `format_version` | TEXT | Bundle format version |
| `created_at` | TEXT | ISO-8601 creation timestamp |
| `status` | TEXT | 'success' or 'error' |
| `duration_ms` | INTEGER | Total execution time |
| `error_count` | INTEGER | Number of errors encountered |
| `prompt_count` | INTEGER | Number of prompts processed |
| `tool_call_count` | INTEGER | Number of tool invocations |
| `provider_call_count` | INTEGER | Number of LLM API calls |
| `input_tokens` | INTEGER | Total input tokens |
| `output_tokens` | INTEGER | Total output tokens |
| `cached_tokens` | INTEGER | Tokens served from cache |

### Artifacts Table Columns

The `artifacts` table exposes the manifest's artifact catalog:

| Column | Type | Description |
|--------|------|-------------|
| `artifact_id` | TEXT | Logical artifact ID (e.g., 'logs', 'session_after') |
| `path` | TEXT | File path within bundle |
| `kind` | TEXT | 'json', 'jsonl', 'text', or 'directory' |
| `content_type` | TEXT | MIME type |
| `size_bytes` | INTEGER | Artifact size |
| `sha256` | TEXT | Content checksum |
| `time_range_start` | TEXT | First timestamp (for logs) |
| `time_range_end` | TEXT | Last timestamp (for logs) |
| `record_count` | INTEGER | Number of records (for logs) |
| `files_captured` | INTEGER | Files archived (for filesystem) |
| `files_skipped` | INTEGER | Files skipped (for filesystem) |
| `total_bytes_captured` | INTEGER | Total bytes (for filesystem) |
| `schema_type` | TEXT | Schema type (for session) |
| `schema_version` | TEXT | Schema version (for session) |

### Optional Tables

| Table | Source | Present When |
|-------|--------|--------------|
| `prompt_overrides` | `prompt_overrides.json` | File exists |
| `eval` | `eval.json` | EvalLoop bundle |

### Dynamic Slice Tables

For each slice type in `session/after.jsonl`, a typed table is created:

```
slice_{normalized_type_name}
```

Example: `myapp.state:AgentPlan` → `slice_agentplan`

## Schema Output

```bash
wink query ./bundle.zip --schema
```

The schema output includes summary statistics for quick bundle assessment:

```json
{
  "bundle_id": "abc123",
  "status": "error",
  "created_at": "2024-01-15T10:30:00Z",
  "duration_ms": 45230,
  "error_count": 1,
  "tool_call_count": 12,
  "artifact_count": 6,
  "log_record_count": 342,
  "tables": [
    {
      "name": "manifest",
      "description": "Bundle metadata and execution summary",
      "row_count": 1,
      "columns": [
        {"name": "bundle_id", "type": "TEXT", "description": ""},
        {"name": "status", "type": "TEXT", "description": ""},
        {"name": "duration_ms", "type": "INTEGER", "description": ""}
      ]
    },
    {
      "name": "artifacts",
      "description": "Artifact catalog with sizes, types, and counts",
      "row_count": 6,
      "columns": [
        {"name": "artifact_id", "type": "TEXT", "description": ""},
        {"name": "path", "type": "TEXT", "description": ""},
        {"name": "size_bytes", "type": "INTEGER", "description": ""}
      ]
    },
    {
      "name": "errors",
      "description": "Aggregated errors",
      "row_count": 1,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": ""},
        {"name": "source", "type": "TEXT", "description": ""},
        {"name": "error_type", "type": "TEXT", "description": ""},
        {"name": "message", "type": "TEXT", "description": ""}
      ]
    }
  ]
}
```

The top-level fields (`duration_ms`, `error_count`, `tool_call_count`, `artifact_count`,
`log_record_count`) provide a quick overview without querying individual tables.

## Example Queries

```sql
-- Quick summary from manifest
SELECT status, duration_ms, error_count, tool_call_count,
       input_tokens, output_tokens
FROM manifest

-- List all artifacts with sizes
SELECT artifact_id, kind, size_bytes, record_count
FROM artifacts ORDER BY size_bytes DESC

-- Check if logs were captured and how many records
SELECT artifact_id, record_count, time_range_start, time_range_end
FROM artifacts WHERE artifact_id = 'logs'

-- Filesystem capture stats
SELECT files_captured, files_skipped, total_bytes_captured
FROM artifacts WHERE artifact_id = 'filesystem'

-- Errors
SELECT error_type, message FROM errors

-- Tool performance
SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms
FROM tool_calls GROUP BY tool_name

-- Failed tools
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0

-- Error logs
SELECT timestamp, message FROM logs WHERE level = 'ERROR'

-- JSON extraction
SELECT json_extract(context, '$.tool_name') FROM logs

-- Session state
SELECT slice_type, COUNT(*) FROM session_slices GROUP BY slice_type

-- File search
SELECT path FROM files WHERE content LIKE '%TODO%'
```

## Caching

Bundle is parsed once and cached as SQLite:

```
./bundle.zip           → input
./bundle.zip.sqlite    → cache
```

Cache invalidated when bundle mtime > cache mtime.

## Implementation

### Database Creation

1. Check cache validity (mtime comparison)
1. If stale/missing, create SQLite at `<bundle>.sqlite`:
   - `manifest.json` → `manifest` (includes summary fields)
   - `manifest.artifacts` → `artifacts` (artifact catalog)
   - `logs/app.jsonl` → `logs`
   - `session/after.jsonl` → `session_slices` + `slice_*`
   - `config.json` → `config` (flattened)
   - `metrics.json` → `metrics`
   - `run_context.json` → `run_context`
   - `filesystem/` → `files`
   - Derive `tool_calls` from logs
   - Derive `errors` from logs + `error.json` + failed tools
   - Optional: `prompt_overrides.json`, `eval.json`
1. Execute query
1. Return JSON or table

### Error Aggregation

`errors` table combines:

- Log entries where `level = 'ERROR'`
- `error.json` contents (if present)
- Tool calls where `success = 0`

Each row has `source` column indicating origin.

### Dynamic Slice Tables

For each `__type__` in session JSONL:

1. Normalize: `myapp.state:AgentPlan` → `slice_agentplan`
1. Infer columns from JSON keys
1. Create table, insert items

## Limitations

- Single bundle only
- SQLite built-in functions only
- Entire bundle loaded on first query

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format
- `specs/SESSIONS.md` - Session state
