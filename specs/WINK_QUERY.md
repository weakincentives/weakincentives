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
| `manifest` | `manifest.json` | Bundle metadata |
| `logs` | `logs/app.jsonl` | Log entries |
| `tool_calls` | derived from logs | Tool invocations |
| `errors` | derived | Aggregated errors |
| `session_slices` | `session/after.jsonl` | Session state items |
| `files` | `filesystem/` | Workspace files |
| `config` | `config.json` | Flattened configuration |
| `metrics` | `metrics.json` | Token usage and timing |
| `run_context` | `run_context.json` | Execution IDs |

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

```json
{
  "bundle_id": "abc123",
  "status": "error",
  "created_at": "2024-01-15T10:30:00Z",
  "tables": [
    {
      "name": "manifest",
      "description": "Bundle metadata",
      "row_count": 1,
      "columns": [
        {"name": "bundle_id", "type": "TEXT", "description": "Bundle identifier"},
        {"name": "status", "type": "TEXT", "description": "'success' or 'error'"},
        {"name": "created_at", "type": "TEXT", "description": "ISO-8601 timestamp"}
      ]
    },
    {
      "name": "errors",
      "description": "Aggregated errors",
      "row_count": 1,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Sequence"},
        {"name": "source", "type": "TEXT", "description": "'log', 'error.json', 'tool_call'"},
        {"name": "error_type", "type": "TEXT", "description": "Exception type"},
        {"name": "message", "type": "TEXT", "description": "Error message"},
        {"name": "traceback", "type": "TEXT", "description": "Stack trace"}
      ]
    }
  ]
}
```

## Example Queries

```sql
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

-- Token usage
SELECT input_tokens, output_tokens, total_ms FROM metrics

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
   - `manifest.json` → `manifest`
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
