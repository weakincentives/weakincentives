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
wink query <BUNDLE> "<SQL>" --table --no-truncate
wink query <BUNDLE> --export-jsonl [logs|session]
```

| Option | Description |
|--------|-------------|
| `--schema` | Output schema as JSON and exit |
| `--table` | Output as ASCII table (default: JSON) |
| `--no-truncate` | Disable column truncation in table output |
| `--export-jsonl` | Export raw JSONL (logs or session) to stdout |

## Usage

```bash
# Always start with schema to discover tables and hints
wink query ./bundle.zip --schema

# Query (JSON output)
wink query ./bundle.zip "SELECT * FROM errors"

# Query (table output)
wink query ./bundle.zip "SELECT * FROM errors" --table

# Full values without truncation
wink query ./bundle.zip "SELECT * FROM tool_calls" --table --no-truncate

# Export raw logs for jq processing
wink query ./bundle.zip --export-jsonl | jq '.event'
wink query ./bundle.zip --export-jsonl=session | jq 'select(.__type__)'
```

## Tables

### Core Tables

| Table | Source | Description |
|-------|--------|-------------|
| `manifest` | `manifest.json` | Bundle metadata |
| `logs` | `logs/app.jsonl` | Log entries (with `seq` column for log_aggregator events) |
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

### Views

Pre-built views for common query patterns:

| View | Description |
|------|-------------|
| `tool_timeline` | Tool calls ordered by timestamp with command extraction |
| `native_tool_calls` | Claude Code native tools from log_aggregator events |
| `error_summary` | Errors with truncated traceback |

## Logs Table

The `logs` table includes a `seq` column extracted from `log_aggregator.log_line`
events. This enables range queries on native tool executions:

```sql
-- Query native tool logs by sequence range
SELECT seq, json_extract(context, '$.content') as content
FROM logs
WHERE event = 'log_aggregator.log_line'
  AND seq BETWEEN 360 AND 370
ORDER BY seq
```

For non-log_aggregator events, `seq` is NULL.

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
      "name": "tool_timeline",
      "description": "View: Tool calls ordered by timestamp",
      "row_count": 5,
      "columns": [...]
    }
  ],
  "hints": {
    "json_extraction": [
      "json_extract(context, '$.tool_name')",
      "json_extract(context, '$.content')",
      "json_extract(params, '$.command')"
    ],
    "common_queries": {
      "native_tools": "SELECT seq, json_extract(context, '$.content') as content FROM logs WHERE event = 'log_aggregator.log_line' AND seq BETWEEN 100 AND 200 ORDER BY seq",
      "tool_timeline": "SELECT * FROM tool_timeline WHERE duration_ms > 1000",
      "error_context": "SELECT timestamp, message, context FROM logs WHERE level = 'ERROR'"
    }
  }
}
```

The `hints` section provides:

- **json_extraction**: Common `json_extract()` patterns for nested JSON data
- **common_queries**: Ready-to-use SQL queries for typical analysis tasks

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

-- JSON extraction (SQLite native function)
SELECT json_extract(context, '$.tool_name') FROM logs

-- Token usage
SELECT input_tokens, output_tokens, total_ms FROM metrics

-- Session state
SELECT slice_type, COUNT(*) FROM session_slices GROUP BY slice_type

-- File search
SELECT path FROM files WHERE content LIKE '%TODO%'

-- Native tools by sequence (Claude Code Bash, Read, Write, etc.)
SELECT seq, json_extract(context, '$.content') as content
FROM logs
WHERE event = 'log_aggregator.log_line'
  AND seq BETWEEN 100 AND 200
ORDER BY seq

-- Use pre-built views
SELECT * FROM tool_timeline WHERE duration_ms > 1000
SELECT * FROM error_summary
SELECT * FROM native_tool_calls LIMIT 10
```

## Raw JSONL Export

For power users who prefer `jq` over SQL:

```bash
# Export logs
wink query ./bundle.zip --export-jsonl | jq 'select(.event == "tool.execution.start")'

# Export session state
wink query ./bundle.zip --export-jsonl=session | jq 'select(.__type__ | contains("Plan"))'
```

## Caching

Bundle is parsed once and cached as SQLite:

```
./bundle.zip           → input
./bundle.zip.sqlite    → cache
```

Cache invalidated when:

- Bundle mtime > cache mtime
- Schema version mismatch (ensures upgrades rebuild caches automatically)

## Implementation

### Database Creation

1. Check cache validity (mtime + schema version)
1. If stale/missing, create SQLite at `<bundle>.sqlite`:
   - `manifest.json` → `manifest`
   - `logs/app.jsonl` → `logs` (with `seq` extraction)
   - `session/after.jsonl` → `session_slices` + `slice_*`
   - `config.json` → `config` (flattened)
   - `metrics.json` → `metrics`
   - `run_context.json` → `run_context`
   - `filesystem/` → `files`
   - Derive `tool_calls` from logs
   - Derive `errors` from logs + `error.json` + failed tools
   - Create views: `tool_timeline`, `native_tool_calls`, `error_summary`
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
- Native tool call content is in nested JSON (use `json_extract()`)
- Extended thinking content is signature-encoded (not accessible as plaintext)

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format
- `specs/SESSIONS.md` - Session state
