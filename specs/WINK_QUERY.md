# Wink Query Specification

## Purpose

`wink query` is a CLI command enabling AI coding agents to explore debug bundles
via SQL. It loads bundle contents into a SQLite database with a dynamic schema
based on bundle contents.

**Primary use case:** AI agents diagnosing failures in agent executions.

**Implementation:** (not yet implemented)

## Design Principles

- **SQL-first**: Standard SQL (SQLite dialect) as the query language
- **Self-describing**: `--schema` provides everything needed to construct queries
- **Zero configuration**: Point at bundle, start querying
- **Cached**: Bundle parsed once, cached as SQLite file for fast repeated queries

## Quick Start

```bash
# Get schema (always start here)
wink query ./bundle.zip --schema

# Run query (JSON output, default)
wink query ./bundle.zip "SELECT * FROM errors"

# Run query (table output for humans)
wink query ./bundle.zip "SELECT * FROM errors" --table
```

## CLI Reference

### Synopsis

```
wink query <BUNDLE> --schema
wink query <BUNDLE> "<SQL>"
wink query <BUNDLE> "<SQL>" --table
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE` | Path to `.zip` debug bundle |
| `SQL` | SQL query to execute |

### Options

| Option | Description |
|--------|-------------|
| `--schema` | Output schema as JSON and exit |
| `--table` | Output results as ASCII table (default: JSON) |

## Schema Output

The `--schema` flag outputs schema information as JSON.

```bash
wink query ./bundle.zip --schema
```

```json
{
  "bundle_id": "debug_abc123_2024-01-15",
  "status": "error",
  "created_at": "2024-01-15T10:30:00Z",
  "tables": [
    {
      "name": "manifest",
      "description": "Bundle metadata",
      "row_count": 1,
      "columns": [
        {"name": "bundle_id", "type": "TEXT", "description": "Unique bundle identifier"},
        {"name": "status", "type": "TEXT", "description": "'success' or 'error'"},
        {"name": "created_at", "type": "TEXT", "description": "ISO-8601 timestamp"},
        {"name": "duration_ms", "type": "INTEGER", "description": "Execution duration"}
      ]
    },
    {
      "name": "logs",
      "description": "Log entries from logs/app.jsonl",
      "row_count": 1523,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Entry sequence number"},
        {"name": "timestamp", "type": "TEXT", "description": "ISO-8601 timestamp"},
        {"name": "level", "type": "TEXT", "description": "DEBUG, INFO, WARNING, ERROR, CRITICAL"},
        {"name": "logger", "type": "TEXT", "description": "Logger name"},
        {"name": "message", "type": "TEXT", "description": "Log message"},
        {"name": "context", "type": "TEXT", "description": "JSON object; use json_extract()"}
      ]
    },
    {
      "name": "tool_calls",
      "description": "Tool invocations",
      "row_count": 47,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Call sequence number"},
        {"name": "tool_name", "type": "TEXT", "description": "Tool identifier"},
        {"name": "started_at", "type": "TEXT", "description": "Start timestamp"},
        {"name": "duration_ms", "type": "INTEGER", "description": "Execution time"},
        {"name": "success", "type": "INTEGER", "description": "1=success, 0=failure"},
        {"name": "error_code", "type": "TEXT", "description": "Error code if failed"},
        {"name": "params", "type": "TEXT", "description": "JSON parameters"},
        {"name": "result", "type": "TEXT", "description": "JSON result"}
      ]
    },
    {
      "name": "errors",
      "description": "Aggregated errors from all sources",
      "row_count": 1,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Error sequence"},
        {"name": "source", "type": "TEXT", "description": "'log', 'error.json', or 'tool_call'"},
        {"name": "timestamp", "type": "TEXT", "description": "When error occurred"},
        {"name": "error_type", "type": "TEXT", "description": "Exception type"},
        {"name": "message", "type": "TEXT", "description": "Error message"},
        {"name": "traceback", "type": "TEXT", "description": "Stack trace if available"}
      ]
    },
    {
      "name": "session_slices",
      "description": "All session state items",
      "row_count": 12,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Item sequence"},
        {"name": "slice_type", "type": "TEXT", "description": "Qualified type (e.g., 'myapp.state:Plan')"},
        {"name": "data", "type": "TEXT", "description": "JSON item; use json_extract()"}
      ]
    },
    {
      "name": "files",
      "description": "Filesystem snapshot",
      "row_count": 25,
      "columns": [
        {"name": "path", "type": "TEXT", "description": "Relative file path"},
        {"name": "size", "type": "INTEGER", "description": "Size in bytes"},
        {"name": "content", "type": "TEXT", "description": "File content (NULL if binary)"}
      ]
    },
    {
      "name": "config",
      "description": "Flattened configuration",
      "row_count": 8,
      "columns": [
        {"name": "key", "type": "TEXT", "description": "Dot-notation path"},
        {"name": "value", "type": "TEXT", "description": "Configuration value"}
      ]
    },
    {
      "name": "slice_agentplan",
      "description": "Typed view: myapp.state:AgentPlan",
      "row_count": 3,
      "columns": [
        {"name": "rowid", "type": "INTEGER", "description": "Item sequence"},
        {"name": "goal", "type": "TEXT", "description": "Field: str"},
        {"name": "steps", "type": "TEXT", "description": "Field: tuple[str, ...] (JSON)"},
        {"name": "status", "type": "TEXT", "description": "Field: Literal['pending', 'active', 'complete']"}
      ]
    }
  ]
}
```

## Tables

### Core Tables

| Table | Description |
|-------|-------------|
| `manifest` | Bundle metadata (status, timestamps, duration) |
| `logs` | Log entries from `logs/app.jsonl` |
| `tool_calls` | Tool invocations with timing and results |
| `errors` | Aggregated errors from logs, error.json, and failed tools |
| `session_slices` | All session state as JSON items |
| `files` | Filesystem snapshot with content |
| `config` | Flattened configuration key-value pairs |

### Dynamic Slice Tables

For each slice type in session state, a typed table is generated:

```
slice_{normalized_type_name}
```

Example: `myapp.state:AgentPlan` → `slice_agentplan`

Columns are derived from dataclass fields. Complex types are JSON-encoded.

## Output Formats

### JSON (default)

```bash
wink query ./bundle.zip "SELECT tool_name, duration_ms FROM tool_calls"
```

```json
[
  {"tool_name": "read_file", "duration_ms": 12},
  {"tool_name": "write", "duration_ms": 45}
]
```

### Table

```bash
wink query ./bundle.zip "SELECT tool_name, duration_ms FROM tool_calls" --table
```

```
┌───────────┬─────────────┐
│ tool_name │ duration_ms │
├───────────┼─────────────┤
│ read_file │ 12          │
│ write     │ 45          │
└───────────┴─────────────┘
```

## SQL Functions

Only SQLite built-in functions are available. Key functions for bundle analysis:

| Function | Example |
|----------|---------|
| `json_extract(col, path)` | `json_extract(context, '$.tool_name')` |
| `json_array_length(col)` | `json_array_length(steps)` |
| `json_each(col)` | `SELECT * FROM json_each(steps)` |
| `instr(str, substr)` | `instr(message, 'timeout')` |
| `substr(str, start, len)` | `substr(traceback, 1, 500)` |

## Example Queries

```sql
-- Find all errors
SELECT error_type, message FROM errors

-- Tool performance
SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms
FROM tool_calls GROUP BY tool_name ORDER BY calls DESC

-- Failed tools
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0

-- Error logs
SELECT timestamp, message FROM logs WHERE level = 'ERROR'

-- Extract from JSON context
SELECT json_extract(context, '$.request_id') as req_id FROM logs LIMIT 5

-- Session state types
SELECT slice_type, COUNT(*) as items FROM session_slices GROUP BY slice_type

-- Search file contents
SELECT path FROM files WHERE content LIKE '%TODO%'
```

## Caching

On first query, the bundle is parsed and loaded into a SQLite database file:

```
./bundle.zip           → input bundle
./bundle.zip.sqlite    → cached SQLite database
```

Subsequent queries use the cached database directly. The cache is invalidated
if the bundle's modification time is newer than the cache file.

**Cache location:** Same directory as the input bundle.

## Implementation Notes

### Database Creation

1. Extract bundle to temp directory (or read from zip directly)
1. Create SQLite database at `<bundle-path>.sqlite`
1. Create tables and insert data:
   - Parse `manifest.json` → `manifest` table
   - Parse `logs/app.jsonl` → `logs` table
   - Parse `session/after.jsonl` → `session_slices` table + dynamic `slice_*` tables
   - Derive `tool_calls` from logs (filter by event type)
   - Derive `errors` from logs + `error.json` + failed tool calls
   - Parse `config.json` → `config` table (flattened)
   - Read `filesystem/` → `files` table
1. Execute query against database
1. Return results as JSON or table

### Dynamic Slice Tables

For each unique `__type__` in `session/after.jsonl`:

1. Normalize type name: `myapp.state:AgentPlan` → `slice_agentplan`
1. Infer columns from first item's JSON keys
1. Create table with inferred schema
1. Insert all items of that type

### Error Aggregation

The `errors` table combines:

- Log entries where `level = 'ERROR'`
- Contents of `error.json` (if present)
- Tool calls where `success = 0`

Each row includes `source` column indicating origin.

## Limitations

- **Single bundle**: No multi-bundle queries
- **Read-only**: Cannot modify bundle contents
- **SQLite built-ins only**: No custom functions
- **Memory**: Entire bundle loaded into SQLite on first query

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format and contents
- `specs/SESSIONS.md` - Session state structure
- `specs/SLICES.md` - Slice storage and types
