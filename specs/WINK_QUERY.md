# Wink Query Specification

## Purpose

`wink query` is a CLI command enabling AI coding agents to explore debug bundles
programmatically. It exposes bundle contents through a SQL interface with dynamic
schemas, making complex investigations expressible as declarative queries.

**Primary use cases:**

- AI agents diagnosing failures in agent executions
- Automated analysis of evaluation runs
- Cross-bundle comparison and trend analysis
- Root cause investigation with structured queries

**Implementation:** (not yet implemented)

## Design Principles

- **SQL-first**: Standard SQL (SQLite dialect) as the universal query language
- **Self-describing**: Schema discovery built-in; no external documentation required
- **Zero configuration**: Point at bundle, start querying
- **Agent-friendly**: Output formats optimized for LLM consumption
- **Composable**: Unix-style piping and scriptability
- **Read-only**: Bundles are immutable; queries never modify source data

## Quick Start

```bash
# Get full schema information (start here)
wink query ./bundle.zip --schema

# Run a query
wink query ./bundle.zip "SELECT * FROM logs WHERE level = 'ERROR'"

# Query multiple bundles
wink query ./debug/ "SELECT bundle_id, status FROM manifest"

# Export results as JSON
wink query ./bundle.zip "SELECT * FROM tool_calls" --format json
```

## SQL Interface

### Virtual Database

Each bundle is loaded as a virtual SQLite database with tables derived from
bundle artifacts. The schema is generated dynamically based on actual bundle
contents.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Virtual Database                              │
├─────────────────────────────────────────────────────────────────┤
│  Core Tables (always present)                                    │
│  ├── manifest          # Bundle metadata                         │
│  ├── logs              # Log entries from logs/app.jsonl         │
│  ├── session_slices    # All session state items                 │
│  ├── tool_calls        # Extracted tool invocations              │
│  ├── files             # Filesystem manifest                     │
│  └── config            # Flattened configuration                 │
├─────────────────────────────────────────────────────────────────┤
│  Derived Tables (computed on load)                               │
│  ├── errors            # Aggregated errors and exceptions        │
│  ├── timeline          # Unified chronological event stream      │
│  ├── token_usage       # Token consumption by phase              │
│  └── state_changes     # Session mutations over time             │
├─────────────────────────────────────────────────────────────────┤
│  Typed Slice Tables (dynamic, per slice type)                    │
│  ├── slice_{type_name} # One table per registered slice type     │
│  └── ...                                                         │
├─────────────────────────────────────────────────────────────────┤
│  Optional Tables (when artifact present)                         │
│  ├── error             # error.json contents                     │
│  ├── eval              # eval.json contents                      │
│  └── metrics           # metrics.json contents                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Tables

#### `manifest`

Bundle metadata from `manifest.json`.

| Column | Type | Description |
|--------|------|-------------|
| `bundle_id` | TEXT | Unique bundle identifier |
| `format_version` | TEXT | Bundle format version |
| `created_at` | TEXT | ISO-8601 creation timestamp |
| `request_id` | TEXT | Associated request ID |
| `session_id` | TEXT | Associated session ID |
| `status` | TEXT | `success` or `error` |
| `started_at` | TEXT | Execution start time |
| `ended_at` | TEXT | Execution end time |
| `duration_ms` | INTEGER | Computed execution duration |
| `capture_mode` | TEXT | `minimal`, `standard`, or `full` |
| `prompt_ns` | TEXT | Prompt namespace |
| `prompt_key` | TEXT | Prompt key |
| `adapter` | TEXT | Adapter name |

#### `logs`

Log entries from `logs/app.jsonl`.

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Entry sequence number |
| `timestamp` | TEXT | ISO-8601 timestamp |
| `timestamp_ms` | INTEGER | Unix timestamp in milliseconds |
| `level` | TEXT | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `logger` | TEXT | Logger name |
| `event` | TEXT | Event name (e.g., `tool.execution.complete`) |
| `message` | TEXT | Log message |
| `context` | TEXT | JSON-encoded context object |

Indexes: `timestamp_ms`, `level`, `event`, `logger`.

**Context extraction**: Nested context fields accessible via JSON functions:

```sql
SELECT json_extract(context, '$.tool_name') AS tool
FROM logs
WHERE event = 'tool.execution.complete';
```

#### `session_slices`

Unified view of all session state from `session/after.jsonl`.

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Item sequence number |
| `slice_type` | TEXT | Qualified type name (e.g., `myapp.state:Plan`) |
| `slice_name` | TEXT | Short type name |
| `timestamp` | TEXT | When item was added (if available) |
| `data` | TEXT | JSON-encoded item |

#### `tool_calls`

Extracted tool invocations (derived from logs and session).

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Call sequence number |
| `tool_name` | TEXT | Tool identifier |
| `started_at` | TEXT | Invocation start time |
| `ended_at` | TEXT | Invocation end time |
| `duration_ms` | INTEGER | Execution duration |
| `success` | INTEGER | 1 if successful, 0 if failed |
| `error_code` | TEXT | Error code if failed |
| `params` | TEXT | JSON-encoded parameters |
| `result` | TEXT | JSON-encoded result (truncated if large) |

Indexes: `tool_name`, `success`, `started_at`.

#### `files`

Filesystem snapshot manifest from `filesystem/`.

| Column | Type | Description |
|--------|------|-------------|
| `path` | TEXT | Relative file path |
| `size` | INTEGER | File size in bytes |
| `modified_at` | TEXT | Last modification time |
| `is_binary` | INTEGER | 1 if binary, 0 if text |
| `truncated` | INTEGER | 1 if content was truncated |

#### `config`

Flattened configuration from `config.json`.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | Dot-notation config path |
| `value` | TEXT | Configuration value |
| `type` | TEXT | Value type (`string`, `number`, `boolean`, `null`) |

### Derived Tables

#### `errors`

Aggregated error information across all sources.

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Error sequence |
| `source` | TEXT | `log`, `error.json`, `tool_call` |
| `timestamp` | TEXT | When error occurred |
| `error_type` | TEXT | Exception or error type |
| `message` | TEXT | Error message |
| `phase` | TEXT | Execution phase (`render`, `call`, `parse`, `tool`) |
| `traceback` | TEXT | Stack trace if available |
| `context` | TEXT | JSON-encoded context |

#### `timeline`

Unified chronological event stream.

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Event sequence |
| `timestamp` | TEXT | ISO-8601 timestamp |
| `timestamp_ms` | INTEGER | Unix timestamp in milliseconds |
| `event_type` | TEXT | `log`, `tool_call`, `state_change`, `error` |
| `event_name` | TEXT | Specific event identifier |
| `summary` | TEXT | Human-readable summary |
| `details` | TEXT | JSON-encoded full event data |

Indexes: `timestamp_ms`, `event_type`.

#### `token_usage`

Token consumption breakdown.

| Column | Type | Description |
|--------|------|-------------|
| `phase` | TEXT | `input`, `output`, `cached`, `total` |
| `tokens` | INTEGER | Token count |
| `cost_usd` | REAL | Estimated cost (if pricing available) |

#### `state_changes`

Session state mutations over time (requires `session/before.jsonl`).

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Change sequence |
| `timestamp` | TEXT | When change occurred |
| `slice_type` | TEXT | Affected slice type |
| `operation` | TEXT | `append`, `replace`, `clear` |
| `before_count` | INTEGER | Items before change |
| `after_count` | INTEGER | Items after change |
| `diff` | TEXT | JSON-encoded change summary |

### Dynamic Slice Tables

For each unique slice type in session state, a typed table is generated:

```
slice_{normalized_type_name}
```

Type names are normalized: `myapp.state:AgentPlan` becomes `slice_agentplan`.

The table schema is derived from the dataclass fields:

```python
@dataclass(frozen=True)
class AgentPlan:
    goal: str
    steps: tuple[str, ...]
    status: Literal["pending", "active", "complete"]
```

Generates:

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | INTEGER | Item sequence |
| `goal` | TEXT | String field |
| `steps` | TEXT | JSON-encoded tuple |
| `status` | TEXT | Literal value |
| `_raw` | TEXT | Full JSON representation |

Nested objects and complex types are JSON-encoded with accessor support.

### Optional Tables

#### `error` (singular)

Detailed error information from `error.json` (only present when execution failed).

| Column | Type | Description |
|--------|------|-------------|
| `error_type` | TEXT | Exception class name |
| `message` | TEXT | Error message |
| `phase` | TEXT | Execution phase when error occurred |
| `traceback` | TEXT | Full stack trace |
| `context` | TEXT | JSON-encoded error context |

#### `eval`

Evaluation metadata from `eval.json` (only present for eval bundles).

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | TEXT | Evaluation sample identifier |
| `experiment` | TEXT | Experiment name |
| `score` | REAL | Evaluation score |
| `passed` | INTEGER | 1 if passed threshold |
| `judge_output` | TEXT | Judge reasoning (if LLM judge) |
| `metadata` | TEXT | JSON-encoded additional metadata |

#### `metrics`

Performance metrics from `metrics.json`.

| Column | Type | Description |
|--------|------|-------------|
| `name` | TEXT | Metric name |
| `value` | REAL | Metric value |
| `unit` | TEXT | Unit of measurement |
| `labels` | TEXT | JSON-encoded label set |

## Multi-Bundle Queries

When pointing at a directory, all bundles are loaded into a unified database
with `bundle_id` columns for disambiguation.

```bash
# Query across all bundles in directory
wink query ./debug/ "
  SELECT bundle_id, COUNT(*) as error_count
  FROM errors
  GROUP BY bundle_id
  ORDER BY error_count DESC
  LIMIT 10
"
```

Each table gains a `bundle_id` column when multiple bundles are loaded.

### Bundle Selection

```sql
-- Filter by bundle
SELECT * FROM logs WHERE bundle_id = 'abc123'

-- Compare bundles
SELECT
  bundle_id,
  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
  SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
FROM tool_calls
GROUP BY bundle_id
```

## CLI Reference

### Synopsis

```
wink query [OPTIONS] <BUNDLE_PATH> <QUERY>
wink query [OPTIONS] <BUNDLE_PATH> --schema
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to `.zip` bundle or directory containing bundles |
| `QUERY` | SQL query to execute (required unless `--schema` is used) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--schema` | - | Output full schema information as JSON and exit |
| `--format`, `-f` | `table` | Output format: `table`, `json`, `jsonl`, `csv`, `markdown` |
| `--limit`, `-n` | `1000` | Maximum rows to return (0 for unlimited) |
| `--no-headers` | false | Omit column headers (csv/table) |
| `--load-file FILE` | - | Execute SQL from file instead of QUERY argument |
| `--output`, `-o` | stdout | Write results to file |
| `--quiet`, `-q` | false | Suppress informational messages |
| `--timing` | false | Show query execution time |
| `--explain` | false | Show query plan instead of results |

### Output Formats

#### `table` (default)

Human-readable ASCII table:

```
┌──────────┬───────┬─────────────────────────────┐
│ tool_name│ count │ avg_duration_ms             │
├──────────┼───────┼─────────────────────────────┤
│ read_file│    42 │ 12.3                        │
│ write    │    15 │ 45.7                        │
│ execute  │     8 │ 1523.4                      │
└──────────┴───────┴─────────────────────────────┘
```

#### `json`

JSON array of objects:

```json
[
  {"tool_name": "read_file", "count": 42, "avg_duration_ms": 12.3},
  {"tool_name": "write", "count": 15, "avg_duration_ms": 45.7}
]
```

#### `jsonl`

Newline-delimited JSON (streaming-friendly):

```json
{"tool_name": "read_file", "count": 42, "avg_duration_ms": 12.3}
{"tool_name": "write", "count": 15, "avg_duration_ms": 45.7}
```

#### `csv`

Standard CSV:

```csv
tool_name,count,avg_duration_ms
read_file,42,12.3
write,15,45.7
```

#### `markdown`

GitHub-flavored markdown table:

```markdown
| tool_name | count | avg_duration_ms |
|-----------|-------|-----------------|
| read_file | 42    | 12.3            |
| write     | 15    | 45.7            |
```

## Schema Discovery

The `--schema` flag outputs comprehensive schema information as JSON, providing
everything an AI agent needs to construct valid queries.

```bash
wink query ./bundle.zip --schema
```

### Schema Output Format

```json
{
  "bundle": {
    "bundle_id": "debug_abc123_2024-01-15",
    "format_version": "1.0",
    "status": "error",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "tables": [
    {
      "name": "manifest",
      "description": "Bundle metadata from manifest.json",
      "row_count": 1,
      "category": "core",
      "columns": [
        {
          "name": "bundle_id",
          "type": "TEXT",
          "description": "Unique bundle identifier",
          "indexed": false
        },
        {
          "name": "status",
          "type": "TEXT",
          "description": "Execution status: 'success' or 'error'",
          "indexed": false
        }
      ],
      "indexes": [],
      "sample_queries": [
        "SELECT status, duration_ms FROM manifest"
      ]
    },
    {
      "name": "logs",
      "description": "Log entries from logs/app.jsonl",
      "row_count": 1523,
      "category": "core",
      "columns": [
        {
          "name": "_rowid",
          "type": "INTEGER",
          "description": "Entry sequence number",
          "indexed": false
        },
        {
          "name": "timestamp",
          "type": "TEXT",
          "description": "ISO-8601 timestamp",
          "indexed": false
        },
        {
          "name": "timestamp_ms",
          "type": "INTEGER",
          "description": "Unix timestamp in milliseconds",
          "indexed": true
        },
        {
          "name": "level",
          "type": "TEXT",
          "description": "Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
          "indexed": true
        },
        {
          "name": "event",
          "type": "TEXT",
          "description": "Structured event name (e.g., 'tool.execution.complete')",
          "indexed": true
        },
        {
          "name": "logger",
          "type": "TEXT",
          "description": "Logger name",
          "indexed": true
        },
        {
          "name": "message",
          "type": "TEXT",
          "description": "Log message text",
          "indexed": false
        },
        {
          "name": "context",
          "type": "TEXT",
          "description": "JSON-encoded context object; use json_extract() to access fields",
          "indexed": false
        }
      ],
      "indexes": [
        {"name": "idx_logs_timestamp_ms", "columns": ["timestamp_ms"]},
        {"name": "idx_logs_level", "columns": ["level"]},
        {"name": "idx_logs_event", "columns": ["event"]},
        {"name": "idx_logs_logger", "columns": ["logger"]}
      ],
      "sample_queries": [
        "SELECT timestamp, level, message FROM logs WHERE level = 'ERROR'",
        "SELECT json_extract(context, '$.tool_name') FROM logs WHERE event = 'tool.execution.complete'"
      ]
    },
    {
      "name": "errors",
      "description": "Aggregated errors from all sources (logs, error.json, tool failures)",
      "row_count": 1,
      "category": "derived",
      "columns": [
        {"name": "_rowid", "type": "INTEGER", "description": "Error sequence", "indexed": false},
        {"name": "source", "type": "TEXT", "description": "Error source: 'log', 'error.json', 'tool_call'", "indexed": false},
        {"name": "timestamp", "type": "TEXT", "description": "When error occurred", "indexed": false},
        {"name": "error_type", "type": "TEXT", "description": "Exception or error type name", "indexed": false},
        {"name": "message", "type": "TEXT", "description": "Error message", "indexed": false},
        {"name": "phase", "type": "TEXT", "description": "Execution phase: 'render', 'call', 'parse', 'tool'", "indexed": false},
        {"name": "traceback", "type": "TEXT", "description": "Stack trace if available", "indexed": false},
        {"name": "context", "type": "TEXT", "description": "JSON-encoded error context", "indexed": false}
      ],
      "indexes": [],
      "sample_queries": [
        "SELECT error_type, message, phase FROM errors",
        "SELECT * FROM errors ORDER BY timestamp"
      ]
    },
    {
      "name": "slice_agentplan",
      "description": "Session state: AgentPlan instances from myapp.state:AgentPlan",
      "row_count": 3,
      "category": "slice",
      "source_type": "myapp.state:AgentPlan",
      "columns": [
        {"name": "_rowid", "type": "INTEGER", "description": "Item sequence", "indexed": false},
        {"name": "goal", "type": "TEXT", "description": "Field: str", "indexed": false},
        {"name": "steps", "type": "TEXT", "description": "Field: tuple[str, ...] (JSON-encoded)", "indexed": false},
        {"name": "status", "type": "TEXT", "description": "Field: Literal['pending', 'active', 'complete']", "indexed": false},
        {"name": "_raw", "type": "TEXT", "description": "Full JSON representation", "indexed": false}
      ],
      "indexes": [],
      "sample_queries": [
        "SELECT goal, status FROM slice_agentplan",
        "SELECT json_extract(steps, '$[0]') as first_step FROM slice_agentplan"
      ]
    }
  ],
  "functions": [
    {
      "name": "file_content",
      "signature": "file_content(path TEXT) -> TEXT",
      "description": "Read file content from filesystem snapshot; returns NULL if binary or missing"
    },
    {
      "name": "json_extract",
      "signature": "json_extract(json TEXT, path TEXT) -> ANY",
      "description": "SQLite built-in: extract value from JSON using path like '$.field'"
    },
    {
      "name": "json_pretty",
      "signature": "json_pretty(json TEXT) -> TEXT",
      "description": "Pretty-print JSON string with indentation"
    },
    {
      "name": "duration_fmt",
      "signature": "duration_fmt(ms INTEGER) -> TEXT",
      "description": "Format milliseconds as human-readable (e.g., '1.5s', '2m 30s')"
    },
    {
      "name": "truncate",
      "signature": "truncate(text TEXT, max_len INTEGER) -> TEXT",
      "description": "Truncate text with ellipsis if exceeds max_len"
    }
  ],
  "suggested_queries": {
    "error_investigation": [
      "SELECT error_type, message, phase FROM errors",
      "SELECT * FROM timeline WHERE timestamp_ms BETWEEN (SELECT timestamp_ms - 10000 FROM errors LIMIT 1) AND (SELECT timestamp_ms + 2000 FROM errors LIMIT 1) ORDER BY timestamp_ms"
    ],
    "tool_analysis": [
      "SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms, SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures FROM tool_calls GROUP BY tool_name ORDER BY calls DESC",
      "SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0"
    ],
    "token_usage": [
      "SELECT phase, tokens FROM token_usage"
    ],
    "state_analysis": [
      "SELECT slice_type, COUNT(*) as items FROM session_slices GROUP BY slice_type"
    ]
  }
}
```

### Schema Categories

| Category | Description |
|----------|-------------|
| `core` | Always present: manifest, logs, session_slices, tool_calls, files, config |
| `derived` | Computed on load: errors, timeline, token_usage, state_changes |
| `slice` | Dynamic per slice type: slice\_{typename} |
| `optional` | Present when artifact exists: error, eval, metrics |

The schema output includes row counts and sample queries for each table,
enabling agents to understand both structure and content at a glance.

## Common Query Patterns

### Error Investigation

```sql
-- Find all errors
SELECT timestamp, error_type, message, phase
FROM errors
ORDER BY timestamp;

-- Get context around first error
SELECT *
FROM timeline
WHERE timestamp_ms BETWEEN
  (SELECT timestamp_ms - 10000 FROM errors ORDER BY timestamp LIMIT 1)
  AND
  (SELECT timestamp_ms + 2000 FROM errors ORDER BY timestamp LIMIT 1)
ORDER BY timestamp_ms;

-- Tool failures
SELECT tool_name, error_code, params
FROM tool_calls
WHERE success = 0;
```

### Performance Analysis

```sql
-- Slowest tool calls
SELECT tool_name, duration_ms, params
FROM tool_calls
ORDER BY duration_ms DESC
LIMIT 10;

-- Tool call frequency
SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms
FROM tool_calls
GROUP BY tool_name
ORDER BY calls DESC;

-- Time spent in each phase
SELECT
  'render' as phase,
  SUM(json_extract(context, '$.render_ms')) as ms
FROM logs WHERE event = 'adapter.call'
UNION ALL
SELECT 'llm_call', SUM(json_extract(context, '$.call_ms'))
FROM logs WHERE event = 'adapter.call'
UNION ALL
SELECT 'tools', SUM(duration_ms)
FROM tool_calls;
```

### State Analysis

```sql
-- What state types exist?
SELECT slice_type, COUNT(*) as items
FROM session_slices
GROUP BY slice_type;

-- Examine specific state
SELECT * FROM slice_agentplan;

-- State changes over time
SELECT timestamp, slice_type, operation, before_count, after_count
FROM state_changes
ORDER BY timestamp;
```

### Log Analysis

```sql
-- Error and warning logs
SELECT timestamp, logger, message
FROM logs
WHERE level IN ('ERROR', 'WARNING')
ORDER BY timestamp;

-- Logs from specific component
SELECT timestamp, event, message, context
FROM logs
WHERE logger LIKE 'weakincentives.adapters%'
ORDER BY timestamp;

-- Search log messages
SELECT timestamp, level, message
FROM logs
WHERE message LIKE '%timeout%'
ORDER BY timestamp;
```

### Multi-Bundle Analysis

```sql
-- Success rate across bundles
SELECT
  bundle_id,
  (SELECT status FROM manifest WHERE manifest.bundle_id = t.bundle_id) as status,
  COUNT(*) as tool_calls,
  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM tool_calls t
GROUP BY bundle_id;

-- Find bundles with specific errors
SELECT DISTINCT bundle_id
FROM errors
WHERE error_type = 'TimeoutError';

-- Compare token usage
SELECT bundle_id, phase, tokens
FROM token_usage
WHERE phase = 'total'
ORDER BY tokens DESC;
```

## File Content Access

For bundles with filesystem snapshots, file contents can be queried:

```sql
-- List all files
SELECT path, size, modified_at
FROM files
WHERE NOT is_binary
ORDER BY modified_at DESC;

-- Find large files
SELECT path, size
FROM files
ORDER BY size DESC
LIMIT 10;
```

File contents are accessed via a special function:

```sql
-- Read file content (returns TEXT, NULL if binary/missing)
SELECT file_content('workspace/src/main.py') as content;

-- Search in files
SELECT path
FROM files
WHERE NOT is_binary
  AND file_content(path) LIKE '%TODO%';
```

## Extension Functions

Custom SQL functions for bundle analysis:

| Function | Description |
|----------|-------------|
| `file_content(path)` | Read file content from filesystem snapshot |
| `json_pretty(json)` | Pretty-print JSON string |
| `duration_fmt(ms)` | Format milliseconds as human-readable |
| `timestamp_fmt(iso)` | Format ISO timestamp for display |
| `truncate(text, len)` | Truncate text with ellipsis |
| `extract_error(json)` | Extract error message from context |
| `path_basename(path)` | Extract filename from path |
| `path_dirname(path)` | Extract directory from path |
| `path_ext(path)` | Extract file extension |

## Agent Integration

### Structured Output for LLMs

The `--format json` and `--format jsonl` options produce output suitable for
LLM consumption:

```bash
# Get structured error info for agent analysis
wink query ./bundle.zip "SELECT * FROM errors" --format json
```

### Scripted Analysis

```bash
#!/bin/bash
# analyze-bundle.sh - Extract key metrics for agent consumption

BUNDLE=$1

echo "=== Bundle Summary ==="
wink query "$BUNDLE" "SELECT * FROM manifest" --format json

echo "=== Errors ==="
wink query "$BUNDLE" "SELECT * FROM errors" --format json

echo "=== Tool Performance ==="
wink query "$BUNDLE" "
  SELECT tool_name, COUNT(*) as calls,
         AVG(duration_ms) as avg_ms,
         SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
  FROM tool_calls
  GROUP BY tool_name
" --format json

echo "=== Token Usage ==="
wink query "$BUNDLE" "SELECT * FROM token_usage" --format json
```

### Diagnostic Workflow

Recommended investigation sequence for AI agents:

```bash
# 1. Get schema (includes bundle status, row counts, and suggested queries)
wink query ./bundle.zip --schema

# 2. Check for errors
wink query ./bundle.zip "SELECT * FROM errors" --format json

# 3. If errors found, get timeline context
wink query ./bundle.zip "
  SELECT event_type, event_name, timestamp, summary
  FROM timeline
  WHERE timestamp_ms BETWEEN
    (SELECT timestamp_ms - 10000 FROM errors ORDER BY timestamp LIMIT 1)
    AND
    (SELECT timestamp_ms + 2000 FROM errors ORDER BY timestamp LIMIT 1)
  ORDER BY timestamp_ms
" --format json

# 4. Analyze tool calls
wink query ./bundle.zip "
  SELECT tool_name, success, duration_ms, error_code
  FROM tool_calls
  ORDER BY started_at
" --format json

# 5. Check final state
wink query ./bundle.zip "
  SELECT slice_type, COUNT(*) as items
  FROM session_slices
  GROUP BY slice_type
" --format json
```

## Implementation Notes

### Virtual Table Architecture

Tables are implemented as SQLite virtual tables using
[APSW](https://rogerbinns.github.io/apsw/) (Another Python SQLite Wrapper).
APSW is chosen over the standard library `sqlite3` module because it provides
complete access to SQLite's virtual table API.

**Why APSW:**

- Full virtual table support via `apsw.VTTable` and `apsw.VTModule` classes
- Access to `xBestIndex` for query planner hints and constraint propagation
- Proper `xFilter` implementation for predicate pushdown
- Support for `xColumn` lazy evaluation (columns computed on demand)
- Complete SQLite C API exposure (standard `sqlite3` has limited vtable support)

**Virtual Table Implementation:**

```python
import apsw

class BundleModule:
    """APSW virtual table module for debug bundle tables."""

    def __init__(self, bundle: DebugBundle) -> None:
        self._bundle = bundle

    def Create(
        self,
        db: apsw.Connection,
        modulename: str,
        dbname: str,
        tablename: str,
        *args: str,
    ) -> tuple[str, "BundleTable"]:
        # Return schema DDL and table instance
        schema = self._bundle.schema_for(tablename)
        return schema.to_ddl(), BundleTable(self._bundle, tablename, schema)

    Connect = Create  # Same behavior for CREATE and connect


class BundleTable:
    """Virtual table backed by bundle artifact."""

    def __init__(
        self, bundle: DebugBundle, name: str, schema: TableSchema
    ) -> None:
        self._bundle = bundle
        self._name = name
        self._schema = schema

    def BestIndex(
        self, constraints: list[tuple[int, int]], orderbys: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, bool]], int, str, bool, int]:
        # Inform query planner about usable indexes
        # Return: (constraint_usage, index_num, index_str, order_satisfied, cost)
        ...

    def Open(self) -> "BundleCursor":
        return BundleCursor(self._bundle, self._name, self._schema)


class BundleCursor:
    """Cursor for iterating bundle data with predicate pushdown."""

    def Filter(
        self, indexnum: int, indexstring: str, constraintargs: tuple
    ) -> None:
        # Apply pushed-down predicates before iteration
        self._predicate = self._build_predicate(indexnum, constraintargs)
        self._iterator = self._bundle.iterate(self._name, self._predicate)
        self._current = None
        self.Next()

    def Next(self) -> None:
        self._current = next(self._iterator, None)

    def Eof(self) -> bool:
        return self._current is None

    def Column(self, n: int) -> Any:
        # Lazy column evaluation - only compute requested columns
        return self._schema.extract_column(self._current, n)

    def Rowid(self) -> int:
        return self._current._rowid
```

**Key capabilities enabled by APSW:**

- Lazy loading of large artifacts (rows loaded during iteration)
- Memory-efficient streaming (no full materialization)
- Predicate pushdown via `BestIndex`/`Filter` (skip non-matching rows early)
- Custom indexing strategies communicated to query planner
- Column-level lazy evaluation (complex JSON parsing deferred)

### Memory Management

- Large JSONL files streamed, not fully loaded
- File content function reads on-demand
- Result pagination via LIMIT/OFFSET
- Configurable row limits prevent runaway queries

### Performance Considerations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Table scan | O(n) | Full file read for JSONL |
| Indexed lookup | O(log n) | For indexed columns |
| JSON extraction | O(m) | Per-row JSON parsing |
| File content | O(k) | Decompression + read |
| Cross-bundle join | O(n\*m) | Consider filtering first |

### Caching

- Schema cached per bundle
- Parsed JSONL cached for repeated queries
- File content LRU cache (configurable size)

## Limitations

- **Read-only**: Cannot modify bundle contents
- **SQLite dialect**: Standard SQL with SQLite extensions
- **Memory bounds**: Large bundles may require streaming queries
- **No live updates**: Bundle state fixed at load time
- **JSON column limits**: Deeply nested JSON requires multiple extractions

## Security Considerations

- Bundles may contain sensitive data (API keys, PII)
- File content function respects bundle boundaries (no path traversal)
- Query output may expose sensitive information
- Consider access controls when sharing query results

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format and contents
- `specs/WINK_DEBUG.md` - Web UI for bundle exploration
- `specs/SESSIONS.md` - Session state structure
- `specs/SLICES.md` - Slice storage and types
- `specs/LOGGING.md` - Log record format
- `specs/METRICS.md` - Metrics structure
