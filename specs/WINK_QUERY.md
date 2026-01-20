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
# Open interactive SQL shell
wink query ./bundle.zip

# Run single query
wink query ./bundle.zip "SELECT * FROM logs WHERE level = 'ERROR'"

# Query multiple bundles
wink query ./debug/ "SELECT bundle_id, status FROM manifest"

# Discover schema
wink query ./bundle.zip ".schema"

# Export results
wink query ./bundle.zip "SELECT * FROM tool_calls" --format csv > tools.csv
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
wink query [OPTIONS] <BUNDLE_PATH> [QUERY]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to `.zip` bundle or directory containing bundles |
| `QUERY` | Optional SQL query; if omitted, opens interactive shell |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--format`, `-f` | `table` | Output format: `table`, `json`, `jsonl`, `csv`, `markdown` |
| `--limit`, `-n` | `1000` | Maximum rows to return (0 for unlimited) |
| `--no-headers` | false | Omit column headers (csv/table) |
| `--schema` | false | Print schema and exit |
| `--tables` | false | List available tables and exit |
| `--describe TABLE` | - | Show detailed schema for TABLE |
| `--load-file FILE` | - | Execute SQL from file |
| `--output`, `-o` | stdout | Write results to file |
| `--quiet`, `-q` | false | Suppress informational messages |
| `--timing` | false | Show query execution time |
| `--explain` | false | Show query plan |

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

## Interactive Shell

When no query is provided, an interactive SQL shell opens:

```
wink query ./bundle.zip
Bundle: debug_abc123_2024-01-15.zip
Tables: 12 (run .tables to list)
Type .help for commands, .quit to exit

sql> SELECT COUNT(*) FROM logs;
┌──────────┐
│ COUNT(*) │
├──────────┤
│ 1523     │
└──────────┘

sql> .schema logs
CREATE TABLE logs (
  _rowid INTEGER PRIMARY KEY,
  timestamp TEXT,
  timestamp_ms INTEGER,
  level TEXT,
  ...
);

sql>
```

### Shell Commands

| Command | Description |
|---------|-------------|
| `.help` | Show available commands |
| `.quit`, `.exit` | Exit shell |
| `.tables` | List all tables |
| `.schema [TABLE]` | Show schema (all or specific table) |
| `.describe TABLE` | Detailed table description with stats |
| `.indexes [TABLE]` | Show indexes |
| `.sample TABLE [N]` | Show N sample rows (default 5) |
| `.count TABLE` | Show row count |
| `.explain QUERY` | Show query execution plan |
| `.timer on|off` | Toggle query timing |
| `.format FORMAT` | Set output format |
| `.output FILE` | Redirect output to file |
| `.output` | Reset output to stdout |
| `.load FILE` | Execute SQL from file |
| `.dump [TABLE]` | Dump table as SQL INSERT statements |
| `.bundle` | Show current bundle info |
| `.bundles` | List all loaded bundles (multi-bundle mode) |
| `.switch BUNDLE_ID` | Switch active bundle (multi-bundle mode) |

## Self-Documentation Features

The query interface is designed to be self-explanatory for AI agents.

### Schema Discovery

```sql
sql> .tables
manifest        logs            session_slices  tool_calls
files           config          errors          timeline
token_usage     state_changes   slice_agentplan slice_toolresult
error           eval            metrics
```

```sql
sql> .describe logs
Table: logs
Description: Log entries from execution (logs/app.jsonl)
Rows: 1523
Source: logs/app.jsonl

Columns:
  _rowid       INTEGER  Entry sequence number
  timestamp    TEXT     ISO-8601 timestamp
  timestamp_ms INTEGER  Unix timestamp in milliseconds (indexed)
  level        TEXT     Log level (indexed)
  event        TEXT     Event name (indexed)
  logger       TEXT     Logger name (indexed)
  message      TEXT     Log message
  context      TEXT     JSON-encoded context object

Indexes:
  idx_logs_timestamp_ms ON (timestamp_ms)
  idx_logs_level ON (level)
  idx_logs_event ON (event)
  idx_logs_logger ON (logger)

Sample:
  timestamp              level  event                     message
  2024-01-15T10:30:00Z  DEBUG  session.dispatch          Dispatching event
  2024-01-15T10:30:01Z  INFO   tool.execution.complete   Tool completed
```

### Contextual Help

```sql
sql> .help timeline
Table: timeline
Unified chronological event stream combining logs, tool calls, state changes,
and errors into a single time-ordered view.

Use this table to understand the sequence of events during execution:

  -- What happened in the first 10 seconds?
  SELECT event_type, event_name, summary
  FROM timeline
  WHERE timestamp_ms < (SELECT MIN(timestamp_ms) + 10000 FROM timeline)
  ORDER BY timestamp_ms;

  -- Find events around an error
  SELECT *
  FROM timeline
  WHERE timestamp_ms BETWEEN
    (SELECT timestamp_ms - 5000 FROM errors WHERE _rowid = 1)
    AND
    (SELECT timestamp_ms + 1000 FROM errors WHERE _rowid = 1)
  ORDER BY timestamp_ms;
```

### Query Suggestions

The shell provides contextual suggestions based on table content:

```sql
sql> .suggest
Based on this bundle's contents:

1. Error Investigation (1 error found):
   SELECT * FROM error;
   SELECT * FROM timeline
   WHERE timestamp_ms BETWEEN
     (SELECT timestamp_ms - 5000 FROM errors LIMIT 1)
     AND (SELECT timestamp_ms + 1000 FROM errors LIMIT 1);

2. Tool Performance (47 tool calls):
   SELECT tool_name, COUNT(*) as calls,
          AVG(duration_ms) as avg_ms,
          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
   FROM tool_calls
   GROUP BY tool_name
   ORDER BY calls DESC;

3. Token Usage:
   SELECT phase, tokens FROM token_usage;

4. State Evolution (3 slice types):
   SELECT slice_type, COUNT(*) FROM session_slices GROUP BY slice_type;
```

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
# 1. Get overview
wink query ./bundle.zip ".describe manifest"
wink query ./bundle.zip "SELECT status, duration_ms FROM manifest"

# 2. Check for errors
wink query ./bundle.zip "SELECT COUNT(*) FROM errors"
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

Tables are implemented as SQLite virtual tables using the `sqlite3` module's
`create_function` and `create_module` capabilities. This enables:

- Lazy loading of large artifacts
- Memory-efficient streaming
- Custom indexing strategies

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
