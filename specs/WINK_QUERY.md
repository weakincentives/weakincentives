# Wink Query Specification

## Purpose

`wink query` enables AI coding agents to explore debug bundles through a SQL interface
for deep analysis, issue diagnosis, and pattern detection. Built on DuckDB with
automatic schema generation from bundle contents.

**Primary use case**: AI-driven analysis of agent execution for debugging, optimization,
and understanding complex multi-step behaviors.

## Principles

- **SQL-first interface**: All exploration through standard SQL queries
- **Self-documenting**: Schema discovery built into the tool
- **Automatic preprocessing**: Bundle data extracted to queryable tables on demand
- **Agent-friendly**: Rich metadata and examples in schema definitions
- **Incremental loading**: Process bundles on first query, cache for subsequent use
- **Zero configuration**: Works immediately with any valid debug bundle
- **Portable**: Query cache can be shared, versioned, or archived

## Architecture

```
wink query [bundle-path]
    │
    ├── Bundle Loader
    │   ├── Validate manifest
    │   ├── Check cache (.wink-query/ directory)
    │   └── Trigger preprocessing if needed
    │
    ├── Preprocessor
    │   ├── Extract JSONL files → tables
    │   ├── Parse nested structures → normalized relations
    │   ├── Build indexes on common query patterns
    │   └── Write .duckdb file
    │
    └── Query Engine (DuckDB)
        ├── Interactive REPL with history
        ├── .schema, .tables, .describe commands
        └── Export results (JSON, CSV, Parquet)
```

## Bundle Preprocessing

### Extraction Pipeline

When a bundle is first queried, the preprocessor:

1. **Load manifest** - Read bundle ID, format version, file inventory
1. **Extract core artifacts** - Session snapshots, logs, metrics, request I/O
1. **Normalize structures** - Flatten nested JSON into relational tables
1. **Create views** - Convenient joins for common queries
1. **Build indexes** - Speed up filtering by timestamp, type, status
1. **Write metadata** - Schema descriptions, sample queries, statistics

### Output Structure

```
.wink-query/
  {bundle_id}.duckdb           # DuckDB database file
  {bundle_id}.schema.json      # Schema metadata for discovery
  {bundle_id}.stats.json       # Preprocessing statistics
```

Cache lives in bundle directory or `~/.wink/query-cache/` for zip files.

## Schema Design

### Core Tables

All tables include AI-friendly column descriptions stored as comments.

#### manifest

Bundle metadata from `manifest.json`.

| Column | Type | Description |
|--------|------|-------------|
| `bundle_id` | `VARCHAR` | Unique bundle identifier (UUID) |
| `format_version` | `VARCHAR` | Bundle format version |
| `created_at` | `TIMESTAMP` | Bundle creation timestamp (UTC) |
| `request_id` | `VARCHAR` | MainLoop request ID |
| `session_id` | `VARCHAR` | Session ID |
| `status` | `VARCHAR` | Request status: success, error |
| `started_at` | `TIMESTAMP` | Request start time |
| `ended_at` | `TIMESTAMP` | Request end time |
| `duration_ms` | `INTEGER` | Total request duration |
| `capture_mode` | `VARCHAR` | minimal, standard, or full |
| `prompt_ns` | `VARCHAR` | Prompt namespace |
| `prompt_key` | `VARCHAR` | Prompt key |
| `adapter` | `VARCHAR` | Provider adapter used |

#### session_state

Session state slices from `session/after.jsonl` (and `before.jsonl` if present).

| Column | Type | Description |
|--------|------|-------------|
| `slice_type` | `VARCHAR` | Slice type name (e.g., Plan, ToolInvoked) |
| `item_index` | `INTEGER` | Item position within slice |
| `timestamp` | `TIMESTAMP` | Event timestamp |
| `phase` | `VARCHAR` | before or after execution |
| `data` | `JSON` | Full item as JSON |

#### logs

Structured log entries from `logs/app.jsonl`.

| Column | Type | Description |
|--------|------|-------------|
| `log_id` | `INTEGER` | Auto-incrementing log entry ID |
| `timestamp` | `TIMESTAMP` | Log record timestamp |
| `level` | `VARCHAR` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `event` | `VARCHAR` | Structured event name |
| `message` | `VARCHAR` | Human-readable message |
| `logger` | `VARCHAR` | Logger name (module path) |
| `context` | `JSON` | Additional structured context |

#### tool_invocations

Extracted from `session_state` where `slice_type = 'ToolInvoked'`.

| Column | Type | Description |
|--------|------|-------------|
| `invocation_id` | `INTEGER` | Auto-incrementing invocation ID |
| `tool_name` | `VARCHAR` | Tool name |
| `timestamp` | `TIMESTAMP` | Invocation timestamp |
| `params` | `JSON` | Tool parameters |
| `result` | `JSON` | Tool result |
| `success` | `BOOLEAN` | Whether tool succeeded |
| `duration_ms` | `INTEGER` | Tool execution duration |
| `error_message` | `VARCHAR` | Error message if failed |

#### metrics

Performance metrics from `metrics.json`.

| Column | Type | Description |
|--------|------|-------------|
| `phase` | `VARCHAR` | Execution phase |
| `started_at` | `TIMESTAMP` | Phase start |
| `ended_at` | `TIMESTAMP` | Phase end |
| `duration_ms` | `INTEGER` | Phase duration |
| `input_tokens` | `INTEGER` | Tokens in prompt |
| `output_tokens` | `INTEGER` | Tokens in response |
| `total_tokens` | `INTEGER` | Total tokens consumed |
| `cost_usd` | `DECIMAL(10,6)` | Estimated cost in USD |

#### request_io

Request input and output from `request/*.json`.

| Column | Type | Description |
|--------|------|-------------|
| `direction` | `VARCHAR` | input or output |
| `content_type` | `VARCHAR` | Type of content (e.g., code_review_request) |
| `data` | `JSON` | Full request/response data |

#### filesystem

Files from `filesystem/` directory (if captured).

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | `VARCHAR` | Relative path within workspace |
| `size_bytes` | `INTEGER` | File size |
| `modified_at` | `TIMESTAMP` | Last modification time |
| `content_type` | `VARCHAR` | Detected MIME type |
| `content` | `BLOB` | File contents (text or binary) |
| `content_text` | `VARCHAR` | Text content (null for binary files) |

#### config

Configuration from `config.json`.

| Column | Type | Description |
|--------|------|-------------|
| `key` | `VARCHAR` | Configuration key (dot-separated path) |
| `value` | `JSON` | Configuration value |
| `value_type` | `VARCHAR` | Type of value (string, int, bool, etc.) |

#### errors

Error details from `error.json` (if present).

| Column | Type | Description |
|--------|------|-------------|
| `error_type` | `VARCHAR` | Exception type |
| `error_message` | `VARCHAR` | Exception message |
| `phase` | `VARCHAR` | Phase where error occurred |
| `traceback` | `VARCHAR` | Full Python traceback |
| `context` | `JSON` | Additional error context |

### Convenience Views

Pre-defined views for common query patterns.

#### tool_failures

Tools that returned errors.

```sql
CREATE VIEW tool_failures AS
SELECT
    invocation_id,
    tool_name,
    timestamp,
    error_message,
    params
FROM tool_invocations
WHERE success = FALSE
ORDER BY timestamp;
```

#### slow_tools

Tool invocations taking longer than 1 second.

```sql
CREATE VIEW slow_tools AS
SELECT
    tool_name,
    timestamp,
    duration_ms,
    params,
    success
FROM tool_invocations
WHERE duration_ms > 1000
ORDER BY duration_ms DESC;
```

#### error_timeline

Chronological view of all errors (logs + tool failures).

```sql
CREATE VIEW error_timeline AS
SELECT
    timestamp,
    'log' AS source,
    event AS error_type,
    message AS error_message,
    context AS details
FROM logs
WHERE level IN ('ERROR', 'CRITICAL')
UNION ALL
SELECT
    timestamp,
    'tool' AS source,
    tool_name AS error_type,
    error_message,
    params AS details
FROM tool_invocations
WHERE success = FALSE
ORDER BY timestamp;
```

#### token_usage_by_phase

Token consumption broken down by execution phase.

```sql
CREATE VIEW token_usage_by_phase AS
SELECT
    phase,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    SUM(total_tokens) AS total_tokens,
    SUM(cost_usd) AS total_cost_usd,
    COUNT(*) AS num_calls
FROM metrics
GROUP BY phase
ORDER BY total_tokens DESC;
```

#### session_evolution

Track how session state slices grew over time.

```sql
CREATE VIEW session_evolution AS
SELECT
    slice_type,
    COUNT(*) AS item_count,
    MIN(timestamp) AS first_seen,
    MAX(timestamp) AS last_seen
FROM session_state
WHERE phase = 'after'
GROUP BY slice_type
ORDER BY item_count DESC;
```

### Dynamic Schema Generation

For polymorphic session slices (e.g., different event types), the preprocessor:

1. **Detect slice types** - Scan `__type__` fields in session JSONL
1. **Generate slice tables** - Create `session_slice_{type}` for each unique type
1. **Extract common fields** - Promote frequently-queried fields to columns
1. **Preserve full data** - Keep original JSON in `data` column
1. **Document schema** - Write descriptions to `.schema.json`

Example: `session_slice_plan` table for `Plan` slice type.

| Column | Type | Description |
|--------|------|-------------|
| `item_index` | `INTEGER` | Item position in slice |
| `timestamp` | `TIMESTAMP` | Event timestamp |
| `objective` | `VARCHAR` | Plan objective |
| `status` | `VARCHAR` | Plan status (active, completed) |
| `num_steps` | `INTEGER` | Total number of steps |
| `num_completed` | `INTEGER` | Number of completed steps |
| `data` | `JSON` | Full plan data |

## CLI Interface

### Interactive Mode (Default)

```bash
wink query ./debug/bundle.zip
```

Launches an interactive REPL with:

- **Command history** - Arrow keys, readline support
- **Tab completion** - Table names, column names, SQL keywords
- **Multi-line queries** - Automatic detection of incomplete statements
- **Result pagination** - Configurable page size for large result sets
- **Output formatting** - Pretty-printed tables with column alignment

### Special Commands

All commands start with `.` (DuckDB convention).

| Command | Description |
|---------|-------------|
| `.help` | Show all available commands |
| `.tables` | List all tables |
| `.schema [TABLE]` | Show CREATE TABLE statement(s) |
| `.describe TABLE` | Show column details with descriptions |
| `.views` | List all views |
| `.indexes` | List all indexes |
| `.stats` | Show preprocessing statistics |
| `.export FORMAT` | Set export format (table, json, csv, parquet) |
| `.output FILE` | Redirect output to file |
| `.width N` | Set table column width |
| `.quit` or `.exit` | Exit REPL |

### Query Examples

Built-in `.examples` command shows common queries:

```sql
-- Example 1: Find all tool failures
SELECT tool_name, timestamp, error_message
FROM tool_failures
ORDER BY timestamp;

-- Example 2: Most expensive operations
SELECT phase, duration_ms, total_tokens, cost_usd
FROM metrics
ORDER BY cost_usd DESC
LIMIT 10;

-- Example 3: Find files modified during execution
SELECT file_path, size_bytes, modified_at
FROM filesystem
WHERE modified_at BETWEEN
    (SELECT started_at FROM manifest) AND
    (SELECT ended_at FROM manifest)
ORDER BY modified_at;

-- Example 4: Session state at specific timestamp
SELECT slice_type, COUNT(*) AS count
FROM session_state
WHERE timestamp <= '2024-01-15T10:30:00Z'
    AND phase = 'after'
GROUP BY slice_type;

-- Example 5: Tool invocation patterns
SELECT
    tool_name,
    COUNT(*) AS invocations,
    AVG(duration_ms) AS avg_duration_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successes,
    SUM(CASE WHEN success THEN 0 ELSE 1 END) AS failures
FROM tool_invocations
GROUP BY tool_name
ORDER BY invocations DESC;

-- Example 6: Log message frequency
SELECT event, COUNT(*) AS count
FROM logs
WHERE level IN ('WARNING', 'ERROR', 'CRITICAL')
GROUP BY event
ORDER BY count DESC;

-- Example 7: Correlated tool and log errors
SELECT
    t.timestamp,
    t.tool_name,
    t.error_message AS tool_error,
    l.message AS log_message
FROM tool_failures t
LEFT JOIN logs l
    ON l.timestamp BETWEEN t.timestamp - INTERVAL 1 SECOND
        AND t.timestamp + INTERVAL 1 SECOND
    AND l.level = 'ERROR'
ORDER BY t.timestamp;
```

### Non-Interactive Mode

Execute queries from command line or files:

```bash
# Single query
wink query ./bundle.zip -c "SELECT COUNT(*) FROM logs WHERE level='ERROR'"

# From file
wink query ./bundle.zip -f analysis.sql

# Pipe query
echo "SELECT * FROM manifest" | wink query ./bundle.zip

# Export results
wink query ./bundle.zip -c "SELECT * FROM tool_invocations" --format csv > tools.csv
```

### Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--command` | `-c` | Execute SQL command and exit |
| `--file` | `-f` | Execute SQL from file |
| `--format` | | Output format: table, json, csv, parquet (default: table) |
| `--output` | `-o` | Write results to file instead of stdout |
| `--no-cache` | | Force reprocessing (ignore existing .duckdb file) |
| `--cache-dir` | | Override cache directory location |
| `--page-size` | | Number of rows per page (default: 50) |
| `--no-pager` | | Disable pagination, print all results |
| `--timing` | | Show query execution time |
| `--explain` | | Show query plan instead of executing |
| `--readonly` | | Open database in readonly mode (default: true) |

## AI Agent Integration

### Self-Documentation

The tool is designed for AI agents to use autonomously:

1. **Schema discovery**: `.describe` command includes natural language descriptions
1. **Example queries**: `.examples` provides templates for common tasks
1. **Error messages**: Detailed, actionable error messages with suggestions
1. **Statistics**: `.stats` shows preprocessing details for understanding data

### Typical Agent Workflow

```
Agent: User reports tool failure in bundle X
      → Run: wink query bundle.zip
      → Check: .stats to understand bundle size/content
      → Query: SELECT * FROM tool_failures
      → Analyze: Find correlation with logs
      → Query: error_timeline view for context
      → Diagnose: Identify root cause
      → Report: Findings with SQL queries as evidence
```

### Prompt Template for Agents

When an agent encounters a debug bundle, it should:

```markdown
I need to analyze debug bundle at {bundle_path} to {objective}.

Steps:
1. Launch wink query and check .stats for bundle overview
2. Use .tables and .views to understand available data
3. Run exploratory queries to find relevant information:
   - Check tool_failures for tool errors
   - Check error_timeline for error chronology
   - Check slow_tools for performance issues
   - Check session_evolution for state growth patterns
4. Formulate hypothesis about issue
5. Write targeted queries to test hypothesis
6. Extract evidence (timestamps, error messages, parameters)
7. Report findings with reproducible queries
```

## Preprocessing Details

### Session State Normalization

Session JSONL files contain heterogeneous slice types. The preprocessor:

1. **Parse each line** - Deserialize JSON with `__type__` field
1. **Group by slice type** - Collect items by `__type__` value
1. **Analyze structure** - Find common fields across all items of a type
1. **Create table** - Generate schema with typed columns
1. **Populate rows** - Insert items with type casting
1. **Index by timestamp** - Speed up temporal queries

### Log Preprocessing

Structured logs from `logs/app.jsonl`:

1. **Extract common fields** - timestamp, level, event, message, logger
1. **Preserve context** - Store additional fields as JSON
1. **Create indexes** - On timestamp, level, event for fast filtering
1. **Compute statistics** - Log level distribution, unique events

### Tool Invocation Extraction

Extract from session state where `slice_type = 'ToolInvoked'`:

1. **Parse tool data** - Extract name, params, result, timestamp
1. **Compute duration** - If timing information available
1. **Detect success** - Parse result structure for success field
1. **Extract errors** - Pull error messages from failed results
1. **Index by name** - Speed up tool-specific queries

### Filesystem Ingestion

For bundles with `filesystem/` directory:

1. **Walk directory tree** - Recursively find all files
1. **Detect content type** - Use magic bytes or extension
1. **Read content** - Store as BLOB
1. **Extract text** - For text files, duplicate to `content_text` column
1. **Compute stats** - Size, modification time
1. **Skip large files** - Configurable threshold (default: 10MB)

### Metrics Extraction

Parse `metrics.json`:

1. **Flatten phases** - Each phase becomes a row
1. **Compute durations** - From start/end timestamps
1. **Extract token counts** - Input, output, total
1. **Calculate costs** - Using provider-specific pricing

### Index Strategy

Automatically created indexes:

- `logs(timestamp)` - Temporal filtering
- `logs(level, timestamp)` - Level + time queries
- `logs(event)` - Event-specific queries
- `tool_invocations(timestamp)` - Tool timeline
- `tool_invocations(tool_name, timestamp)` - Per-tool queries
- `tool_invocations(success)` - Failure analysis
- `session_state(slice_type, timestamp)` - Slice queries
- `filesystem(file_path)` - Path lookups
- `manifest(bundle_id)` - Bundle lookups (for multi-bundle queries)

## Multi-Bundle Queries

When given a directory of bundles:

```bash
wink query ./debug/
```

The preprocessor:

1. **Discover bundles** - Find all `.zip` files
1. **Process each** - Create individual `.duckdb` files
1. **Create union database** - Aggregate all tables with `bundle_id` column
1. **Enable cross-bundle queries** - Compare executions, track regressions

Example multi-bundle queries:

```sql
-- Compare token usage across runs
SELECT bundle_id, SUM(total_tokens) AS tokens
FROM metrics
GROUP BY bundle_id
ORDER BY tokens DESC;

-- Find consistently failing tools
SELECT tool_name, COUNT(DISTINCT bundle_id) AS bundles_affected
FROM tool_failures
GROUP BY tool_name
HAVING COUNT(DISTINCT bundle_id) > 1;

-- Track performance regression
SELECT
    m1.bundle_id AS current_bundle,
    m1.phase,
    m1.duration_ms AS current_duration,
    m2.duration_ms AS previous_duration,
    ((m1.duration_ms - m2.duration_ms) * 100.0 / m2.duration_ms) AS pct_change
FROM metrics m1
JOIN metrics m2 ON m1.phase = m2.phase
WHERE m1.bundle_id = 'latest'
    AND m2.bundle_id = 'previous';
```

## Performance Considerations

### Preprocessing Time

| Bundle Size | Preprocessing Time | Cache Size |
|-------------|-------------------|------------|
| Minimal (< 1 MB) | < 100ms | ~500KB |
| Standard (1-10 MB) | < 500ms | ~2-5MB |
| Full (10-50 MB) | 1-3 seconds | ~10-20MB |
| Large (50-100 MB) | 3-10 seconds | ~20-50MB |

### Query Performance

DuckDB optimizations:

- **Columnar storage** - Efficient scanning of large tables
- **Vectorized execution** - Fast aggregations and filters
- **Automatic parallelism** - Multi-core query execution
- **Lazy evaluation** - Only load required columns
- **Compression** - Reduced I/O for large result sets

Typical query response times:

- Simple filters (e.g., `WHERE level='ERROR'`): < 10ms
- Aggregations (e.g., `GROUP BY tool_name`): < 50ms
- Joins (e.g., tool invocations + logs): < 100ms
- Complex analytics: < 500ms

## Cache Management

### Cache Location

By default:

- Zip bundles: `~/.wink/query-cache/{bundle_id}.duckdb`
- Directory bundles: `.wink-query/` subdirectory

### Cache Invalidation

Cache is invalidated when:

- Bundle file modified (zip mtime changed)
- `--no-cache` flag used
- Format version mismatch
- Preprocessing error occurred

### Cache Cleanup

```bash
# Remove all cached databases
wink query --clear-cache

# Remove cache for specific bundle
wink query --clear-cache ./bundle.zip

# Show cache statistics
wink query --cache-info
```

## Schema Metadata Format

`.schema.json` contains AI-readable schema documentation:

```json
{
  "bundle_id": "550e8400-e29b-41d4-a716-446655440000",
  "format_version": "1.0.0",
  "preprocessed_at": "2024-01-15T10:30:00+00:00",
  "tables": {
    "manifest": {
      "description": "Bundle metadata and execution summary",
      "row_count": 1,
      "columns": {
        "bundle_id": {
          "type": "VARCHAR",
          "description": "Unique bundle identifier (UUID)",
          "nullable": false
        },
        "status": {
          "type": "VARCHAR",
          "description": "Request status: success or error",
          "values": ["success", "error"]
        }
      }
    },
    "logs": {
      "description": "Structured log entries from execution",
      "row_count": 1543,
      "columns": { ... },
      "indexes": ["timestamp", "level"]
    }
  },
  "views": {
    "tool_failures": {
      "description": "All failed tool invocations with error details",
      "base_tables": ["tool_invocations"]
    }
  },
  "statistics": {
    "total_tables": 10,
    "total_views": 6,
    "total_rows": 2847,
    "preprocessing_duration_ms": 234
  }
}
```

## Error Handling

### Bundle Validation Errors

```
Error: Invalid bundle format
  → manifest.json not found
  → Run: wink debug ./bundle.zip
  → Check: bundle integrity
```

### Preprocessing Errors

```
Error: Failed to parse session state
  → Invalid JSON in session/after.jsonl:142
  → Cause: Unexpected token '}' at position 53
  → Action: Check bundle integrity or report bug
```

### Query Errors

```
Error: Table 'tool_invocations' not found
  → Available tables: manifest, logs, session_state, metrics
  → This bundle may not contain tool invocations
  → Try: .tables to see all available tables
```

## Limitations

- **Preprocessing required**: First query has startup latency
- **Cache disk usage**: Grows with number of unique bundles
- **Large bundles**: Preprocessing may be slow for > 100MB bundles
- **Filesystem content**: Text extraction limited to common encodings
- **JSON depth**: Deeply nested structures remain as JSON columns
- **Readonly by default**: No support for modifying bundles via SQL
- **No streaming**: Entire bundle processed before first query
- **DuckDB version dependency**: Requires DuckDB >= 0.9.0

## Security Considerations

Bundles may contain sensitive information:

- **Local execution**: Query engine runs locally, no network calls
- **Readonly by default**: Cannot modify bundle contents
- **Cache isolation**: Per-user cache directory with restricted permissions
- **No SQL injection**: Parameterized queries in agent integrations
- **Audit trail**: All queries logged (if logging enabled)

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format and structure
- `specs/WINK_DEBUG.md` - Web UI for bundle exploration
- `specs/SESSIONS.md` - Session state model
- `specs/LOGGING.md` - Structured logging format
- `specs/TOOLS.md` - Tool invocation semantics
- `specs/METRICS.md` - Performance metrics schema

## Future Enhancements

Potential future capabilities:

- **Streaming preprocessing** - Process bundles incrementally
- **Schema inference** - Auto-detect query patterns and suggest indexes
- **Query templates** - Saved queries for common diagnostics
- **Diff mode** - Compare two bundles side-by-side
- **Graph queries** - Tool dependency analysis, state transition graphs
- **Time-travel queries** - Query session state at arbitrary timestamps
- **Export to other formats** - Convert bundles to other analysis tools
- **Embedding support** - Semantic search over logs and tool calls
- **Interactive visualization** - Generate plots from query results
- **Collaborative analysis** - Share query sessions with other developers
