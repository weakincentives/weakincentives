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

- SQLite built-in functions only
- Entire bundle loaded on first query
- Native tool call content is in nested JSON (use `json_extract()`)
- Extended thinking content is signature-encoded (not accessible as plaintext)

______________________________________________________________________

## Multi-Bundle Queries

### Purpose

Query across multiple debug bundles as a single logical dataset. Essential for
eval analysis where results from many runs need aggregation, comparison, and
pattern detection.

### CLI Extension

```
wink query --bundles-from <FILE> "<SQL>"
wink query --bundles-from <FILE> --schema
```

| Option | Description |
|--------|-------------|
| `--bundles-from` | Read bundle paths from file (one per line) |

The file contains one bundle path per line. Empty lines and lines starting with
`#` are ignored.

### Bundle Resolution

When `--bundles-from` is specified:

1. Each path is resolved via existing `resolve_bundle_path()` logic
2. Duplicate bundle IDs are rejected with an error
3. Bundles with incompatible schema versions are rejected

```bash
# Create bundle list (shell handles discovery)
ls ./eval-results/debug_bundle_*.zip > bundles.txt
find ./runs -name "*.zip" -mtime -1 > bundles.txt

# Query across bundles
wink query --bundles-from bundles.txt "SELECT * FROM errors"
wink query --bundles-from bundles.txt --schema
```

### Schema Additions

#### `bundles` Table

Master table listing all bundles in the query set:

| Column | Type | Description |
|--------|------|-------------|
| `bundle_id` | TEXT | Unique bundle identifier (PK) |
| `bundle_path` | TEXT | Resolved path to bundle file |
| `bundle_idx` | INTEGER | 0-based index in load order |
| `status` | TEXT | 'success' or 'error' |
| `created_at` | TEXT | ISO-8601 timestamp |
| `prompt_ns` | TEXT | Prompt namespace |
| `prompt_key` | TEXT | Prompt key |
| `input_tokens` | INTEGER | Total input tokens |
| `output_tokens` | INTEGER | Total output tokens |
| `total_ms` | INTEGER | Total execution time |

#### Bundle Identification Columns

All existing tables gain a `bundle_id` column as first column:

```sql
-- Single bundle (unchanged)
SELECT tool_name, COUNT(*) FROM tool_calls GROUP BY tool_name

-- Multi-bundle: group by bundle
SELECT bundle_id, tool_name, COUNT(*)
FROM tool_calls
GROUP BY bundle_id, tool_name

-- Multi-bundle: aggregate across all
SELECT tool_name, COUNT(*), AVG(duration_ms)
FROM tool_calls
GROUP BY tool_name
```

Primary keys become composite: `(bundle_id, <existing_pk>)`.

### Aggregate Views

New views for cross-bundle analysis:

| View | Description |
|------|-------------|
| `bundle_summary` | One row per bundle with key metrics |
| `bundle_errors` | Error counts and types per bundle |
| `bundle_tools` | Tool usage aggregated per bundle |
| `bundle_comparison` | Side-by-side metrics for quick comparison |

#### `bundle_summary` View

```sql
CREATE VIEW bundle_summary AS
SELECT
    b.bundle_id,
    b.status,
    b.prompt_key,
    b.input_tokens,
    b.output_tokens,
    b.total_ms,
    (SELECT COUNT(*) FROM tool_calls t WHERE t.bundle_id = b.bundle_id) as tool_count,
    (SELECT COUNT(*) FROM errors e WHERE e.bundle_id = b.bundle_id) as error_count,
    (SELECT COUNT(*) FROM tool_calls t WHERE t.bundle_id = b.bundle_id AND t.success = 0) as failed_tools
FROM bundles b
```

#### `bundle_comparison` View

```sql
CREATE VIEW bundle_comparison AS
SELECT
    bundle_id,
    status,
    prompt_key,
    input_tokens + output_tokens as total_tokens,
    total_ms,
    ROUND(output_tokens * 1000.0 / NULLIF(total_ms, 0), 1) as tokens_per_sec,
    (SELECT COUNT(*) FROM errors e WHERE e.bundle_id = b.bundle_id) as errors
FROM bundles b
ORDER BY created_at
```

### Multi-Bundle Schema Output

```bash
wink query --bundles-from bundles.txt --schema
```

```json
{
  "bundle_count": 2,
  "bundles": [
    {"bundle_id": "abc123", "status": "success", "created_at": "..."},
    {"bundle_id": "def456", "status": "error", "created_at": "..."}
  ],
  "tables": [...],
  "hints": {
    "json_extraction": [...],
    "common_queries": {...},
    "multi_bundle_queries": {
      "success_rate": "SELECT COUNT(*) FILTER (WHERE status = 'success') * 100.0 / COUNT(*) as pct FROM bundles",
      "compare_bundles": "SELECT * FROM bundle_comparison",
      "error_patterns": "SELECT error_type, COUNT(*) as occurrences, COUNT(DISTINCT bundle_id) as affected_bundles FROM errors GROUP BY error_type ORDER BY occurrences DESC",
      "tool_performance": "SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms, COUNT(DISTINCT bundle_id) as used_in_bundles FROM tool_calls GROUP BY tool_name"
    }
  }
}
```

### Example Multi-Bundle Queries

```sql
-- Success/failure rate across eval
SELECT
    status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM bundles), 1) as pct
FROM bundles
GROUP BY status

-- Most common errors across runs
SELECT error_type, message, COUNT(*) as occurrences
FROM errors
GROUP BY error_type, message
ORDER BY occurrences DESC
LIMIT 10

-- Tool reliability
SELECT
    tool_name,
    COUNT(*) as total_calls,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
    ROUND(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as success_rate
FROM tool_calls
GROUP BY tool_name
HAVING COUNT(*) >= 5
ORDER BY success_rate ASC

-- Compare token usage by prompt variant
SELECT
    prompt_key,
    COUNT(*) as runs,
    AVG(input_tokens) as avg_input,
    AVG(output_tokens) as avg_output,
    AVG(total_ms) as avg_ms
FROM bundles
GROUP BY prompt_key

-- Find bundles where a specific error occurred
SELECT b.bundle_id, b.bundle_path, e.message
FROM bundles b
JOIN errors e ON e.bundle_id = b.bundle_id
WHERE e.error_type = 'ValidationError'

-- Session slice counts by type across bundles
SELECT
    slice_type,
    COUNT(*) as total_items,
    COUNT(DISTINCT bundle_id) as bundles_with_slice,
    AVG(item_count) as avg_items_per_bundle
FROM (
    SELECT bundle_id, slice_type, COUNT(*) as item_count
    FROM session_slices
    GROUP BY bundle_id, slice_type
)
GROUP BY slice_type
```

### Caching Strategy

Multi-bundle databases are cached using a composite key:

```
Cache path: <first_bundle>.multi-<count>-<hash>.sqlite
Hash input: sorted(bundle_path + ":" + str(mtime) for each bundle)
```

Cache invalidated when:

- Any bundle's mtime changes
- Bundle list changes (different hash)
- Schema version mismatch

For large eval sets (>100 bundles), consider using `--bundles-from` with a
manifest file that tracks bundle mtimes to avoid repeated stat calls.

### Implementation Notes

#### Database Building

1. Parse bundle list and validate all paths exist
2. Load each bundle and verify unique `bundle_id`
3. Create unified schema with `bundle_id` columns
4. Insert data from each bundle with `bundle_id` foreign key
5. Create aggregate views after all data loaded
6. Build indexes on `(bundle_id, ...)` for common join patterns

#### Memory Considerations

For large eval sets:

- Stream bundles one at a time during database build
- Close each `DebugBundle` after extraction to release memory
- Consider `--limit N` flag to cap number of bundles loaded

#### Indexes

Automatically created for multi-bundle queries:

```sql
CREATE INDEX idx_tool_calls_bundle ON tool_calls(bundle_id);
CREATE INDEX idx_errors_bundle ON errors(bundle_id);
CREATE INDEX idx_logs_bundle ON logs(bundle_id);
CREATE INDEX idx_session_slices_bundle ON session_slices(bundle_id);
```

### Backwards Compatibility

Single-bundle queries work unchanged:

```bash
# Single bundle (existing behavior)
wink query ./bundle.zip "SELECT * FROM errors"
```

The original CLI syntax is preserved. Multi-bundle mode is only activated when
`--bundles-from` is specified.

______________________________________________________________________

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Bundle format
- `specs/SESSIONS.md` - Session state
