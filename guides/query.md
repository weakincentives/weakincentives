# Querying Debug Bundles

This guide covers `wink query`, a SQL-based tool for exploring debug bundles.
It's the primary way to analyze what happened during an agent execution.

*Canonical spec: [specs/WINK_QUERY.md](../specs/WINK_QUERY.md)*

## Prerequisites

Install the CLI extra:

```bash
pip install "weakincentives[wink]"
```

## Why SQL?

Debug bundles contain structured data—logs, tool calls, session state, errors.
SQL provides a familiar, powerful interface for filtering and aggregating this
data without writing custom scripts. The query command loads your bundle into
SQLite, giving you the full power of SQL for analysis.

## Basic Usage

```bash
# Discover what tables are available
wink query ./bundle.zip --schema

# Run a query (JSON output by default)
wink query ./bundle.zip "SELECT * FROM errors"

# Get formatted table output
wink query ./bundle.zip "SELECT * FROM errors" --table
```

Always start with `--schema` to see what tables exist and what columns they
have.

## Schema Discovery

The `--schema` flag outputs JSON describing all tables, views, and columns:

```bash
wink query ./bundle.zip --schema
```

Output includes:

- **tables**: Each table with name, description, row count, and column info
- **hints**: Ready-to-use JSON extraction patterns and common queries

The hints section is especially useful—it shows you exactly how to query nested
JSON data:

```json
{
  "hints": {
    "json_extraction": [
      "json_extract(context, '$.tool_name')",
      "json_extract(params, '$.command')",
      "json_extract(context, '$.content')",
      "json_extract(parsed, '$.message.content')",
      "json_extract(parsed, '$.tool_use_id')"
    ],
    "common_queries": {
      "native_tools": "SELECT * FROM native_tool_calls ORDER BY timestamp",
      "transcript": "SELECT transcript_source, sequence_number, entry_type, role, content FROM transcript ORDER BY rowid",
      "tool_timeline": "SELECT * FROM tool_timeline WHERE duration_ms > 1000",
      "error_context": "SELECT timestamp, message, context FROM logs WHERE level = 'ERROR'"
    }
  }
}
```

## Tables

### Core Tables

These tables are always present:

| Table | Source | What It Contains |
|-------|--------|------------------|
| `manifest` | `manifest.json` | Bundle ID, status, timestamps |
| `logs` | `logs/app.jsonl` | All log entries with event names and context |
| `transcript` | derived from logs | Transcript entries from TranscriptCollector |
| `tool_calls` | derived from logs | Tool invocations with params, results, timing |
| `errors` | aggregated | All errors from logs, error.json, and failed tools |
| `session_slices` | `session/after.jsonl` | Session state items |
| `files` | `filesystem/` | Workspace files with paths and content |
| `config` | `config.json` | Configuration as flattened key-value pairs |
| `metrics` | `metrics.json` | Token usage and timing metrics |
| `run_context` | `run_context.json` | Request ID, session ID, trace ID |

### Optional Tables

| Table | Present When |
|-------|--------------|
| `prompt_overrides` | `prompt_overrides.json` exists |
| `eval` | Bundle is from EvalLoop |

### Dynamic Slice Tables

For each slice type in your session, a dedicated table is created:

```
slice_{normalized_type_name}
```

Example: A slice type `myapp.state:AgentPlan` becomes `slice_agentplan`. These
tables have columns inferred from the JSON structure, making it easy to query
typed session state.

## Environment Tables

These tables capture the execution environment at bundle creation time:

| Table | Contents |
|-------|----------|
| `env_system` | OS name, kernel version, architecture, CPU count, memory |
| `env_python` | Python version, implementation, executable, virtualenv status |
| `env_git` | Repository root, commit SHA, branch, dirty state, remotes, tags |
| `env_container` | Container runtime, ID, image, cgroup path |
| `env_vars` | Filtered environment variables (name-value pairs) |
| `environment` | Flattened key-value view of all environment data |

Example queries:

```sql
-- System info
SELECT * FROM env_system

-- Git state at execution time
SELECT commit_sha, branch, is_dirty FROM env_git

-- Python environment
SELECT version, is_virtualenv, executable FROM env_python

-- Check if running in container
SELECT runtime, container_id FROM env_container

-- Specific environment variable
SELECT value FROM env_vars WHERE name = 'HOME'
```

## Views

Pre-built views for common analysis patterns:

| View | Description |
|------|-------------|
| `tool_timeline` | Tool calls ordered by timestamp with extracted commands |
| `native_tool_calls` | Native tool calls from transcripts (and legacy log aggregator) |
| `transcript_entries` | TranscriptCollector entries extracted from logs |
| `transcript_flow` | Conversation flow with message previews |
| `transcript_tools` | Tool usage analysis with paired calls and results |
| `transcript_thinking` | Thinking blocks with preview and length |
| `transcript_agents` | Agent hierarchy and activity metrics |
| `error_summary` | Errors with truncated tracebacks |

Use views directly in queries:

```sql
SELECT * FROM tool_timeline WHERE duration_ms > 1000
SELECT * FROM error_summary
SELECT * FROM native_tool_calls LIMIT 20
SELECT * FROM transcript_entries LIMIT 20

-- Transcript analysis views
SELECT * FROM transcript_flow ORDER BY rowid DESC LIMIT 50
SELECT * FROM transcript_tools WHERE tool_name IS NOT NULL
SELECT * FROM transcript_thinking WHERE thinking_length > 1000
SELECT * FROM transcript_agents WHERE transcript_source != 'main'
```

## Common Queries

### Finding Errors

```sql
-- All errors with their source
SELECT source, error_type, message FROM errors

-- Error logs with timestamps
SELECT timestamp, message FROM logs WHERE level = 'ERROR'

-- Failed tool calls
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0
```

### Analyzing Tool Performance

```sql
-- Tool call counts and average duration
SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms
FROM tool_calls
GROUP BY tool_name
ORDER BY calls DESC

-- Slowest tool calls
SELECT tool_name, duration_ms, params
FROM tool_calls
ORDER BY duration_ms DESC
LIMIT 10
```

### Inspecting Session State

```sql
-- What slice types exist
SELECT slice_type, COUNT(*) FROM session_slices GROUP BY slice_type

-- Query a specific slice table (created dynamically)
SELECT * FROM slice_agentplan
```

### Working with Logs

```sql
-- Find specific events
SELECT timestamp, event, context FROM logs WHERE event LIKE 'tool.%'

-- Extract nested JSON data
SELECT
    timestamp,
    json_extract(context, '$.tool_name') as tool,
    json_extract(context, '$.duration_ms') as duration
FROM logs
WHERE event = 'tool.execution.complete'
```

### Querying Native Tool Calls

Native tool calls (Bash, Read, Write, etc.) are captured via transcripts (and
legacy log aggregator events in older bundles). Use the `native_tool_calls`
view to query them consistently:

```sql
-- Get native tool logs by sequence number
SELECT sequence_number, source, tool_name, content
FROM native_tool_calls
WHERE sequence_number BETWEEN 100 AND 200
ORDER BY sequence_number
```

The `logs.seq` column is populated for events that include
`context.sequence_number` (e.g. legacy `log_aggregator.log_line`,
`transcript.collector.entry`); it's NULL for other log entries.

### Querying Transcripts

When using the Claude Agent SDK adapter with transcript collection enabled,
each transcript entry is emitted as a structured log event and normalized into
the `transcript` table (the `transcript_entries` view is an alias):

```sql
SELECT transcript_source, sequence_number, entry_type, role, content
FROM transcript
ORDER BY rowid
```

### Token Usage

```sql
-- Get token metrics
SELECT key, value FROM metrics
WHERE key IN ('input_tokens', 'output_tokens', 'total_ms')
```

### Searching Files

```sql
-- Find files containing specific text
SELECT path FROM files WHERE content LIKE '%TODO%'

-- Find files by extension
SELECT path, LENGTH(content) as size FROM files WHERE path LIKE '%.py'
```

## Output Formats

### JSON (Default)

Returns results as a JSON array:

```bash
wink query ./bundle.zip "SELECT * FROM manifest"
# [{"bundle_id": "abc123", "status": "success", ...}]
```

Good for piping to `jq` or processing programmatically.

### Table

Human-readable ASCII table:

```bash
wink query ./bundle.zip "SELECT source, error_type, message FROM errors" --table
```

```
source | error_type | message
-------|------------|--------
log    | ValueError | Invalid config
tool   | TimeoutErr | Request timed out
```

By default, long values are truncated. Use `--no-truncate` for full content:

```bash
wink query ./bundle.zip "SELECT * FROM logs" --table --no-truncate
```

## Raw JSONL Export

For power users who prefer `jq` over SQL, export raw JSONL directly:

```bash
# Export logs (default)
wink query ./bundle.zip --export-jsonl

# Export session state
wink query ./bundle.zip --export-jsonl=session
```

Pipe to `jq` for filtering:

```bash
# Find specific events
wink query ./bundle.zip --export-jsonl | jq 'select(.event == "tool.execution.start")'

# Find session slices by type
wink query ./bundle.zip --export-jsonl=session | jq 'select(.__type__ | contains("Plan"))'
```

## Caching

The query command caches the SQLite database next to the bundle:

```
./bundle.zip           → input
./bundle.zip.sqlite    → cache
```

Cache is automatically invalidated when:

- Bundle modification time is newer than cache
- Schema version changes (after upgrades)

You don't need to manage the cache manually. Delete the `.sqlite` file if you
want to force a rebuild.

## Workflow Tips

**Start with schema**: Always run `--schema` first. The hints section shows you
exactly how to query the bundle.

**Use views first**: The pre-built views (`tool_timeline`, `error_summary`,
`native_tool_calls`) handle common patterns. Check these before writing custom
queries.

**Extract JSON early**: Log context and tool params are JSON. Extract the
fields you need in your SELECT:

```sql
SELECT
    timestamp,
    json_extract(context, '$.tool_name') as tool
FROM logs
```

**Filter then aggregate**: SQLite is fast, but filtering before aggregation
helps with large bundles:

```sql
SELECT tool_name, AVG(duration_ms)
FROM tool_calls
WHERE success = 1  -- Filter first
GROUP BY tool_name
```

## Debugging Scenarios

### "Why did the agent fail?"

```sql
-- Check the error summary
SELECT * FROM error_summary

-- Look at error logs with context
SELECT timestamp, message, context FROM logs WHERE level = 'ERROR'

-- Check if a tool failed
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0
```

### "Why was the agent slow?"

```sql
-- Find slow tool calls
SELECT * FROM tool_timeline WHERE duration_ms > 5000

-- Aggregate by tool
SELECT tool_name, COUNT(*), SUM(duration_ms) as total_ms
FROM tool_calls
GROUP BY tool_name
ORDER BY total_ms DESC
```

### "What did the agent actually do?"

```sql
-- Timeline of all tool calls
SELECT * FROM tool_timeline

-- Transcript timeline
SELECT transcript_source, sequence_number, entry_type, role, content
FROM transcript
ORDER BY rowid

-- For Claude Code native tools
SELECT * FROM native_tool_calls
```

### "What was the session state?"

```sql
-- List all slice types
SELECT DISTINCT slice_type FROM session_slices

-- Query specific slice (table name from schema)
SELECT * FROM slice_agentplan
```

## Limitations

- Single bundle at a time (no cross-bundle queries)
- SQLite functions only (no custom functions)
- Entire bundle is loaded on first query
- Native tool content is nested JSON (use `json_extract()`)
- Extended thinking content is signature-encoded (not readable)

## Next Steps

- [Debugging](debugging.md): Broader debugging and observability patterns
- [Debug Bundle spec](../specs/DEBUG_BUNDLE.md): Bundle format details
- [Sessions](sessions.md): Understanding session state structure
