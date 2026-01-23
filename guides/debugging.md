# Debugging and Observability

This guide covers how to understand what your agent is doing, why it's doing it,
and what went wrong when things fail.

## Structured Logging

*Canonical spec: [specs/LOGGING.md](../specs/LOGGING.md)*

```python nocheck
from weakincentives.runtime import configure_logging, get_logger

configure_logging(level="INFO", json_mode=True)
logger = get_logger(__name__)
logger.info("hello", event="demo.hello", context={"foo": "bar"})
```

Logs include structured `event` and `context` fields for downstream routing and
analysis. JSON mode makes logs machine-parseable.

**Event taxonomy:**

Every non-error log record carries an `event` key for predictable downstream
routing. The naming convention is `module.action`:

| Event Pattern | Description |
| ----------------------------------- | ------------------------------------ |
| `prompt.render.*` | Prompt rendering lifecycle |
| `tool.execution.*` | Tool invocation and results |
| `session.dispatch`, `session.reset` | Session state mutations |
| `adapter.init`, `evaluate.*` | Provider adapter lifecycle |
| `hook.*` | Claude Agent SDK execution hooks |
| `bridge.*` | MCP tool bridging |

**Enabling DEBUG logs:**

```bash
export WEAKINCENTIVES_LOG_LEVEL=DEBUG
```

DEBUG-level logs include full tool arguments, message content, and lifecycle
events—invaluable for understanding exactly what happened during an evaluation.

## Session Events

Sessions subscribe to the event dispatcher and capture telemetry events:

- `PromptRendered`: emitted when a prompt is rendered
- `ToolInvoked`: emitted when a tool is called (includes params, result, timing)
- `PromptExecuted`: emitted when a prompt evaluation completes (includes token
  usage)
- `TokenUsage`: token counts from provider responses

You can use these for your own tracing pipeline. Subscribe to the dispatcher and
route events wherever you need them.

## Inspecting What Was Sent

To see exactly what's being sent to the model:

```python nocheck
rendered = prompt.render(session=session)
print(rendered.text)  # Full prompt markdown
print([t.name for t in rendered.tools])  # Tool names
```

This is deterministic—same inputs produce the same output. You can use this for
debugging or for snapshot tests.

## Debug Bundles

*Canonical spec: [specs/DEBUG_BUNDLE.md](../specs/DEBUG_BUNDLE.md)*

Debug bundles capture execution state in a self-contained zip archive:

```python nocheck
from weakincentives.debug import BundleConfig
from weakincentives.runtime import MainLoopConfig

# Automatic bundling per-request via MainLoop
config = MainLoopConfig(
    debug_bundle=BundleConfig(
        target="./debug_bundles/",
    ),
)
```

**What's captured:**

- Session state before and after execution
- Log records during evaluation
- Request input and output
- Configuration and run context
- Filesystem snapshots (workspace state)

For manual bundle creation:

```python nocheck
from weakincentives.debug import BundleWriter

with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
    writer.write_request_input(request)
    with writer.capture_logs():
        response = adapter.evaluate(prompt, session=session)
    writer.write_request_output(response)
    writer.write_session_after(session)
```

## The Debug UI

**Install:** `pip install "weakincentives[wink]"`

**Run:**

```bash
wink debug debug_bundles/  # Opens most recent bundle (directory mode)
wink debug debug_bundles/<session_id>.zip  # Opens specific bundle
```

This starts a local server that renders the prompt/tool timeline for inspection.
You can see:

- Exactly what was sent to the model
- What tools were called and what they returned
- How state evolved over the session
- Token usage at each step

The debug UI is your primary tool for understanding agent behavior after the
fact.

### SQLite Caching

The debug UI shares SQLite database infrastructure with `wink query`. When a
cached database exists from a previous query run, the debug UI starts instantly
without re-parsing bundle contents:

```
./bundle.zip           → input
./bundle.zip.sqlite    → cache (shared by debug UI and query)
```

Key benefits:

- **Unified caching**: Both `wink debug` and `wink query` use the same `.sqlite`
  cache file
- **Thread-safe access**: Database operations use locking for FastAPI
  compatibility
- **SQL-powered pagination**: Filtering handled by SQLite instead of in-memory
  Python lists

### Log Filtering

The debug UI provides powerful log filtering capabilities:

| Filter | Description |
| --- | --- |
| `level` | Filter by log level (DEBUG, INFO, WARNING, ERROR) |
| `logger` | Filter by logger name |
| `event` | Filter by event type |
| `exclude_logger` | Exclude specific loggers |
| `exclude_event` | Exclude specific event types |
| `search` | Full-text search across log messages |

The filter facets API returns counts per logger/event/level for autocomplete in
the UI.

## Querying Debug Bundles

*Canonical spec: [specs/WINK_QUERY.md](../specs/WINK_QUERY.md)*

The `wink query` command enables SQL-based exploration of debug bundles. Bundle
contents are loaded into a cached SQLite database.

```bash
# Always start with schema to discover tables
wink query ./bundle.zip --schema

# Query with JSON output (default)
wink query ./bundle.zip "SELECT * FROM errors"

# Query with ASCII table output
wink query ./bundle.zip "SELECT * FROM tool_calls" --table

# Full values without truncation
wink query ./bundle.zip "SELECT * FROM tool_calls" --table --no-truncate
```

### SQL Views

Pre-built views provide common analysis patterns:

| View | Description |
| --- | --- |
| `tool_timeline` | Tool calls ordered by timestamp with command extraction |
| `native_tool_calls` | Claude Code native tools from log_aggregator events |
| `error_summary` | Errors with truncated traceback for quick debugging |

```sql
-- Use views directly
SELECT * FROM tool_timeline WHERE duration_ms > 1000
SELECT * FROM error_summary
SELECT * FROM native_tool_calls LIMIT 10
```

### Sequence Number Tracking

The `logs` table includes a `seq` column extracted from `log_aggregator.log_line`
events, enabling range queries on native tool executions:

```sql
-- Query native tool logs by sequence range
SELECT seq, json_extract(context, '$.content') as content
FROM logs
WHERE event = 'log_aggregator.log_line'
  AND seq BETWEEN 100 AND 200
ORDER BY seq
```

For non-log_aggregator events, `seq` is NULL.

### JSONL Export

For power users who prefer `jq` over SQL:

```bash
# Export logs
wink query ./bundle.zip --export-jsonl | jq 'select(.event == "tool.execution.start")'

# Export session state
wink query ./bundle.zip --export-jsonl=session | jq 'select(.__type__ | contains("Plan"))'
```

### Schema Hints

The `--schema` output includes helpful hints for querying:

- **json_extraction**: Common `json_extract()` patterns for nested JSON data
- **common_queries**: Ready-to-use SQL queries for typical analysis tasks

```bash
wink query ./bundle.zip --schema | jq '.hints'
```

**Loading bundles programmatically:**

```python nocheck
from weakincentives.debug import DebugBundle

bundle = DebugBundle.load("./debug_bundles/abc123.zip")
print(bundle.manifest.request.status)
print(bundle.request_input)
print(bundle.logs)
```

## Common Debugging Patterns

**"Why did the model do X?"**

1. Dump the session snapshot
1. Open the debug UI
1. Look at the prompt that was sent immediately before X
1. Check the tool results the model saw before deciding

**"Why did this tool call fail?"**

1. Look at the `ToolInvoked` event
1. Check the params the model sent
1. Check the error message in the result
1. Look at the tool handler code

**"Why is the agent in the wrong state?"**

1. Look at the session slices
1. Trace back through the events that modified that slice
1. Find the event that caused the unexpected state

**"Why did tokens spike?"**

1. Look at the `PromptExecuted` events
1. Check which prompts had large token counts
1. Consider using progressive disclosure to reduce initial prompt size

## Debug CLI Commands

```bash
# Start the debug UI server
wink debug <bundle_path> [options]
wink debug <directory>   # Auto-selects newest bundle

# Query bundles with SQL
wink query <bundle_path> --schema
wink query <bundle_path> "<SQL>"
wink query <bundle_path> "<SQL>" --table
wink query <bundle_path> --export-jsonl

# Access bundled documentation
wink docs --guide       # Print guides (usage guide)
wink docs --reference   # Print llms.md (API reference)
wink docs --specs       # Print all spec files concatenated
wink docs --changelog   # Print CHANGELOG.md
```

**Debug UI options:**

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `8000` | Port to bind |
| `--open-browser` | `true` | Open browser automatically |
| `--no-open-browser` | - | Disable auto-open |
| `--log-level` | `INFO` | Log verbosity |
| `--json-logs` | `true` | Emit structured JSON logs |

**Query options:**

| Option | Description |
| --- | --- |
| `--schema` | Output schema as JSON and exit |
| `--table` | Output as ASCII table (default: JSON) |
| `--no-truncate` | Disable column truncation in table output |
| `--export-jsonl` | Export raw JSONL (`logs` or `session`) to stdout |

## Accessing Bundled Documentation

*Canonical spec: [specs/WINK_DOCS.md](../specs/WINK_DOCS.md)*

The `wink docs` command provides access to WINK documentation for users who
install the package via pip. This supports LLM-assisted development workflows
where documentation can be piped directly to tools.

```bash
# Copy API reference to clipboard
wink docs --reference | pbcopy

# Pipe to an LLM context
wink docs --guide | llm "Summarize the key concepts"

# Export all specs to a file
wink docs --specs > all-specs.md

# Combine multiple flags
wink docs --reference --specs  # Outputs both, separated by ---
```

**Why this exists:** When you install WINK via `pip install weakincentives`, you
don't have access to the repository's documentation files. The docs subcommand
bundles documentation inside the package and exposes it via CLI for easy access.

## Next Steps

- [Testing](testing.md): Write tests that catch issues before they hit
  production
- [Evaluation](evaluation.md): Systematically test agent behavior
- [Sessions](sessions.md): Understand how state is structured
