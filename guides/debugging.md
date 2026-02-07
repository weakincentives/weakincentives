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
from weakincentives.runtime import AgentLoopConfig

# Automatic bundling per-request via AgentLoop
config = AgentLoopConfig(
    debug_bundle=BundleConfig(
        target="./debug_bundles/",
    ),
)

# Or for evaluations via EvalLoop
from weakincentives.evals import EvalLoopConfig

eval_config = EvalLoopConfig(
    debug_bundle=BundleConfig(target="./eval_bundles/"),
)
```

EvalLoop bundles include all standard bundle contents plus an `eval.json` file
with score, experiment name, and latency. See [Evaluation](evaluation.md) for
details on eval-specific debugging.

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
wink debug debug_bundles/           # Opens most recent bundle in directory
wink debug debug_bundles/abc123.zip # Opens specific bundle
```

This starts a local web server with a visual interface for bundle inspection.

**What you can inspect:**

- **Session state**: Browse session slices with markdown rendering
- **Logs**: Filter by level, logger, event, or search term
- **Transcripts**: View conversation flow, tool calls, thinking blocks
- **Tool calls**: See params, results, timing, and success status
- **Files**: Browse workspace snapshots including image preview support
- **Environment**: System info, Python version, Git state, container runtime
- **Metrics**: Token usage and timing data
- **Errors**: Full error details when execution fails

**Bundle management:**

When pointed at a directory, the debug UI can:

- Auto-detect and load the most recent bundle
- List and switch between all bundles in the directory
- Reload bundles after re-running evaluations

The debug UI is your primary tool for visual exploration of agent behavior.

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

### wink debug

Start the visual debug UI server:

```bash
wink debug <bundle_path> [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `8000` | Port to bind |
| `--open-browser` | `true` | Open browser automatically |
| `--no-open-browser` | - | Disable auto-open |
| `--log-level` | `INFO` | Log verbosity |
| `--json-logs` | `true` | Emit structured JSON logs |

### wink query

SQL-based analysis of debug bundles. See [Query Guide](query.md) for full
details.

```bash
wink query <bundle_path> --schema                    # Discover tables and hints
wink query <bundle_path> "SELECT * FROM errors"     # Run SQL query (JSON output)
wink query <bundle_path> "SQL" --table              # ASCII table output
wink query <bundle_path> "SQL" --table --no-truncate # Full values
wink query <bundle_path> --export-jsonl             # Export raw logs as JSONL
wink query <bundle_path> --export-jsonl=session     # Export session state
```

**Key tables:**

| Table | Contents |
| --- | --- |
| `manifest` | Bundle metadata (ID, status, timestamps) |
| `logs` | All log entries with structured context |
| `transcript` | Conversation entries from TranscriptCollector |
| `tool_calls` | Tool invocations with params, results, timing |
| `errors` | Aggregated errors from all sources |
| `files` | Workspace file contents |
| `metrics` | Token usage and timing |

**Environment tables** (inspect execution context):

| Table | Contents |
| --- | --- |
| `env_system` | OS, architecture, CPU, memory |
| `env_python` | Python version, virtualenv, executable |
| `env_git` | Commit, branch, dirty state, remotes |
| `env_container` | Docker/K8s runtime info |
| `env_vars` | Filtered environment variables |

**Pre-built views:**

| View | Purpose |
| --- | --- |
| `tool_timeline` | Tool calls ordered by timestamp |
| `native_tool_calls` | Native tool calls from transcripts |
| `transcript_flow` | Conversation flow with message previews |
| `transcript_tools` | Tool usage with paired calls and results |
| `transcript_thinking` | Thinking blocks with preview and length |
| `transcript_agents` | Agent hierarchy and activity metrics |
| `error_summary` | Errors with truncated tracebacks |

**Common queries:**

```sql
-- Find all errors
SELECT * FROM error_summary

-- Slow tool calls
SELECT * FROM tool_timeline WHERE duration_ms > 1000

-- Conversation flow (last 50 messages)
SELECT * FROM transcript_flow ORDER BY rowid DESC LIMIT 50

-- Tool usage by frequency
SELECT tool_name, COUNT(*) as calls FROM tool_calls GROUP BY tool_name

-- Thinking blocks over 1000 chars
SELECT * FROM transcript_thinking WHERE thinking_length > 1000

-- Sub-agent activity
SELECT * FROM transcript_agents WHERE transcript_source != 'main'

-- Environment info
SELECT * FROM env_system
SELECT * FROM env_git
```

### wink docs

Access bundled documentation:

```bash
wink docs --guide       # Print guides (usage guide)
wink docs --reference   # Print llms.md (API reference)
wink docs --specs       # Print all spec files concatenated
wink docs --changelog   # Print CHANGELOG.md
```

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

## Instructing Coding Agents to Use wink query

When working with a coding agent (like Claude Code) to debug WINK agent
executions, you can instruct it to use `wink query` for systematic analysis.
This is more effective than manually exploring logs.

### Prompt Instructions for Agents

Include the following guidance in your prompt or system instructions:

> **Debug Bundle Analysis**: When analyzing debug bundles, use `wink query` for
> SQL-based exploration. Always start with `wink query ./bundle.zip --schema`
> to discover tables, columns, and pre-built queries.

**Key views to use:**

- `error_summary` - All errors with truncated tracebacks
- `tool_timeline` - Tool calls ordered by timestamp
- `transcript_flow` - Conversation flow with message previews
- `transcript_tools` - Tool usage with paired calls and results
- `transcript_thinking` - Thinking blocks (for extended thinking)
- `transcript_agents` - Sub-agent hierarchy and activity

**Investigation queries to provide:**

```sql
-- What went wrong?
SELECT * FROM error_summary

-- What did the agent do?
SELECT * FROM transcript_flow ORDER BY rowid

-- Which tools were slow?
SELECT * FROM tool_timeline WHERE duration_ms > 1000

-- Failed tool calls
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0
```

**Output formats to mention:**

- Default: JSON (good for parsing)
- `--table`: ASCII table (good for reading)
- `--table --no-truncate`: Full values without truncation

### Key Points for Agents

When instructing a coding agent to debug, emphasize:

1. **Schema first**: The `--schema` output includes ready-to-use queries in the
   `hints.common_queries` section. The agent should start here.

1. **Views over raw tables**: Pre-built views like `transcript_flow` and
   `tool_timeline` are designed for the most common analysis tasks.

1. **JSON extraction**: Log context and tool params are JSON. Use SQLite's
   `json_extract()` function for nested data extraction.

1. **Sequence ordering**: Transcript entries have `sequence_number` for
   ordering. Native tool calls can be queried by sequence range.

1. **Environment context**: When debugging environment-specific issues, query
   the `env_*` tables (`env_system`, `env_git`, `env_python`, `env_container`).

### Example Agent Workflow

A well-instructed coding agent should follow this pattern:

1. Run `wink query ./bundle.zip --schema` to understand the bundle
1. Check `SELECT * FROM error_summary` if investigating a failure
1. Use `transcript_flow` to trace the conversation
1. Query specific tables based on the schema hints
1. Use `--table` output for human-readable results when sharing findings

This structured approach ensures thorough analysis while minimizing token usage
from reading raw log files.

## Next Steps

- [Query](query.md): SQL-based analysis of debug bundles (comprehensive guide)
- [Testing](testing.md): Write tests that catch issues before they hit
  production
- [Evaluation](evaluation.md): Systematically test agent behavior
- [Sessions](sessions.md): Understand how state is structured
