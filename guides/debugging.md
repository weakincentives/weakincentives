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
wink debug debug_bundles/  # Opens most recent bundle
wink debug debug_bundles/<session_id>.jsonl  # Opens specific snapshot
```

This starts a local server that renders the prompt/tool timeline for inspection.
You can see:

- Exactly what was sent to the model
- What tools were called and what they returned
- How state evolved over the session
- Token usage at each step

The debug UI is your primary tool for understanding agent behavior after the
fact.

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
wink debug <snapshot_path> [options]

# Access bundled documentation
wink docs --guide       # Print guides (usage guide)
wink docs --reference   # Print llms.md (API reference)
wink docs --specs       # Print all spec files concatenated
wink docs --changelog   # Print CHANGELOG.md
```

**Debug options:**

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `8000` | Port to bind |
| `--open-browser` | `true` | Open browser automatically |
| `--no-open-browser` | - | Disable auto-open |
| `--log-level` | `INFO` | Log verbosity |
| `--json-logs` | `true` | Emit structured JSON logs |

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
