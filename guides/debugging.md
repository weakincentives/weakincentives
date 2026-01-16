# Debugging and Observability

This guide covers how to understand what your agent is doing, why it's doing it,
and what went wrong when things fail.

## Structured Logging

```python
from weakincentives.runtime import configure_logging, get_logger

configure_logging(level="INFO", json_mode=True)
logger = get_logger(__name__)
logger.info("hello", event="demo.hello", context={"foo": "bar"})
```

Logs include structured `event` and `context` fields for downstream routing and
analysis. JSON mode makes logs machine-parseable.

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

```python
rendered = prompt.render(session=session)
print(rendered.text)  # Full prompt markdown
print([t.name for t in rendered.tools])  # Tool names
```

This is deterministic—same inputs produce the same output. You can use this for
debugging or for snapshot tests.

## Dumping Snapshots to JSONL

Use `weakincentives.debug.dump_session(...)` to persist a session tree:

```python
from weakincentives.debug import dump_session

path = dump_session(session, target="snapshots/")  # writes <session_id>.jsonl
```

Each line is one serialized session snapshot (root → leaves). The JSONL format
is stable and human-readable.

**What's captured:**

- All session slices (state and logs)
- All events that flowed through the dispatcher
- Timestamps for each snapshot

## The Debug UI

**Install:** `pip install "weakincentives[wink]"`

**Run:**

```bash
wink debug snapshots/<session_id>.jsonl
```

This starts a local server that renders the prompt/tool timeline for inspection.
You can see:

- Exactly what was sent to the model
- What tools were called and what they returned
- How state evolved over the session
- Token usage at each step

The debug UI is your primary tool for understanding agent behavior after the
fact.

## Common Debugging Patterns

**"Why did the model do X?"**

1. Dump the session snapshot
2. Open the debug UI
3. Look at the prompt that was sent immediately before X
4. Check the tool results the model saw before deciding

**"Why did this tool call fail?"**

1. Look at the `ToolInvoked` event
2. Check the params the model sent
3. Check the error message in the result
4. Look at the tool handler code

**"Why is the agent in the wrong state?"**

1. Look at the session slices
2. Trace back through the events that modified that slice
3. Find the event that caused the unexpected state

**"Why did tokens spike?"**

1. Look at the `PromptExecuted` events
2. Check which prompts had large token counts
3. Consider using progressive disclosure to reduce initial prompt size

## Debug CLI Commands

```bash
# Start the debug UI server
wink debug <snapshot_path> [options]

# Access bundled documentation
wink docs --guide       # Print WINK_GUIDE.md
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

## Next Steps

- [Testing](testing.md): Write tests that catch issues before they hit
  production
- [Evaluation](evaluation.md): Systematically test agent behavior
- [Sessions](sessions.md): Understand how state is structured
