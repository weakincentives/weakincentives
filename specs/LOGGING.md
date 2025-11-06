# Logging Schema and Conventions

This document records the current logging surface in the runtime modules and
establishes conventions for future events. All loggers are module scoped and
created via `logging.getLogger(__name__)`.

## Current Logger Usage

| Module | Logger Variable | Level | Message / Event | Context Fields |
| --- | --- | --- | --- | --- |
| `src/weakincentives/events.py` | `logger` | `exception` (ERROR) | "Error delivering event %s to handler %r" | `event_type` (`type(event).__name__`), `handler` |
| `src/weakincentives/session/session.py` | `logger` | `exception` (ERROR) | "Reducer %r failed for data type %s" | `reducer`, `data_type` |
| `src/weakincentives/adapters/shared.py` | `logger` (or override) | `exception` (ERROR) | "Tool '%s' raised an unexpected exception." | `tool_name` |
| `src/weakincentives/prompt/local_prompt_overrides_store.py` | `_LOGGER` | `debug` | Missing override file, empty override payload, persistence success, delete-miss, unknown section/tool, stale hashes | `ns`, `prompt_key`, `tag`, `section_path`, `expected_hash`, `found_hash`, `tool_name` |
| `src/weakincentives/tools/asteval.py` | `_logger` | `debug` | event="asteval.run" | `event`, `mode`, `stdout_len`, `stderr_len`, `write_count`, `code_preview` |

### Module Notes

- **events.py**: Exceptions from subscriber handlers are logged at ERROR and the
  publish operation continues, collecting the failures for the caller.
- **session/session.py**: Reducer failures are logged at ERROR. The session
  suppresses the exception, skips the reducer, and continues dispatching.
- **adapters/shared.py**: Unexpected tool handler failures are logged at ERROR.
  The adapter converts the exception into a failed `ToolResult` and continues.
- **prompt/local_prompt_overrides_store.py**: Diagnostic messages for override
  lookups, persistence, and validation run at DEBUG. They use positional
  formatting to emit namespace, prompt key, tag, and mismatch details.
- **tools/asteval.py**: Tool runs emit a DEBUG record with an explicit
  `event` field (`"asteval.run"`) and additional telemetry describing the run.

## Required Context Keys

To support downstream consumers (CLI output, structured log collectors, and
third-party analytics), logging calls SHOULD include the following fields when
available:

- `event`: A stable event name that categorizes the log entry (mandatory for
  structured DEBUG/INFO events; optional for exception paths that rely on
  message templates).
- `prompt_name`: Name of the prompt being evaluated when the log is tied to a
  prompt lifecycle event (e.g., publish failures, adapter execution).
- `adapter`: Adapter identifier for events emitted from provider adapters.
- `tool`: Tool identifier when reporting tool invocation outcomes.
- `mode`: Execution mode for tools that support multiple behaviors (e.g.,
  `"expr"` vs `"statements"` in the asteval tool).

When a module cannot provide a field (for example, there is no active prompt),
omit it rather than emitting empty placeholders.

## Severity Conventions

- Use `DEBUG` for diagnostic and lifecycle messages that assist with local
  development or verbose tracing (e.g., prompt override resolution, tool run
  summaries).
- Use `INFO` for high-level lifecycle events that should appear in default logs
  (e.g., successful prompt execution summaries once implemented).
- Use `WARNING` for recoverable conditions that may require operator attention
  (e.g., automatic fallbacks, deprecated configuration usage).
- Use `ERROR` for unexpected exceptions that were caught and converted into a
  fallback path (e.g., reducer failures, tool handler crashes).
- Use `CRITICAL` only when the process is about to exit or enter a degraded
  state that cannot self-recover.

`logging.exception()` automatically records a stack trace and SHOULD be used for
exception paths where execution continues after capturing the error.

## Structured Context Delivery

Always pass structured fields via the logger's `extra` mapping (or the
repository's helper wrappers) instead of formatting them into the message
string. This keeps the message stable while downstream collectors receive the
full context payload.

```python
logger.info(
    "Tool execution completed",
    extra={
        "event": "tool.run",
        "prompt_name": prompt.name,
        "adapter": adapter_id,
        "tool": tool.name,
        "mode": tool.mode,
    },
)
```

When reporting exceptions, continue using `logging.exception` (or
`logger.exception`) and include the same `extra` mapping so the structured
fields propagate alongside the traceback.

## Error Handling Expectations

- Publishing events MUST NOT raise from subscriber failures; the bus records
  each exception, logs it, and exposes failures through the `PublishResult`.
- Reducers that raise are logged and skipped, leaving the previous state in
  place.
- Tool handlers that raise are logged and converted into `ToolResult` instances
  with `success=False` so adapters can continue execution.
- Prompt override operations surface validation issues by raising
  `PromptOverridesError`; DEBUG logs are available to help diagnose stale or
  missing overrides.
- Tool executions that succeed log a structured DEBUG record (`event`
  `"asteval.run"`) so telemetry pipelines can aggregate success metrics.

## Backwards-Compatibility & Maintainer Review

- Existing third-party integrations may parse `event="asteval.run"` and the
  accompanying fields; maintain backwards compatibility when renaming or
  expanding this payload.
- Introduce new structured events by adding an `event` key and documenting the
  schema here before release.
- Coordinate any breaking changes to log messages or context keys with the
  maintainer team. Schedule a review to confirm the schema and gather feedback
  on consumer expectations before landing future modifications.
