# Logging Schema and Conventions

This document describes the runtime logging mini-framework that lives inside
the `weakincentives` library. It is an **internal** facility shared by runtime
modules; callers do not consume it directly. The goals are to:

- give maintainers a stable contract for structured log shapes across runtime
  modules;
- make downstream collectors and CLI presentation code resilient to module-level
  change by reusing common field names; and
- keep loggers module scoped via `weakincentives.runtime.logging.get_logger(__name__)`
  so signals stay attributable without a global registry.

Changes SHOULD extend this file first to capture intent before altering
logger usage. Treat the schemas and conventions below as the single source of
truth for the in-repo logging framework.

## Scope

The logging framework covers runtime-facing modules (event bus, sessions,
adapters, prompt overrides, built-in tools). It intentionally excludes:

- build and CI scripts;
- test-only utilities; and
- external consumers (applications integrating the library must layer their own
  logging policies on top of what the runtime emits).

Within scope, log records SHOULD favor structured payloads over formatted
messages so the runtime can evolve without breaking observability pipelines.

## Design Intent (Internal Framework)

The logging framework is intentionally minimal: it provides shared semantics for
event names, severity, and structured payloads without wrapping the standard
library API. The design choices are:

- **Module isolation**: each module owns its logger instance; cross-module
  helpers SHOULD rely on shared field names rather than shared logger objects.
- **Structured-first**: prefer stable key/value pairs (`extra`) over message
  formatting to keep downstream parsing simple.
- **Event taxonomy**: every non-error record SHOULD carry an `event` key so
  collectors and CLI renderers can bucket logs predictably.
- **Non-breaking evolvability**: new context fields SHOULD extend existing
  schemas; breaking changes require updating this document and coordinating
  migrations.

These rules keep the runtime logs cohesive while letting maintainers adjust
message wording or add fields without surprising internal consumers.

## Current Implementation (Module-Level Loggers)

Runtime modules attach to Python's standard library logging without custom
handlers by default, using `StructuredLogger` (a `logging.LoggerAdapter`) to
enforce an `{event, context}` schema. The table below captures the current
surface area and should be kept in sync with code changes.

| Module | Logger Variable | Level | Event Name | Context Fields |
| --------------------------------- | --------------- | ----------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| `runtime/events/__init__.py` | `logger` | `exception` | `event_delivery_failed` | `event_type`, `handler` |
| `runtime/session/session.py` | `logger` | `exception` | `session_reducer_failed` | `reducer`, `data_type`, `slice_type` |
| `runtime/session/session.py` | `logger` | `exception` | `session_observer_failed` | `observer`, `slice_type` |
| `adapters/shared.py` | `log` | `exception` | `tool_handler_exception` | `provider_payload` |
| `adapters/shared.py` | `log` | `info` | `prompt_execution_started` | `tool_count` |
| `adapters/shared.py` | `log` | `info` | `prompt_execution_succeeded` | `tool_count`, `has_output`, `text_length`, `structured_output`, `handler_count` |
| `adapters/shared.py` | `log` | `error` | `prompt_execution_publish_failed` | `failure_count`, `failed_handlers` |
| `adapters/shared.py` | `log` | `debug` | `prompt_rendered_published` | `handler_count` |
| `adapters/shared.py` | `log` | `error` | `prompt_rendered_publish_failed` | `failure_count`, `failed_handlers` |
| `adapters/shared.py` | `log` | `debug` | `prompt_tool_calls_detected` | `count` |
| `adapters/shared.py` | `log` | `info` | `tool_handler_completed` | `success`, `has_value` |
| `adapters/shared.py` | `log` | `warning` | `tool_validation_failed` | `reason` |
| `adapters/shared.py` | `log` | `warning` | `prompt_throttled` | `kind`, `delay_seconds`, `attempt`, `retry_after_seconds` |
| `adapters/shared.py` | `log` | `warning` | `session_rollback_due_to_publish_failure` | (none) |
| `adapters/shared.py` | `log` | `error` | `tool_event_publish_failed` | `failure_count`, `failed_handlers` |
| `adapters/shared.py` | `log` | `debug` | `tool_event_published` | `handler_count` |
| `prompt/overrides/local_store.py` | `_LOGGER` | `info` | `prompt_override_resolved` | `ns`, `prompt_key`, `tag` |
| `prompt/overrides/local_store.py` | `_LOGGER` | `info` | `prompt_override_persisted` | `ns`, `prompt_key`, `tag` |
| `prompt/overrides/local_store.py` | `_LOGGER` | `debug` | `prompt_override_missing` | `ns`, `prompt_key`, `tag` |
| `prompt/overrides/local_store.py` | `_LOGGER` | `debug` | `prompt_override_empty` | `ns`, `prompt_key`, `tag` |
| `prompt/overrides/local_store.py` | `_LOGGER` | `debug` | `prompt_override_delete_missing` | `ns`, `prompt_key`, `tag` |
| `contrib/tools/asteval.py` | `_LOGGER` | `debug` | `asteval_run` | `stdout_len`, `stderr_len`, `write_count`, `code_preview` |

### Module Notes and Caveats

- **events.py**: Exceptions from subscriber handlers are logged at ERROR and the
  publish operation continues, collecting the failures for the caller. The
  structured context exposes the event class name and handler reference to
  support debugging misbehaving subscribers.
- **session/session.py**: Reducer failures are logged at ERROR with structured
  context including `reducer`, `data_type`, and `slice_type`. The session
  suppresses the exception, skips the reducer, and continues dispatching.
- **adapters/shared.py**: Unexpected tool handler failures are logged at ERROR.
  The adapter converts the exception into a failed `ToolResult` and continues.
  Adapter implementations MAY supply their own logger if they wish to enrich
  context fields, but SHOULD preserve the message for compatibility.
- **prompt/overrides/local_store.py**: Override resolution and persistence
  are logged at INFO; diagnostic messages for missing files run at DEBUG.
  Structured context includes namespace, prompt key, and tag.
- **tools/asteval.py**: Tool runs emit a DEBUG record with event `asteval_run`
  and telemetry describing the run. Note: event name uses underscore, not period.

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
    event="tool.run",
    context={
        "prompt_name": prompt.name,
        "adapter": adapter_id,
        "tool": tool.name,
    },
)
```

When reporting exceptions, continue using `logging.exception` (or
`logger.exception`) and include the same `extra` mapping so the structured
fields propagate alongside the traceback.

## Error Handling Expectations

- Publishing events MUST NOT raise from subscriber failures; the bus records
  each exception, logs it, and exposes failures through the `DispatchResult`.
- Reducers that raise are logged and skipped, leaving the previous state in
  place.
- Tool handlers that raise are logged and converted into `ToolResult` instances
  with `success=False` so adapters can continue execution.
- Prompt override operations surface validation issues by raising
  `PromptOverridesError`; DEBUG logs are available to help diagnose stale or
  missing overrides.
- Tool executions that succeed log a structured DEBUG record (`event`
  `"asteval_run"`) so telemetry pipelines can aggregate success metrics.

## Backwards-Compatibility & Maintainer Review

- Existing third-party integrations may parse `event="asteval_run"` and the
  accompanying fields; maintain backwards compatibility when renaming or
  expanding this payload.
- Introduce new structured events by adding an `event` key and documenting the
  schema here before release.
- Coordinate any breaking changes to log messages or context keys with the
  maintainer team. Schedule a review to confirm the schema and gather feedback
  on consumer expectations before landing future modifications.
