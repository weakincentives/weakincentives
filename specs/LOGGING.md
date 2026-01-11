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

The logging framework covers runtime-facing modules (event dispatcher, sessions,
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

### Adapter DEBUG Logging

All adapters provide extensive DEBUG-level logging for troubleshooting. These
events are suppressed by default and enabled via `WEAKINCENTIVES_LOG_LEVEL=DEBUG`.

#### Claude Agent SDK Adapter

The Claude Agent SDK adapter provides the most extensive debug logging due to its
subprocess-based execution model.

| Module | Event | Context Fields |
| ------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `adapters/claude_agent_sdk/adapter.py` | `adapter.init` | `model`, `permission_mode`, `cwd`, `max_turns`, `max_budget_usd`, `suppress_stderr`, `stop_on_structured_output`, `has_isolation_config`, `allowed_tools`, `disallowed_tools`, `max_thinking_tokens` |
| `adapters/claude_agent_sdk/adapter.py` | `evaluate.entry` | `prompt_name`, `prompt_ns`, `prompt_key`, `has_deadline`, `deadline_remaining_seconds`, `has_budget`, `has_budget_tracker` |
| `adapters/claude_agent_sdk/adapter.py` | `evaluate.deadline_expired` | `prompt_name` |
| `adapters/claude_agent_sdk/adapter.py` | `evaluate.rendered` | `prompt_text_length`, `tool_count`, `tool_names`, `has_output_type`, `output_type` |
| `adapters/claude_agent_sdk/adapter.py` | `evaluate.temp_workspace_created` | `temp_workspace_dir` |
| `adapters/claude_agent_sdk/adapter.py` | `evaluate.filesystem_bound` | `effective_cwd` |
| `adapters/claude_agent_sdk/adapter.py` | `run_context.entry` | `prompt_name`, `effective_cwd`, `has_output_format` |
| `adapters/claude_agent_sdk/adapter.py` | `run_context.bridged_tools` | `bridged_tool_count`, `bridged_tool_names` |
| `adapters/claude_agent_sdk/adapter.py` | `run_context.isolation` | `ephemeral_home_path`, `workspace_path`, `network_policy`, `sandbox_enabled`, `has_api_key_override`, `include_host_env`, `skill_count` |
| `adapters/claude_agent_sdk/adapter.py` | `run_context.visibility_expansion_required` | `prompt_name` |
| `adapters/claude_agent_sdk/adapter.py` | `run_context.sdk_error` | `prompt_name`, `error_type`, `error_message`, `stderr_output`, `exit_code` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.entry` | `prompt_text_preview`, `has_output_format`, `bridged_tool_count` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.env_configured` | `home_override`, `has_api_key`, `env_var_count`, `env_keys` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.mcp_server_configured` | `mcp_server_name` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.stderr` | `line` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.hooks_registered` | `hook_types` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.options` | `model`, `cwd`, `permission_mode`, `max_turns`, `max_budget_usd`, `max_thinking_tokens`, `has_output_format`, `allowed_tools`, `disallowed_tools`, `has_mcp_servers`, `betas` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.executing` | `prompt_name` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.message_received` | `message_type`, `message_index`, `role`, `content`, `content_blocks`, `result`, `structured_output`, `usage`, `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `has_thinking`, `thinking_preview`, `thinking_length`, `cumulative_input_tokens`, `cumulative_output_tokens` |
| `adapters/claude_agent_sdk/adapter.py` | `sdk_query.complete` | `message_count`, `stderr_line_count`, `stats_tool_count`, `stats_turn_count`, `stats_subagent_count`, `stats_compact_count`, `stats_input_tokens`, `stats_output_tokens`, `stats_hook_errors` |
| `adapters/claude_agent_sdk/_errors.py` | `error.normalizing` | `error_type`, `error_module`, `error_message`, `has_stderr_output`, `stderr_preview` |
| `adapters/claude_agent_sdk/_errors.py` | `error.cli_not_found` | `prompt_name` |
| `adapters/claude_agent_sdk/_errors.py` | `error.cli_connection_error` | `prompt_name`, `stderr_output` |
| `adapters/claude_agent_sdk/_errors.py` | `error.process_error` | `prompt_name`, `exit_code`, `stderr`, `stderr_captured` |
| `adapters/claude_agent_sdk/_errors.py` | `error.json_decode_error` | `prompt_name`, `stderr_output` |
| `adapters/claude_agent_sdk/_errors.py` | `error.max_turns_exceeded` | `prompt_name` |
| `adapters/claude_agent_sdk/_errors.py` | `error.unknown` | `prompt_name`, `is_sdk_error`, `error_type`, `stderr_output` |

**Stderr Capture**: The Claude Agent SDK adapter captures all stderr output from the
underlying Claude Code process, even when `suppress_stderr=True`. This captured output
is logged at DEBUG level via the `sdk_query.stderr` event and included in error payloads
when process failures occur. This is particularly useful for debugging CLI issues.

**Full Content Logging**: The `sdk_query.message_received` event logs full message content
including text, tool uses, tool results, and structured output. This provides complete
visibility into the conversation flow for debugging without content truncation.

#### MCP Bridge

| Module | Event | Context Fields |
| ----------------------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.create_bridged_tools` | `tool_count`, `tool_names`, `prompt_name` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.tool_call.start` | `tool_name`, `prompt_name`, `arguments` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.tool_call.complete` | `tool_name`, `success`, `message`, `value_type`, `output_text` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.validation_error` | `tool_name`, `error` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.handler_error` | `tool_name`, `error` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.tool_registered` | `tool_name` |
| `adapters/claude_agent_sdk/_bridge.py` | `bridge.mcp_server_created` | `server_name`, `tool_count` |

#### SDK Hooks

The SDK hooks provide extensive DEBUG logging for tracking execution flow,
constraints, and cumulative statistics throughout Claude Code execution.

| Module | Event | Context Fields |
| ----------------------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.pre_tool_use` | `tool_name`, `tool_use_id`, `input_data`, `elapsed_ms`, `tool_count`, `deadline_remaining_ms`, `budget_consumed_input`, `budget_consumed_output`, `budget_consumed_total`, `budget_max_total`, `budget_remaining` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.deadline_exceeded` | `tool_name`, `elapsed_ms`, `tool_count` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.budget_exhausted` | `tool_name`, `consumed_total`, `max_total`, `elapsed_ms` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.snapshot_taken` | `tool_name`, `tool_use_id`, `hook_duration_ms` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.tool_invoked` | `tool_name`, `success`, `call_id`, `tool_input`, `tool_response`, `output_text`, `elapsed_ms`, `tool_count`, `turn_count` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.state_restored` | `tool_name`, `tool_use_id`, `reason`, `hook_duration_ms`, `elapsed_ms` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.structured_output_stop` | `tool_name`, `elapsed_ms`, `tool_count` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.turn_start` | `turn_number`, `session_id`, `prompt_preview`, `prompt_length`, `elapsed_ms`, `tool_count`, `cumulative_input_tokens`, `cumulative_output_tokens` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.stop` | `stop_reason`, `sdk_num_turns`, `sdk_duration_ms`, `result_preview`, `elapsed_ms`, `stats_tool_count`, `stats_turn_count`, `stats_subagent_count`, `stats_compact_count`, `stats_input_tokens`, `stats_output_tokens`, `stats_thinking_tokens`, `stats_hook_errors` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.subagent_start` | `subagent_number`, `subagent_type`, `subagent_id`, `description`, `elapsed_ms`, `tool_count`, `payload` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.subagent_stop` | `subagent_id`, `result_preview`, `subagent_duration_ms`, `subagent_tool_count`, `transcript_entries`, `elapsed_ms`, `parent_tool_count`, `subagent_count`, `payload` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.pre_compact` | `compact_number`, `context_tokens`, `max_context_tokens`, `utilization_pct`, `message_count`, `compaction_reason`, `elapsed_ms`, `tool_count`, `turn_count`, `payload` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.notification` | `notification_type`, `notification_level`, `message_preview`, `elapsed_ms`, `tool_count`, `payload` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.deadline_error_caught` | `error_type`, `hook_errors`, `elapsed_ms` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.budget_error_caught` | `error_type`, `hook_errors`, `elapsed_ms` |
| `adapters/claude_agent_sdk/_hooks.py` | `hook.error` | `error`, `error_type`, `hook_errors`, `elapsed_ms`, `tool_count` |

#### OpenAI Adapter

| Module | Event | Context Fields |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------- |
| `adapters/openai.py` | `adapter.init` | `model`, `tool_choice`, `has_model_config`, `has_client_config`, `used_explicit_client`, `temperature`, `max_tokens` |
| `adapters/openai.py` | `evaluate.entry` | `prompt_name`, `has_deadline`, `deadline_remaining_seconds`, `has_budget`, `has_budget_tracker` |
| `adapters/openai.py` | `evaluate.setup_complete` | `prompt_name`, `has_response_format`, `tool_count`, `tool_names` |
| `adapters/openai.py` | `provider.request` | `prompt_name`, `model`, `message_count`, `tool_count`, `tool_names`, `tool_choice`, `has_response_format` |
| `adapters/openai.py` | `provider.response` | `prompt_name`, `response_type`, `has_output`, `has_usage` |
| `adapters/openai.py` | `provider.error` | `prompt_name`, `error_type`, `error_message`, `status_code`, `code` |
| `adapters/openai.py` | `provider.throttle_detected` | `prompt_name`, `throttle_kind`, `retry_after_seconds` |

#### LiteLLM Adapter

| Module | Event | Context Fields |
| --------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `adapters/litellm.py` | `adapter.init` | `model`, `tool_choice`, `has_model_config`, `has_completion_config`, `used_explicit_completion`, `used_completion_factory`, `temperature`, `max_tokens` |
| `adapters/litellm.py` | `evaluate.entry` | `prompt_name`, `has_deadline`, `deadline_remaining_seconds`, `has_budget`, `has_budget_tracker` |
| `adapters/litellm.py` | `evaluate.setup_complete` | `prompt_name`, `has_response_format`, `tool_count`, `tool_names` |
| `adapters/litellm.py` | `provider.request` | `prompt_name`, `model`, `message_count`, `tool_count`, `tool_names`, `tool_choice`, `has_response_format` |
| `adapters/litellm.py` | `provider.response` | `prompt_name`, `response_type`, `has_choices`, `has_usage` |
| `adapters/litellm.py` | `throttle.analyzing` | `prompt_name`, `error_type`, `error_message`, `status_code`, `code` |
| `adapters/litellm.py` | `throttle.not_throttle` | `prompt_name`, `error_type` |
| `adapters/litellm.py` | `throttle.detected` | `prompt_name`, `throttle_kind`, `retry_after_seconds` |

### Session Dispatch

The session module provides DEBUG-level logging for all state mutations and
reducer executions.

| Module | Event | Context Fields |
| --------------------------------- | ------------------------------ | ----------------------------------------------------------------- |
| `runtime/session/session.py` | `session.dispatch` | `session_id`, `event_type` |
| `runtime/session/session.py` | `session.dispatch_data_event` | `session_id`, `data_type`, `reducer_count` |
| `runtime/session/session.py` | `session.reducer_applied` | `session_id`, `reducer`, `slice_type`, `operation` |
| `runtime/session/session.py` | `session.register_reducer` | `session_id`, `data_type`, `slice_type`, `reducer`, `policy` |
| `runtime/session/session.py` | `session.initialize_slice` | `session_id`, `slice_type`, `value_count` |
| `runtime/session/session.py` | `session.clear_slice` | `session_id`, `slice_type`, `has_predicate` |
| `runtime/session/session.py` | `session.reset` | `session_id`, `slice_count`, `slice_types` |
| `runtime/session/session.py` | `session.restore` | `session_id`, `preserve_logs`, `snapshot_slice_count`, `registered_slice_count` |

### Prompt Rendering

The prompt rendering pipeline provides DEBUG-level logging for section rendering
and tool collection.

| Module | Event | Context Fields |
| -------------------------- | -------------------------- | ----------------------------------------------------------------- |
| `prompt/rendering.py` | `prompt.render.start` | `descriptor`, `param_types`, `override_count`, `tool_override_count` |
| `prompt/rendering.py` | `prompt.render.section` | `section_path`, `section_type`, `visibility`, `has_override`, `depth` |
| `prompt/rendering.py` | `prompt.render.complete` | `descriptor`, `section_count`, `tool_count`, `text_length`, `has_structured_output` |

### Tool Execution

The tool executor provides DEBUG-level logging for all tool invocations including
full argument and result capture.

| Module | Event | Context Fields |
| --------------------------- | ---------------------------- | ----------------------------------------------------------------- |
| `adapters/tool_executor.py` | `tool.execution.start` | `tool_name`, `call_id`, `prompt_name`, `arguments` |
| `adapters/tool_executor.py` | `tool.execution.complete` | `tool_name`, `success`, `message`, `value`, `value_type` |
| `adapters/tool_executor.py` | `tool_handler_completed` | `success`, `has_value` |
| `adapters/tool_executor.py` | `tool_event_dispatched` | `handler_count` |
| `adapters/tool_executor.py` | `tool_event_dispatch_failed` | `failure_count`, `failed_handlers` |

### Resource Lifecycle

The resource context provides DEBUG-level logging for resource construction,
cleanup, and scoping.

| Module | Event | Context Fields |
| ----------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| `resources/context.py` | `resource.construct.start` | `protocol`, `scope` |
| `resources/context.py` | `resource.construct.complete` | `protocol`, `scope`, `instance_type` |
| `resources/context.py` | `resource.close` | `protocol`, `scope` |
| `resources/context.py` | `resource.context.close.start` | `resource_count` |
| `resources/context.py` | `resource.context.close.complete` | (none) |
| `resources/context.py` | `resource.tool_scope.enter` | (none) |
| `resources/context.py` | `resource.tool_scope.exit` | `closed_count` |

### Mailbox Operations

The mailbox module provides DEBUG-level logging for message send, receive, and
acknowledgment operations.

| Module | Event | Context Fields |
| ------------------------------ | ---------------------- | ----------------------------------------------------------------- |
| `runtime/mailbox/_in_memory.py` | `mailbox.send` | `mailbox`, `message_id`, `reply_to`, `body_type` |
| `runtime/mailbox/_in_memory.py` | `mailbox.receive` | `mailbox`, `message_count`, `message_ids` |
| `runtime/mailbox/_in_memory.py` | `mailbox.acknowledge` | `mailbox`, `message_id`, `receipt_handle` |

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

- Publishing events MUST NOT raise from subscriber failures; the dispatcher records
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
