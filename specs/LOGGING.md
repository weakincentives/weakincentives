# Logging Specification

Internal structured logging framework for runtime modules.

**Source:** `src/weakincentives/runtime/logging.py`

## Principles

- **Module isolation**: Each module owns its logger via `get_logger(__name__)`
- **Structured-first**: Use `extra` dict for stable key/value pairs
- **Event taxonomy**: Non-error records carry an `event` key
- **Non-breaking**: New fields extend schemas; breaking changes require coordination

## StructuredLogger

```python
from weakincentives.runtime import get_logger

logger = get_logger(__name__)
logger.info(
    "Tool execution completed",
    event="tool.run",
    context={"prompt_name": prompt.name, "tool": tool.name},
)
```

## Severity Conventions

| Level | Usage |
|-------|-------|
| `DEBUG` | Diagnostic/lifecycle messages, verbose tracing |
| `INFO` | High-level lifecycle events (prompt execution) |
| `WARNING` | Recoverable conditions requiring attention |
| `ERROR` | Caught exceptions converted to fallback paths |
| `CRITICAL` | Process exit or unrecoverable degraded state |

## Required Context Keys

- `event`: Stable event name categorizing the log entry
- `prompt_name`: Name of prompt being evaluated (when applicable)
- `adapter`: Adapter identifier (for provider events)
- `tool`: Tool identifier (for tool invocation outcomes)

## Core Event Names

| Module | Event | Level |
|--------|-------|-------|
| `runtime/events` | `event_delivery_failed` | exception |
| `runtime/session` | `session_reducer_failed` | exception |
| `adapters/shared` | `prompt_execution_started` | info |
| `adapters/shared` | `prompt_execution_succeeded` | info |
| `adapters/shared` | `tool_handler_completed` | info |
| `adapters/shared` | `prompt_throttled` | warning |

## Adapter DEBUG Events

Enable via `WEAKINCENTIVES_LOG_LEVEL=DEBUG`.

| Adapter | Key Events |
|---------|------------|
| Claude Agent SDK | `adapter.init`, `evaluate.entry`, `sdk_query.*`, `hook.*` |
| OpenAI | `adapter.init`, `evaluate.entry`, `provider.request`, `provider.response` |
| LiteLLM | `adapter.init`, `evaluate.entry`, `provider.request`, `throttle.*` |

## Error Handling

- Publishing events: Exceptions logged, publish continues, failures in `DispatchResult`
- Reducers: Exceptions logged and skipped, previous state preserved
- Tool handlers: Exceptions logged, converted to `ToolResult(success=False)`

## Scope

Covers runtime modules (events, sessions, adapters, tools). Excludes build/CI scripts, tests, and external consumers.
