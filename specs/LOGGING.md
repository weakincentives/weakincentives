# Logging Specification

## Purpose

Structured logging with stable event names and context payloads for observability.
Internal framework shared by runtime modules.

**Implementation:** `src/weakincentives/runtime/logging.py`

## Design Principles

- **Module isolation**: Each module owns its logger instance
- **Structured-first**: Prefer key/value pairs over formatted messages
- **Event taxonomy**: Every non-error record carries `event` key
- **Non-breaking evolvability**: New fields extend existing schemas

## Log Record Structure

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "level": "INFO",
  "logger": "weakincentives.adapters.openai",
  "event": "tool.execution.complete",
  "message": "Tool execution completed",
  "context": {"tool_name": "read_file", "success": true}
}
```

## Severity Conventions

| Level | Use Case |
|-------|----------|
| DEBUG | Diagnostic, lifecycle messages |
| INFO | High-level lifecycle events |
| WARNING | Recoverable conditions |
| ERROR | Caught exceptions with fallback |
| CRITICAL | Process exit or degraded state |

## Required Context Keys

| Key | When |
|-----|------|
| `event` | All structured DEBUG/INFO events |
| `prompt_name` | Prompt lifecycle events |
| `adapter` | Adapter events |
| `tool` | Tool invocation events |

## Event Categories

### Adapter Events

| Event | Level | Context |
|-------|-------|---------|
| `adapter.init` | INFO | model, config |
| `evaluate.entry` | DEBUG | prompt_name |
| `provider.request` | DEBUG | model, tool_count |
| `provider.error` | ERROR | error_type, status_code |

### Session Events

| Event | Level | Context |
|-------|-------|---------|
| `session.dispatch` | DEBUG | event_type |
| `session.reducer_applied` | DEBUG | reducer, operation |
| `session.restore` | INFO | snapshot_slice_count |

### Tool Events

| Event | Level | Context |
|-------|-------|---------|
| `tool.execution.start` | DEBUG | tool_name, arguments |
| `tool.execution.complete` | DEBUG | tool_name, success |
| `tool_handler_exception` | ERROR | provider_payload |

### Transcript Events

| Event | Level | Context |
|-------|-------|---------|
| `transcript.entry` | DEBUG | prompt_name, sequence_number, entry_type |
| `transcript.start` | DEBUG | prompt_name, source |
| `transcript.stop` | DEBUG | prompt_name, entry_count |
| `transcript.error` | WARNING | prompt_name, error |
| `transcript.path_discovered` | DEBUG | prompt_name, path |
| `transcript.subagent_discovered` | DEBUG | prompt_name, subagent_id |

### Resource Events

| Event | Level | Context |
|-------|-------|---------|
| `resource.construct.start` | DEBUG | protocol, scope |
| `resource.close` | DEBUG | protocol |

## Configuration

```bash
WEAKINCENTIVES_LOG_LEVEL=DEBUG
WEAKINCENTIVES_LOG_FORMAT=json
```

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Debug bundle specification
