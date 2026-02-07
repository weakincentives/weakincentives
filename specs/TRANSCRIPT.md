# Transcript Specification

## Purpose

A **Transcript** is a unified, adapter-agnostic log of everything that happens
during a single `evaluate()` call. It provides a chronological record of
conversation turns, tool invocations, reasoning, and system events in a common
format regardless of which adapter (Claude Agent SDK, Codex App Server, or
future adapters) produced it.

The Claude adapter already emits transcript entries by tailing the SDK's JSONL
files. This spec extends the same pattern to the Codex adapter (which produces
entries from stdio notifications) and defines the shared schema that both
converge on.

**Implementation:**

- Transcript types: `src/weakincentives/runtime/transcript.py` (new)
- Claude collector: `src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py`
- Codex emitter: `src/weakincentives/adapters/codex_app_server/_transcript.py` (new)
- Debug bundle: `src/weakincentives/debug/bundle.py` (existing `logs/app.jsonl`)

## Principles

- **One schema, many sources.** Transcript entries share a common envelope.
  Adapters populate them from whatever source they have (file tailing, stdio
  notifications, in-memory events).
- **Logs are the transcript.** Entries are emitted as DEBUG-level structured
  log records with event name `transcript.entry`. Existing log infrastructure
  (debug bundles, log viewers, `wink query`) consumes them with zero changes.
- **Adapter-specific detail preserved.** The common envelope carries an opaque
  `detail` dict for adapter-specific fields. Consumers that know the adapter
  can inspect it; generic consumers ignore it.
- **Chronologically ordered.** Entries are emitted in the order they occur.
  Sequence numbers are per-source (main vs subagent), monotonically increasing.
- **Non-blocking.** Transcript emission never fails the evaluation. Errors are
  logged at WARNING level and skipped.

## Transcript Entry Schema

Every transcript entry is emitted as a structured log record with
`event="transcript.entry"` and the following context keys:

### Common Envelope

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `prompt_name` | `str` | Yes | Prompt being evaluated |
| `adapter` | `str` | Yes | Adapter name (`"claude_agent_sdk"` or `"codex_app_server"`) |
| `entry_type` | `str` | Yes | Canonical entry type (see table below) |
| `sequence_number` | `int` | Yes | Monotonically increasing per source |
| `source` | `str` | Yes | Entry source: `"main"`, `"subagent:{id}"` |
| `timestamp` | `str` | Yes | ISO-8601 UTC when entry was captured |
| `session_id` | `str` | No | Session UUID if available |
| `detail` | `dict` | No | Adapter-specific payload (opaque to generic consumers) |
| `raw` | `str` | No | Raw source data (JSONL line or JSON notification) |

### Canonical Entry Types

These entry types form the common vocabulary. Both adapters map their
native events to these types.

| Entry Type | Description | Emitted By |
|------------|-------------|------------|
| `user_message` | User/system prompt delivered to the model | Both |
| `assistant_message` | Model response text (complete or accumulated) | Both |
| `tool_use` | Model requests a tool call | Both |
| `tool_result` | Tool execution result returned to model | Both |
| `thinking` | Extended thinking / reasoning block | Both |
| `system_event` | Lifecycle events (compaction, context window, etc.) | Both |
| `token_usage` | Token consumption update | Both |
| `error` | Error during evaluation | Both |
| `unknown` | Unrecognized entry (preserved for forward compatibility) | Both |

## Adapter Mapping

### Claude Agent SDK

The Claude adapter derives transcript entries from the SDK's JSONL transcript
files via `TranscriptCollector`. The collector already emits
`transcript.collector.entry` logs. Under this spec, those entries are
**re-keyed** to the unified `transcript.entry` event and enriched with the
common envelope.

| SDK Transcript `type` | Canonical `entry_type` | Notes |
|-----------------------|----------------------|-------|
| `user` | `user_message` | `message.content` |
| `assistant` | `assistant_message` | `message.content`, may include tool_use blocks |
| `tool_result` | `tool_result` | `tool_use_id`, `content` |
| `thinking` | `thinking` | `thinking` text |
| `summary` | `system_event` | Compaction summary, `detail.subtype = "compaction"` |
| `system` | `system_event` | SDK-level events |
| (other) | `unknown` | Forward compatibility |

**Source mapping:**

- Main transcript (`{sessionId}.jsonl`) -> `source: "main"`
- Subagent transcripts (`{sessionId}/subagents/agent-{id}.jsonl`) -> `source: "subagent:{id}"`

**Detail payload** includes the full parsed JSON entry from the JSONL line,
keyed under `detail.sdk_entry`. The raw JSONL line is in `raw` (when
`emit_raw_json=True`).

### Codex App Server

The Codex adapter derives transcript entries from stdio JSON-RPC notifications
received during `_stream_turn()`. Entries are emitted synchronously as each
notification is processed—no file tailing required.

| Codex Notification | Canonical `entry_type` | Notes |
|--------------------|----------------------|-------|
| `turn/start` (input text sent) | `user_message` | The rendered prompt text |
| `item/agentMessage/delta` | `assistant_message` | Accumulated deltas; emit on `item/completed` for `agentMessage` |
| `item/started` (`commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`) | `tool_use` | Model decided to use a tool |
| `item/completed` (`commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`) | `tool_result` | Tool finished with result |
| `item/tool/call` (server request for dynamic tool) | `tool_use` | WINK bridged tool call request |
| Dynamic tool response sent | `tool_result` | WINK bridged tool result |
| `item/reasoning/delta` or `item/reasoning/completed` | `thinking` | Reasoning summary text |
| `thread/tokenUsage/updated` | `token_usage` | Per-turn token counts |
| `turn/completed` with `status: "failed"` | `error` | Turn failure with error info |
| `item/completed` for `contextCompaction` | `system_event` | `detail.subtype = "compaction"` |
| (other `item/*`, `turn/*`) | `unknown` | Forward compatibility |

**Source mapping:**

- All Codex notifications -> `source: "main"` (Codex does not expose subagent
  boundaries through the app-server protocol)

**Detail payload** includes the full notification `params` dict, keyed under
`detail.notification`. The raw JSON notification is in `raw` (when configured).

### Entry Emission Points

The Codex adapter emits entries at these points in `_stream_turn()`:

1. **Before `turn/start`:** Emit `user_message` with the rendered prompt text.
2. **On `item/started`:** Emit `tool_use` with tool name and parameters.
3. **On `item/completed` for tool items:** Emit `tool_result` with outcome.
4. **On `item/completed` for `agentMessage`:** Emit `assistant_message` with
   accumulated text.
5. **On `item/tool/call`:** Emit `tool_use` for WINK bridged tool.
6. **After bridged tool execution:** Emit `tool_result` for WINK bridged tool.
7. **On `item/reasoning/completed`:** Emit `thinking` with summary text.
8. **On `thread/tokenUsage/updated`:** Emit `token_usage`.
9. **On `turn/completed` with failure:** Emit `error`.

## Log Event Taxonomy

### Unified Events (Both Adapters)

| Event | Level | Description |
|-------|-------|-------------|
| `transcript.entry` | DEBUG | A single transcript entry (common envelope) |
| `transcript.start` | DEBUG | Transcript collection/emission started |
| `transcript.stop` | DEBUG | Transcript collection/emission stopped with summary |
| `transcript.error` | WARNING | Non-fatal transcript error (file I/O, parse failure) |

### Claude-Specific Events (Retained)

These existing events remain as sub-events of the collection mechanism:

| Event | Level | Description |
|-------|-------|-------------|
| `transcript.collector.path_discovered` | DEBUG | Main transcript path found via hook |
| `transcript.collector.subagent_discovered` | DEBUG | Subagent transcript found |

### Migration from `transcript.collector.entry`

The current `transcript.collector.entry` event is **replaced** by
`transcript.entry`. The context schema changes from:

```json
{
  "event": "transcript.collector.entry",
  "context": {
    "prompt_name": "...",
    "transcript_source": "main",
    "entry_type": "assistant",
    "sequence_number": 5,
    "raw_json": "...",
    "parsed": { "type": "assistant", "message": { ... } }
  }
}
```

To:

```json
{
  "event": "transcript.entry",
  "context": {
    "prompt_name": "code-review",
    "adapter": "claude_agent_sdk",
    "entry_type": "assistant_message",
    "sequence_number": 5,
    "source": "main",
    "timestamp": "2024-01-15T10:30:00.123+00:00",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "detail": {
      "sdk_entry": { "type": "assistant", "message": { "role": "assistant", "content": "..." } }
    },
    "raw": "{\"type\": \"assistant\", ...}"
  }
}
```

## Example Log Output

### Claude Adapter — Assistant Message

```json
{
  "timestamp": "2024-01-15T10:30:00.123+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.adapters.claude_agent_sdk._transcript_collector",
  "event": "transcript.entry",
  "message": "transcript entry: assistant_message",
  "context": {
    "prompt_name": "code-review",
    "adapter": "claude_agent_sdk",
    "entry_type": "assistant_message",
    "sequence_number": 3,
    "source": "main",
    "timestamp": "2024-01-15T10:30:00.123+00:00",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "detail": {
      "sdk_entry": {
        "type": "assistant",
        "message": { "role": "assistant", "content": "I'll analyze the code..." }
      }
    }
  }
}
```

### Codex Adapter — Tool Result

```json
{
  "timestamp": "2024-01-15T10:30:01.456+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.adapters.codex_app_server._transcript",
  "event": "transcript.entry",
  "message": "transcript entry: tool_result",
  "context": {
    "prompt_name": "code-review",
    "adapter": "codex_app_server",
    "entry_type": "tool_result",
    "sequence_number": 7,
    "source": "main",
    "timestamp": "2024-01-15T10:30:01.456+00:00",
    "session_id": "660e8400-e29b-41d4-a716-446655440001",
    "detail": {
      "notification": {
        "type": "commandExecution",
        "status": "completed",
        "command": "ls -la",
        "aggregatedOutput": "total 42\ndrwxr-xr-x ..."
      }
    }
  }
}
```

### Codex Adapter — Thinking

```json
{
  "timestamp": "2024-01-15T10:30:00.789+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.adapters.codex_app_server._transcript",
  "event": "transcript.entry",
  "message": "transcript entry: thinking",
  "context": {
    "prompt_name": "code-review",
    "adapter": "codex_app_server",
    "entry_type": "thinking",
    "sequence_number": 2,
    "source": "main",
    "timestamp": "2024-01-15T10:30:00.789+00:00",
    "detail": {
      "notification": {
        "summary": "Analyzing repository structure and identifying key files..."
      }
    }
  }
}
```

## TranscriptEmitter Protocol

Both adapters use an internal emitter that encapsulates entry construction and
logging. This is not a public API — it is a shared helper to enforce the
envelope schema.

```python
class TranscriptEmitter:
    """Emit transcript entries as structured DEBUG logs."""

    def __init__(
        self,
        *,
        prompt_name: str,
        adapter: str,
        session_id: str | None = None,
        logger: StructuredLogger,
        emit_raw: bool = True,
    ) -> None: ...

    def emit(
        self,
        entry_type: str,
        *,
        source: str = "main",
        detail: dict[str, Any] | None = None,
        raw: str | None = None,
    ) -> None:
        """Emit a transcript entry. Thread-safe, never raises."""
        ...
```

The emitter:

- Assigns monotonically increasing `sequence_number` per source
- Captures `timestamp` at emit time (UTC)
- Logs at DEBUG level with `event="transcript.entry"`
- Catches and logs any emission errors (never propagates)

### Claude Adapter Integration

`TranscriptCollector._emit_entry()` is updated to:

1. Construct the canonical `entry_type` from the SDK entry's `type` field
2. Call `TranscriptEmitter.emit()` with the mapped type and SDK-specific detail
3. The collector continues to handle file tailing, rotation, and subagent
   discovery as before

### Codex Adapter Integration

A new `CodexTranscriptEmitter` wraps `TranscriptEmitter` and is called from
`_process_notification()` and `_handle_server_request()`:

1. `_process_notification()` calls `emitter.on_notification(method, params)`
2. `_handle_tool_call()` calls `emitter.on_tool_call(params)` before execution
   and `emitter.on_tool_result(params, result)` after
3. Before `turn/start`, the adapter calls `emitter.on_user_message(text)`

## Configuration

### Claude Adapter

No new configuration. `TranscriptCollectorConfig` already controls raw JSON
emission and entry parsing. The unified entry format is always enabled when
transcript collection is active.

### Codex Adapter

A new optional field on `CodexAppServerClientConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transcript_emission` | `bool` | `True` | Emit transcript entries as DEBUG logs |
| `transcript_emit_raw` | `bool` | `True` | Include raw notification JSON in entries |

Transcript emission is **enabled by default** to match Claude adapter behavior.
Set `transcript_emission=False` to disable.

## Debug Bundle Integration

Debug bundles already capture all structured logs in `logs/app.jsonl`. Since
transcript entries are emitted as DEBUG logs, they are automatically captured
in bundles with zero changes to the bundle writer.

Consumers can filter transcript entries from the log stream:

```python
# Extract transcript from debug bundle logs
transcript = [
    entry for entry in bundle.logs
    if entry.get("event") == "transcript.entry"
]
```

The `wink query` CLI can also filter:

```sql
SELECT context->>'entry_type', context->>'source', context->>'detail'
FROM logs
WHERE event = 'transcript.entry'
ORDER BY context->>'sequence_number'
```

## Transcript Reconstruction

A transcript can be reconstructed from log records for analysis or replay:

```python
from weakincentives.runtime.transcript import reconstruct_transcript

# From debug bundle
transcript = reconstruct_transcript(bundle.logs)

# Returns list of TranscriptEntry dataclasses, ordered by (source, sequence_number)
for entry in transcript:
    print(f"[{entry.source}:{entry.sequence_number}] {entry.entry_type}: ...")
```

### TranscriptEntry

```python
@dataclass(slots=True, frozen=True)
class TranscriptEntry:
    """A single entry in a reconstructed transcript."""

    prompt_name: str
    adapter: str
    entry_type: str
    sequence_number: int
    source: str
    timestamp: datetime
    session_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None
```

### TranscriptSummary

```python
@dataclass(slots=True, frozen=True)
class TranscriptSummary:
    """Summary statistics for a transcript."""

    total_entries: int
    entries_by_type: dict[str, int]
    entries_by_source: dict[str, int]
    sources: tuple[str, ...]
    adapter: str
    prompt_name: str
    first_timestamp: datetime | None
    last_timestamp: datetime | None
```

## Invariants

1. **Envelope completeness**: Every `transcript.entry` log record contains all
   required envelope keys (`prompt_name`, `adapter`, `entry_type`,
   `sequence_number`, `source`, `timestamp`).

2. **Sequence monotonicity**: Within a `(prompt_name, source)` pair,
   `sequence_number` is strictly increasing with no gaps.

3. **Type vocabulary**: `entry_type` is always one of the canonical types
   listed above. Unknown source entries map to `"unknown"`.

4. **Non-blocking emission**: Transcript emission never raises exceptions to
   the adapter's evaluation path. Errors are logged and the entry is skipped.

5. **Adapter labeling**: `adapter` field always matches the adapter's
   registered name.

6. **Source consistency**: Claude adapter may produce multiple sources (main +
   subagents). Codex adapter produces `"main"` only.

7. **Timestamp ordering**: Within a source, entries are emitted in
   chronological order. Across sources, timestamps reflect wall-clock capture
   time (not causal ordering).

## Testing

### Unit Tests

- Verify `TranscriptEmitter` produces correct envelope for each entry type
- Verify sequence numbers increment per source
- Verify `entry_type` mapping from SDK transcript types (Claude)
- Verify `entry_type` mapping from Codex notifications (Codex)
- Verify `raw` field presence controlled by configuration
- Verify emission errors are caught and logged (never raised)

### Integration Tests

- Claude adapter: verify `transcript.entry` events appear in captured logs
  with correct envelope during `evaluate()`
- Codex adapter: verify `transcript.entry` events appear in captured logs
  with correct envelope during `evaluate()`
- Debug bundle: verify transcript entries are present in `logs/app.jsonl`
- `reconstruct_transcript()`: verify round-trip from log records

### Cross-Adapter Consistency Tests

- Given the same logical operation (e.g., "model calls a bash tool"), verify
  both adapters produce entries with the same `entry_type` sequence:
  `tool_use` followed by `tool_result`
- Verify `TranscriptSummary` produces comparable statistics for equivalent
  evaluation runs across adapters

## Migration

### Phase 1: Types and Emitter

1. Add `TranscriptEntry`, `TranscriptSummary`, `TranscriptEmitter` to
   `src/weakincentives/runtime/transcript.py`
2. Add `reconstruct_transcript()` utility

### Phase 2: Claude Adapter

1. Update `TranscriptCollector._emit_entry()` to use `TranscriptEmitter`
2. Map SDK entry types to canonical types
3. Change event name from `transcript.collector.entry` to `transcript.entry`
4. Retain `transcript.collector.*` events for collector lifecycle

### Phase 3: Codex Adapter

1. Add `CodexTranscriptEmitter` to
   `src/weakincentives/adapters/codex_app_server/_transcript.py`
2. Wire into `_process_notification()` and `_handle_server_request()`
3. Add `transcript_emission` config fields to `CodexAppServerClientConfig`
4. Emit `transcript.start` / `transcript.stop` around turn streaming

### Phase 4: Debug Bundle Enhancement (Optional)

1. Add a dedicated `transcript/` section to debug bundles (extracted from
   `logs/app.jsonl` for convenience)
2. Add transcript panel to `wink debug` web UI

## Related Specifications

- `specs/TRANSCRIPT_COLLECTION.md` — Claude-specific transcript collection (superseded by this spec for schema; collection mechanism retained)
- `specs/CLAUDE_AGENT_SDK.md` — Claude adapter
- `specs/CODEX_APP_SERVER.md` — Codex adapter
- `specs/LOGGING.md` — Structured logging format
- `specs/DEBUG_BUNDLE.md` — Debug bundle captures transcript entries in `logs/app.jsonl`
- `specs/SESSIONS.md` — Session events (PromptRendered, ToolInvoked, PromptExecuted)
- `specs/ADAPTERS.md` — Provider adapter protocol
