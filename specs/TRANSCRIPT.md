# Transcript Specification

## Purpose

A **Transcript** is a unified, adapter-agnostic log of everything that happens
during a single `evaluate()` call. It provides a chronological record of
conversation turns, tool invocations, reasoning, and system events in a common
format regardless of which adapter produced it.

This spec is the single source of truth for transcript format and emission.
It **replaces** `specs/TRANSCRIPT_COLLECTION.md`, which is deleted.

**Implementation:**

- Transcript types and emitter: `src/weakincentives/runtime/transcript.py`
- Claude collector: `src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py`
- Codex transcript: `src/weakincentives/adapters/codex_app_server/_transcript.py`
- Debug bundle integration: `src/weakincentives/debug/bundle.py`

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
| `adapter` | `str` | Yes | Adapter name (`"claude_agent_sdk"`, `"codex_app_server"`) |
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
| `unknown` | Unrecognized entry (forward compatibility) | Both |

## Adapter Mapping

### Claude Agent SDK

The Claude adapter derives transcript entries from the SDK's JSONL transcript
files via `TranscriptCollector`. The collector tails `.jsonl` files, parses
each line, maps the SDK `type` to a canonical `entry_type`, and emits through
`TranscriptEmitter`.

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
`emit_raw=True`).

### Codex App Server

The Codex adapter derives transcript entries from stdio JSON-RPC notifications
received during `_stream_turn()`. Entries are emitted synchronously as each
notification is processed — no file tailing required.

| Codex Notification | Canonical `entry_type` | Notes |
|--------------------|----------------------|-------|
| Input text sent before `turn/start` | `user_message` | The rendered prompt text |
| `item/agentMessage/delta` (accumulated, emitted on `item/completed`) | `assistant_message` | Full accumulated text |
| `item/started` (`commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`) | `tool_use` | Model decided to use a tool |
| `item/completed` (`commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`) | `tool_result` | Tool finished with result |
| `item/tool/call` (server request for bridged tool) | `tool_use` | WINK bridged tool call request |
| Bridged tool response sent | `tool_result` | WINK bridged tool result |
| `item/reasoning/completed` | `thinking` | Reasoning summary text |
| `thread/tokenUsage/updated` | `token_usage` | Per-turn token counts |
| `turn/completed` with `status: "failed"` | `error` | Turn failure with error info |
| `item/completed` for `contextCompaction` | `system_event` | `detail.subtype = "compaction"` |
| (other `item/*`, `turn/*`) | `unknown` | Forward compatibility |

**Source mapping:**

- All Codex notifications -> `source: "main"` (Codex does not expose subagent
  boundaries through the app-server protocol)

**Detail payload** includes the full notification `params` dict, keyed under
`detail.notification`. The raw JSON notification is in `raw` (when
`emit_raw=True`).

### Entry Emission Points (Codex)

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

All transcript-related log events use the `transcript.*` prefix.

| Event | Level | Description |
|-------|-------|-------------|
| `transcript.entry` | DEBUG | A single transcript entry (common envelope) |
| `transcript.start` | DEBUG | Transcript emission started for an evaluation |
| `transcript.stop` | DEBUG | Transcript emission stopped with summary statistics |
| `transcript.error` | WARNING | Non-fatal transcript error (file I/O, parse failure) |
| `transcript.path_discovered` | DEBUG | Claude: main transcript path found via hook |
| `transcript.subagent_discovered` | DEBUG | Claude: subagent transcript found |

### Example: `transcript.entry`

```json
{
  "timestamp": "2024-01-15T10:30:00.123+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.runtime.transcript",
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

### Example: `transcript.entry` (Codex)

```json
{
  "timestamp": "2024-01-15T10:30:01.456+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.runtime.transcript",
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

## TranscriptEmitter

Shared helper that both adapters use to construct and emit entries. Lives in
the runtime layer so adapters don't duplicate envelope logic. Not a public
API — internal to the framework.

```python
class TranscriptEmitter:
    """Emit transcript entries as structured DEBUG logs."""

    def __init__(
        self,
        *,
        prompt_name: str,
        adapter: str,
        session_id: str | None = None,
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

    def start(self) -> None:
        """Emit transcript.start event."""
        ...

    def stop(self) -> None:
        """Emit transcript.stop event with summary statistics."""
        ...
```

The emitter:

- Assigns monotonically increasing `sequence_number` per source
- Captures `timestamp` at emit time (UTC)
- Logs at DEBUG level with `event="transcript.entry"`
- Catches and logs any emission errors (never propagates)
- Tracks per-source entry counts for summary statistics

### Claude Adapter Integration

`TranscriptCollector` uses `TranscriptEmitter` internally:

1. On entry parsed from JSONL, maps SDK `type` to canonical `entry_type`
2. Calls `emitter.emit()` with the mapped type and `detail.sdk_entry`
3. Collector still owns file tailing, rotation detection, and subagent
   discovery — emitter only handles entry formatting and logging

### Codex Adapter Integration

A `CodexTranscriptBridge` wraps `TranscriptEmitter` with Codex-specific
mapping logic. Called from `_process_notification()` and
`_handle_server_request()`:

1. `bridge.on_notification(method, params)` — maps notification to entry type,
   emits via `TranscriptEmitter`
2. `bridge.on_tool_call(params)` — emits `tool_use` before bridged tool
   execution
3. `bridge.on_tool_result(params, result)` — emits `tool_result` after bridged
   tool execution
4. `bridge.on_user_message(text)` — emits `user_message` before turn start

## Configuration

### Claude Adapter

`TranscriptCollectorConfig` controls transcript behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `poll_interval` | `float` | `0.25` | Seconds between file polls |
| `subagent_discovery_interval` | `float` | `1.0` | Seconds between subagent scans |
| `max_read_bytes` | `int` | `65536` | Maximum bytes per read cycle |
| `emit_raw` | `bool` | `True` | Include raw JSONL in `raw` field |

Set `ClaudeAgentSDKClientConfig.transcript_collection = None` to disable.

### Codex Adapter

`CodexAppServerClientConfig` gains transcript fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transcript` | `bool` | `True` | Emit transcript entries |
| `transcript_emit_raw` | `bool` | `True` | Include raw notification JSON in `raw` field |

Set `transcript=False` to disable.

## Debug Bundle Integration

Debug bundles capture all structured logs in `logs/app.jsonl`. Transcript
entries are DEBUG logs, so they appear there automatically.

Additionally, bundles gain a `transcript.jsonl` file that contains only
transcript entries, extracted from the log stream for direct access:

```
debug_bundle/
  ...
  transcript.jsonl    # transcript.entry records only, ordered
  logs/
    app.jsonl         # all logs (including transcript entries)
```

`BundleWriter` extracts entries where `event == "transcript.entry"` from the
captured logs and writes them to `transcript.jsonl` during finalization.

### Querying

```python
# From debug bundle
for entry in bundle.transcript:
    print(f"[{entry['context']['source']}:{entry['context']['sequence_number']}] "
          f"{entry['context']['entry_type']}")
```

```sql
SELECT context->>'entry_type', context->>'source', context->>'detail'
FROM transcript
ORDER BY context->>'sequence_number'
```

## Transcript Reconstruction

Reconstruct a typed transcript from log records:

```python
from weakincentives.runtime.transcript import reconstruct_transcript

transcript = reconstruct_transcript(bundle.transcript)

for entry in transcript:
    print(f"[{entry.source}:{entry.sequence_number}] {entry.entry_type}")
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
    detail: Mapping[str, Any] = field(default_factory=dict)
    raw: str | None = None
```

### TranscriptSummary

```python
@dataclass(slots=True, frozen=True)
class TranscriptSummary:
    """Summary statistics for a transcript."""

    total_entries: int
    entries_by_type: Mapping[str, int]
    entries_by_source: Mapping[str, int]
    sources: tuple[str, ...]
    adapter: str
    prompt_name: str
    first_timestamp: datetime | None
    last_timestamp: datetime | None
```

## What Gets Deleted

This is a clean break. The following are removed entirely:

| Removed | Replacement |
|---------|-------------|
| `specs/TRANSCRIPT_COLLECTION.md` | This spec |
| `transcript.collector.entry` event name | `transcript.entry` |
| `transcript.collector.start` event name | `transcript.start` |
| `transcript.collector.stop` event name | `transcript.stop` |
| `transcript.collector.error` event name | `transcript.error` |
| `transcript.collector.path_discovered` event name | `transcript.path_discovered` |
| `transcript.collector.subagent_discovered` event name | `transcript.subagent_discovered` |
| `TranscriptCollectorConfig.emit_raw_json` field | `emit_raw` (renamed) |
| `TranscriptCollectorConfig.parse_entries` field | Removed — entries are always parsed |
| Context key `transcript_source` | `source` |
| Context key `raw_json` | `raw` |
| Context key `parsed` | `detail.sdk_entry` |
| Log context without `adapter` or `timestamp` | Always present in new envelope |

No shims, no aliases, no deprecation warnings. Old event names and context
keys stop existing.

## What Gets Added

| Added | Where |
|-------|-------|
| `TranscriptEmitter` class | `src/weakincentives/runtime/transcript.py` |
| `TranscriptEntry` dataclass | `src/weakincentives/runtime/transcript.py` |
| `TranscriptSummary` dataclass | `src/weakincentives/runtime/transcript.py` |
| `reconstruct_transcript()` function | `src/weakincentives/runtime/transcript.py` |
| `CodexTranscriptBridge` class | `src/weakincentives/adapters/codex_app_server/_transcript.py` |
| `transcript` config field | `CodexAppServerClientConfig` |
| `transcript_emit_raw` config field | `CodexAppServerClientConfig` |
| `transcript.jsonl` bundle artifact | `src/weakincentives/debug/bundle.py` |
| `bundle.transcript` property | `DebugBundle` |

## What Gets Modified

| Modified | Change |
|----------|--------|
| `TranscriptCollector._emit_entry()` | Uses `TranscriptEmitter` instead of direct logging |
| `TranscriptCollector._emit_entries()` | Maps SDK types to canonical types |
| `TranscriptCollectorConfig` | `emit_raw_json` -> `emit_raw`, `parse_entries` removed |
| All `transcript.collector.*` log events in collector | Renamed to `transcript.*` |
| Codex `_process_notification()` | Calls `CodexTranscriptBridge` for each notification |
| Codex `_handle_tool_call()` | Calls bridge before/after tool execution |
| `BundleWriter` finalization | Extracts `transcript.jsonl` from captured logs |
| `DebugBundle` | Adds `transcript` property |

## Invariants

1. **Envelope completeness**: Every `transcript.entry` log record contains all
   required envelope keys (`prompt_name`, `adapter`, `entry_type`,
   `sequence_number`, `source`, `timestamp`).

2. **Sequence monotonicity**: Within a `(prompt_name, source)` pair,
   `sequence_number` is strictly increasing with no gaps.

3. **Type vocabulary**: `entry_type` is always one of the canonical types
   listed above. Unrecognized source entries map to `"unknown"`.

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

- `TranscriptEmitter`: correct envelope, sequence numbering, error suppression
- `entry_type` mapping from SDK transcript types (Claude)
- `entry_type` mapping from Codex notifications (Codex)
- `raw` field presence controlled by `emit_raw` configuration
- `reconstruct_transcript()` round-trip from log records to `TranscriptEntry`
- Claude adapter `evaluate()` produces `transcript.entry` logs with correct
  envelope
- Codex adapter `evaluate()` produces `transcript.entry` logs with correct
  envelope
- Debug bundle `transcript.jsonl` contains exactly the transcript entries
- Cross-adapter: same logical operation (tool call) produces same `entry_type`
  sequence (`tool_use` then `tool_result`) from both adapters

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` — Claude adapter
- `specs/CODEX_APP_SERVER.md` — Codex adapter
- `specs/LOGGING.md` — Structured logging format
- `specs/DEBUG_BUNDLE.md` — Debug bundle format
- `specs/SESSIONS.md` — Session events (PromptRendered, ToolInvoked, PromptExecuted)
- `specs/ADAPTERS.md` — Provider adapter protocol
