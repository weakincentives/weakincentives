# Transcript Specification

## Purpose

A **Transcript** is a unified, adapter-agnostic log of everything that happens
during a single `evaluate()` call. It provides a chronological record of
conversation turns, tool invocations, reasoning, and system events in a common
format regardless of which adapter produced it.

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

- Main transcript (`{sessionId}.jsonl`) → `source: "main"`
- Subagent transcripts → `source: "subagent:{id}"`
- `detail.sdk_entry` contains the full parsed JSON entry; `raw` contains the
  raw JSONL line (when `emit_raw=True`).

### Codex App Server

The Codex adapter derives transcript entries from stdio JSON-RPC notifications
received during `_stream_turn()`. Entries are emitted synchronously as each
notification is processed — no file tailing required.

| Codex Notification | Canonical `entry_type` | Notes |
|--------------------|----------------------|-------|
| Input text sent before `turn/start` | `user_message` | The rendered prompt text |
| `item/agentMessage/delta` (accumulated, emitted on `item/completed`) | `assistant_message` | Full accumulated text |
| `item/started` (`commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`) | `tool_use` | Model decided to use a tool |
| `item/completed` (same types) | `tool_result` | Tool finished with result |
| `item/tool/call` (server request for bridged tool) | `tool_use` | WINK bridged tool call request |
| Bridged tool response sent | `tool_result` | WINK bridged tool result |
| `item/reasoning/completed` | `thinking` | Reasoning summary text |
| `thread/tokenUsage/updated` | `token_usage` | Per-turn token counts |
| `turn/completed` with `status: "failed"` | `error` | Turn failure with error info |
| `item/completed` for `contextCompaction` | `system_event` | `detail.subtype = "compaction"` |
| (other `item/*`, `turn/*`) | `unknown` | Forward compatibility |

All Codex notifications → `source: "main"`. `detail.notification` contains
the full notification `params` dict.

## Log Event Taxonomy

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
  "event": "transcript.entry",
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

## TranscriptEmitter

Shared helper at `src/weakincentives/runtime/transcript.py` that both adapters
use to construct and emit entries. Not a public API — internal to the framework.
It assigns monotonically increasing `sequence_number` per source, captures
`timestamp` at emit time, logs at DEBUG level, catches emission errors (never
propagates), and tracks per-source entry counts for summary statistics.

### Claude Adapter Integration

`TranscriptCollector` uses `TranscriptEmitter` internally. The collector owns
file tailing, rotation detection, and subagent discovery; the emitter handles
entry formatting and logging.

### Codex Adapter Integration

`CodexTranscriptBridge` at `src/weakincentives/adapters/codex_app_server/_transcript.py`
wraps `TranscriptEmitter` with Codex-specific mapping logic, called from
`_process_notification()` and `_handle_server_request()`.

## Transcript Types

At `src/weakincentives/runtime/transcript.py`:

- `TranscriptEntry` — frozen dataclass with all envelope fields plus `detail`
  and `raw`
- `TranscriptSummary` — aggregate statistics: `total_entries`,
  `entries_by_type`, `entries_by_source`, `sources`, timestamps
- `reconstruct_transcript(records)` — builds a typed transcript from log records

## Configuration

### Claude Adapter

`TranscriptCollectorConfig` at `src/weakincentives/adapters/claude_agent_sdk/config.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `poll_interval` | `float` | `0.25` | Seconds between file polls |
| `subagent_discovery_interval` | `float` | `1.0` | Seconds between subagent scans |
| `max_read_bytes` | `int` | `65536` | Maximum bytes per read cycle |
| `emit_raw` | `bool` | `True` | Include raw JSONL in `raw` field |

Set `ClaudeAgentSDKClientConfig.transcript_collection = None` to disable.

### Codex Adapter

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transcript` | `bool` | `True` | Emit transcript entries |
| `transcript_emit_raw` | `bool` | `True` | Include raw notification JSON in `raw` field |

## Debug Bundle Integration

Transcript entries are DEBUG logs, so they appear in `logs/app.jsonl`
automatically. Bundles also include a dedicated `transcript.jsonl` extracted
from the log stream during finalization:

```
debug_bundle/
  transcript.jsonl    # transcript.entry records only, ordered
  logs/
    app.jsonl         # all logs (including transcript entries)
```

Query with `wink query`:

```sql
SELECT context->>'entry_type', context->>'source'
FROM transcript
ORDER BY context->>'sequence_number'
```

## Invariants

1. **Envelope completeness**: Every `transcript.entry` contains all required
   envelope keys.

1. **Sequence monotonicity**: Within a `(prompt_name, source)` pair,
   `sequence_number` is strictly increasing with no gaps.

1. **Type vocabulary**: `entry_type` is always one of the canonical types.
   Unrecognized source entries map to `"unknown"`.

1. **Non-blocking emission**: Transcript emission never raises exceptions to
   the evaluation path. Errors are logged and the entry is skipped.

1. **Adapter labeling**: `adapter` field always matches the adapter's
   registered name.

1. **Source consistency**: Claude adapter may produce multiple sources (main +
   subagents). Codex adapter produces `"main"` only.

## Testing

- `TranscriptEmitter`: correct envelope, sequence numbering, error suppression
- `entry_type` mapping from SDK transcript types (Claude)
- `entry_type` mapping from Codex notifications (Codex)
- `reconstruct_transcript()` round-trip from log records to `TranscriptEntry`
- Debug bundle `transcript.jsonl` contains exactly the transcript entries
- Cross-adapter: same logical operation produces same `entry_type` sequence

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` — Claude adapter
- `specs/CODEX_APP_SERVER.md` — Codex adapter
- `specs/LOGGING.md` — Structured logging format
- `specs/DEBUG_BUNDLE.md` — Debug bundle format
- `specs/SESSIONS.md` — Session events
- `specs/ADAPTERS.md` — Provider adapter protocol
