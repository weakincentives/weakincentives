# Transcript Collection Specification

## Purpose

Real-time collection and logging of Claude Agent SDK transcripts from the main
session and all sub-agent sessions. Replaces the generic log aggregator with a
focused system that captures structured transcript entries as DEBUG-level logs
in WINK.

**Implementation:** `src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py`

## Background

The Claude Agent SDK writes transcripts to JSONL files on disk:

- **Main transcript**: `~/.claude/projects/{project}/{sessionId}.jsonl`
- **Sub-agent transcripts**: `~/.claude/projects/{project}/{sessionId}/subagents/agent-{agentId}.jsonl`

These transcripts contain the canonical record of all conversation turns,
tool invocations, thinking blocks, and system events. The SDK exposes the
`transcript_path` via hook input data.

## Principles

- **Transcript-focused**: Capture only transcript files, not generic logs
- **Hook-driven discovery**: Use `transcript_path` from SDK hooks, not polling
- **Sub-agent aware**: Automatically discover and tail sub-agent transcripts
- **Structured emission**: Parse JSONL entries and emit with typed context
- **Real-time streaming**: Tail files as they grow, emit lines immediately
- **Non-blocking**: File I/O in executor, never blocks the event loop
- **Graceful degradation**: File access errors logged, never fail the query

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ClaudeAgentSDKAdapter                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  TranscriptCollector                        │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌──────────────────────────────────┐   │ │
│  │  │ Hook CB     │───>│ _remember_transcript_path()      │   │ │
│  │  │ (UserPrompt │    │ • Extracts transcript_path       │   │ │
│  │  │  Submit,    │    │ • Derives session_dir            │   │ │
│  │  │  PreToolUse,│    │ • Starts tailer if not running   │   │ │
│  │  │  etc.)      │    └──────────────────────────────────┘   │ │
│  │  └─────────────┘                                            │ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │               Background Tailer Task                  │  │ │
│  │  │                                                       │  │ │
│  │  │  ┌─────────────────┐  ┌───────────────────────────┐  │  │ │
│  │  │  │ Main Transcript │  │ Sub-agent Transcripts     │  │  │ │
│  │  │  │ Tailer          │  │ Discovery + Tailers       │  │  │ │
│  │  │  │                 │  │                           │  │  │ │
│  │  │  │ {session}.jsonl │  │ {session}/subagents/      │  │  │ │
│  │  │  │                 │  │   agent-{id}.jsonl        │  │  │ │
│  │  │  └────────┬────────┘  └────────────┬──────────────┘  │  │ │
│  │  │           │                        │                  │  │ │
│  │  │           └───────────┬────────────┘                  │  │ │
│  │  │                       ▼                               │  │ │
│  │  │              ┌─────────────────┐                      │  │ │
│  │  │              │ _emit_entry()   │                      │  │ │
│  │  │              │ DEBUG log       │                      │  │ │
│  │  │              └─────────────────┘                      │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### TranscriptCollectorConfig

See `src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py` for
the full dataclass definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `poll_interval` | `float` | `0.25` | Seconds between file polls |
| `subagent_discovery_interval` | `float` | `1.0` | Seconds between subagent scans |
| `max_read_bytes` | `int` | `65536` | Maximum bytes per read cycle |
| `emit_raw_json` | `bool` | `True` | Include raw JSON in log context |
| `parse_entries` | `bool` | `True` | Parse and type transcript entries |

### Enabled by Default

Transcript collection is **enabled by default** in `ClaudeAgentSDKClientConfig`.
The `transcript_collection` field defaults to `TranscriptCollectorConfig()` (with
default settings). Set to `None` to disable collection entirely.

See `src/weakincentives/adapters/claude_agent_sdk/config.py` for the client config.

## Transcript Entry Types

Claude Agent SDK transcripts contain JSONL entries with a `type` field:

| Type | Description | Key Fields |
|------|-------------|------------|
| `user` | User message | `message.content` |
| `assistant` | Assistant response | `message.content`, `message.tool_use` |
| `tool_result` | Tool execution result | `tool_use_id`, `content` |
| `system` | System event | `event`, `details` |
| `thinking` | Extended thinking block | `thinking` |
| `summary` | Compaction summary | `summary`, `dropped_count` |

## API

### TranscriptCollector

See `src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py` for the
full class definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_name` | `str` | (required) | Name of the prompt being evaluated (for log context) |
| `config` | `TranscriptCollectorConfig` | `TranscriptCollectorConfig()` | Collector configuration |

Properties: `main_entry_count`, `subagent_count`, `total_entries`, `transcript_paths`.

### Hook Callback

The collector provides `hook_callback()` for SDK integration. It extracts
`transcript_path` from hook input data and starts tailing when discovered.
Returns an empty dict (no modifications to hook behavior).

### Hook Registration

`hooks_config()` returns a hook configuration dict registering the collector's
callback for all supported SDK events:

- `UserPromptSubmit` (earliest discovery)
- `PreToolUse`, `PostToolUse`
- `SubagentStop`
- `Stop`
- `PreCompact`

Note: The SDK does not support `SubagentStart` or `Notification` hooks.

## Adapter Integration

### ClaudeAgentSDKAdapter Changes

The adapter integrates `TranscriptCollector` during SDK query execution.
See `src/weakincentives/adapters/claude_agent_sdk/adapter.py` for the full
adapter implementation.

The integration flow:

1. Adapter creates a `TranscriptCollector` with the prompt name and config
1. Collector hooks are merged with existing adapter hooks (not replaced)
1. Collector runs as an async context manager during SDK query execution
1. Both adapter hooks (for tool bridging) and collector hooks (for transcript
   discovery) receive callbacks concurrently

## Log Events

### Transcript Events

| Event | Level | When |
|-------|-------|------|
| `transcript.collector.start` | DEBUG | Collector started |
| `transcript.collector.path_discovered` | DEBUG | Main transcript path found |
| `transcript.collector.subagent_discovered` | DEBUG | Sub-agent transcript found |
| `transcript.collector.entry` | DEBUG | Each transcript entry |
| `transcript.collector.stop` | DEBUG | Collector stopped |
| `transcript.collector.error` | WARNING | File access error (recoverable) |

### Entry Log Context

Each `transcript.collector.entry` event includes:

| Key | Type | Description |
|-----|------|-------------|
| `prompt_name` | `str` | Prompt being evaluated |
| `transcript_source` | `str` | `"main"` or `"subagent:{agent_id}"` |
| `entry_type` | `str` | Entry type (user, assistant, etc.) |
| `sequence_number` | `int` | Entry number within transcript |
| `raw_json` | `str \| None` | Raw JSON (if `emit_raw_json=True`) |
| `parsed` | `dict \| None` | Parsed fields (if `parse_entries=True`) |

### Example Log Output

```json
{
  "timestamp": "2024-01-15T10:30:00.123+00:00",
  "level": "DEBUG",
  "logger": "weakincentives.adapters.claude_agent_sdk._transcript_collector",
  "event": "transcript.collector.entry",
  "message": "Transcript entry: assistant",
  "context": {
    "prompt_name": "code-review",
    "transcript_source": "main",
    "entry_type": "assistant",
    "sequence_number": 5,
    "parsed": {
      "type": "assistant",
      "message": {
        "role": "assistant",
        "content": "I'll analyze the code..."
      }
    }
  }
}
```

## Sub-agent Discovery

Sub-agent transcripts are discovered by scanning the session directory.
See `_discover_subagents()` in
`src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py`.

### Discovery Pattern

```
transcript_path = /home/user/.claude/projects/myproj/abc123.jsonl
                                                      ^^^^^^
                                                   session ID

session_dir = /home/user/.claude/projects/myproj/abc123/
                                                 ^^^^^^
                                               derived from path

subagent_pattern = {session_dir}/subagents/agent-*.jsonl
```

## File Tailing

### Tailer State

The `_TailerState` dataclass tracks per-file tailing state. See
`src/weakincentives/adapters/claude_agent_sdk/_transcript_collector.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `Path` | (required) | Absolute path to transcript file |
| `source` | `str` | (required) | Source identifier: `'main'` or `'subagent:{id}'` |
| `position` | `int` | `0` | Current read position in bytes |
| `inode` | `int` | `0` | Inode for rotation detection |
| `partial_line` | `str` | `""` | Incomplete line buffer |
| `entry_count` | `int` | `0` | Entries emitted from this file |

### Rotation Handling

The tailer handles file rotation and truncation:

1. **Inode change**: File was rotated (new file created)
1. **Size decrease**: File was truncated in place (copytruncate)

Both cases reset position to 0 and clear the partial line buffer.

## Compaction Awareness

Claude Code compacts long conversations. The collector handles this:

1. **PreCompact hook**: Snapshot transcript state before compaction
1. **Summary entries**: Log compaction summary with dropped count
1. **Position reset**: If file size shrinks, reset tailer position

## Usage Example

### Basic Usage (Enabled by Default)

Transcript collection is enabled by default. No configuration needed:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)

# Default: transcript_collection=TranscriptCollectorConfig()
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(),
)

# Transcripts automatically collected during evaluate()
response = adapter.evaluate(prompt, session=session)

# All transcript entries emitted as DEBUG logs during execution
```

### Custom Configuration

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    TranscriptCollectorConfig,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        transcript_collection=TranscriptCollectorConfig(
            poll_interval=0.5,         # Slower polling
            emit_raw_json=False,       # Skip raw JSON in logs
        ),
    ),
)
```

### Disabling Collection

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        transcript_collection=None,  # Disable collection
    ),
)
```

### Manual Integration

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from weakincentives.adapters.claude_agent_sdk import TranscriptCollector

collector = TranscriptCollector(prompt_name="my-eval")

options = ClaudeAgentOptions(
    hooks=collector.hooks_config(),
    # ... other options
)

async with collector.run():
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            process_message(message)
# Transcripts emitted as DEBUG logs throughout execution
```

### Accessing Collected State

```python
async with collector.run():
    # ... SDK execution ...
    pass

# After execution, access collection statistics
print(f"Main transcript entries: {collector.main_entry_count}")
print(f"Sub-agent count: {collector.subagent_count}")
print(f"Total entries: {collector.total_entries}")
print(f"Transcripts discovered: {list(collector.transcript_paths)}")
```

## Migration from Log Aggregator

### Removed Components

| Component | Replacement |
|-----------|-------------|
| `ClaudeLogAggregator` | `TranscriptCollector` |
| `_EXCLUDED_PATHS` | Not needed (transcript-only) |
| `_LOG_EXTENSIONS` | `.jsonl` only |
| `_is_log_file()` | Not needed (explicit paths) |
| Directory polling | Hook-driven discovery |

### Configuration Mapping

| Log Aggregator | Transcript Collector |
|----------------|---------------------|
| `claude_dir` | Derived from `transcript_path` |
| `poll_interval` | `config.poll_interval` |
| N/A | `config.subagent_discovery_interval` |
| N/A | `config.emit_raw_json` |
| N/A | `config.parse_entries` |

### Behavioral Changes

1. **Discovery**: Immediate via hook vs polling directory
1. **Scope**: Transcripts only vs all log-like files
1. **Structure**: Parsed JSONL entries vs raw text lines
1. **Sub-agents**: Explicit directory scanning vs generic file discovery

## Invariants

1. **Hook registration**: Collector hooks registered before query starts
1. **Path derivation**: Session dir correctly derived from transcript path
1. **Entry ordering**: Entries emitted in file order within each transcript
1. **Sequence isolation**: Sequence numbers per-transcript, not global
1. **No data loss**: All complete JSONL lines emitted (partial buffered)
1. **Graceful errors**: File errors logged as WARNING, never raise

## Performance Considerations

- **Poll interval**: 250ms balances latency vs CPU usage
- **Sub-agent scan**: 1s interval reduces filesystem operations
- **Max read bytes**: 64KB prevents memory spikes on large entries
- **Executor I/O**: File reads in thread pool, non-blocking
- **Entry parsing**: Optional to reduce CPU overhead if not needed

## Testing

Tests verify:

- Main transcript discovery from hook callback
- Sub-agent discovery via directory scanning
- Entry emission as DEBUG logs with correct context
- File rotation handling (inode change, truncation)
- Compaction handling (PreCompact hook)
- Graceful error handling (file errors logged, not raised)

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` - Adapter specification
- `specs/LOGGING.md` - Structured logging format
- `specs/DEBUG_BUNDLE.md` - Debug bundle captures transcripts
