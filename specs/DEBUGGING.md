# Debugging Specification

This document describes debugging tools and patterns available in the
`weakincentives` library for inspecting prompt evaluation, session state, and
runtime behavior.

## Overview

The library provides several debugging surfaces:

| Surface | Purpose | Module |
| ----------------------- | ----------------------------------------- | ------------------------------ |
| **Log Collector** | Capture logs during prompt evaluation | `weakincentives.debug` |
| **Filesystem Archive** | Archive VFS contents to zip | `weakincentives.debug` |
| **Session Dump** | Persist session snapshots to JSONL | `weakincentives.debug` |
| **Session Slices** | Query stored prompt/tool events | `weakincentives.runtime` |
| **Debug Web UI** | Visual snapshot explorer | `wink debug` CLI |
| **Structured Logging** | Event-based observability | `weakincentives.runtime.logging` |

## Log Collector

The `collect_all_logs` context manager captures all log records emitted during
the context and writes them to a JSONL file for later analysis.

### Usage

```python
from weakincentives.debug import collect_all_logs

with collect_all_logs("./logs/session.log") as collector:
    response = adapter.evaluate(prompt, session=session)

print(f"Logs written to: {collector.path}")
```

### Collector Options

```python
collect_all_logs(
    target: str | Path,
    *,
    level: int = logging.DEBUG,
)
```

| Parameter | Default | Description |
| --------- | ---------------- | ---------------------------------------------------- |
| `target` | (required) | Path to output JSONL file (appends if exists; directories created automatically) |
| `level` | `logging.DEBUG` | Minimum log level to capture |

### File Format

Logs are written in JSONL format (one JSON object per line) using the standard
structured logging schema:

```json
{"timestamp": "2024-01-15T10:30:00+00:00", "level": "INFO", "logger": "weakincentives.adapters.openai", "event": "tool.execution.complete", "message": "Tool execution completed", "context": {"tool_name": "read_file", "success": true}}
{"timestamp": "2024-01-15T10:30:01+00:00", "level": "DEBUG", "logger": "weakincentives.prompt", "event": "prompt.render.complete", "message": "Prompt rendered", "context": {"char_count": 1234}}
```

| Field | Type | Description |
| ----------- | -------- | ------------------------------------------ |
| `timestamp` | `string` | ISO 8601 timestamp (UTC) |
| `level` | `string` | Log level name (DEBUG, INFO, WARNING, etc) |
| `logger` | `string` | Logger name (e.g., `weakincentives.adapters.openai`) |
| `event` | `string` | Structured event name (e.g., `tool.execution.complete`) |
| `message` | `string` | Human-readable log message |
| `context` | `object` | Structured context payload |

### Context Object

The context manager yields an object with a `path` property:

```python
with collect_all_logs("./logs/debug.jsonl") as collector:
    # ... run code ...
    pass

print(f"Logs written to: {collector.path}")
```

### Per-Session Log Files

A common pattern is to create a log file per session:

```python
from weakincentives.debug import collect_all_logs

log_path = f"./logs/{session.session_id}.log"
with collect_all_logs(log_path):
    response = adapter.evaluate(prompt, session=session)
```

### Thread Safety

The collector uses Python's logging infrastructure, which is thread-safe. The
handler is attached only for the duration of the context manager and safely
removed on exit.

## Filesystem Archive

The `archive_filesystem` function creates a zip archive of all files in a
`Filesystem` instance. This is useful for capturing the complete state of a
virtual filesystem used during agent execution.

### Usage

```python
from weakincentives.debug import archive_filesystem

# Archive the filesystem to a zip file
archive_path = archive_filesystem(fs, "./snapshots/")
if archive_path:
    print(f"Filesystem archived to: {archive_path}")
```

### Function Signature

```python
archive_filesystem(
    fs: Filesystem,
    target: str | Path,
    *,
    archive_id: UUID | None = None,
) -> Path | None
```

| Parameter | Default | Description |
| ------------ | ----------- | ---------------------------------------------------- |
| `fs` | (required) | Filesystem instance to archive |
| `target` | (required) | Target directory (or file path whose parent is used) |
| `archive_id` | `None` | UUID for the archive filename; generated if not provided |

### Return Value

Returns the `Path` to the created archive file (`<archive_id>.zip`), or `None`
if the filesystem is empty.

### Error Handling

- Files that cannot be read (permission errors, deleted during archive) are
  skipped with a warning log
- If the archive cannot be created (e.g., permission denied), an `OSError` is
  raised and any partial archive is cleaned up

### Archive Contents

The zip archive contains all files from the filesystem with their relative
paths preserved. Directory structure is maintained within the archive.

```python
# Example: Archive a VFS with tool-generated files
from weakincentives.debug import archive_filesystem
from weakincentives.contrib.tools import VirtualFilesystem

vfs = VirtualFilesystem()
# ... agent writes files to vfs ...

archive_path = archive_filesystem(vfs, f"./debug/{session.session_id}/")
# Creates: ./debug/<session_id>/<uuid>.zip
```

## Session Event Slices

The session automatically captures high-level lifecycle events during prompt
evaluation. These are stored as session slices and available for inspection.

### Automatic Events

| Event Type | When Emitted | Key Fields |
| ---------------- | ---------------------------- | ------------------------------------- |
| `PromptRendered` | Before sending to provider | `rendered_prompt`, `tool_count` |
| `ToolInvoked` | After each tool execution | `name`, `params`, `result` |
| `PromptExecuted` | After evaluation completes | `result`, `usage` |

### Querying Events

```python
from weakincentives.runtime import PromptRendered, ToolInvoked, PromptExecuted

# All rendered prompts in this session
for event in session[PromptRendered].all():
    print(f"Rendered {event.prompt_name}: {len(event.rendered_prompt)} chars")

# Tool invocations with results
for event in session[ToolInvoked].all():
    print(f"Tool {event.name}: success={event.result.success}")

# Final execution result
executed = session[PromptExecuted].latest()
if executed:
    print(f"Output: {executed.result}")
```

## Debug Web UI

The `wink debug` command provides a browser-based interface for exploring
session snapshots. See `specs/WINK_DEBUG.md` for full documentation.

### Quick Start

```bash
# Start debug server with a snapshot file
wink debug ./snapshots/session.jsonl

# Opens browser to http://127.0.0.1:8000
```

### Capabilities

- Browse all snapshots in a directory
- Inspect slice contents with syntax highlighting
- View raw JSON payloads
- Navigate between snapshot entries
- Automatic Markdown rendering for text fields

## Structured Logging

All runtime modules emit structured log records with stable `event` names and
`context` payloads. See `specs/LOGGING.md` for the complete event taxonomy.

### Enabling Debug Logs

```bash
# Via environment variable
WEAKINCENTIVES_LOG_LEVEL=DEBUG python my_agent.py

# Via API
from weakincentives.runtime.logging import configure_logging
configure_logging(level="DEBUG")
```

### Key Event Categories

| Prefix | Module | Examples |
| -------------- | ----------------- | ------------------------------------------ |
| `adapter.*` | Provider adapters | `adapter.init`, `evaluate.entry` |
| `session.*` | Session state | `session.dispatch`, `session.restore` |
| `tool.*` | Tool execution | `tool.execution.start`, `tool.execution.complete` |
| `prompt.*` | Prompt rendering | `prompt.render.start`, `prompt.render.complete` |
| `resource.*` | Resource lifecycle | `resource.construct.start`, `resource.close` |

### JSON Log Output

For structured log aggregation (e.g., to a log management system):

```bash
WEAKINCENTIVES_LOG_FORMAT=json python my_agent.py
```

Output format:

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "level": "INFO",
  "logger": "weakincentives.adapters.openai",
  "message": "Tool execution completed",
  "event": "tool.execution.complete",
  "context": {
    "tool_name": "read_file",
    "success": true
  }
}
```

## Debugging Patterns

### Diagnosing Tool Failures

```python
from weakincentives.debug import collect_all_logs

log_path = f"./logs/{session.session_id}.log"
with collect_all_logs(log_path):
    try:
        response = adapter.evaluate(prompt, session=session)
    except Exception:
        # Logs are already persisted to file for post-mortem analysis
        print(f"Logs available at: {log_path}")
        raise
```

### Analyzing Token Usage

```python
from weakincentives.runtime import PromptExecuted

for event in session[PromptExecuted].all():
    if event.usage:
        print(f"Prompt: {event.prompt_name}")
        print(f"  Input tokens: {event.usage.input_tokens}")
        print(f"  Output tokens: {event.usage.output_tokens}")
        print(f"  Total: {event.usage.total_tokens}")
```

### Inspecting Session State

```python
# Snapshot current state for debugging
from weakincentives.runtime import Snapshot

snapshot = session.snapshot()

# Examine slice contents
for slice_type, items in snapshot.slices.items():
    print(f"{slice_type}: {len(items)} items")

# Restore to a previous state
session.restore(snapshot)
```

### Complete Debug Capture

Combine all three debug utilities for comprehensive post-mortem analysis:

```python
from weakincentives.debug import (
    archive_filesystem,
    collect_all_logs,
    dump_session,
)

debug_dir = f"./debug/{session.session_id}"

# Capture logs during evaluation
with collect_all_logs(f"{debug_dir}/prompt.log"):
    response = adapter.evaluate(prompt, session=session)

# Dump session state
dump_session(session, debug_dir)

# Archive any files created by tools
if fs := prompt.resources.get(Filesystem, None):
    archive_filesystem(fs, debug_dir)

# Debug directory now contains:
# - prompt.log: detailed log timeline (JSONL)
# - <session_id>.jsonl: session state snapshots
# - <uuid>.zip: filesystem contents archive
```

This pattern captures:

- **Logs**: Every structured log event with timestamps and context
- **Session state**: All slice data, events, and reducers
- **File artifacts**: Any files the agent created or modified

## Implementation Notes

- The log collector attaches a `logging.Handler` to the root logger for the
  duration of the context manager, capturing logs from all hierarchies
- Log entries are written to disk immediately (flushed after each entry)
- Handler removal is guaranteed via `try/finally` even if exceptions occur
- The collector does not interfere with existing logging configuration
- Parent directories are created automatically if they don't exist
