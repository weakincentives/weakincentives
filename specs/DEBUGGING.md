# Debugging Specification

## Purpose

Debugging tools for inspecting prompt evaluation, session state, and runtime
behavior.

**Implementation:** `src/weakincentives/debug/`

## Debugging Surfaces

| Surface | Purpose | Module |
|---------|---------|--------|
| Log Collector | Capture logs during evaluation | `weakincentives.debug` |
| Filesystem Archive | Archive VFS to zip | `weakincentives.debug` |
| Session Dump | Persist snapshots to JSONL | `weakincentives.debug` |
| Session Slices | Query prompt/tool events | `weakincentives.runtime` |
| Debug Web UI | Visual snapshot explorer | `wink debug` CLI |

## Log Collector

```python
from weakincentives.debug import collect_all_logs

with collect_all_logs("./logs/session.log") as collector:
    response = adapter.evaluate(prompt, session=session)
```

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target` | required | Output JSONL path |
| `level` | DEBUG | Minimum log level |

### File Format

```json
{"timestamp": "...", "level": "INFO", "logger": "...", "event": "...", "message": "...", "context": {...}}
```

## Filesystem Archive

```python
from weakincentives.debug import archive_filesystem

archive_path = archive_filesystem(fs, "./snapshots/")
```

Returns `Path` to created zip, or `None` if filesystem empty.

## Session Event Slices

| Event Type | When | Key Fields |
|------------|------|------------|
| `PromptRendered` | Before sending | `rendered_prompt`, `prompt_ns`, `prompt_key` |
| `ToolInvoked` | After tool execution | `name`, `params`, `result` |
| `PromptExecuted` | After evaluation | `result`, `usage` |

Query via session indexing:

```python
for event in session[ToolInvoked].all():
    print(f"Tool {event.name}: success={event.result.success}")
```

## Debug Web UI

```bash
wink debug <snapshot_path> [--host HOST] [--port PORT]
```

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | HTML index page |
| `/api/meta` | GET | Snapshot metadata |
| `/api/entries` | GET | List entries |
| `/api/slices/{type}` | GET | Slice items |
| `/api/raw` | GET | Raw JSON payload |
| `/api/reload` | POST | Reload from disk |
| `/api/snapshots` | GET | List files |
| `/api/select` | POST | Select entry |
| `/api/switch` | POST | Switch file |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Normal stop |
| 2 | Validation failed |
| 3 | Server failed to start |

## Complete Debug Capture

```python
from weakincentives.debug import archive_filesystem, collect_all_logs, dump_session

debug_dir = f"./debug/{session.session_id}"

with collect_all_logs(f"{debug_dir}/prompt.log"):
    response = adapter.evaluate(prompt, session=session)

dump_session(session, debug_dir)

if fs := prompt.resources.get(Filesystem, None):
    archive_filesystem(fs, debug_dir)
```

Captures logs (JSONL), session state, and filesystem contents.

## Environment Variables

```bash
WEAKINCENTIVES_LOG_LEVEL=DEBUG
WEAKINCENTIVES_LOG_FORMAT=json
```

## Related Specifications

- `specs/LOGGING.md` - Event taxonomy
- `specs/SESSIONS.md` - Session state
