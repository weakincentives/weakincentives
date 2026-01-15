# Debugging Specification

Tools for inspecting prompt evaluation, session state, and runtime behavior.

**Source:** `src/weakincentives/debug/`

## Debugging Surfaces

| Surface | Purpose |
|---------|---------|
| `collect_all_logs` | Capture logs to JSONL |
| `archive_filesystem` | Archive VFS to zip |
| `dump_session` | Persist snapshots to JSONL |
| Session slices | Query events |
| `wink debug` | Web UI for snapshots |

## Log Collector

```python
from weakincentives.debug import collect_all_logs

with collect_all_logs("./logs/session.log") as collector:
    response = adapter.evaluate(prompt, session=session)
```

JSONL format with `timestamp`, `level`, `logger`, `event`, `message`, `context`.

## Filesystem Archive

```python
from weakincentives.debug import archive_filesystem

archive_path = archive_filesystem(fs, "./snapshots/")
```

Returns `Path` to zip file, or `None` if empty.

## Session Event Slices

```python
from weakincentives.runtime import PromptRendered, ToolInvoked, PromptExecuted

for event in session[ToolInvoked].all():
    print(f"Tool {event.name}: success={event.result.success}")

executed = session[PromptExecuted].latest()
```

## Debug Web UI

```bash
wink debug <snapshot_path> [--port 8000]
```

| Route | Purpose |
|-------|---------|
| `/api/meta` | Current snapshot metadata |
| `/api/entries` | List all entries |
| `/api/slices/{type}` | Slice contents |
| `/api/reload` | Reload file |

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

## Structured Logging

```bash
WEAKINCENTIVES_LOG_LEVEL=DEBUG python my_agent.py
WEAKINCENTIVES_LOG_FORMAT=json python my_agent.py
```

| Prefix | Examples |
|--------|----------|
| `adapter.*` | `adapter.init`, `evaluate.entry` |
| `session.*` | `session.dispatch`, `session.restore` |
| `tool.*` | `tool.execution.start/complete` |
