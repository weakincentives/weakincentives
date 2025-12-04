# TODO Tool Specification

## Purpose

A minimal session-scoped todo list: just ordered text strings. No priorities, deadlines, or nested plans—only a way to write a list and read it back.

## Data Model

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class TodoList:
    items: list[str] = field(default_factory=list)
```

- `items` are stored exactly as provided; empty lists are allowed.
- Keep validation minimal: ensure every entry is a string, optionally trim surrounding whitespace, and reject empty strings if trimming was applied.

## Session Behaviour

- State is **session-local** and cleared when the session ends.
- The session keeps **every `TodoList` snapshot**. Writes append to the slice (via the `append` reducer) instead of replacing the latest value.
- Reading uses the most recent snapshot (`select_latest(session, TodoList)`) but the full slice history remains available to the orchestrator if needed.

## Tools

| Tool | Parameters | Result | Behaviour |
| ---- | ---------- | ------ | --------- |
| `todo_write` | `TodoList` | `TodoList` | Validate `items`, normalise to an immutable copy, and append the snapshot to the session slice. Returns the stored snapshot. |
| `todo_read` | none | `TodoList` | Return the latest stored list via `select_latest`, or an empty `TodoList` if none exists. |

## Integration Notes

- Implemented in `weakincentives.tools.todo`.
- Use `ToolValidationError` for bad inputs.
- `TodoToolsSection` registers the `append` reducer for `TodoList` and exposes brief prompt copy reminding callers to read before writing.
