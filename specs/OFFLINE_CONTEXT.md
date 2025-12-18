# Offline Context Storage Specification

> **Status**: Proposed
> **Version**: 1.0

## Overview

Append tool invocations to a context file on the filesystem. Read it back when
needed. That's it.

## Core Concept

```
Tool invocation happens
       ↓
Append to .wink/context.md
       ↓
Later: Agent reads file to recall previous work
```

## Data Structure

One frozen dataclass representing a context entry:

```python
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.types import JSONValue


@FrozenDataclass()
class ContextEntry:
    """A single entry in the offline context store."""

    tool_name: str
    params: JSONValue
    result: JSONValue
    success: bool = True
    summary: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: UUID = field(default_factory=uuid4)
```

## ContextHistory

Minimal interface - append and read:

```python
from dataclasses import dataclass
from typing import Sequence

from weakincentives.prompt.tool import Filesystem
from weakincentives.serde import dump, parse


@dataclass(slots=True)
class ContextHistory:
    """Append-only context history backed by filesystem."""

    filesystem: Filesystem
    path: str = ".wink/context.jsonl"

    def append(self, entry: ContextEntry) -> None:
        """Append entry to context file."""
        line = dump(entry) + "\n"
        mode = "append" if self.filesystem.exists(self.path) else "create"
        self.filesystem.write(self.path, line, mode=mode)

    def read(self, limit: int = 50) -> Sequence[ContextEntry]:
        """Read recent entries."""
        if not self.filesystem.exists(self.path):
            return []

        content = self.filesystem.read(self.path).content
        entries = []
        for line in content.strip().split("\n"):
            if line:
                entries.append(parse(ContextEntry, line))

        return entries[-limit:]
```

## Tool

One tool to read context:

```python
from dataclasses import dataclass

from weakincentives.prompt.tool import Tool, ToolContext, ToolResult, ToolExample


@dataclass(slots=True, frozen=True)
class ReadContextParams:
    limit: int = 20


def read_context_handler(
    params: ReadContextParams,
    *,
    context: ToolContext,
) -> ToolResult[list[ContextEntry]]:
    """Read recent entries from offline context."""
    history = context.resources.get(ContextHistory)
    if history is None:
        return ToolResult(
            message="Context history not available.",
            value=None,
            success=False,
        )

    entries = list(history.read(params.limit))
    return ToolResult(
        message=f"Retrieved {len(entries)} context entries.",
        value=entries,
        success=True,
    )


read_context = Tool[ReadContextParams, list[ContextEntry]](
    name="read_context",
    description="Read recent tool invocations from offline context history.",
    handler=read_context_handler,
    examples=(
        ToolExample(
            description="Read last 10 context entries",
            input=ReadContextParams(limit=10),
            output=[
                ContextEntry(
                    tool_name="grep",
                    params={"pattern": "auth", "path": "src/"},
                    result={"matches": 15},
                    summary="Searched for auth patterns",
                ),
            ],
        ),
    ),
)
```

## Recording

Record tool invocations via the session event bus:

```python
from weakincentives.runtime.events import ToolInvoked


def auto_record(history: ContextHistory) -> Callable[[ToolInvoked], None]:
    """Create observer that records tool invocations."""

    def handler(event: ToolInvoked) -> None:
        entry = ContextEntry(
            tool_name=event.tool_name,
            params=event.params,
            result=event.result,
            success=event.success,
            summary=f"Called {event.tool_name}",
        )
        history.append(entry)

    return handler


# Setup
history = ContextHistory(filesystem=fs)
bus.subscribe(ToolInvoked, auto_record(history))
```

## Storage Format

Simple JSONL file at `.wink/context.jsonl`:

```jsonl
{"tool_name":"grep","params":{"pattern":"auth"},"result":{"matches":15},"success":true,"summary":"Searched for auth","timestamp":"2025-12-18T10:30:00Z","id":"..."}
{"tool_name":"read_file","params":{"path":"src/auth.py"},"result":{"content":"..."},"success":true,"summary":"Read auth module","timestamp":"2025-12-18T10:31:00Z","id":"..."}
```

## Usage

```
Agent needs to recall earlier work
       ↓
Calls read_context tool
       ↓
Gets list of previous tool invocations
       ↓
Avoids redundant work
```

## Future Extensions (Not MVP)

- Search/filter entries
- Summarization
- Compaction for long sessions
- Index for faster lookups
- Manual recording of decisions

## Related Specs

| Spec | Relationship |
|------|--------------|
| `specs/FILESYSTEM.md` | Storage backend |
| `specs/SESSIONS.md` | Event bus for recording |
| `specs/TOOLS.md` | Tool patterns |
