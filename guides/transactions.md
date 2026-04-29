# Transactional Tools

Every tool call wrapped by `execute_tool` runs inside a transaction.
On any exception or `ToolResult.error` outcome, the spine restores:

- The session's typed slices.
- Any `Snapshotable` resource registered with the call (filesystem,
  workspace, etc.).

The model still sees a structured failure message — only the side
effects roll back.

## Filesystem rollback

```python
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Session,
    Tool,
    ToolContext,
    ToolResult,
    execute_tool,
)
from weakincentives.filesystem import InMemoryFilesystem

fs = InMemoryFilesystem()
fs.write_text("/notes.txt", "original")


def write_then_fail(params: object, context: ToolContext) -> ToolResult[None]:
    fs.write_text("/notes.txt", "DESTROYED")
    return ToolResult.error("simulated tool failure")


tool = Tool(name="bad", description="Writes then fails", handler=write_then_fail)
prompt = Prompt(
    ns="demo",
    key="rollback",
    sections=(MarkdownSection[None](title="T", key="t", tools=(tool,)),),
)

session = Session()
result = execute_tool(prompt, session, "bad", None, snapshotables=(fs,))

assert result.success is False
assert fs.read_text("/notes.txt") == "original"  # rolled back
```

## How it works

`execute_tool` calls `core.capture(session)` and `fs.snapshot()` before
running the handler. Any failure path — raised `WinkError`, raised
generic exception, or `ToolResult.error` — restores both. The
`tool_transaction` context manager exposes the same semantics for
hand-rolled flows.

## Pre/post hook tracking

Adapters that drive native tool calls via callback hooks (rather than
`execute_tool`) use `core.PendingToolTracker`:

```python
from weakincentives.core import PendingToolTracker

tracker = PendingToolTracker(session=session, snapshotables=(fs,))

# Pre-tool hook:
tracker.begin("call-1", "bad")

# … tool runs …

# Post-tool hook:
rolled_back = tracker.end("call-1", success=False)
```

The tracker takes the snapshot in `begin`, holds it across the gap,
and restores in `end` when the success flag is `False`.
