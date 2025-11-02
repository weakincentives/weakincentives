# Prompt Subagent Dispatch Tool Specification

## Overview

`dispatch_subagent` lets a prompt run a bounded task in a throwaway child session. The parent keeps planning context while the child inherits every enabled tool, completes the work, and reports a compact summary. No child conversation history or reducer state survives after the call returns.

## Module Surface

- Tool code lives in `weakincentives.tools.prompt_subagent`.
- `PromptSubagentToolsSection` exposes the markdown guidance and tool registration.
- Validation errors reuse `ToolValidationError` from `weakincentives.tools.errors`.
- The implementation depends on the session helpers described in `specs/SESSIONS.md` and snapshot rules in `specs/SESSION_SNAPSHOTS.md`.

## Data Contracts

Payloads stay immutable so reducers and serde helpers can reuse them without copying.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SubagentMode = Literal["plan_step", "ad_hoc"]


@dataclass(slots=True, frozen=True)
class DispatchSubagent:
    mode: SubagentMode
    prompt_ns: str
    prompt_key: str
    instructions: str
    expected_artifacts: tuple[str, ...] = field(default_factory=tuple)
    plan_step_id: str | None = None


@dataclass(slots=True, frozen=True)
class DispatchSubagentResult:
    message_summary: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    tools_used: tuple[str, ...] = field(default_factory=tuple)
```

Key validation rules:

- `prompt_ns` and `prompt_key` must match a prompt registered in `weakincentives.prompt.registry`.
- `instructions` is trimmed ASCII (1–2,000 characters).
- `expected_artifacts` elements are ASCII ≤160 characters.
- When `mode == "plan_step"`, callers must provide `plan_step_id` so planning results can reconcile.
- Snapshot capture and hydration are implementation details; callers never provide serializer metadata.

## Tool Definition

| Name | Parameters | Result | Summary |
| ---- | ---------- | ------ | ------- |
| `dispatch_subagent` | `DispatchSubagent` | `DispatchSubagentResult` | Run a prompt in an isolated child session and return the summary payload. |

`PromptSubagentToolsSection` renders concise guidance covering:

1. When to offload work (deep research, large drafting, or plan steps).
1. The guarantee that child runs inherit the full tool set.
1. The requirement to keep `instructions` specific and to enumerate expected artifacts.

The section requires the parent `Session` so it can compute snapshots when the tool fires.

## Execution Outline

Orchestrators follow the pattern below when handling a tool call:

1. Validate params and coerce sequences into tuples; raise `ToolValidationError` on failure.
1. Capture the parent snapshot via `session.snapshot()` (see `specs/SESSION_SNAPSHOTS.md`).
1. Create a new `Session` with the same reducers and event bus, then hydrate it with `rollback(parent_snapshot)`.
1. Load the child prompt identified by `(prompt_ns, prompt_key)` and render it with the provided instructions plus any planner metadata.
1. Run the child agent. It may call any tool that the parent prompt exposed.
1. Collect the child’s final message, emitted artifacts, and observed tool usage. Package them into `DispatchSubagentResult` and return `ToolResult(message_summary, payload)`.
1. Tear down the child session regardless of success so no state leaks into the parent conversation.

Only the summary text and structured payload return to the parent. Reducer state created during the child run is discarded with the session.

## Error Handling

- Registry lookup failures, invalid params, or snapshot hydration errors raise `ToolValidationError`.
- Runtime issues (LLM failure, tool crash) surface through a domain-specific `DispatchSubagentError`. The handler must still dispose of the child session and return a concise failure message to the parent.
- Child sessions unsubscribe from the event bus inside `finally` blocks to avoid lingering listeners.

## Testing Checklist

- Child sessions created from a snapshot expose the same reducer slices as the parent before any child tool runs.
- Tool registration tests confirm the section renders guidance and exposes exactly one tool definition.
- Execution tests assert that running tools inside the child leaves the parent session untouched after teardown.
- Structured output tests cover artifact propagation and the alignment between `ToolResult.message` and `DispatchSubagentResult.message_summary`.
- Failure tests simulate prompt lookup errors, snapshot mismatches, and runtime exceptions while ensuring child sessions are always cleaned up.
