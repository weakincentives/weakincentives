# Prompt Subagent Dispatch Tool Specification

## Overview

The prompt subagent tool lets a running prompt offload a block of work to a
fresh child prompt without growing the root conversation. A root prompt may
plan a sequence of tasks, pick one, and dispatch the work to an isolated
subagent that receives a snapshot of the current `Session` plus the full tool
suite. The child runs to completion, returns a structured result, and its
session is discarded so the root keeps only the summarized output in context.

## Goals

- **Context isolation** – Keep large investigative turns out of the root
  prompt history while still letting the subagent reuse every registered tool.
- **Deterministic state handoff** – The child receives an immutable snapshot of
  the parent session so tool reducers observe a consistent baseline.
- **Single-entry orchestration** – The tool surfaces a uniform payload that
  orchestration layers can translate into provider-specific prompt runs.
- **Disposable execution** – Child sessions never leak state; only the final
  `ToolResult` message and payload survive the dispatch.

## Module Surface

- Tool implementation lives in `weakincentives.tools.prompt_subagent`.
- The section exposing the tool is `PromptSubagentToolsSection`.
- The module reuses existing session helpers from `weakincentives.session` and
  prompt loaders from `weakincentives.prompt.registry`.
- Validation errors raise `ToolValidationError` from
  `weakincentives.tools.errors`.

## Data Model

The tool contract uses frozen dataclasses so payloads integrate cleanly with
session reducers and serde helpers.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

SubagentMode = Literal["plan_step", "ad_hoc"]


@dataclass(slots=True, frozen=True)
class DispatchSubagent:
    mode: SubagentMode
    prompt_ns: str
    prompt_key: str
    instructions: str
    expected_artifacts: tuple[str, ...] = field(default_factory=tuple)
    plan_step_id: str | None = None
    snapshot_version: str | None = None


@dataclass(slots=True, frozen=True)
class DispatchSubagentResult:
    prompt_ns: str
    prompt_key: str
    message_summary: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    tools_used: tuple[str, ...] = field(default_factory=tuple)
```

Key rules:

- `prompt_ns` and `prompt_key` select the child prompt via the registry. They
  must match an enabled prompt definition or validation fails.
- `instructions` is required ASCII content (1–2,000 characters). The handler
  trims whitespace before dispatching the run request.
- `expected_artifacts` describes deliverables the orchestrator should surface
  back to the parent. Items are ASCII strings up to 160 characters each.
- When `mode == "plan_step"`, `plan_step_id` must be provided so orchestrators
  can reconcile results with the planning tool.
- `snapshot_version` captures the serialized snapshot version string so
  orchestration layers can validate compatibility before hydration.

`DispatchSubagentResult` encodes the child output that the root prompt sees. The
`message_summary` becomes the textual reply in the parent conversation, while
`artifacts` captures any structured follow-up (for example file paths or
citations) produced by the subagent.

## Tool Definition

`PromptSubagentToolsSection` registers a single tool:

| Name | Params | Result | Description |
| ---- | ------ | ------ | ----------- |
| `dispatch_subagent` | `DispatchSubagent` | `DispatchSubagentResult` | Spawn an isolated subagent run with the full tool suite and return a summarized result. |

The section ships with concise, model-facing instructions that explain when to
invoke the tool (e.g. when a plan step requires deep investigation or when the
root conversation nears the context limit). The section requires a `Session`
instance during initialization so it can compute snapshots.

## Execution Flow

Orchestrators implement the following lifecycle when the tool executes:

1. **Validate inputs** – The handler enforces ASCII constraints, non-empty
   instructions, and prompt registry membership. Invalid payloads raise
   `ToolValidationError`.
2. **Capture snapshot** – Call `session.snapshot()` to obtain an immutable view
   of all slices along with the schema version.
3. **Spawn child session** – Create a fresh `Session` instance, register the
   same reducers as the parent, and immediately invoke
   `child.rollback(parent_snapshot)` so the child starts with identical state.
   The child subscribes to the same event bus so all tools remain available.
4. **Render child prompt** – Load the prompt identified by `(prompt_ns,
   prompt_key)`, render it with any configured defaults, and include the
   `instructions`, plan metadata, and snapshot summary in the runtime overrides.
   The rendered prompt exposes the full list of tool definitions; pass them to
   the child orchestration loop unchanged.
5. **Execute subagent** – Run the prompt inside the child session. The child may
   invoke any tool the root had enabled. Tool events populate the child session
   only; the parent session stays untouched.
6. **Collect result** – When the child completes, gather its final message,
   structured output, tool usage list, and any artifacts recorded by reducers
   (e.g. files emitted by VFS tools). Format these into a
   `DispatchSubagentResult`.
7. **Tear down** – Unsubscribe and discard the child session. Do not persist its
   snapshot. The handler returns a `ToolResult` whose `message` restates
   `DispatchSubagentResult.message_summary`.

Only the summary text and structured payload flow back to the parent prompt. No
child conversation turns or tool transcripts remain in context, keeping the root
history small.

## Session and Snapshot Semantics

- The parent session never mutates during child execution. All tool outputs stay
  scoped to the child session and vanish when it is discarded.
- Orchestrators may hydrate additional derived reducers after `rollback()` (for
  example metrics aggregators) so long as they derive deterministically from the
  snapshot contents.
- If snapshot hydration fails, abort the tool call and surface a validation
  error. The parent session remains untouched.
- Child runs may take a fresh snapshot before completing if they need to attach
  metadata to the result; these snapshots are local and not persisted.

## Error Handling

- Prompt lookup failures, snapshot incompatibilities, or orchestration errors
  return a `ToolResult` whose message explains the failure and whose payload is
  omitted. The handler still raises `ToolValidationError` for invalid inputs.
- Downstream execution failures (model errors, tool crashes) propagate through a
  `DispatchSubagentError` exception. The root conversation should receive a
  concise failure message indicating that the subagent run was aborted.
- Handlers must ensure the child session unsubscribes from the bus even when
  errors occur.

## Testing Checklist

- **Snapshot parity** – After rollback the child session exposes identical
  slices to the parent snapshot before any child tool runs.
- **Tool propagation** – The child prompt receives the same tool definitions as
  the parent rendered prompt; orchestrators can invoke them without additional
  registration.
- **Isolation** – Running a tool in the child session leaves the parent session
  unchanged. After the run, the child session is garbage collected and no longer
  receives events.
- **Structured output** – Successful runs produce a `DispatchSubagentResult`
  whose `message_summary` matches the `ToolResult` message and whose artifact
  list reflects child outputs.
- **Failure reporting** – Simulate prompt lookup failures, snapshot hydration
  issues, and downstream execution errors; confirm they raise the documented
  exceptions and leave the parent session untouched.
