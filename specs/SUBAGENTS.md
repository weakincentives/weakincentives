# Subagent Dispatch Specification

## Overview

`dispatch_subagent` lets a prompt run a bounded task in a throwaway child session. The parent keeps planning context while the child inherits every enabled tool, completes the work, and reports a compact summary. No child conversation history or reducer state survives after the call returns. The current implementation builds the child prompt dynamically inside the caller-supplied runner, so no global registry is required.

## Module Surface

- Tool code lives in `weakincentives.tools.prompt_subagent`.
- `PromptSubagentToolsSection` exposes the markdown guidance and tool registration.
- Validation errors reuse `ToolValidationError` from `weakincentives.tools.errors`.
- The implementation depends on the session helpers described in `specs/SESSIONS.md` and snapshot rules in `specs/SESSION_SNAPSHOTS.md`.
- Runners conform to the `SubagentRunner` protocol and receive the normalized `DispatchSubagent` payload along with a cloned `Session` and an isolated `EventBus`.

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

- `instructions` is trimmed ASCII (1–2,000 characters).
- `expected_artifacts` elements are ASCII ≤160 characters.
- When `mode == "plan_step"`, callers must provide `plan_step_id` so planning results can reconcile.
- Snapshot capture and hydration are implementation details; callers never provide serializer metadata.

Each field mirrors the tool interface so runners can pipe the dataclass directly into a child prompt or re-render it into instructions. Runners receive the normalized payload (stripped ASCII strings, tuples for artifacts) and should avoid mutating it in place.

## Normalization & Validation

`DispatchSubagent` inputs are validated before any session work happens:

- `_normalize_text` trims whitespace, enforces ASCII, and checks length limits for `instructions`, `plan_step_id`, and artifact labels. Errors surface as `ToolValidationError` describing the offending field.
- `_normalize_artifacts` maps incoming sequences into tuples of normalized labels.
- `_normalize_params` threads the above helpers and ensures that `plan_step_id` is provided whenever `mode == "plan_step"`.

The handler copies the normalized dataclass before handing it to the runner so downstream code can assume all invariants hold.

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
1. Instantiate a child `Session` wired to an `InProcessEventBus`. `_clone_session_structure` copies reducer registrations and seed slices so selectors behave the same way as in the parent.
1. Hydrate the child session with the parent snapshot via `rollback(snapshot)`.
1. Subscribe a `_ToolUsageRecorder` to the child bus so tool invocations can be deduplicated and surfaced in `DispatchSubagentResult.tools_used`.
1. Invoke the runner supplied to `PromptSubagentToolsSection`. The runner receives the cloned session, bus, and normalized params and is responsible for constructing and evaluating the child prompt.
1. Package the returned `DispatchSubagentResult` as the payload in a `ToolResult`. Any non-dataclass result triggers a validation failure.
1. Tear down the child session regardless of success so no state leaks into the parent conversation.

### Runner Contract

Runners must:

- Build or resolve the child prompt using the provided `DispatchSubagent` dataclass. The prompt should expose whatever tools the subagent needs and may embed the instructions directly.
- Evaluate the child prompt through the chosen adapter, passing the cloned session and bus so events remain isolated. The runner may mutate the cloned session freely.
- Return `DispatchSubagentResult` with:
  - `message_summary`: concise ASCII summary suitable for user-visible output.
  - `artifacts`: tuple of delivered artifact identifiers (often echoing `expected_artifacts`).
  - `tools_used`: optional; callers may supply their own telemetry, but the tool will fill it if omitted.
- Raise `ToolValidationError` for caller errors (e.g., unsupported mode) and `DispatchSubagentError` for runtime issues that should be reported back to the parent as `success=False`.
- Reuse state seeded by the parent session (for example virtual filesystem mounts) instead of reinitializing host mounts; the cloned session already holds the materialized data.

### Child Session Lifecycle

- Child sessions reuse the parent `session_id`/`created_at` for continuity but maintain their own state store and reducers.
- Reducer registrations are copied by type; when a reducer targets a different slice type the helper ensures that slice is pre-seeded in the child.
- Tool invocations recorded during the child run flow through the cloned session and are captured by the recorder for telemetry but do not leak back into the parent.
- After the runner returns (or raises), the handler converts the recorder results into a deduplicated tuple preserving first-seen order.

### Tool Result Semantics

- Success path returns `ToolResult(message_summary, value=DispatchSubagentResult, success=True)`.
- For `DispatchSubagentError` and unexpected exceptions the handler converts the error into a plain `ToolResult` with `success=False` and clears the payload. This keeps the parent reducer pipeline consistent.
- Non-`DispatchSubagentResult` payloads from the runner trigger an explicit failure message so adapters cannot silently misconfigure.

Only the summary text and structured payload return to the parent. Reducer state created during the child run is discarded with the session.

## Error Handling

- Invalid params or snapshot hydration errors raise `ToolValidationError`.
- Runtime issues (LLM failure, tool crash) surface through a domain-specific `DispatchSubagentError`. The handler must still dispose of the child session and return a concise failure message to the parent.
- Child sessions unsubscribe from the event bus inside `finally` blocks to avoid lingering listeners.

## Testing Checklist

- Child sessions created from a snapshot expose the same reducer slices as the parent before any child tool runs.
- Tool registration tests confirm the section renders guidance and exposes exactly one tool definition.
- Execution tests assert that running tools inside the child leaves the parent session untouched after teardown and that telemetry (`tools_used`) reflects the recorded calls.
- Structured output tests cover artifact propagation and the alignment between `ToolResult.message` and `DispatchSubagentResult.message_summary`.
- Failure tests simulate snapshot serialization/restore errors, runner validation failures, and unexpected exceptions to ensure the handler converts them into deterministic `ToolResult` payloads.
