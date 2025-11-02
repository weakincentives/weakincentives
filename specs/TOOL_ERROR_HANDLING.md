# Error Handling Revamp

## Overview

This spec revisits error handling across prompt executions and tool invocations. The current implementation short-circuits prompt evaluation on most handler failures and returns tool parameters instead of result payloads when validation fails. The redesign promotes a consistent, typed contract that allows the LLM to recover after tooling issues while keeping downstream consumers informed.

## Goals

- Provide a uniform `ToolResult` shape that captures success or failure without violating type annotations.
- Ensure adapters always surface tooling failures to the LLM so the model can decide how to proceed.
- Maintain reliable telemetry for sessions and reducers even when tools fail.

## Non-Goals

- Changing how prompt sections render or how prompts choose tools.
- Introducing retries, backoff strategies, or automatic error correction.
- Modifying provider-specific error payloads beyond what is needed to satisfy this spec.

## Current Behavior

- `ToolResult.value` is required and typed as the tool's `ResultT`. When a handler raises `ToolValidationError`, adapters currently stuff the rejected params into `value`, violating the advertised type.
- Any non-validation exception raised during tool execution escalates as `PromptEvaluationError`, aborting the evaluation before the LLM receives feedback.
- Session reducers depend on `ToolInvoked.result.value` being a dataclass instance; failures that diverge from this shape are dropped.

## Proposed Changes

### ToolResult Contract

- Extend `ToolResult` with a `success: bool` flag. `True` indicates a normal payload; `False` indicates a failure of any kind.
- Allow `ToolResult.value` to be `None`. Successful executions must continue returning the documented result dataclass. Failures must set `value=None` unless a structured error payload is supplied by the tool.
- Continue using `message` for human-readable context while relying on `success` for programmatic branching.
- Update `ToolInvoked` events, session reducers, and any other consumers to respect the new contract.

### Adapter Handling

- Wrap all tool handler exceptions—validation errors and unexpected exceptions—and convert them into `ToolResult` instances with `success=False`, `value=None`, and a descriptive `message`.
- The adapter must still log or attach the original exception for observability but avoid raising `PromptEvaluationError` for tool-level issues.
- Continue short-circuiting only when provider communication fails or prompt parsing is impossible.
- Append a `role: "tool"` message containing the error message to the conversation so the LLM can adjust its plan.

### Session and Telemetry

- Update the session reducer logic to record failed tool executions, even though `value` might be `None`. Reducers can decide whether to track failures in a separate slice.
- Ensure `ToolInvoked.result.value` may be `None` without dropping the event.
- Consider emitting a dedicated `ToolFailed` event in the future; for now, the `success` flag is sufficient.

## Acceptance Criteria

- Adapters never abort evaluation on tool handler failures. Instead, they return `ToolResult(success=False, value=None, message="…")` to the LLM.
- All unit tests updated to assert the new `success` semantics and nullable value.
- Session reducers continue operating without crashing when `ToolResult.value is None`.
- Documentation in `AGENTS.md`, relevant tool specs, and adapter docs references the new contract.

## Design Notes

- We will not introduce a structured error payload type at this stage; the existing `message` string paired with the `success` flag is sufficient for consumers.
- Retry policies remain the responsibility of the LLM. Adapters do not attempt tool-level retries.
- No provider currently requires extra metadata in the tool message. Revisit only if a specific integration surfaces new constraints.
