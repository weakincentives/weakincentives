# Subagents Specification

## Goal

Enable a parent run to fork lightweight child runs in parallel without losing access to the parent's tools, session state, or telemetry. The feature should feel like a single tool call that spawns fully powered agents sharing the same persistent context.

## Building Blocks

- **`SubagentsSection`** – a prompt section that introduces the delegation tool and reminds the model that parallel work MUST be dispatched instead of executed serially.
- **`dispatch_subagents` tool** – accepts a batch of child descriptions, runs each child prompt in parallel, and returns one result object per child.

No other sections, knobs, or pre-baked templates are required. The parent prompt stays declarative; the language model decides when and how to dispatch.

## Section Requirements

`SubagentsSection` lives in `src/weakincentives/prompt/subagents.py` and is exported from `weakincentives.prompt`. When rendered it MUST:

1. Briefly explain that the parent agent can offload parallelizable steps by calling `dispatch_subagents`.
1. State that every delegation must include recap bullets so the parent can audit the child’s plan.
1. List only the `dispatch_subagents` tool in `tools()`.
1. Avoid runtime configuration—no static dispatch payloads, flags, or template data.

## Tool Contract

The tool handler resides in `src/weakincentives/tools/subagents.py` and exports:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from weakincentives.prompt import DelegationSummaryParams, Tool, ToolResult
from weakincentives.prompt._types import SupportsDataclass


@dataclass(slots=True)
class SubagentDispatch:
    summary: DelegationSummaryParams
    recap_lines: tuple[str, ...]


@dataclass(slots=True)
class DispatchSubagentsParams:
    dispatches: Sequence[SubagentDispatch]


@dataclass(slots=True)
class SubagentResult(SupportsDataclass):
    output: str
    success: bool
    error: str | None = None


dispatch_subagents: Tool[DispatchSubagentsParams, Sequence[SubagentResult]]
```

Key rules:

- Parameters are constructed entirely by the LLM at call time. No defaults live in the section.
- Results mirror the input order.
- Each child's `output` is a plain string so downstream reducers can concatenate or summarize without extra coercion.
- Failures are captured per child via `success`/`error` while allowing healthy siblings to return normally.

## Runtime Flow

Every tool invocation MUST execute the following steps:

1. **Require the rendered parent prompt**. `context.rendered_prompt` is mandatory. Treat a missing prompt as an orchestrator bug and respond with a failing `ToolResult`.
1. **Share state with children**. For each dispatch, reuse the original `context.session` instead of cloning it so children mutate the exact same state object the parent uses. ALWAYS pass the parent's event bus reference so observers receive child telemetry in real time. When `context.session` is `None`, still hand each child the parent's bus reference.
1. **Wrap the child prompt**. Build a `DelegationPrompt` using the rendered parent prompt and the provided recap lines. Propagate response format metadata and tool descriptions exactly as described in `PROMPTS_COMPOSITION.md`.
1. **Run in parallel**. Evaluate each child through `context.adapter.evaluate` using a `ThreadPoolExecutor`. Use `min(len(dispatches), default_max_workers)` where `default_max_workers` matches Python's executor default when `None`.
1. **Collect per-child outcomes**. Successful executions populate `output` and set `success=True`. Exceptions are caught and converted into `success=False` with an error string, without cancelling other children.
1. **Return a structured result**. On handler-level success, return `ToolResult(value=tuple(results), success=True)`. When preconditions fail (for example, prompt not rendered, cloning unsupported), surface `ToolResult(success=False, value=None, message=...)`.

## Implementation Notes

- Keep the handler synchronous; the executor supplies concurrency.
- Avoid bespoke telemetry plumbing—sharing the event bus is sufficient.
- Ensure the shared session instance is safe for concurrent access and already carries any adapters or configuration the parent relies on.
- Tests should cover mixed success/failure batches and verify that state written by a child is visible to the parent after completion.

## Minimal Usage Sketch

```python
from dataclasses import dataclass

from weakincentives.prompt import DelegationSummaryParams, MarkdownSection, Prompt
from weakincentives.prompt.subagents import SubagentsSection
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentDispatch,
    dispatch_subagents,
)


@dataclass(slots=True)
class UpdateParams:
    body: str


daily_update = Prompt[str](
    ns="weekly.review",
    key="team-update",
    sections=(
        MarkdownSection[UpdateParams](
            title="Team Update",
            key="team-update-body",
            template="${body}",
        ),
        SubagentsSection(),
    ),
)

rendered = daily_update.render(UpdateParams(body="Summarize blockers, then plan execution."))

params = DispatchSubagentsParams(
    dispatches=(
        SubagentDispatch(
            summary=DelegationSummaryParams(
                reason="Gather product A metrics",
                expected_result="Table of the latest KPIs",
                may_delegate_further="no",
            ),
            recap_lines=("Pull weekly dashboard numbers", "Highlight major swings"),
        ),
    ),
)

result = dispatch_subagents(params, context=...)
```
