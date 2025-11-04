# Subagents Specification

## Purpose and Scope

Introduce a builtin prompt section that makes parallel delegation a first-class, mandatory workflow whenever a plan exposes tasks that can run concurrently. The new `SubagentsSection` contributes a `dispatch_subagents` tool and the surrounding instructional copy that forces the language model to identify parallelizable work, package it for subagents, and run those subagents in parallel. This document focuses strictly on that surface area.

## Section Authoring Guidance

`SubagentsSection` lives in `src/weakincentives/prompt/subagents.py` as a `Section` specialization and is re-exported from `weakincentives.prompt`. When rendered inside a parent prompt it MUST:

- Introduce the dispatch tool using concise prose, then explicitly state that **whenever the plan contains tasks that can execute in parallel, the model MUST call `dispatch_subagents` instead of running the steps sequentially**.
- Remind the model that every delegated child needs a recap block. The section should mention that recap bullets are mandatory and must summarize the desired checkpoints for each child run.
- Avoid static configuration: do **not** accept dispatch payloads or templated task definitions as section parameters. The section exists solely to prime the model; the actual dispatch list is constructed dynamically when the model decides to call the tool.
- Surface the single tool `dispatch_subagents` in its `tools()` implementation so the parent prompt automatically exposes it to adapters. No other tools or subsections are added by this feature.

## Tool Contract

The dispatch handler lives in a new module `src/weakincentives/tools/subagents.py` alongside the other builtin tools. Export the following dataclasses and tool instance from that module:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from weakincentives.prompt import DelegationSummaryParams, Tool, ToolResult
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.composition import DelegationPrompt
from weakincentives.prompt.tool import ToolContext


@dataclass(slots=True)
class SubagentDispatch:
    summary: DelegationSummaryParams
    recap_lines: tuple[str, ...]
    tool: str | None = None
    files: Sequence[str] | None = None


@dataclass(slots=True)
class DispatchSubagentsParams:
    dispatches: Sequence[SubagentDispatch]


@dataclass(slots=True)
class SubagentResult(SupportsDataclass):
    dispatch: SubagentDispatch
    output: object | None
    success: bool
    error: str | None = None


@dataclass(slots=True)
class SubagentResults(SupportsDataclass):
    items: tuple[SubagentResult, ...]


dispatch_subagents: Tool[DispatchSubagentsParams, SubagentResults]
```

Key aspects of the contract:

- Tool parameters are provided **only** at call time. The LLM builds a `DispatchSubagentsParams` instance when it triggers the tool; the section never supplies defaults.
- Every `SubagentDispatch` carries the metadata mandated by `PROMPTS_COMPOSITION.md`. The handler reconstructs the delegation wrapper by combining the `DelegationSummaryParams`, the parent prompt text from `ToolContext.rendered_prompt`, and the recap bullet list. Optional metadata such as `tool` and `files` is preserved so downstream systems can audit or enrich follow-up work, and the handler normalizes `files` into an immutable tuple before execution.
- The tool returns one `SubagentResult` per input in matching order. Each result wraps the original dispatch metadata, the subagent output (if evaluation succeeded), and failure information when a child run raises. Results are wrapped in a `SubagentResults` container that behaves like a tuple for iteration and indexing.

## Execution Requirements

The handler must follow these steps for every invocation:

1. **Read the parent prompt**: fetch `context.rendered_prompt`. If it is `None`, treat that as a bug in the orchestrator and surface an error—the orchestrator must render the prompt and populate `ToolContext` before tools execute (per `TOOL_CONTEXT.md`).
1. **Reuse the parent context**: execute every child with `context.session` and `context.event_bus`. Sharing the parent session and bus keeps telemetry contiguous and avoids fragmenting state across clones. When `context.session` is `None`, continue dispatching with a `None` session but still rely on the parent bus.
1. **Build the wrapper**: instantiate `DelegationPrompt` from `src/weakincentives/prompt/composition.py`, passing the parent prompt, the rendered text, and `recap_lines=dispatch.recap_lines` so the recap section always renders (even for empty tuples).
1. **Evaluate in parallel**: submit each child prompt to `context.adapter.evaluate` using a `ThreadPoolExecutor` so the children run concurrently. Size the pool to `min(len(dispatches), max_workers_default)` where `max_workers_default` aligns with Python’s default when `None`. Propagate `session=context.session` and `bus=context.event_bus` for each call so reducers and telemetry observers see a unified stream.
1. **Collect results**: capture successful outputs and wrap them in `ToolResult.success == True`. If any evaluation raises, mark the corresponding `SubagentResult.success` as `False`, populate the `error` field with the exception string, and keep the other children unaffected.
1. **Publish tool completion**: return `ToolResult(value=results, success=True)` when all children finished. If cloning fails or the parent prompt is unavailable, return `ToolResult(success=False, value=None, message=...)` so downstream reducers can handle the failure.

Throughout the execution flow, propagate the parent prompt’s tool descriptions and response format metadata to the `DelegationPrompt` so subagents inherit every adapter capability exactly as described in `PROMPTS_COMPOSITION.md`.

## Implementation Sketch

The snippet below highlights how the handler orchestrates the dispatch. It is illustrative rather than executable but uses the actual runtime APIs described above.

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, cast

from weakincentives.prompt import DelegationPrompt, ToolResult
from weakincentives.prompt.tool import ToolContext
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentDispatch,
    SubagentResult,
    SubagentResults,
)


def _run_single_dispatch(
    dispatch: SubagentDispatch,
    *,
    context: ToolContext,
) -> SubagentResult:
    rendered_parent = context.rendered_prompt
    if rendered_parent is None:
        raise RuntimeError("Parent prompt was not rendered before dispatch_subagents.")

    parent_output_type = cast(type[object], rendered_parent.output_type or object)
    child_prompt_cls = DelegationPrompt[parent_output_type, object]
    child_prompt = child_prompt_cls(
        parent_prompt=context.prompt,
        rendered_parent=rendered_parent,
        include_response_format=rendered_parent.container is not None,
        recap_lines=dispatch.recap_lines,
    )

    result = context.adapter.evaluate(
        child_prompt.prompt,
        dispatch.summary,
        bus=context.event_bus,
        session=context.session,
    )
    return SubagentResult(
        dispatch=dispatch,
        output=result.output,
        success=True,
    )


def handle_dispatch_subagents(
    params: DispatchSubagentsParams,
    *,
    context: ToolContext,
) -> ToolResult[SubagentResults]:
    try:
        with ThreadPoolExecutor(max_workers=len(params.dispatches) or None) as executor:
            futures = [
                executor.submit(_run_single_dispatch, dispatch, context=context)
                for dispatch in params.dispatches
            ]
        results = [future.result() for future in futures]
    except Exception as error:
        return ToolResult(success=False, message=str(error))
    payload = SubagentResults(items=tuple(results))
    return ToolResult(value=payload, success=True)
```

The production implementation must tighten error handling (for example, using per-child `try` blocks to keep healthy dispatches alive) and rely on the shared session and event bus exposed by the parent context.

## Usage Example

The following example shows how to add the section to a prompt and how an agent might invoke the tool at runtime. All imports rely on existing modules.

```python
from __future__ import annotations

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

# Later, during execution, the language model recognizes two parallel tasks and
# issues the following tool call (constructed on the fly):
tool_call = DispatchSubagentsParams(
    dispatches=(
        SubagentDispatch(
            summary=DelegationSummaryParams(
                reason="Gather product A metrics",
                expected_result="Table of the latest KPIs",
                may_delegate_further="no",
            ),
            recap_lines=("Pull weekly dashboard numbers", "Highlight major swings"),
            tool="python_evaluate",
        ),
        SubagentDispatch(
            summary=DelegationSummaryParams(
                reason="Digest customer feedback",
                expected_result="Bullet summary of key themes",
                may_delegate_further="no",
            ),
            recap_lines=("Cluster similar feedback", "Escalate urgent issues"),
            files=("sunfish/README.md",),
        ),
    ),
)

current_tool_context = ...  # Provided by the orchestrator at runtime.
result = dispatch_subagents(tool_call, context=current_tool_context)
```

The example emphasizes that:

- The prompt author only instantiates `SubagentsSection()`—no dispatch data is threaded into the prompt upfront.
- The language model is responsible for calling the tool whenever its plan reveals parallelizable steps, building `DispatchSubagentsParams` dynamically with summary metadata and recap bullets.
- Each dispatched child inherits the full rendered parent prompt while the handler manages parallel execution, shared session telemetry, and recap propagation automatically.
