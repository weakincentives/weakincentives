# Subagents Specification

## Goal

Enable a parent run to dispatch lightweight child runs in parallel. Each child
receives a pre-populated Plan in its isolated session and is instructed to
complete all steps before returning. Full isolation is the default mode.

## Guiding Principles and Scope

- The feature lives entirely in `SubagentsSection` plus the `dispatch_subagents`
  tool; no additional knobs, prompt templates, or helper sections are in scope.
- Dispatch is centered around the `Plan` object from the planning tools. Each
  task becomes a Plan in the subagent's cloned session.
- Full isolation is the default: each child gets its own cloned session and
  event bus, preventing state or telemetry from crossing run boundaries.
- Children complete all plan steps before returning, enabling structured
  parallel task decomposition.
- The specification is a living reference rather than a contract for every edge
  case. Confirm runtime behaviour in code when referencing these notes, and
  prefer `SubagentsSection` as the single entry point for future extensions.

## Building Blocks

- **`SubagentsSection`** – a prompt section that introduces the delegation tool
  and reminds the model that parallel work MUST be dispatched instead of
  executed serially.
- **`dispatch_subagents` tool** – accepts a batch of tasks, runs each in
  parallel with its own Plan, and returns one result object per task.
- **`SubagentTask`** – describes a task as an objective and ordered steps, which
  become a Plan in the child's session.

No other sections, knobs, or pre-baked templates are required. The parent prompt
stays declarative; the language model decides when and how to dispatch.

## Section Requirements

`SubagentsSection` lives alongside the tool in
`src/weakincentives/tools/subagents.py` and is exported from
`weakincentives.tools`. When rendered it MUST:

1. Briefly explain that the parent agent can offload parallelizable steps by
   calling `dispatch_subagents`.
1. State that each task becomes a Plan in the subagent's isolated session.
1. List only the `dispatch_subagents` tool in `tools()`.
1. Avoid runtime configuration—no static dispatch payloads, flags, or template
   data.

`SubagentsSection` accepts optional `isolation_level` and `accepts_overrides`
arguments. The section instantiates the dispatch tool with a closure that
captures the requested isolation mode. Full isolation is the default.

```python
section = SubagentsSection(
    isolation_level=SubagentIsolationLevel.FULL_ISOLATION,  # default
    accepts_overrides=False,  # default
)
```

## Tool Contract

The tool handler resides in `src/weakincentives/tools/subagents.py` and exports:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.prompt import Tool, ToolResult


class SubagentIsolationLevel(Enum):
    NO_ISOLATION = auto()
    FULL_ISOLATION = auto()


@FrozenDataclass()
class SubagentTask:
    """A task to delegate to a subagent, expressed as a plan."""
    objective: str
    steps: tuple[str, ...] = ()


@FrozenDataclass()
class DispatchSubagentsParams:
    """Parameters describing the tasks to delegate."""
    tasks: tuple[SubagentTask, ...]


@FrozenDataclass()
class SubagentResult:
    output: str
    success: bool
    error: str | None = None

    def render(self) -> str:
        """Render result for display."""
        ...


def build_dispatch_subagents_tool(
    *,
    isolation_level: SubagentIsolationLevel = SubagentIsolationLevel.FULL_ISOLATION,
    accepts_overrides: bool = False,
) -> Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]: ...


dispatch_subagents: Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]
```

Key rules:

- Parameters are constructed entirely by the LLM at call time. No defaults live
  in the section.
- `DispatchSubagentsParams` coerces the provided tasks into a tuple for
  immutability and deterministic ordering.
- Each `SubagentTask` specifies an objective and steps that form a Plan for the
  subagent to complete.
- Results mirror the input order.
- Each child's `output` is a plain string so downstream reducers can concatenate
  or summarize without extra coercion.
- Failures are captured per child via `success`/`error` while allowing healthy
  siblings to return normally.

## Isolation Levels

Isolation levels describe how much access a child run has to parent state and
telemetry surfaces. They are expressed with the `SubagentIsolationLevel` enum.

- **Full Isolation (default)** – Each child runs inside a cloned session with a
  fresh event bus. The child receives a pre-populated Plan containing the task's
  objective and steps. The cloned session MUST NOT share mutable state with the
  parent, and the fresh event bus MUST prevent telemetry from crossing run
  boundaries.
- **No Isolation** – Children inherit the exact `Session` instance, event bus,
  and tool access the parent uses. Tool calls and state mutations occur against
  the shared objects. Plan injection is skipped since children share state.

## Plan Injection

When using Full Isolation (the default), each child session receives:

1. A cloned session via `session.clone()` with the parent as the parent
   reference.
1. Planning reducers initialized via
   `PlanningToolsSection._initialize_session()`.
1. A Plan pre-populated with:
   - `objective`: The task's objective
   - `status`: "active"
   - `steps`: One PlanStep per task step, with incrementing IDs and "pending"
     status

The delegation prompt instructs the child to complete all plan steps before
returning, using planning tools to track progress.

## Current Behaviour and Caveats

- Spawning uses `ThreadPoolExecutor` with default worker sizing to launch
  children in parallel; results track the order of the provided tasks.
- In Full Isolation mode (default), children get cloned sessions with injected
  Plans. Sessions that cannot be cloned will cause a precondition failure.
- In No Isolation mode, children share the parent's session and event bus, so
  mutations and telemetry propagate immediately.
- The handler requires a rendered parent prompt and will fail fast if it is
  missing. Downstream consumers should treat this document as guidance; defer to
  `dispatch_subagents` and `SubagentsSection` implementations for authoritative
  runtime details.

## Runtime Flow

Every tool invocation MUST execute the following steps:

1. **Require the rendered parent prompt**. `context.rendered_prompt` is
   mandatory. Treat a missing prompt as an orchestrator bug and respond with a
   failing `ToolResult`.
1. **Resolve the isolation level**. The section wires the requested
   `SubagentIsolationLevel` into the tool handler via a closure. In **Full
   Isolation** mode (default), clone the parent session and construct a fresh
   event bus for each child, then inject the task's Plan. In **No Isolation**
   mode, reuse the original `context.session` and event bus.
1. **Build the delegation prompt**. Create a prompt that instructs the child to
   complete its pre-populated Plan using planning tools. Include parent tools
   for the child to use.
1. **Run in parallel**. Evaluate each child through `context.adapter.evaluate`
   using a `ThreadPoolExecutor`. Use `min(len(tasks), default_max_workers)`
   where `default_max_workers` matches Python's executor default.
1. **Collect per-child outcomes**. Successful executions populate `output` and
   set `success=True`. Exceptions are caught and converted into `success=False`
   with an error string, without cancelling other children.
1. **Return a structured result**. On handler-level success, return
   `ToolResult(value=tuple(results), success=True)`. When preconditions fail,
   surface `ToolResult(success=False, value=None, message=...)`.

## Implementation Notes

- Keep the handler synchronous; the executor supplies concurrency.
- Full isolation provisions a cloned session with an injected Plan for each
  child; No Isolation reuses the shared session without Plan injection.
- Budget tracker is always shared with children regardless of isolation mode.
- Ensure the session instance is safe for concurrent access.
- Tests should cover mixed success/failure batches, verify Plan injection in
  Full Isolation, and confirm that No Isolation shares state.

## Minimal Usage Sketch

```python
from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.tools import SubagentsSection
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    SubagentTask,
    build_dispatch_subagents_tool,
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
        SubagentsSection(),  # Full isolation is the default
    ),
)

rendered = daily_update.render(UpdateParams(body="Summarize blockers."))

params = DispatchSubagentsParams(
    tasks=(
        SubagentTask(
            objective="Gather product A metrics",
            steps=(
                "Pull weekly dashboard numbers",
                "Highlight major swings",
            ),
        ),
        SubagentTask(
            objective="Draft FAQ responses",
            steps=(
                "Review ticket summaries",
                "Draft five FAQ answers",
            ),
        ),
    ),
)

# Default behaviour uses Full Isolation with Plan injection
tool = build_dispatch_subagents_tool()
result = tool.handler(params, context=...)

# Opt into No Isolation by configuring the SubagentsSection
shared_section = SubagentsSection(
    isolation_level=SubagentIsolationLevel.NO_ISOLATION
)
```
