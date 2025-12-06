# Subagents Specification

## Goal

Enable a parent run to fork lightweight child runs in parallel without losing access to the parent's tools, session state, or telemetry. The feature should feel like a single tool call that spawns fully powered agents sharing the same persistent context.

## Guiding Principles and Scope

- The feature lives entirely in `SubagentsSection` plus the `dispatch_subagents`
  tool; no additional knobs, prompt templates, or helper sections are in scope.
- Delegation is always explicit: the language model decides when to dispatch and
  constructs every payload field at call time, keeping the parent prompt
  declarative and free of baked-in defaults.
- Children should behave like peers inside the same orchestration context unless
  a stricter isolation level is configured. Sharing tools, event streams, and
  state is the default to preserve observability and avoid setup drift.
- The specification is a living reference rather than a contract for every edge
  case. Confirm runtime behaviour in code when referencing these notes, and
  prefer `SubagentsSection` as the single entry point for future extensions.

## Building Blocks

- **`SubagentsSection`** – a prompt section that introduces the delegation tool and reminds the model that parallel work MUST be dispatched instead of executed serially.
- **`dispatch_subagents` tool** – accepts a batch of child descriptions, runs each child prompt in parallel, and returns one result object per child.

No other sections, knobs, or pre-baked templates are required. The parent prompt stays declarative; the language model decides when and how to dispatch.

## Section Requirements

`SubagentsSection` lives alongside the tool in `src/weakincentives/tools/subagents.py` and is exported from `weakincentives.tools`. When rendered it MUST:

1. Briefly explain that the parent agent can offload parallelizable steps by calling `dispatch_subagents`.
1. State that every delegation must include recap bullets so the parent can audit the child’s plan.
1. List only the `dispatch_subagents` tool in `tools()`.
1. Avoid runtime configuration—no static dispatch payloads, flags, or template data.

`SubagentsSection` accepts optional `isolation_level` and `accepts_overrides`
arguments. The section instantiates the dispatch tool with a closure that
captures the requested isolation mode so every handler invocation enforces the
configured behavior.

```python
section = SubagentsSection(
    isolation_level=SubagentIsolationLevel.NO_ISOLATION,  # default
    accepts_overrides=False,  # default
)
```

## Tool Contract

The tool handler resides in `src/weakincentives/tools/subagents.py` and exports:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from weakincentives.prompt import DelegationParams, Tool, ToolResult


class SubagentIsolationLevel(Enum):
    NO_ISOLATION = auto()
    FULL_ISOLATION = auto()


@FrozenDataclass()
class DispatchSubagentsParams:
    delegations: tuple[DelegationParams, ...]


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
    isolation_level: SubagentIsolationLevel = SubagentIsolationLevel.NO_ISOLATION,
    accepts_overrides: bool = False,
) -> Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]: ...


dispatch_subagents: Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]
```

Key rules:

- Parameters are constructed entirely by the LLM at call time. No defaults live in the section.
- `DispatchSubagentsParams` coerces the provided delegations into a tuple so downstream code can rely on immutability and
  deterministic ordering.
- Results mirror the input order.
- Each child's `output` is a plain string so downstream reducers can concatenate or summarize without extra coercion.
- Failures are captured per child via `success`/`error` while allowing healthy siblings to return normally.
- `DelegationParams` replaces the old `DelegationSummaryParams` and owns both the recap lines and summary fields so call sites have a single payload to construct.

## Isolation Levels

Isolation levels describe how much access a child run has to parent state and telemetry surfaces. They are expressed with the `SubagentIsolationLevel` enum defined alongside the tool contract. Unless a caller opts into a different mode, subagents run with **No Isolation** (`SubagentIsolationLevel.NO_ISOLATION`).

- **No Isolation (default)** – Children inherit the exact `Session` instance, event bus, and tool access the parent uses. Tool calls and state mutations occur against the shared objects so observers can watch every update in real time.
- **Full Isolation** – Each child runs inside a brand new environment. Clone the parent session via `context.session.clone()` (or the equivalent repository helper) and back it with a newly created event bus. The cloned session MUST NOT share mutable state with the parent, and the fresh event bus MUST prevent telemetry from crossing run boundaries.

## Current Behaviour and Caveats

- Spawning uses `ThreadPoolExecutor` with default worker sizing to launch
  children in parallel; results track the order of the provided delegations.
- In No Isolation mode, children share the parent's session and event bus, so
  mutations and telemetry propagate immediately. Full Isolation relies on
  session cloning and separate buses; implementations that cannot clone the
  session should treat that limitation as a precondition failure.
- The handler requires a rendered parent prompt and will fail fast if it is
  missing. Downstream consumers should treat this document as guidance; defer to
  `dispatch_subagents` and `SubagentsSection` implementations for authoritative
  runtime details.

## Runtime Flow

Every tool invocation MUST execute the following steps:

1. **Require the rendered parent prompt**. `context.rendered_prompt` is mandatory. Treat a missing prompt as an orchestrator bug and respond with a failing `ToolResult`.
1. **Resolve the isolation level**. The section wires the requested `SubagentIsolationLevel` into the tool handler via a closure. In **No Isolation** mode the handler reuses the original `context.session` and event bus so children mutate the exact same objects the parent uses and observers receive child telemetry in real time. When configured for **Full Isolation** (`SubagentIsolationLevel.FULL_ISOLATION`), clone the parent session (`context.session.clone()`) and construct a fresh event bus for each child so no state or telemetry leaks across runs.
1. **Wrap the child prompt**. Build a `DelegationPrompt` using the rendered parent prompt and the recap lines carried inside `DelegationParams`. Propagate response format metadata and tool descriptions as described in `PROMPTS.md`.
1. **Run in parallel**. Evaluate each child through `context.adapter.evaluate` using a `ThreadPoolExecutor`. Use `min(len(delegations), default_max_workers)` where `default_max_workers` matches Python's executor default when `None`.
1. **Collect per-child outcomes**. Successful executions populate `output` and set `success=True`. Exceptions are caught and converted into `success=False` with an error string, without cancelling other children.
1. **Return a structured result**. On handler-level success, return `ToolResult(value=tuple(results), success=True)`. When preconditions fail (for example, prompt not rendered, cloning unsupported), surface `ToolResult(success=False, value=None, message=...)`.

## Implementation Notes

- Keep the handler synchronous; the executor supplies concurrency.
- Avoid bespoke telemetry plumbing—reuse the shared bus in No Isolation and provision a new one in Full Isolation with the canonical cloning helper.
- Ensure the shared session instance is safe for concurrent access and already carries any adapters or configuration the parent relies on.
- Tests should cover mixed success/failure batches, verify that state written by a child is visible to the parent after completion in No Isolation, and confirm that Full Isolation leaves parent state untouched.

## Minimal Usage Sketch

```python
from dataclasses import dataclass, replace

from weakincentives.prompt import DelegationParams, MarkdownSection, Prompt
from weakincentives.tools import SubagentsSection
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
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
        SubagentsSection(),
    ),
)

rendered = daily_update.render(UpdateParams(body="Summarize blockers, then plan execution."))

params = DispatchSubagentsParams(
    delegations=(
        DelegationParams(
            reason="Gather product A metrics",
            expected_result="Table of the latest KPIs",
            may_delegate_further="no",
            recap_lines=("Pull weekly dashboard numbers", "Highlight major swings"),
        ),
    ),
)

# Default behaviour uses No Isolation, reusing the parent session and event bus.
shared_tool = build_dispatch_subagents_tool()
result = shared_tool.handler(params, context=...)

# Opt into Full Isolation by configuring the SubagentsSection itself.
isolated_section = SubagentsSection(
    isolation_level=SubagentIsolationLevel.FULL_ISOLATION
)
isolated_prompt = replace(
    daily_update,
    sections=(daily_update.sections[0], isolated_section),
)
isolated_rendered = isolated_prompt.render(
    UpdateParams(body="Summarize blockers, then plan execution.")
)
isolated_tool = build_dispatch_subagents_tool(
    isolation_level=SubagentIsolationLevel.FULL_ISOLATION
)
isolated_result = isolated_tool.handler(params, context=...)
```
