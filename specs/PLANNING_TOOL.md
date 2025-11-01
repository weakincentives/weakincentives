# Planning Tool Suite Specification

## Overview

The planning tool suite lets background agents maintain a single session-scoped todo list. It is designed for lightweight
execution planning: capture the objective, keep a few ordered steps, and observe progress while staying inside the
existing session/state infrastructure. Nothing in this suite persists beyond the session and no calendar, deadline, or
multi-user concepts are exposed.

## Module Surface

- Planning code lives in `weakincentives.tools.planning`.
- Tool validation errors use `ToolValidationError` from `weakincentives.tools.errors`. This exception acts as the shared
  params validation failure for every built-in tool.
- `PlanningToolsSection` is the public entry point. It owns the deskilled prompt copy and registers all tool definitions
  internally, so callers integrate the suite by adding the section rather than importing individual tool functions.

## Session Integration

- `PlanningToolsSection` requires a `Session` instance from `weakincentives.session`.
- During initialisation the section registers `replace_latest` for the `Plan` slice so every tool result replaces the
  current plan snapshot automatically.
- Reducers produce copy-on-write `Plan` instances; every invocation builds a new snapshot that replaces the previous
  tuple in the session store so orchestrators always observe a single, immutable plan.
- Orchestrators obtain the latest plan via `select_latest(session, Plan)` and may fall back to `None` when no plan exists.

## Data Model

The schemas below are defined as frozen dataclasses to keep reducer snapshots immutable and trivially comparable. Lists
exposed through tool params are normalised into tuples when persisted.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

PlanStatus = Literal["active", "completed", "abandoned"]
StepStatus = Literal["pending", "in_progress", "blocked", "done"]


@dataclass(slots=True, frozen=True)
class PlanStep:
    step_id: str
    title: str
    details: str | None
    status: StepStatus
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class Plan:
    objective: str
    status: PlanStatus
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class NewPlanStep:
    title: str
    details: str | None = None


@dataclass(slots=True, frozen=True)
class SetupPlan:
    objective: str
    initial_steps: tuple[NewPlanStep, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class AddStep:
    steps: tuple[NewPlanStep, ...]


@dataclass(slots=True, frozen=True)
class UpdateStep:
    step_id: str
    title: str | None = None
    details: str | None = None


@dataclass(slots=True, frozen=True)
class MarkStep:
    step_id: str
    status: StepStatus
    note: str | None = None


@dataclass(slots=True, frozen=True)
class ClearPlan:
    pass


@dataclass(slots=True, frozen=True)
class ReadPlan:
    pass
```

Implementation detail notes:

- Tool handlers perform parameter validation and return informative messages alongside the original params dataclass.
- Reducers are responsible for merging tool params into the latest `Plan` snapshot, constructing a fresh instance on
  every invocation before persisting it via `replace_latest`.
- `ReadPlan` is the sole tool that returns a `Plan`. Its handler is defined as a closure over the target `Session`
  instance so it can call `select_latest` and render a short summary message for the LLM.
- `NewPlanStep` sequences supplied by callers can be any `Sequence` at runtime; handlers coerce them into tuples before
  storing the new snapshot.
- `UpdateStep` rejects empty updates; at least one of `title` or `details` must be non-`None`.
- String validation rules enforce ASCII payloads, trim surrounding whitespace, and limit lengths to:
  - `objective`: 1–240 characters
  - `title`: 1–160 characters
  - `details` and notes: ≤512 characters each
- `PlanStep.step_id` remains stable across updates to keep `MarkStep` invocations idempotent for the caller.

## Tool Contracts

All tools raise `ToolValidationError` when inputs are invalid. Every successful call writes the updated `Plan` through
the session reducer pipeline.

| Tool | Summary | Parameters | Result | Behaviour highlights |
| ---- | ------- | ---------- | ------ | -------------------- |
| `planning_setup_plan` | Create or replace the session plan | `SetupPlan` | `SetupPlan` | Emits validated setup payload; reducer replaces any existing plan, seeds sequential `step_id`s (`S001`, `S002`, …), sets status to `active`. |
| `planning_add_step` | Append one or more steps | `AddStep` | `AddStep` | Fails if no plan or plan not active, appends validated steps in order when reducer applies the update. |
| `planning_update_step` | Edit an existing step description | `UpdateStep` | `UpdateStep` | Rejects empty updates, trims strings; reducer mutates the target step before persisting snapshot. |
| `planning_mark_step` | Change a step status | `MarkStep` | `MarkStep` | Validates step/status, attaches optional note; reducer auto-completes plan when all steps done. |
| `planning_clear_plan` | Abandon the current plan | `ClearPlan` | `ClearPlan` | Signals abandonment; reducer marks status `abandoned` and resets to an empty step list. |
| `planning_read_plan` | Retrieve current plan state | `ReadPlan` | `Plan` | Returns the latest reducer-managed plan or raises when the plan has never been initialised. |

## Prompt Template Guidance

`PlanningToolsSection` emits markdown that must:

1. Explain when to engage planning (multi-step or stateful work) versus responding directly.
1. Describe how to initialise the plan with `planning_setup_plan` and keep the objective concise.
1. Document how to expand and refine the plan (`planning_add_step`, `planning_update_step`).
1. Outline status tracking with `planning_mark_step` and how to inspect progress with `planning_read_plan`.
1. Warn that `planning_clear_plan` discards the current plan and should be used sparingly.
1. Remind agents to stay brief, ASCII-only, and avoid planning trivial single-step tasks.

The section follows the standard prompt system rules (see `specs/PROMPTS.md`) and contributes tool definitions so
`RenderedPrompt.tools` contains the planning suite without additional orchestration code.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.session import Session
from weakincentives.session.selectors import select_latest
from weakincentives.tools import planning

@dataclass
class BehaviourParams:
    objective: str

bus = InProcessEventBus()
session = Session(bus=bus)

prompt = Prompt(
    ns="agents/background",
    key="session-plan",
    name="session-plan",
    sections=[
        MarkdownSection[BehaviourParams](
            title="Behaviour",
            template="Stay focused on ${objective} and call planning tools for multi-step work.",
        ),
        planning.PlanningToolsSection(session=session),
    ],
)

rendered = prompt.render(BehaviourParams(objective="triage open bug reports"))
latest_plan = select_latest(session, planning.Plan)
```

Adapters dispatch `ToolInvoked` events onto the shared bus; the session handles persistence automatically via the reducer
registered by the section.

## Telemetry

No custom telemetry is required. The planning tools rely solely on the default `ToolInvoked` events defined in
`specs/EVENTS.md`.

## Testing Checklist

- Tool unit tests covering success paths and validation errors (missing plan, invalid step ID, empty patch, etc.).
- Session integration tests ensuring the reducer stores exactly one `Plan` snapshot per invocation.
- Prompt snapshot tests verifying `PlanningToolsSection` renders the instructional template and exposes all tools.
- End-to-end test showing an agent setting up a plan, updating it, marking completion, and clearing it at the end.

## Documentation Tasks

- Add `examples/openai_runner.py` demonstrating the typical flow using the openai adapter.
- Update the README to reference the planning suite and link to this specification.
- Generate API reference entries for all dataclasses and tool functions in `weakincentives.tools.planning`.
