# Planning Tool Suite Specification

## Overview

The planning tool suite lets background agents maintain a single session-scoped todo list. It captures an objective, tracks ordered steps, and observes progress within the session infrastructure. Nothing persists beyond the session.

## Rationale

- **Lightweight coordination**: Smallest possible layer for agents needing to track steps across tool calls.
- **Session-only**: Plans exist only in a `Session`; they die with the session.
- **Single-plan focus**: One plan at a time per session.
- **No scheduling**: Deadlines, reminders, or dependency graphs are out-of-scope.

## Data Model

```python
from dataclasses import dataclass, field
from typing import Literal

StepStatus = Literal["pending", "in_progress", "done"]
PlanStatus = Literal["active", "completed"]


@dataclass(slots=True, frozen=True)
class PlanStep:
    step_id: int
    title: str
    status: StepStatus


@dataclass(slots=True, frozen=True)
class Plan:
    objective: str
    status: PlanStatus
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)
```

## Tools

The suite provides 4 tools:

| Tool | Purpose | Notes |
| ---------------------- | ------------------------------- | -------------------------------------------------------------- |
| `planning_setup_plan` | Create or replace the plan | Sets objective and optional initial steps; step IDs start at 1 |
| `planning_add_step` | Append steps to active plan | Fails if no plan exists or plan is completed |
| `planning_update_step` | Modify a step's title or status | Auto-completes plan when all steps are done |
| `planning_read_plan` | Retrieve current plan state | Fails if no plan exists |

### Tool Params

```python
@dataclass(slots=True, frozen=True)
class SetupPlan:
    objective: str
    initial_steps: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class AddStep:
    steps: tuple[str, ...]  # step titles


@dataclass(slots=True, frozen=True)
class UpdateStep:
    step_id: int
    title: str | None = None
    status: StepStatus | None = None


@dataclass(slots=True, frozen=True)
class ReadPlan:
    pass
```

## Behaviour

- **Plan lifecycle**: `planning_setup_plan` creates or replaces the plan. Other tools require an existing plan.
- **Step IDs**: Simple incrementing integers (1, 2, 3...). IDs are never reused within a plan.
- **Auto-completion**: When all steps reach `done`, plan status becomes `completed`.
- **Validation**: Titles must be non-empty and \<= 500 characters. At least one of `title` or `status` required for updates.

## Session Integration

`PlanningToolsSection` registers reducers on construction. Every tool invocation produces a fresh `Plan` snapshot via `replace_latest`.

```python
from weakincentives.runtime.session import Session, select_latest
from weakincentives.tools import Plan, PlanningToolsSection

session = Session(bus=bus)
section = PlanningToolsSection(session=session)
# ... after tool calls ...
plan = select_latest(session, Plan)
```

## Prompt Guidance

The section emits markdown instructing agents to:

1. Use planning for multi-step work, skip for trivial tasks
1. Initialize with `planning_setup_plan`
1. Add steps with `planning_add_step`
1. Update progress with `planning_update_step`
1. Check state with `planning_read_plan`

## Testing Checklist

- Tool success paths and validation errors
- Step ID continuity across add operations
- Auto-completion when all steps done
- Session stores single Plan snapshot
