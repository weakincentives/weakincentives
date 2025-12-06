# Planning Tool Suite Specification

## Overview

The planning tool suite lets background agents maintain a single session-scoped
todo list. It is designed for lightweight execution planning: capture the
objective, keep a few ordered steps, and observe progress while staying inside
the existing session/state infrastructure. Nothing in this suite persists beyond
the session and no calendar, deadline, or multi-user concepts are exposed.

## Guiding Principles

- **Keep orchestration lightweight**: The suite is the smallest possible
  coordination layer for background agents that need to remember a handful of
  steps across tool calls.
- **Session-only provenance**: Plans exist only inside a `Session`; they die
  with the session and never write to external stores.
- **Single-plan focus**: Each session tracks exactly one plan at a time. Tools
  replace or mutate the existing snapshot but never spawn branches.
- **No scheduling semantics**: Deadlines, reminders, or dependency graphs are
  out-of-scope.
- **Deterministic state transitions**: Reducers are the only place that mutate
  plan state. Every reducer invocation emits a fresh immutable snapshot.
- **Tool-first validation**: Handlers eagerly validate inputs and normalise
  strings/collections before dispatching to reducers.
- **Minimal, stable identifiers**: Step IDs are numeric suffixes in the `S###`
  format, generated exclusively by reducers.
- **Prompt clarity over verbosity**: The prompt copy steers agents toward
  concise, ASCII-only objectives.

## Module Surface

- Planning code lives in `weakincentives.tools.planning`.
- Tool validation errors use `ToolValidationError` from `weakincentives.tools.errors`.
- `PlanningToolsSection` is the public entry point. It owns the prompt copy and
  registers all tool definitions internally.

## Session Integration

- `PlanningToolsSection` must be constructed with a `Session` instance so
  reducers register before any tools run.
- During initialisation the section registers `replace_latest` for the `Plan`
  slice.
- Reducers produce copy-on-write `Plan` instances; every invocation builds a
  new snapshot.
- Orchestrators obtain the latest plan via `select_latest(session, Plan)`.

## Data Model

Schemas are frozen dataclasses to keep reducer snapshots immutable:

```python
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

### Validation Rules

- `PlanStep.step_id` values use the `S###` format, scanning the existing plan
  for the highest numeric suffix.
- `UpdateStep` rejects empty updates; at least one of `title` or `details` must
  be non-`None`.
- String validation: ASCII payloads, trim whitespace, length limits:
  - `objective`: 1-240 characters
  - `title`: 1-160 characters
  - `details` and notes: up to 512 characters each

## Tool Contracts

All tools raise `ToolValidationError` when inputs are invalid.

| Tool | Summary | Parameters | Result | Behaviour |
| ---- | ------- | ---------- | ------ | --------- |
| `planning_setup_plan` | Create/replace the session plan | `SetupPlan` | `SetupPlan` | Replaces any existing plan, seeds sequential `step_id`s, sets status to `active`. |
| `planning_add_step` | Append steps | `AddStep` | `AddStep` | Fails if no plan or plan not active. |
| `planning_update_step` | Edit an existing step | `UpdateStep` | `UpdateStep` | Rejects empty updates. |
| `planning_mark_step` | Change step status | `MarkStep` | `MarkStep` | Auto-completes plan when all steps done. |
| `planning_clear_plan` | Abandon current plan | `ClearPlan` | `ClearPlan` | Marks status `abandoned`, resets steps. |
| `planning_read_plan` | Retrieve current plan | `ReadPlan` | `Plan` | Raises when no plan exists. |

## Planning Strategies

Orchestrators can tailor the planning section's instructional copy to different
reasoning styles via the `strategy` argument on `PlanningToolsSection`.

### API

```python
from enum import Enum, auto

class PlanningStrategy(Enum):
    REACT = auto()
    PLAN_ACT_REFLECT = auto()
    GOAL_DECOMPOSE_ROUTE_SYNTHESISE = auto()
```

The default is `PlanningStrategy.REACT` to preserve legacy behaviour.

### Strategy Templates

Each strategy represents a mindset to nudge the LLM toward. The emitted markdown
observes prompt house style (ASCII, short intro, imperative voice, quiet tone).

#### ReAct (`reason -> act -> observe -> repeat`)

- Mirrors the existing copy: encourage the agent to alternate between short
  reasoning bursts, tool calls, and observation updates.
- Remind the agent to capture observations as plan step notes.

#### Plan -> Act -> Reflect (PAR)

- Instruct the agent to outline the entire plan first, execute steps, then add
  a short reflection after each tool call or completed step.
- Reflections should be appended as plan notes or brief status updates.

#### Goal framing -> Decomposition -> Tool routing -> Synthesis

- Coach the agent to start by restating the goal in their own words.
- Break the goal into concrete sub-problems before assigning tools to each one.
- Close with guidance to synthesise the results back into a cohesive answer.

### Rendering Rules

- The section continues to emit a single markdown heading and ordered lists.
- Swap only the body text that describes the mindset; do not alter the tool
  usage references or ordering.
- The default constructor path must render exactly the existing copy so no
  external prompts change without opting into a strategy.

### Configuration

- Strategies are selected via the `strategy` argument on `PlanningToolsSection`.
- Strategy selection should flow from the session or prompt configuration
  surface.

## Prompt Template Guidance

`PlanningToolsSection` emits markdown that must:

1. Explain when to engage planning versus responding directly.
1. Describe how to initialise the plan with `planning_setup_plan`.
1. Document how to expand and refine the plan.
1. Call out that `step_id`s follow the `S###` pattern and must be referenced
   when updating or marking steps.
1. Outline status tracking with `planning_mark_step`.
1. Warn that `planning_clear_plan` discards the current plan.
1. Remind agents to stay brief, ASCII-only.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.selectors import select_latest
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
            template="Stay focused on ${objective}.",
        ),
        planning.PlanningToolsSection(
            strategy=planning.PlanningStrategy.PLAN_ACT_REFLECT
        ),
    ],
)

rendered = prompt.render(BehaviourParams(objective="triage open bug reports"))
latest_plan = select_latest(session, planning.Plan)
```

## Telemetry

No custom telemetry is required. The planning tools rely solely on the default
`ToolInvoked` events defined in `specs/EVENTS.md`.

## Testing Checklist

- Tool unit tests covering success paths and validation errors.
- Reducer tests asserting step ID continuity.
- Session integration tests ensuring the reducer stores exactly one `Plan`
  snapshot per invocation.
- Prompt snapshot tests verifying `PlanningToolsSection` renders the
  instructional template and exposes all tools.
- Snapshot tests for each strategy's rendered markdown.
- End-to-end test showing plan setup, updates, marking, and clearing.

## Caveats

- **Template drift**: Strategy copy can diverge from the canonical planning tool
  description. Periodically compare rendered markdown with this spec.
- **Section ordering**: Strategies must not assume they are the first or only
  instructions the model sees.
- **Session constraints**: Some orchestration flows may disable planning tools
  in favour of direct tool calls.
