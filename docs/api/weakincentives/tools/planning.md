# `weakincentives.tools.planning`

Session-scoped planning helpers for agents that need a lightweight task list.
These APIs are transient and scoped to a single `Session` instance.

## Data classes

### `Plan`

Tracks the current objective, overall status (`"active"`, `"completed"`, or
`"abandoned"`), and an ordered collection of `PlanStep` entries.

### `PlanStep`

Immutable representation of an individual plan step. Includes a stable
`step_id`, a short title, optional details, current `StepStatus`, and recorded
notes.

### `NewPlanStep`

Input payload used when creating or appending steps before they receive a
`step_id`.

### `SetupPlan`

Parameters for `planning_setup_plan`. Captures the plan objective and optional
initial steps.

### `AddStep`

Parameters for `planning_add_step`. Contains one or more `NewPlanStep`
instances to append to the active plan.

### `UpdateStep`

Parameters for `planning_update_step`. Identifies an existing step and provides
updated title and/or details.

### `MarkStep`

Parameters for `planning_mark_step`. Identifies an existing step, sets a new
status, and optionally appends a note.

### `ClearPlan`

Parameters for `planning_clear_plan`. Signals that the current plan should be
marked as abandoned and cleared.

### `ReadPlan`

Parameters for `planning_read_plan`. Requests the latest plan snapshot from the
session store.

## Tools

### `planning_setup_plan(params: SetupPlan) -> SetupPlan`

Validate and persist a new plan. Replaces any existing plan and seeds step
identifiers starting at `S001`.

### `planning_add_step(params: AddStep) -> AddStep`

Validate appended steps and queue them for persistence. Requires an active
plan.

### `planning_update_step(params: UpdateStep) -> UpdateStep`

Validate a step edit request and persist title/detail changes for the targeted
step.

### `planning_mark_step(params: MarkStep) -> MarkStep`

Validate a step status change, append optional notes, and toggle the plan's
completion status when all steps are done.

### `planning_clear_plan(params: ClearPlan) -> ClearPlan`

Mark the current plan as abandoned and reset the step list.

### `planning_read_plan(params: ReadPlan) -> Plan`

Return the most recent plan snapshot. Raises a validation error when no plan
has been initialised.

## Prompt integration

### `PlanningToolsSection`

Prompt section that registers reducers on the provided `Session`, exposes all
planning tools, and renders concise usage guidance for the language model.
