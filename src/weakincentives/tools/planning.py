# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planning tool suite for session-scoped execution plans."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, cast

from ..prompt import SupportsDataclass
from ..prompt.section import Section
from ..prompt.tool import Tool, ToolResult
from ..session import Session, replace_latest, select_latest
from ..session.session import DataEvent
from .errors import ToolValidationError

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


@dataclass(slots=True, frozen=True)
class _PlanningSectionParams:
    """Placeholder params container for the planning tools section."""

    pass


_ASCII: Final[str] = "ascii"
_MAX_OBJECTIVE_LENGTH: Final[int] = 240
_MAX_TITLE_LENGTH: Final[int] = 160
_MAX_DETAIL_LENGTH: Final[int] = 512
_STEP_ID_PREFIX: Final[str] = "S"


class PlanningToolsSection(Section[_PlanningSectionParams]):
    """Prompt section exposing the planning tool suite."""

    def __init__(self, *, session: Session) -> None:
        self._session = session
        session.register_reducer(Plan, replace_latest)
        session.register_reducer(SetupPlan, _setup_plan_reducer, slice_type=Plan)
        session.register_reducer(AddStep, _add_step_reducer, slice_type=Plan)
        session.register_reducer(UpdateStep, _update_step_reducer, slice_type=Plan)
        session.register_reducer(MarkStep, _mark_step_reducer, slice_type=Plan)
        session.register_reducer(ClearPlan, _clear_plan_reducer, slice_type=Plan)

        tools = _build_tools(session)
        super().__init__(
            title="Planning Tools",
            key="planning.tools",
            default_params=_PlanningSectionParams(),
            tools=tools,
        )

    def render(self, params: _PlanningSectionParams, depth: int) -> str:
        del params, depth
        return (
            "Use planning tools for multi-step or stateful work that requires an "
            "execution plan.\n"
            "- Start with `planning.setup_plan` to set an objective (<=240 ASCII "
            "characters) and optional initial steps.\n"
            "- Keep steps concise (<=160 ASCII characters for titles, <=512 for "
            "details).\n"
            "- Extend plans with `planning.add_step` and refine steps with "
            "`planning.update_step`.\n"
            "- Track progress via `planning.mark_step` (pending, in_progress, "
            "blocked, done).\n"
            "- Inspect the latest plan using `planning.read_plan`.\n"
            "- Use `planning.clear_plan` only when abandoning the objective.\n"
            "Stay brief, ASCII-only, and skip planning for trivial single-step "
            "tasks."
        )


def _build_tools(
    session: Session,
) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
    suite = _PlanningToolSuite(session)
    return (
        Tool[SetupPlan, SetupPlan](
            name="planning.setup_plan",
            description="Create or replace the session plan.",
            handler=suite.setup_plan,
        ),
        Tool[AddStep, AddStep](
            name="planning.add_step",
            description="Append one or more steps to the active plan.",
            handler=suite.add_step,
        ),
        Tool[UpdateStep, UpdateStep](
            name="planning.update_step",
            description="Edit the title or details for an existing step.",
            handler=suite.update_step,
        ),
        Tool[MarkStep, MarkStep](
            name="planning.mark_step",
            description="Update step status and optionally record a note.",
            handler=suite.mark_step,
        ),
        Tool[ClearPlan, ClearPlan](
            name="planning.clear_plan",
            description="Mark the current plan as abandoned.",
            handler=suite.clear_plan,
        ),
        Tool[ReadPlan, Plan](
            name="planning.read_plan",
            description="Return the latest plan snapshot.",
            handler=suite.read_plan,
        ),
    )  # type: ignore[return-value]


class _PlanningToolSuite:
    """Collection of tool handlers bound to a session instance."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def setup_plan(self, params: SetupPlan) -> ToolResult[SetupPlan]:
        objective = _normalize_required_text(
            params.objective,
            field_name="objective",
            max_length=_MAX_OBJECTIVE_LENGTH,
        )
        initial_steps = _normalize_new_steps(params.initial_steps)
        normalized = SetupPlan(objective=objective, initial_steps=initial_steps)
        step_count = len(initial_steps)
        message = (
            f"Plan initialised with {step_count} step{'s' if step_count != 1 else ''}."
        )
        return ToolResult(message=message, value=normalized)

    def add_step(self, params: AddStep) -> ToolResult[AddStep]:
        plan = _require_plan(self._session)
        _ensure_active(plan, "add steps to")
        normalized_steps = _normalize_new_steps(params.steps)
        if not normalized_steps:
            raise ToolValidationError("Provide at least one step to add.")
        message = (
            "Queued"
            f" {len(normalized_steps)} step{'s' if len(normalized_steps) != 1 else ''}"
            " for addition."
        )
        return ToolResult(message=message, value=AddStep(steps=normalized_steps))

    def update_step(self, params: UpdateStep) -> ToolResult[UpdateStep]:
        plan = _require_plan(self._session)
        _ensure_active(plan, "update steps in")
        step_id = params.step_id.strip()
        if not step_id:
            raise ToolValidationError("Step ID must be provided.")
        _ensure_ascii(step_id, "step_id")
        updated_title = (
            _normalize_required_text(
                params.title,
                field_name="title",
                max_length=_MAX_TITLE_LENGTH,
            )
            if params.title is not None
            else None
        )
        updated_details = (
            _normalize_optional_text(
                params.details,
                field_name="details",
                max_length=_MAX_DETAIL_LENGTH,
            )
            if params.details is not None
            else None
        )
        if updated_title is None and updated_details is None:
            raise ToolValidationError(
                "Provide a new title or details to update a step."
            )
        _ensure_step_exists(plan, step_id)
        normalized = UpdateStep(
            step_id=step_id,
            title=updated_title,
            details=updated_details,
        )
        return ToolResult(message=f"Step {step_id} update queued.", value=normalized)

    def mark_step(self, params: MarkStep) -> ToolResult[MarkStep]:
        plan = _require_plan(self._session)
        if plan.status == "abandoned":
            raise ToolValidationError("Cannot mark steps on an abandoned plan.")
        step_id = params.step_id.strip()
        if not step_id:
            raise ToolValidationError("Step ID must be provided.")
        _ensure_ascii(step_id, "step_id")
        _ensure_step_exists(plan, step_id)
        note = _normalize_optional_text(
            params.note,
            field_name="note",
            max_length=_MAX_DETAIL_LENGTH,
            require_content=True,
        )
        normalized = MarkStep(step_id=step_id, status=params.status, note=note)
        return ToolResult(
            message=f"Step {step_id} marked as {params.status}.",
            value=normalized,
        )

    def clear_plan(self, params: ClearPlan) -> ToolResult[ClearPlan]:
        plan = _require_plan(self._session)
        if plan.status == "abandoned":
            raise ToolValidationError("Plan already abandoned.")
        return ToolResult(message="Plan marked as abandoned.", value=params)

    def read_plan(self, params: ReadPlan) -> ToolResult[Plan]:
        del params
        plan = select_latest(self._session, Plan)
        if plan is None:
            raise ToolValidationError("No plan is currently initialised.")
        step_summary = _summarize_steps(plan.steps)
        message = (
            f"Objective: {plan.objective}\nStatus: {plan.status}\nSteps: {step_summary}"
        )
        return ToolResult(message=message, value=plan)


def _setup_plan_reducer(
    slice_values: tuple[Plan, ...], event: DataEvent
) -> tuple[Plan, ...]:
    params = cast(SetupPlan, event.value)
    steps = tuple(
        PlanStep(
            step_id=_format_step_id(index + 1),
            title=step.title,
            details=step.details,
            status="pending",
            notes=(),
        )
        for index, step in enumerate(params.initial_steps)
    )
    plan = Plan(objective=params.objective, status="active", steps=steps)
    return (plan,)


def _add_step_reducer(
    slice_values: tuple[Plan, ...], event: DataEvent
) -> tuple[Plan, ...]:
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    params = cast(AddStep, event.value)
    existing = list(previous.steps)
    next_index = _next_step_index(existing)
    for step in params.steps:
        next_index += 1
        existing.append(
            PlanStep(
                step_id=_format_step_id(next_index),
                title=step.title,
                details=step.details,
                status="pending",
                notes=(),
            )
        )
    updated = Plan(
        objective=previous.objective,
        status="active",
        steps=tuple(existing),
    )
    return (updated,)


def _update_step_reducer(
    slice_values: tuple[Plan, ...], event: DataEvent
) -> tuple[Plan, ...]:
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    params = cast(UpdateStep, event.value)
    updated_steps: list[PlanStep] = []
    for step in previous.steps:
        if step.step_id != params.step_id:
            updated_steps.append(step)
            continue
        new_title = params.title if params.title is not None else step.title
        new_details = params.details if params.details is not None else step.details
        updated_steps.append(
            PlanStep(
                step_id=step.step_id,
                title=new_title,
                details=new_details,
                status=step.status,
                notes=step.notes,
            )
        )
    updated_plan = Plan(
        objective=previous.objective,
        status=previous.status,
        steps=tuple(updated_steps),
    )
    return (updated_plan,)


def _mark_step_reducer(
    slice_values: tuple[Plan, ...], event: DataEvent
) -> tuple[Plan, ...]:
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    params = cast(MarkStep, event.value)
    updated_steps: list[PlanStep] = []
    for step in previous.steps:
        if step.step_id != params.step_id:
            updated_steps.append(step)
            continue
        notes = step.notes
        if params.note is not None:
            notes = (*notes, params.note)
        updated_steps.append(
            PlanStep(
                step_id=step.step_id,
                title=step.title,
                details=step.details,
                status=params.status,
                notes=notes,
            )
        )
    plan_status: PlanStatus
    if not updated_steps or all(step.status == "done" for step in updated_steps):
        plan_status = "completed"
    else:
        plan_status = "active"
    updated_plan = Plan(
        objective=previous.objective,
        status=plan_status,
        steps=tuple(updated_steps),
    )
    return (updated_plan,)


def _clear_plan_reducer(
    slice_values: tuple[Plan, ...], event: DataEvent
) -> tuple[Plan, ...]:
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    del event
    abandoned = Plan(objective=previous.objective, status="abandoned", steps=())
    return (abandoned,)


def _latest_plan(plans: tuple[Plan, ...]) -> Plan | None:
    if not plans:
        return None
    return plans[-1]


def _next_step_index(steps: Iterable[PlanStep]) -> int:
    max_index = 0
    for step in steps:
        suffix = step.step_id[len(_STEP_ID_PREFIX) :]
        try:
            max_index = max(max_index, int(suffix))
        except ValueError:
            continue
    return max_index


def _format_step_id(index: int) -> str:
    return f"{_STEP_ID_PREFIX}{index:03d}"


def _normalize_new_steps(steps: Sequence[NewPlanStep]) -> tuple[NewPlanStep, ...]:
    normalized: list[NewPlanStep] = []
    for step in steps:
        title = _normalize_required_text(
            step.title,
            field_name="title",
            max_length=_MAX_TITLE_LENGTH,
        )
        details = _normalize_optional_text(
            step.details,
            field_name="details",
            max_length=_MAX_DETAIL_LENGTH,
        )
        normalized.append(NewPlanStep(title=title, details=details))
    return tuple(normalized)


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ToolValidationError(f"{field_name.title()} must not be empty.")
    if len(stripped) > max_length:
        raise ToolValidationError(
            f"{field_name.title()} must be <= {max_length} characters."
        )
    _ensure_ascii(stripped, field_name)
    return stripped


def _normalize_optional_text(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
    require_content: bool = False,
) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        if require_content:
            raise ToolValidationError(
                f"{field_name.title()} must not be empty when provided."
            )
        return None
    if len(stripped) > max_length:
        raise ToolValidationError(
            f"{field_name.title()} must be <= {max_length} characters."
        )
    _ensure_ascii(stripped, field_name)
    return stripped


def _ensure_ascii(value: str, field_name: str) -> None:
    try:
        value.encode(_ASCII)
    except UnicodeEncodeError as error:  # pragma: no cover - defensive
        raise ToolValidationError(f"{field_name.title()} must be ASCII.") from error


def _require_plan(session: Session) -> Plan:
    plan = select_latest(session, Plan)
    if plan is None:
        raise ToolValidationError("No plan is currently initialised.")
    return plan


def _ensure_active(plan: Plan, action: str) -> None:
    if plan.status != "active":
        raise ToolValidationError(
            f"Plan must be active to {action}. Current status: {plan.status}."
        )


def _ensure_step_exists(plan: Plan, step_id: str) -> None:
    if not any(step.step_id == step_id for step in plan.steps):
        raise ToolValidationError(f"Step {step_id} does not exist.")


def _summarize_steps(steps: Sequence[PlanStep]) -> str:
    if not steps:
        return "no steps recorded"
    summaries = [f"{step.step_id}:{step.status}" for step in steps]
    return ", ".join(summaries)


__all__ = [
    "Plan",
    "PlanStatus",
    "PlanStep",
    "StepStatus",
    "NewPlanStep",
    "SetupPlan",
    "AddStep",
    "UpdateStep",
    "MarkStep",
    "ClearPlan",
    "ReadPlan",
    "PlanningToolsSection",
]
