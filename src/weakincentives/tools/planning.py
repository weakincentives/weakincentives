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
from enum import Enum
from typing import Final, Literal, cast, override
from weakref import WeakSet

from ..prompt import SupportsDataclass
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..session import ReducerEvent, Session, replace_latest, select_latest
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


class PlanningStrategy(Enum):
    """Predefined guidance templates for the planning section."""

    REACT = "react"
    PLAN_ACT_REFLECT = "plan_act_reflect"
    GOAL_DECOMPOSE_ROUTE_SYNTHESISE = "goal_decompose_route_synthesise"


_PLANNING_SECTION_HEADER: Final[str] = (
    "Use planning tools for multi-step or stateful work that requires an"
    " execution plan.\n"
)
_PLANNING_SECTION_BODY: Final[str] = (
    "- Start with `planning_setup_plan` to set an objective (<=240 ASCII"
    " characters) and optional initial steps.\n"
    "- Keep steps concise (<=160 ASCII characters for titles, <=512 for"
    " details).\n"
    "- Extend plans with `planning_add_step` and refine steps with"
    " `planning_update_step`.\n"
    "- Track progress via `planning_mark_step` (pending, in_progress,\n"
    " blocked, done).\n"
    "- Inspect the latest plan using `planning_read_plan`.\n"
    "- Use `planning_clear_plan` only when abandoning the objective.\n"
    "Stay brief, ASCII-only, and skip planning for trivial single-step tasks."
)


def _template_for_strategy(strategy: PlanningStrategy) -> str:
    guidance_map: dict[PlanningStrategy, str] = {
        PlanningStrategy.REACT: "",
        PlanningStrategy.PLAN_ACT_REFLECT: (
            "Follow a plan-act-reflect rhythm: outline the entire plan before"
            " executing any tools, then work through the steps.\n"
            "After each tool call or completed step, append a brief reflection"
            " as plan notes or status updates so progress stays visible.\n"
        ),
        PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE: (
            "Begin by restating the goal in your own words to ensure the"
            " objective is clear.\n"
            "Break the goal into concrete sub-problems before routing tools to"
            " each one, and record the routing in the plan steps.\n"
            "When every tool has run, synthesise the results into a cohesive"
            " answer and update the plan status as part of that synthesis.\n"
        ),
    }
    guidance = guidance_map[strategy]
    if not guidance:
        return f"{_PLANNING_SECTION_HEADER}{_PLANNING_SECTION_BODY}"
    return f"{_PLANNING_SECTION_HEADER}{guidance}{_PLANNING_SECTION_BODY}"


class PlanningToolsSection(MarkdownSection[_PlanningSectionParams]):
    """Prompt section exposing the planning tool suite."""

    def __init__(
        self,
        *,
        strategy: PlanningStrategy = PlanningStrategy.REACT,
        accepts_overrides: bool = False,
    ) -> None:
        self._strategy = strategy
        self._configured_sessions: WeakSet[Session] = WeakSet()

        tools = _build_tools(section=self, accepts_overrides=accepts_overrides)
        super().__init__(
            title="Planning Tools",
            key="planning.tools",
            template=_template_for_strategy(strategy),
            default_params=_PlanningSectionParams(),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    def ensure_session(self, context: ToolContext) -> Session:
        session = context.session
        if not isinstance(session, Session):
            message = "PlanningToolsSection requires ToolContext.session to be a Session instance."
            raise ToolValidationError(message)
        if session not in self._configured_sessions:
            self._initialize_session(session)
            self._configured_sessions.add(session)
        return session

    def _initialize_session(self, session: Session) -> None:
        session.register_reducer(Plan, replace_latest)
        session.register_reducer(SetupPlan, _setup_plan_reducer, slice_type=Plan)
        session.register_reducer(AddStep, _add_step_reducer, slice_type=Plan)
        session.register_reducer(UpdateStep, _update_step_reducer, slice_type=Plan)
        session.register_reducer(MarkStep, _mark_step_reducer, slice_type=Plan)
        session.register_reducer(ClearPlan, _clear_plan_reducer, slice_type=Plan)

    @override
    def render(self, params: _PlanningSectionParams, depth: int) -> str:
        template = _template_for_strategy(self._strategy)
        return self.render_with_template(template, params, depth)

    @override
    def original_body_template(self) -> str:
        return _template_for_strategy(self._strategy)


def _build_tools(
    *,
    section: PlanningToolsSection,
    accepts_overrides: bool,
) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
    suite = _PlanningToolSuite(section=section)
    return cast(
        tuple[Tool[SupportsDataclass, SupportsDataclass], ...],
        (
            Tool[SetupPlan, SetupPlan](
                name="planning_setup_plan",
                description="Create or replace the session plan.",
                handler=suite.setup_plan,
                accepts_overrides=accepts_overrides,
            ),
            Tool[AddStep, AddStep](
                name="planning_add_step",
                description="Append one or more steps to the active plan.",
                handler=suite.add_step,
                accepts_overrides=accepts_overrides,
            ),
            Tool[UpdateStep, UpdateStep](
                name="planning_update_step",
                description="Edit the title or details for an existing step.",
                handler=suite.update_step,
                accepts_overrides=accepts_overrides,
            ),
            Tool[MarkStep, MarkStep](
                name="planning_mark_step",
                description="Update step status and optionally record a note.",
                handler=suite.mark_step,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ClearPlan, ClearPlan](
                name="planning_clear_plan",
                description="Mark the current plan as abandoned.",
                handler=suite.clear_plan,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadPlan, Plan](
                name="planning_read_plan",
                description="Return the latest plan snapshot.",
                handler=suite.read_plan,
                accepts_overrides=accepts_overrides,
            ),
        ),
    )


class _PlanningToolSuite:
    """Collection of tool handlers bound to a section instance."""

    def __init__(self, *, section: PlanningToolsSection) -> None:
        super().__init__()
        self._section = section

    def setup_plan(
        self, params: SetupPlan, *, context: ToolContext
    ) -> ToolResult[SetupPlan]:
        _ = self._section.ensure_session(context)
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

    def add_step(self, params: AddStep, *, context: ToolContext) -> ToolResult[AddStep]:
        session = self._section.ensure_session(context)
        plan = _require_plan(session)
        _ensure_active(plan, "add steps to")
        normalized_steps = _normalize_new_steps(params.steps)
        if not normalized_steps:
            message = "Provide at least one step to add."
            raise ToolValidationError(message)
        message = (
            "Queued"
            f" {len(normalized_steps)} step{'s' if len(normalized_steps) != 1 else ''}"
            " for addition."
        )
        return ToolResult(message=message, value=AddStep(steps=normalized_steps))

    def update_step(
        self, params: UpdateStep, *, context: ToolContext
    ) -> ToolResult[UpdateStep]:
        session = self._section.ensure_session(context)
        plan = _require_plan(session)
        _ensure_active(plan, "update steps in")
        step_id = params.step_id.strip()
        if not step_id:
            message = "Step ID must be provided."
            raise ToolValidationError(message)
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
            message = "Provide a new title or details to update a step."
            raise ToolValidationError(message)
        _ensure_step_exists(plan, step_id)
        normalized = UpdateStep(
            step_id=step_id,
            title=updated_title,
            details=updated_details,
        )
        return ToolResult(message=f"Step {step_id} update queued.", value=normalized)

    def mark_step(
        self, params: MarkStep, *, context: ToolContext
    ) -> ToolResult[MarkStep]:
        session = self._section.ensure_session(context)
        plan = _require_plan(session)
        if plan.status == "abandoned":
            message = "Cannot mark steps on an abandoned plan."
            raise ToolValidationError(message)
        step_id = params.step_id.strip()
        if not step_id:
            message = "Step ID must be provided."
            raise ToolValidationError(message)
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

    def clear_plan(
        self, params: ClearPlan, *, context: ToolContext
    ) -> ToolResult[ClearPlan]:
        session = self._section.ensure_session(context)
        plan = _require_plan(session)
        if plan.status == "abandoned":
            message = "Plan already abandoned."
            raise ToolValidationError(message)
        return ToolResult(message="Plan marked as abandoned.", value=params)

    def read_plan(self, params: ReadPlan, *, context: ToolContext) -> ToolResult[Plan]:
        del params
        session = self._section.ensure_session(context)
        plan = select_latest(session, Plan)
        if plan is None:
            message = "No plan is currently initialised."
            raise ToolValidationError(message)
        step_count = len(plan.steps)
        if step_count == 0:
            message = "Retrieved the current plan (no steps recorded)."
        else:
            message = (
                "Retrieved the current plan "
                f"with {step_count} step{'s' if step_count != 1 else ''}."
            )
        return ToolResult(message=message, value=plan)


def _setup_plan_reducer(
    slice_values: tuple[Plan, ...], event: ReducerEvent
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
    slice_values: tuple[Plan, ...], event: ReducerEvent
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
    slice_values: tuple[Plan, ...], event: ReducerEvent
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
    slice_values: tuple[Plan, ...], event: ReducerEvent
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
    slice_values: tuple[Plan, ...], event: ReducerEvent
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
        message = f"{field_name.title()} must not be empty."
        raise ToolValidationError(message)
    if len(stripped) > max_length:
        message = f"{field_name.title()} must be <= {max_length} characters."
        raise ToolValidationError(message)
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
            message = f"{field_name.title()} must not be empty when provided."
            raise ToolValidationError(message)
        return None
    if len(stripped) > max_length:
        message = f"{field_name.title()} must be <= {max_length} characters."
        raise ToolValidationError(message)
    _ensure_ascii(stripped, field_name)
    return stripped


def _ensure_ascii(value: str, field_name: str) -> None:
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:  # pragma: no cover - defensive
        message = f"{field_name.title()} must be ASCII."
        raise ToolValidationError(message) from error


def _require_plan(session: Session) -> Plan:
    plan = select_latest(session, Plan)
    if plan is None:
        message = "No plan is currently initialised."
        raise ToolValidationError(message)
    return plan


def _ensure_active(plan: Plan, action: str) -> None:
    if plan.status != "active":
        message = f"Plan must be active to {action}. Current status: {plan.status}."
        raise ToolValidationError(message)


def _ensure_step_exists(plan: Plan, step_id: str) -> None:
    if not any(step.step_id == step_id for step in plan.steps):
        message = f"Step {step_id} does not exist."
        raise ToolValidationError(message)


__all__ = [
    "AddStep",
    "ClearPlan",
    "MarkStep",
    "NewPlanStep",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanningToolsSection",
    "ReadPlan",
    "SetupPlan",
    "StepStatus",
    "UpdateStep",
]
