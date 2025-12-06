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

from collections.abc import Sequence
from dataclasses import field
from enum import Enum
from typing import Final, Literal, cast, override

from ..dataclasses import FrozenDataclass
from ..prompt import SupportsDataclass, SupportsToolResult
from ..prompt._visibility import SectionVisibility
from ..prompt.errors import PromptRenderError
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ..runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    ReducerEventWithValue,
    Session,
    replace_latest,
)
from ._context import ensure_context_uses_session
from .errors import ToolValidationError

PlanStatus = Literal["active", "completed"]
StepStatus = Literal["pending", "in_progress", "done"]

_MAX_TITLE_LENGTH: Final[int] = 500


@FrozenDataclass()
class PlanStep:
    """Single actionable step tracked within a plan."""

    step_id: int = field(
        metadata={"description": "Stable identifier for the step (integer)."}
    )
    title: str = field(
        metadata={"description": "Concise summary of the work item (<=500 characters)."}
    )
    status: StepStatus = field(
        metadata={
            "description": "Current progress state: pending, in_progress, or done."
        }
    )

    def render(self) -> str:
        return f"{self.step_id} [{self.status}] {self.title}"


@FrozenDataclass()
class Plan:
    """Immutable snapshot of the active plan."""

    objective: str = field(
        metadata={"description": "Single-sentence objective for the session."}
    )
    status: PlanStatus = field(
        metadata={"description": "Lifecycle state: active or completed."}
    )
    steps: tuple[PlanStep, ...] = field(
        default_factory=tuple,
        metadata={"description": "Ordered collection of plan steps."},
    )

    def render(self) -> str:
        lines = [f"Objective: {self.objective}", f"Status: {self.status}", ""]
        if not self.steps:
            lines.append("<no steps>")
        else:
            lines.extend(step.render() for step in self.steps)
        return "\n".join(lines)


@FrozenDataclass()
class SetupPlan:
    """Initialise or replace the session plan."""

    objective: str = field(
        metadata={"description": "Objective the plan should accomplish."}
    )
    initial_steps: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"description": "Optional step titles to seed the plan with."},
    )

    def render(self) -> str:
        lines = [f"Setup plan: {self.objective}"]
        if self.initial_steps:
            lines.append("Initial steps:")
            lines.extend(f"- {step}" for step in self.initial_steps)
        else:
            lines.append("Initial steps: <none>")
        return "\n".join(lines)


@FrozenDataclass()
class AddStep:
    """Append new steps to the current plan."""

    steps: tuple[str, ...] = field(metadata={"description": "Step titles to add."})

    def render(self) -> str:
        if not self.steps:
            return "AddStep: <no steps provided>"
        lines = ["AddStep:"]
        lines.extend(f"- {step}" for step in self.steps)
        return "\n".join(lines)


@FrozenDataclass()
class UpdateStep:
    """Modify a step's title or status."""

    step_id: int = field(metadata={"description": "Identifier of the step to update."})
    title: str | None = field(
        default=None,
        metadata={"description": "New title for the step (optional)."},
    )
    status: StepStatus | None = field(
        default=None,
        metadata={
            "description": "New status: pending, in_progress, or done (optional)."
        },
    )

    def render(self) -> str:
        changes: list[str] = []
        if self.title is not None:
            changes.append(f"title='{self.title}'")
        if self.status is not None:
            changes.append(f"status={self.status}")
        payload = ", ".join(changes) or "no changes"
        return f"UpdateStep {self.step_id}: {payload}"


@FrozenDataclass()
class ReadPlan:
    """Request the most recent plan snapshot from the session store."""

    def render(self) -> str:  # pragma: no cover - mirrors Request semantics
        return "Read latest plan snapshot."


@FrozenDataclass()
class _PlanningSectionParams:
    """Placeholder params container for the planning tools section."""

    pass


class PlanningStrategy(Enum):
    """Predefined guidance templates for the planning section."""

    REACT = "react"
    PLAN_ACT_REFLECT = "plan_act_reflect"
    GOAL_DECOMPOSE_ROUTE_SYNTHESISE = "goal_decompose_route_synthesise"


_PLANNING_SECTION_HEADER: Final[str] = (
    "Use planning tools for multi-step or stateful work.\n"
)
_PLANNING_SECTION_BODY: Final[str] = (
    "- Start with `planning_setup_plan` to set an objective and optional steps.\n"
    "- Add more steps with `planning_add_step`.\n"
    "- Update step title or status with `planning_update_step`.\n"
    "- Check progress with `planning_read_plan`.\n"
    "- Step IDs are integers (1, 2, 3...).\n"
    "- Plan completes automatically when all steps are done.\n"
    "Skip planning for trivial single-step tasks."
)


def _template_for_strategy(strategy: PlanningStrategy) -> str:
    guidance_map: dict[PlanningStrategy, str] = {
        PlanningStrategy.REACT: "",
        PlanningStrategy.PLAN_ACT_REFLECT: (
            "Follow a plan-act-reflect rhythm: outline the entire plan before"
            " executing any tools, then work through the steps.\n"
            "After each tool call or completed step, update the step status.\n"
        ),
        PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE: (
            "Begin by restating the goal in your own words to ensure the"
            " objective is clear.\n"
            "Break the goal into concrete sub-problems before routing tools to"
            " each one.\n"
            "When every tool has run, synthesise the results into a cohesive"
            " answer.\n"
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
        session: Session,
        strategy: PlanningStrategy = PlanningStrategy.REACT,
        accepts_overrides: bool = False,
    ) -> None:
        self._strategy = strategy
        self._session = session
        self._initialize_session(session)

        tools = _build_tools(section=self, accepts_overrides=accepts_overrides)
        super().__init__(
            title="Planning Tools",
            key="planning.tools",
            template=_template_for_strategy(strategy),
            default_params=_PlanningSectionParams(),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session

    @staticmethod
    def _initialize_session(session: Session) -> None:
        session.register_reducer(Plan, replace_latest)
        session.register_reducer(SetupPlan, _setup_plan_reducer, slice_type=Plan)
        session.register_reducer(AddStep, _add_step_reducer, slice_type=Plan)
        session.register_reducer(UpdateStep, _update_step_reducer, slice_type=Plan)

    @override
    def clone(self, **kwargs: object) -> PlanningToolsSection:
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone PlanningToolsSection."
            raise TypeError(msg)
        return PlanningToolsSection(
            session=session,
            strategy=self._strategy,
            accepts_overrides=self.accepts_overrides,
        )

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del visibility
        if not isinstance(params, _PlanningSectionParams):
            raise PromptRenderError(
                "Planning tools section requires parameters.",
                dataclass_type=_PlanningSectionParams,
            )
        template = _template_for_strategy(self._strategy)
        return self.render_with_template(template, params, depth, number)

    @override
    def original_body_template(self) -> str:
        return _template_for_strategy(self._strategy)


def _build_tools(
    *,
    section: PlanningToolsSection,
    accepts_overrides: bool,
) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]:
    suite = _PlanningToolSuite(section=section)
    return cast(
        tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
        (
            Tool[SetupPlan, SetupPlan](
                name="planning_setup_plan",
                description=(
                    "Create or replace the session plan with an objective and "
                    "optional initial steps."
                ),
                handler=suite.setup_plan,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[SetupPlan, SetupPlan](
                        description="Create a plan with an objective and two steps.",
                        input=SetupPlan(
                            objective="Publish migration guide",
                            initial_steps=(
                                "Audit existing guides",
                                "List breaking changes",
                            ),
                        ),
                        output=SetupPlan(
                            objective="Publish migration guide",
                            initial_steps=(
                                "Audit existing guides",
                                "List breaking changes",
                            ),
                        ),
                    ),
                ),
            ),
            Tool[AddStep, AddStep](
                name="planning_add_step",
                description="Append one or more steps to the active plan.",
                handler=suite.add_step,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[AddStep, AddStep](
                        description="Add two follow-up steps.",
                        input=AddStep(steps=("Draft outline", "Review with team")),
                        output=AddStep(steps=("Draft outline", "Review with team")),
                    ),
                ),
            ),
            Tool[UpdateStep, UpdateStep](
                name="planning_update_step",
                description="Update a step's title or status by its ID.",
                handler=suite.update_step,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[UpdateStep, UpdateStep](
                        description="Mark step 1 as done.",
                        input=UpdateStep(step_id=1, status="done"),
                        output=UpdateStep(step_id=1, status="done"),
                    ),
                    ToolExample[UpdateStep, UpdateStep](
                        description="Rename step 2.",
                        input=UpdateStep(step_id=2, title="Finalize outline"),
                        output=UpdateStep(step_id=2, title="Finalize outline"),
                    ),
                ),
            ),
            Tool[ReadPlan, Plan](
                name="planning_read_plan",
                description="Return the latest plan snapshot.",
                handler=suite.read_plan,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[ReadPlan, Plan](
                        description="Inspect the current plan.",
                        input=ReadPlan(),
                        output=Plan(
                            objective="Publish migration guide",
                            status="active",
                            steps=(
                                PlanStep(
                                    step_id=1, title="Audit guides", status="done"
                                ),
                                PlanStep(
                                    step_id=2, title="Draft outline", status="pending"
                                ),
                            ),
                        ),
                    ),
                ),
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
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        objective = _normalize_text(params.objective, "objective")
        initial_steps = _normalize_step_titles(params.initial_steps)
        normalized = SetupPlan(objective=objective, initial_steps=initial_steps)
        step_count = len(initial_steps)
        message = (
            f"Plan initialised with {step_count} step{'s' if step_count != 1 else ''}."
        )
        return ToolResult(message=message, value=normalized)

    def add_step(self, params: AddStep, *, context: ToolContext) -> ToolResult[AddStep]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        plan = _require_plan(session)
        _ensure_active(plan)
        normalized_steps = _normalize_step_titles(params.steps)
        if not normalized_steps:
            message = "Provide at least one step to add."
            raise ToolValidationError(message)
        message = f"Added {len(normalized_steps)} step{'s' if len(normalized_steps) != 1 else ''}."
        return ToolResult(message=message, value=AddStep(steps=normalized_steps))

    def update_step(
        self, params: UpdateStep, *, context: ToolContext
    ) -> ToolResult[UpdateStep]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        plan = _require_plan(session)
        _ensure_active(plan)
        step_id = params.step_id
        updated_title = (
            _normalize_text(params.title, "title") if params.title is not None else None
        )
        updated_status = params.status
        if updated_title is None and updated_status is None:
            message = "Provide a new title or status to update a step."
            raise ToolValidationError(message)
        _ensure_step_exists(plan, step_id)
        normalized = UpdateStep(
            step_id=step_id,
            title=updated_title,
            status=updated_status,
        )
        return ToolResult(message=f"Step {step_id} updated.", value=normalized)

    def read_plan(self, params: ReadPlan, *, context: ToolContext) -> ToolResult[Plan]:
        del params
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        plan = session.query(Plan).latest()
        if plan is None:
            message = "No plan is currently initialised."
            raise ToolValidationError(message)
        step_count = len(plan.steps)
        if step_count == 0:
            message = "Retrieved the current plan (no steps recorded)."
        else:
            message = (
                f"Retrieved the current plan "
                f"with {step_count} step{'s' if step_count != 1 else ''}."
            )
        return ToolResult(message=message, value=plan)


def _setup_plan_reducer(
    slice_values: tuple[Plan, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[Plan, ...]:
    del context, slice_values
    if not isinstance(event, ReducerEventWithValue):  # pragma: no cover - defensive
        msg = "Planning reducer requires an event with a value payload."
        raise TypeError(msg)
    params = cast(SetupPlan, event.value)
    steps = tuple(
        PlanStep(step_id=index + 1, title=title, status="pending")
        for index, title in enumerate(params.initial_steps)
    )
    plan = Plan(objective=params.objective, status="active", steps=steps)
    return (plan,)


def _add_step_reducer(
    slice_values: tuple[Plan, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[Plan, ...]:
    del context
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    if not isinstance(event, ReducerEventWithValue):  # pragma: no cover - defensive
        msg = "Planning reducer requires an event with a value payload."
        raise TypeError(msg)
    params = cast(AddStep, event.value)
    existing = list(previous.steps)
    next_id = _next_step_id(existing)
    for title in params.steps:
        existing.append(PlanStep(step_id=next_id, title=title, status="pending"))
        next_id += 1
    updated = Plan(
        objective=previous.objective,
        status="active",
        steps=tuple(existing),
    )
    return (updated,)


def _update_step_reducer(
    slice_values: tuple[Plan, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[Plan, ...]:
    del context
    previous = _latest_plan(slice_values)
    if previous is None:
        return slice_values
    if not isinstance(event, ReducerEventWithValue):  # pragma: no cover - defensive
        msg = "Planning reducer requires an event with a value payload."
        raise TypeError(msg)
    params = cast(UpdateStep, event.value)
    updated_steps: list[PlanStep] = []
    for step in previous.steps:
        if step.step_id != params.step_id:
            updated_steps.append(step)
            continue
        new_title = params.title if params.title is not None else step.title
        new_status = params.status if params.status is not None else step.status
        updated_steps.append(
            PlanStep(step_id=step.step_id, title=new_title, status=new_status)
        )
    plan_status: PlanStatus
    if updated_steps and all(step.status == "done" for step in updated_steps):
        plan_status = "completed"
    else:
        plan_status = "active"
    updated_plan = Plan(
        objective=previous.objective,
        status=plan_status,
        steps=tuple(updated_steps),
    )
    return (updated_plan,)


def _latest_plan(plans: tuple[Plan, ...]) -> Plan | None:
    if not plans:
        return None
    return plans[-1]


def _next_step_id(steps: Sequence[PlanStep]) -> int:
    if not steps:
        return 1
    return max(step.step_id for step in steps) + 1


def _normalize_step_titles(titles: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize_text(title, "title") for title in titles)


def _normalize_text(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        message = f"{field_name.title()} must not be empty."
        raise ToolValidationError(message)
    if len(stripped) > _MAX_TITLE_LENGTH:
        message = f"{field_name.title()} must be <= {_MAX_TITLE_LENGTH} characters."
        raise ToolValidationError(message)
    return stripped


def _require_plan(session: Session) -> Plan:
    plan = session.query(Plan).latest()
    if plan is None:
        message = "No plan is currently initialised."
        raise ToolValidationError(message)
    return plan


def _ensure_active(plan: Plan) -> None:
    if plan.status != "active":
        message = f"Plan must be active. Current status: {plan.status}."
        raise ToolValidationError(message)


def _ensure_step_exists(plan: Plan, step_id: int) -> None:
    if not any(step.step_id == step_id for step in plan.steps):
        message = f"Step {step_id} does not exist."
        raise ToolValidationError(message)


__all__ = [
    "AddStep",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanningStrategy",
    "PlanningToolsSection",
    "ReadPlan",
    "SetupPlan",
    "StepStatus",
    "UpdateStep",
]
