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

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.errors import PromptRenderError
from ...prompt.section import SectionVisibility
from ...prompt.markdown import MarkdownSection
from ...prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ...runtime.session import Replace, Session, reducer, replace_latest
from ...types import SupportsDataclass, SupportsToolResult
from ._context import ensure_context_uses_session

PlanStatus = Literal["active", "completed"]
StepStatus = Literal["pending", "in_progress", "done"]

_MAX_TITLE_LENGTH: Final[int] = 500


@FrozenDataclass()
class PlanStep:
    """Single actionable step tracked within a plan.

    Each step has a unique integer identifier, a descriptive title, and a
    status indicating its progress. Steps are created via :class:`SetupPlan`
    or :class:`AddStep` and updated via :class:`UpdateStep`.

    Attributes:
        step_id: Stable integer identifier for the step (1, 2, 3, ...).
        title: Concise summary of the work item (max 500 characters).
        status: Current progress state - one of "pending", "in_progress", or "done".

    Example::

        step = PlanStep(step_id=1, title="Review requirements", status="pending")
        print(step.render())  # "1 [pending] Review requirements"
    """

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
        """Return a human-readable representation of the step.

        Returns:
            A string in the format "{step_id} [{status}] {title}".
        """
        return f"{self.step_id} [{self.status}] {self.title}"


@FrozenDataclass()
class SetupPlan:
    """Event to initialize or replace the session plan.

    Dispatching this event creates a new :class:`Plan` with the given objective
    and optional initial steps. Any existing plan is completely replaced.

    Attributes:
        objective: The goal the plan should accomplish.
        initial_steps: Optional tuple of step titles to seed the plan with.
            Each title becomes a :class:`PlanStep` with status "pending".

    Example::

        event = SetupPlan(
            objective="Deploy new feature",
            initial_steps=("Write tests", "Update docs"),
        )
        session.dispatch(event)
    """

    objective: str = field(
        metadata={"description": "Objective the plan should accomplish."}
    )
    initial_steps: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"description": "Optional step titles to seed the plan with."},
    )

    def render(self) -> str:
        """Return a human-readable description of the setup event.

        Returns:
            A multi-line string showing the objective and initial steps.
        """
        lines = [f"Setup plan: {self.objective}"]
        if self.initial_steps:
            lines.append("Initial steps:")
            lines.extend(f"- {step}" for step in self.initial_steps)
        else:
            lines.append("Initial steps: <none>")
        return "\n".join(lines)


@FrozenDataclass()
class AddStep:
    """Event to append new steps to the current plan.

    Dispatching this event adds one or more steps to an existing active plan.
    Each new step receives a unique ID (continuing from the highest existing ID)
    and starts with status "pending".

    Attributes:
        steps: Tuple of step titles to add. At least one title is required.

    Raises:
        ToolValidationError: If no plan exists or the plan is not active.

    Example::

        event = AddStep(steps=("Implement feature", "Write tests"))
        session.dispatch(event)
    """

    steps: tuple[str, ...] = field(metadata={"description": "Step titles to add."})

    def render(self) -> str:
        """Return a human-readable description of the add-step event.

        Returns:
            A multi-line string listing the steps to be added.
        """
        if not self.steps:
            return "AddStep: <no steps provided>"
        lines = ["AddStep:"]
        lines.extend(f"- {step}" for step in self.steps)
        return "\n".join(lines)


@FrozenDataclass()
class UpdateStep:
    """Event to modify a step's title or status.

    Dispatching this event updates an existing step in the active plan. At least
    one of ``title`` or ``status`` must be provided. When all steps reach "done"
    status, the plan automatically transitions to "completed".

    Attributes:
        step_id: Integer identifier of the step to update.
        title: New title for the step, or None to keep the existing title.
        status: New status ("pending", "in_progress", or "done"), or None to
            keep the existing status.

    Raises:
        ToolValidationError: If no plan exists, the plan is not active, the step
            does not exist, or neither title nor status is provided.

    Example::

        # Mark step 1 as in progress
        session.dispatch(UpdateStep(step_id=1, status="in_progress"))

        # Rename and complete step 2
        session.dispatch(UpdateStep(step_id=2, title="Revised title", status="done"))
    """

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
        """Return a human-readable description of the update event.

        Returns:
            A string showing the step ID and what fields are being changed.
        """
        changes: list[str] = []
        if self.title is not None:
            changes.append(f"title='{self.title}'")
        if self.status is not None:
            changes.append(f"status={self.status}")
        payload = ", ".join(changes) or "no changes"
        return f"UpdateStep {self.step_id}: {payload}"


@FrozenDataclass()
class Plan:
    """Immutable snapshot of a session plan.

    A plan tracks progress toward an objective through a sequence of steps.
    Plans are created via :class:`SetupPlan`, modified via :class:`AddStep`
    and :class:`UpdateStep`, and read via :class:`ReadPlan`.

    The plan status transitions automatically:
    - "active": At least one step is not "done"
    - "completed": All steps have reached "done" status

    Attributes:
        objective: Single-sentence description of what the plan accomplishes.
        status: Lifecycle state - either "active" or "completed".
        steps: Ordered tuple of :class:`PlanStep` instances.

    Example::

        plan = session[Plan].latest()
        if plan is not None:
            print(f"Working on: {plan.objective}")
            for step in plan.steps:
                print(step.render())
    """

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
        """Return a human-readable representation of the entire plan.

        Returns:
            A multi-line string showing the objective, status, and all steps.
        """
        lines = [f"Objective: {self.objective}", f"Status: {self.status}", ""]
        if not self.steps:
            lines.append("<no steps>")
        else:
            lines.extend(step.render() for step in self.steps)
        return "\n".join(lines)

    @reducer(on=SetupPlan)
    def handle_setup(self, event: SetupPlan) -> Replace[Plan]:
        """Handle a SetupPlan event by creating a fresh plan.

        This reducer ignores the current plan state and creates an entirely
        new plan with the given objective and initial steps.

        Args:
            event: The setup event containing the objective and initial steps.

        Returns:
            A Replace operation containing the new Plan instance.
        """
        del self  # SetupPlan creates a new plan, ignoring current state
        steps = tuple(
            PlanStep(step_id=index + 1, title=title, status="pending")
            for index, title in enumerate(event.initial_steps)
        )
        new_plan = Plan(objective=event.objective, status="active", steps=steps)
        return Replace((new_plan,))

    @reducer(on=AddStep)
    def handle_add_step(self, event: AddStep) -> Replace[Plan]:
        """Handle an AddStep event by appending new steps to the plan.

        New steps receive sequential IDs starting from the highest existing
        ID plus one. All new steps start with "pending" status.

        Args:
            event: The add-step event containing titles for new steps.

        Returns:
            A Replace operation containing the updated Plan with new steps.
        """
        existing = list(self.steps)
        next_id = max((step.step_id for step in existing), default=0) + 1
        for title in event.steps:
            existing.append(PlanStep(step_id=next_id, title=title, status="pending"))
            next_id += 1
        new_plan = Plan(
            objective=self.objective,
            status="active",
            steps=tuple(existing),
        )
        return Replace((new_plan,))

    @reducer(on=UpdateStep)
    def handle_update_step(self, event: UpdateStep) -> Replace[Plan]:
        """Handle an UpdateStep event by modifying a step's title or status.

        If all steps reach "done" status after the update, the plan's status
        automatically transitions to "completed".

        Args:
            event: The update event specifying which step to modify and how.

        Returns:
            A Replace operation containing the updated Plan.
        """
        updated_steps: list[PlanStep] = []
        for step in self.steps:
            if step.step_id != event.step_id:
                updated_steps.append(step)
                continue
            new_title = event.title if event.title is not None else step.title
            new_status = event.status if event.status is not None else step.status
            updated_steps.append(
                PlanStep(step_id=step.step_id, title=new_title, status=new_status)
            )
        plan_status: PlanStatus
        if updated_steps and all(step.status == "done" for step in updated_steps):
            plan_status = "completed"
        else:
            plan_status = "active"
        new_plan = Plan(
            objective=self.objective,
            status=plan_status,
            steps=tuple(updated_steps),
        )
        return Replace((new_plan,))


@FrozenDataclass()
class ReadPlan:
    """Request to retrieve the most recent plan snapshot from the session.

    This is a read-only operation that does not modify the plan state.
    Use the ``planning_read_plan`` tool to inspect current progress.

    Raises:
        ToolValidationError: If no plan has been initialized.

    Example::

        result = tool_context.invoke(ReadPlan())
        plan = result.value
        print(plan.render())
    """

    def render(self) -> str:  # noqa: PLR6301 - mirrors Request semantics
        """Return a human-readable description of the read request.

        Returns:
            A static string indicating a plan read operation.
        """
        return "Read latest plan snapshot."


@FrozenDataclass()
class _PlanningSectionParams:
    """Placeholder params container for the planning tools section."""

    pass


class PlanningStrategy(Enum):
    """Predefined guidance templates for the planning section.

    Each strategy provides different instructions to the LLM about how to
    approach planning and execution. Choose based on the agent's workflow.

    Attributes:
        REACT: Minimal guidance - just the basic tool usage instructions.
            Best for agents that already have strong reasoning patterns.
        PLAN_ACT_REFLECT: Encourages a structured workflow where the agent
            outlines the full plan before execution, then updates status
            after each action. Good for methodical, step-by-step work.
        GOAL_DECOMPOSE_ROUTE_SYNTHESISE: Emphasizes goal clarification,
            decomposition into sub-problems, and synthesis of results.
            Suitable for complex tasks requiring careful analysis.

    Example::

        from weakincentives.contrib.tools import PlanningStrategy, PlanningConfig

        config = PlanningConfig(strategy=PlanningStrategy.PLAN_ACT_REFLECT)
    """

    REACT = "react"
    PLAN_ACT_REFLECT = "plan_act_reflect"
    GOAL_DECOMPOSE_ROUTE_SYNTHESISE = "goal_decompose_route_synthesise"


@FrozenDataclass()
class PlanningConfig:
    """Configuration for :class:`PlanningToolsSection`.

    Consolidates constructor arguments for PlanningToolsSection into a single
    configuration object. This avoids accumulating long argument lists as the
    section evolves and enables configuration reuse across sections.

    Attributes:
        strategy: The planning guidance template to use. Defaults to REACT.
        accepts_overrides: Whether the section allows parameter overrides from
            parent prompts. Defaults to False.

    Example::

        from weakincentives.contrib.tools import PlanningConfig, PlanningToolsSection

        config = PlanningConfig(
            strategy=PlanningStrategy.PLAN_ACT_REFLECT,
            accepts_overrides=True,
        )
        section = PlanningToolsSection(session=session, config=config)
    """

    strategy: PlanningStrategy = field(
        default=PlanningStrategy.REACT,
        metadata={"description": "Predefined guidance template for planning behavior."},
    )
    accepts_overrides: bool = field(
        default=False,
        metadata={"description": "Whether the section accepts parameter overrides."},
    )


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
    """Prompt section that provides planning tools for multi-step task execution.

    This section registers four tools with the prompt:

    - ``planning_setup_plan``: Initialize a plan with an objective and steps
    - ``planning_add_step``: Append additional steps to the active plan
    - ``planning_update_step``: Modify a step's title or status
    - ``planning_read_plan``: Retrieve the current plan state

    The section also installs the :class:`Plan` slice into the provided session,
    enabling plan state to persist across tool calls.

    Use :class:`PlanningConfig` to consolidate configuration::

        config = PlanningConfig(strategy=PlanningStrategy.PLAN_ACT_REFLECT)
        section = PlanningToolsSection(session=session, config=config)

    Individual parameters are still accepted for backward compatibility,
    but config takes precedence when provided.

    Attributes:
        session: The session instance where plan state is stored.

    Note:
        Skip planning for trivial single-step tasks. Planning is most valuable
        for complex, multi-step workflows where tracking progress matters.
    """

    def __init__(
        self,
        *,
        session: Session,
        config: PlanningConfig | None = None,
        strategy: PlanningStrategy = PlanningStrategy.REACT,
        accepts_overrides: bool = False,
    ) -> None:
        """Initialize the planning tools section.

        Args:
            session: The session instance where plan state will be stored.
                The section installs the Plan slice automatically.
            config: Configuration object. When provided, strategy and
                accepts_overrides arguments are ignored.
            strategy: Planning guidance template when config is not provided.
                Defaults to REACT.
            accepts_overrides: Whether to allow parameter overrides when
                config is not provided. Defaults to False.
        """
        # Resolve config - explicit config takes precedence
        if config is not None:
            resolved_strategy = config.strategy
            resolved_accepts_overrides = config.accepts_overrides
        else:
            resolved_strategy = strategy
            resolved_accepts_overrides = accepts_overrides

        self._strategy = resolved_strategy
        self._session = session
        self._initialize_session(session)

        # Store config for cloning
        self._config = PlanningConfig(
            strategy=self._strategy,
            accepts_overrides=resolved_accepts_overrides,
        )

        tools = _build_tools(
            section=self,
            accepts_overrides=resolved_accepts_overrides,
        )
        super().__init__(
            title="Planning Tools",
            key="planning.tools",
            template=_template_for_strategy(resolved_strategy),
            default_params=_PlanningSectionParams(),
            tools=tools,
            accepts_overrides=resolved_accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """The session instance where plan state is stored."""
        return self._session

    @staticmethod
    def _initialize_session(session: Session) -> None:
        # Use a dummy initial factory so SetupPlan can create a new plan
        # even when no plan exists yet
        session.install(
            Plan,
            initial=lambda: Plan(objective="", status="active", steps=()),
        )
        # Register replace_latest for Plan type itself (when returned from tools)
        session[Plan].register(Plan, replace_latest)

    @override
    def clone(self, **kwargs: object) -> PlanningToolsSection:
        """Create a copy of this section bound to a different session.

        Args:
            **kwargs: Must include ``session`` with a :class:`Session` instance.
                Other kwargs are ignored.

        Returns:
            A new PlanningToolsSection with the same configuration but
            bound to the provided session.

        Raises:
            TypeError: If ``session`` is not provided or is not a Session.
        """
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone PlanningToolsSection."
            raise TypeError(msg)
        return PlanningToolsSection(
            session=session,
            config=self._config,
        )

    @override
    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        """Render the section body with the configured strategy template.

        Args:
            params: Must be a _PlanningSectionParams instance (internal).
            visibility: Ignored - included for interface compatibility.
            path: Ignored - included for interface compatibility.
            session: Ignored - included for interface compatibility.

        Returns:
            The rendered markdown body containing planning tool instructions.

        Raises:
            PromptRenderError: If params is not the expected type.
        """
        del visibility, path, session
        if not isinstance(params, _PlanningSectionParams):
            raise PromptRenderError(
                "Planning tools section requires parameters.",
                dataclass_type=_PlanningSectionParams,
            )
        template = _template_for_strategy(self._strategy)
        return self._render_template(template, params)

    @override
    def original_body_template(self) -> str:
        """Return the raw template string for this section.

        Returns:
            The strategy-specific template before any rendering.
        """
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
            Tool[SetupPlan, Plan](
                name="planning_setup_plan",
                description=(
                    "Create or replace the session plan with an objective and "
                    "optional initial steps."
                ),
                handler=suite.setup_plan,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[SetupPlan, Plan](
                        description="Create a plan with an objective and two steps.",
                        input=SetupPlan(
                            objective="Publish migration guide",
                            initial_steps=(
                                "Audit existing guides",
                                "List breaking changes",
                            ),
                        ),
                        output=Plan(
                            objective="Publish migration guide",
                            status="active",
                            steps=(
                                PlanStep(
                                    step_id=1,
                                    title="Audit existing guides",
                                    status="pending",
                                ),
                                PlanStep(
                                    step_id=2,
                                    title="List breaking changes",
                                    status="pending",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            Tool[AddStep, Plan](
                name="planning_add_step",
                description="Append one or more steps to the active plan.",
                handler=suite.add_step,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[AddStep, Plan](
                        description="Add two follow-up steps.",
                        input=AddStep(steps=("Draft outline", "Review with team")),
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
                                PlanStep(
                                    step_id=3,
                                    title="Review with team",
                                    status="pending",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            Tool[UpdateStep, Plan](
                name="planning_update_step",
                description="Update a step's title or status by its ID.",
                handler=suite.update_step,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[UpdateStep, Plan](
                        description="Mark step 1 as done.",
                        input=UpdateStep(step_id=1, status="done"),
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
                    ToolExample[UpdateStep, Plan](
                        description="Rename step 2.",
                        input=UpdateStep(step_id=2, title="Finalize outline"),
                        output=Plan(
                            objective="Publish migration guide",
                            status="active",
                            steps=(
                                PlanStep(
                                    step_id=1, title="Audit guides", status="done"
                                ),
                                PlanStep(
                                    step_id=2,
                                    title="Finalize outline",
                                    status="pending",
                                ),
                            ),
                        ),
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
    ) -> ToolResult[Plan]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        objective = _normalize_text(params.objective, "objective")
        initial_steps = _normalize_step_titles(params.initial_steps)
        normalized = SetupPlan(objective=objective, initial_steps=initial_steps)
        session = self._section.session
        _ = session.dispatch(normalized)
        plan = _require_plan(session)
        step_count = len(initial_steps)
        message = (
            f"Plan initialised with {step_count} step{'s' if step_count != 1 else ''}."
        )
        return ToolResult(message=message, value=plan)

    def add_step(self, params: AddStep, *, context: ToolContext) -> ToolResult[Plan]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        plan = _require_plan(session)
        _ensure_active(plan)
        normalized_steps = _normalize_step_titles(params.steps)
        if not normalized_steps:
            message = "Provide at least one step to add."
            raise ToolValidationError(message)
        normalized = AddStep(steps=normalized_steps)
        _ = session.dispatch(normalized)
        plan = _require_plan(session)
        message = f"Added {len(normalized_steps)} step{'s' if len(normalized_steps) != 1 else ''}."
        return ToolResult(message=message, value=plan)

    def update_step(
        self, params: UpdateStep, *, context: ToolContext
    ) -> ToolResult[Plan]:
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
        _ = session.dispatch(normalized)
        plan = _require_plan(session)
        return ToolResult(message=f"Step {step_id} updated.", value=plan)

    def read_plan(self, params: ReadPlan, *, context: ToolContext) -> ToolResult[Plan]:
        del params
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        plan = session[Plan].latest()
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
    plan = session[Plan].latest()
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
    "PlanningConfig",
    "PlanningStrategy",
    "PlanningToolsSection",
    "ReadPlan",
    "SetupPlan",
    "StepStatus",
    "UpdateStep",
]
