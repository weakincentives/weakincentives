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

"""Tooling for dispatching subagents in parallel.

Subagent dispatch is centered around the Plan object from the planning tools.
Each subagent receives a pre-populated plan in its cloned session and is
instructed to complete all steps before returning. Full isolation is the
default mode, meaning each child gets its own session and event bus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field, is_dataclass
from enum import Enum, auto
from typing import Any, Final, cast, override

from ..dataclasses import FrozenDataclass
from ..prompt import Prompt, SupportsDataclass
from ..prompt._visibility import SectionVisibility
from ..prompt.errors import PromptRenderError
from ..prompt.markdown import MarkdownSection
from ..prompt.protocols import (
    PromptProtocol,
    PromptResponseProtocol,
    RenderedPromptProtocol,
)
from ..prompt.tool import Tool, ToolContext, ToolExample
from ..prompt.tool_result import ToolResult
from ..runtime.events import InProcessEventBus
from ..runtime.events._types import EventBus
from ..runtime.session import Session
from ..runtime.session.protocols import SessionProtocol
from ..serde import dump
from .planning import (
    Plan,
    PlanStep,
    build_standalone_planning_tools,
    initialize_planning_session,
)


class SubagentIsolationLevel(Enum):
    """Isolation modes describing how children interact with parent state."""

    NO_ISOLATION = auto()
    FULL_ISOLATION = auto()


@FrozenDataclass()
class SubagentTask:
    """A task to delegate to a subagent, expressed as a plan.

    Each subagent receives a pre-populated Plan in its cloned session
    and is instructed to complete all steps before returning.
    """

    objective: str = field(
        metadata={"description": "Single-sentence goal for this subagent to achieve."}
    )
    steps: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"description": "Ordered steps the subagent should complete."},
    )


def _default_max_workers() -> int:
    with ThreadPoolExecutor() as executor:
        return executor._max_workers


_DEFAULT_MAX_WORKERS: Final[int] = _default_max_workers()


@FrozenDataclass()
class DispatchSubagentsParams:
    """Parameters describing the tasks to delegate.

    Each task becomes a Plan in the subagent's cloned session. The subagent
    is instructed to complete all steps in the plan before returning.
    """

    tasks: tuple[SubagentTask, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Ordered tasks to delegate. Each task specifies an objective "
                "and steps that form a plan for the subagent to complete."
            )
        },
    )


@FrozenDataclass()
class SubagentResult:
    """Outcome captured for an individual delegation."""

    output: str = field(
        metadata={
            "description": (
                "Rendered response from the delegated prompt, including parsed "
                "output when structured modes are enabled."
            )
        }
    )
    success: bool = field(
        metadata={
            "description": (
                "Flag indicating whether the delegated run completed without "
                "adapter or rendering errors."
            )
        }
    )
    error: str | None = field(
        default=None,
        metadata={
            "description": (
                "Optional diagnostic message when the delegation fails. Null "
                "when the run succeeds."
            )
        },
    )

    def render(self) -> str:
        output_text = self.output or ""
        if self.success:
            return f"Subagent succeeded:\n\n{output_text}"
        message = self.error or "Subagent execution failed"
        return f"Subagent failed: {message}\n\n{output_text}"


def _extract_output_text(response: PromptResponseProtocol[Any]) -> str:
    if response.text:
        return response.text
    if response.output is not None:
        try:
            rendered = dump(response.output, exclude_none=True)
            return json.dumps(rendered, ensure_ascii=False)
        except TypeError:
            return str(response.output)
    return ""


def _build_error(message: str) -> str:
    cleaned = message.strip()
    return cleaned or "Subagent execution failed"


def _clone_session(
    session: SessionProtocol,
    *,
    bus: EventBus,
    parent: SessionProtocol | None = None,
) -> SessionProtocol | None:
    clone_method = getattr(session, "clone", None)
    if not callable(clone_method):
        return None
    kwargs: dict[str, object] = {"bus": bus}
    if parent is not None:
        kwargs["parent"] = parent
    return cast(SessionProtocol, clone_method(**kwargs))


def _prepare_child_contexts(
    *,
    tasks: Iterable[SubagentTask],
    session: SessionProtocol,
    bus: EventBus,
    isolation_level: SubagentIsolationLevel,
) -> tuple[tuple[Session, EventBus], ...] | str:
    """Prepare isolated session/bus pairs for each task.

    In FULL_ISOLATION mode (default), each child gets a cloned session with
    the task's Plan pre-populated. In NO_ISOLATION mode, all children share
    the parent session and bus (Plan injection is skipped since there's no
    isolation).
    """
    tasks_list = list(tasks)

    if isolation_level is SubagentIsolationLevel.NO_ISOLATION:
        # No isolation - children share parent session and bus
        # Plan injection is skipped since children share state
        return tuple((cast(Session, session), bus) for _ in tasks_list)

    child_pairs: list[tuple[Session, EventBus]] = []
    for task in tasks_list:
        child_bus = InProcessEventBus()
        try:
            cloned = _clone_session(session, bus=child_bus, parent=session)
        except Exception as error:  # pragma: no cover - defensive
            return _build_error(str(error))
        if cloned is None:
            return "Parent session does not support cloning for full isolation."
        if not isinstance(cloned, Session):
            return "Cloned session must be a Session instance for plan injection."

        # Initialize planning reducers and inject the task's Plan directly
        initialize_planning_session(cloned)
        steps = tuple(
            PlanStep(step_id=idx + 1, title=title, status="pending")
            for idx, title in enumerate(task.steps)
        )
        plan = Plan(objective=task.objective, status="active", steps=steps)
        cloned.seed_slice(Plan, (plan,))

        child_pairs.append((cloned, child_bus))
    return tuple(child_pairs)


@FrozenDataclass()
class _PlanDelegationParams:
    """Parameters for the plan delegation prompt section."""

    objective: str = field(
        metadata={"description": "The objective for this subagent to achieve."}
    )


_PLAN_DELEGATION_TEMPLATE: Final[str] = (
    "You are a subagent executing a delegated task. Your session contains a Plan\n"
    "that you must complete before returning.\n\n"
    "**Objective**: ${objective}\n\n"
    "Use `planning_read_plan` to see your assigned steps. Work through each step\n"
    "in order, updating status as you go:\n"
    "- Mark steps `in_progress` when you start working on them\n"
    "- Mark steps `done` when completed\n\n"
    "Complete ALL steps before returning your final response. The plan will\n"
    "automatically transition to `completed` status when all steps are done."
)


def _build_plan_delegation_prompt(
    *,
    parent_prompt: PromptProtocol[Any],
    rendered_parent: RenderedPromptProtocol[Any],
) -> Prompt[Any]:
    """Build a delegation prompt that instructs the child to complete its plan.

    The prompt includes parent tools and planning tools, with instructions to
    complete all steps in the pre-populated plan.
    """
    from ..prompt import PromptTemplate

    # Combine parent tools with standalone planning tools so subagents can
    # read and update the Plan injected into their sessions
    planning_tools = build_standalone_planning_tools()
    combined_tools = (*rendered_parent.tools, *planning_tools)

    # Create a section for the delegation instructions
    delegation_section = MarkdownSection[_PlanDelegationParams](
        title="Subagent Task",
        key="subagent-task",
        template=_PLAN_DELEGATION_TEMPLATE,
        tools=combined_tools,
    )

    # Get the output type from the parent prompt - already validated as dataclass
    output_type = rendered_parent.output_type

    # Build the delegation prompt with the same output type as the parent
    prompt_cls: type[PromptTemplate[Any]] = PromptTemplate.__class_getitem__(
        output_type
    )
    delegation_prompt = prompt_cls(
        ns=f"{parent_prompt.ns}.subagent",
        key=f"{parent_prompt.key}-delegation",
        sections=(delegation_section,),
        inject_output_instructions=False,
        allow_extra_keys=bool(rendered_parent.allow_extra_keys),
    )

    return Prompt(delegation_prompt)


def build_dispatch_subagents_tool(
    *,
    isolation_level: SubagentIsolationLevel = SubagentIsolationLevel.FULL_ISOLATION,
    accepts_overrides: bool = False,
) -> Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]:
    """Return a configured dispatch tool bound to the desired isolation level.

    Full isolation is the default: each child gets a cloned session with the
    task's Plan pre-populated. The child is instructed to complete all steps
    in the plan before returning.
    """

    examples: tuple[
        ToolExample[DispatchSubagentsParams, tuple[SubagentResult, ...]], ...
    ] = (
        ToolExample(
            description="Fan out competitor research and FAQ drafting in parallel.",
            input=DispatchSubagentsParams(
                tasks=(
                    SubagentTask(
                        objective="Summarize feature gaps across direct competitors.",
                        steps=(
                            "Scan Acme and Globex product docs",
                            "List strengths and weaknesses",
                            "Call out notable differentiators",
                        ),
                    ),
                    SubagentTask(
                        objective="Draft concise FAQ responses from support backlog.",
                        steps=(
                            "Review latest ticket summaries",
                            "Draft five FAQ answers",
                            "Keep each answer under 50 words",
                        ),
                    ),
                )
            ),
            output=(
                SubagentResult(
                    output=(
                        "Acme focuses on analytics depth; Globex excels in billing "
                        "flexibility; both advertise 99.9% uptime SLAs."
                    ),
                    success=True,
                ),
                SubagentResult(
                    output=(
                        "Auth setup, SSO access, refund timeline, data export, and "
                        "plan upgrade FAQs drafted with concise answers."
                    ),
                    success=True,
                ),
            ),
        ),
    )

    def _dispatch_subagents(
        params: DispatchSubagentsParams,
        *,
        context: ToolContext,
    ) -> ToolResult[tuple[SubagentResult, ...]]:
        rendered_parent = context.rendered_prompt
        if rendered_parent is None:
            return ToolResult(
                message="dispatch_subagents requires the parent prompt to be rendered.",
                value=None,
                success=False,
            )

        if not isinstance(rendered_parent.output_type, type) or not is_dataclass(
            rendered_parent.output_type
        ):
            return ToolResult(
                message="Parent prompt must declare a dataclass output type for delegation.",
                value=None,
                success=False,
            )

        tasks = tuple(params.tasks)
        if not tasks:
            empty_results = cast(tuple[SubagentResult, ...], ())
            return ToolResult(
                message="No tasks supplied.",
                value=empty_results,
            )

        contexts = _prepare_child_contexts(
            tasks=tasks,
            session=context.session,
            bus=context.event_bus,
            isolation_level=isolation_level,
        )
        if isinstance(contexts, str):
            return ToolResult(
                message=contexts,
                value=None,
                success=False,
            )

        adapter = context.adapter
        parse_output = rendered_parent.container is not None
        parent_deadline = rendered_parent.deadline
        # Budget tracker is always shared with children regardless of isolation
        parent_budget_tracker = context.budget_tracker

        def _run_child(
            payload: tuple[SubagentTask, Session, EventBus],
        ) -> SubagentResult:
            task, child_session, child_bus = payload
            try:
                # Build a fresh delegation prompt for each child to avoid races
                delegation_prompt = _build_plan_delegation_prompt(
                    parent_prompt=context.prompt,
                    rendered_parent=rendered_parent,
                )
                bound_prompt = cast(
                    PromptProtocol[Any],
                    delegation_prompt.bind(
                        _PlanDelegationParams(objective=task.objective)
                    ),
                )
                response = adapter.evaluate(
                    bound_prompt,
                    parse_output=parse_output,
                    bus=child_bus,
                    session=child_session,
                    deadline=parent_deadline,
                    budget_tracker=parent_budget_tracker,
                )
            except Exception as error:  # pragma: no cover - defensive
                return SubagentResult(
                    output="",
                    success=False,
                    error=_build_error(str(error)),
                )
            return SubagentResult(
                output=_extract_output_text(response),
                success=True,
            )

        payloads: list[tuple[SubagentTask, Session, EventBus]] = [
            (task, child_session, child_bus)
            for task, (child_session, child_bus) in zip(tasks, contexts, strict=True)
        ]

        max_workers = min(len(payloads), _DEFAULT_MAX_WORKERS) or 1
        results: list[SubagentResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_child, payload) for payload in payloads]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:  # pragma: no cover - defensive
                    results.append(
                        SubagentResult(
                            output="",
                            success=False,
                            error=_build_error(str(error)),
                        )
                    )

        return ToolResult(
            message=f"Dispatched {len(results)} subagents.",
            value=tuple(results),
        )

    return Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]](
        name="dispatch_subagents",
        description=(
            "Dispatch tasks to subagents running in parallel. Each task becomes "
            "a Plan in the subagent's isolated session. The subagent completes "
            "all plan steps before returning."
        ),
        handler=_dispatch_subagents,
        accepts_overrides=accepts_overrides,
        examples=examples,
    )


@FrozenDataclass()
class _SubagentsSectionParams:
    """Placeholder params container for the subagents section."""

    pass


_DELEGATION_BODY: Final[str] = (
    "Use `dispatch_subagents` to offload work that can proceed in parallel.\n"
    "Each task becomes a Plan in the subagent's isolated session. Subagents\n"
    "complete all plan steps before returning. Prefer dispatching concurrent\n"
    "tasks over running them sequentially yourself."
)


class SubagentsSection(MarkdownSection[_SubagentsSectionParams]):
    """Explain the delegation workflow and expose the dispatch tool.

    Each subagent receives a pre-populated Plan in its cloned session and is
    instructed to complete all steps before returning. Full isolation is the
    default mode.
    """

    def __init__(
        self,
        *,
        isolation_level: SubagentIsolationLevel | None = None,
        accepts_overrides: bool = False,
    ) -> None:
        effective_level = (
            isolation_level
            if isolation_level is not None
            else SubagentIsolationLevel.FULL_ISOLATION
        )
        self._isolation_level = effective_level
        tool = build_dispatch_subagents_tool(
            isolation_level=effective_level,
            accepts_overrides=accepts_overrides,
        )
        super().__init__(
            title="Delegation",
            key="subagents",
            template=_DELEGATION_BODY,
            default_params=_SubagentsSectionParams(),
            tools=(tool,),
            accepts_overrides=accepts_overrides,
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
        if not isinstance(params, _SubagentsSectionParams):
            raise PromptRenderError(
                "Subagents section requires parameters.",
                dataclass_type=_SubagentsSectionParams,
            )
        return super().render(params, depth, number, visibility=visibility)

    @override
    def clone(self, **kwargs: object) -> SubagentsSection:
        return SubagentsSection(
            isolation_level=self._isolation_level,
            accepts_overrides=self.accepts_overrides,
        )


dispatch_subagents = build_dispatch_subagents_tool()


__all__ = [
    "DispatchSubagentsParams",
    "SubagentIsolationLevel",
    "SubagentResult",
    "SubagentTask",
    "SubagentsSection",
    "build_dispatch_subagents_tool",
    "dispatch_subagents",
]
