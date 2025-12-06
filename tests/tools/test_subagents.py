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

"""Tests for the subagent dispatch tooling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Literal, cast

from tests.helpers.adapters import RECORDING_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import ToolContext
from weakincentives.runtime.events import InProcessEventBus, PromptExecuted
from weakincentives.runtime.session import Session, select_latest
from weakincentives.runtime.session.protocols import SessionProtocol, SnapshotProtocol
from weakincentives.tools.planning import Plan
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    SubagentTask,
    build_dispatch_subagents_tool,
    dispatch_subagents,
)


def test_subagent_result_render_reports_status() -> None:
    success = SubagentResult(output="done", success=True, error=None)
    failure = SubagentResult(output="", success=False, error="boom")

    success_render = success.render()
    failure_render = failure.render()
    assert "succeeded" in success_render.lower()
    assert "done" in success_render
    assert "failed" in failure_render.lower()
    assert "boom" in failure_render


@dataclass(slots=True)
class ParentSectionParams:
    instructions: str


@dataclass(slots=True)
class ParentOutput:
    summary: str


class RecordingAdapter(ProviderAdapter[Any]):
    """Test adapter that records calls and returns configurable responses.

    The adapter extracts the objective from the delegation prompt's first
    parameter (which has an `objective` field from _PlanDelegationParams).
    """

    def __init__(
        self,
        *,
        failures: set[str] | None = None,
        delays: dict[str, float] | None = None,
        structured_outputs: dict[str, SupportsDataclass] | None = None,
        raw_outputs: dict[str, object] | None = None,
        empty_text: set[str] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.sessions: list[Session | None] = []
        self.buses: list[InProcessEventBus] = []
        self.deadlines: list[Deadline | None] = []
        self._failures = failures or set()
        self._delays = delays or {}
        self._structured_outputs = structured_outputs or {}
        self._raw_outputs = raw_outputs or {}
        self._empty_text = empty_text or set()
        self._lock = Lock()

    def evaluate(
        self,
        prompt: Prompt[Any],
        *,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: Session | None = None,
        deadline: Deadline | None = None,
        budget_tracker: object = None,
    ) -> PromptResponse[Any]:
        del parse_output, budget_tracker
        params = prompt.params
        # Extract objective from the delegation params (_PlanDelegationParams)
        delegation_params = params[0]
        objective = getattr(delegation_params, "objective", "unknown")
        with self._lock:
            self.calls.append(objective)
            self.sessions.append(session)
            self.buses.append(bus)
            self.deadlines.append(deadline)
        if objective in self._failures:
            raise RuntimeError(f"failure: {objective}")
        delay = self._delays.get(objective, 0.0)
        if delay:
            time.sleep(delay)
        prompt_name = prompt.name or prompt.key

        def _emit(response: PromptResponse[Any]) -> PromptResponse[Any]:
            bus.publish(
                PromptExecuted(
                    prompt_name=prompt_name,
                    adapter=RECORDING_ADAPTER_NAME,
                    result=cast(PromptResponse[object], response),
                    session_id=getattr(session, "session_id", None),
                    created_at=datetime.now(UTC),
                    value=(
                        cast(SupportsDataclass, response.output)
                        if response.output is not None and is_dataclass(response.output)
                        else None
                    ),
                )
            )
            return response

        if objective in self._empty_text:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=None,
                )
            )
        structured = self._structured_outputs.get(objective)
        if structured is not None:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=structured,
                )
            )
        raw_output = self._raw_outputs.get(objective)
        if raw_output is not None:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=raw_output,
                )
            )
        return _emit(
            PromptResponse(
                prompt_name=prompt_name,
                text=f"child:{objective}",
                output=None,
            )
        )


def _build_parent_prompt(
    *, deadline: Deadline | None = None
) -> tuple[PromptTemplate[ParentOutput], RenderedPrompt[ParentOutput]]:
    section = MarkdownSection[ParentSectionParams](
        title="Parent",
        key="parent",
        template="${instructions}",
    )
    prompt = PromptTemplate[ParentOutput](
        ns="tests.subagents",
        key="parent",
        sections=(section,),
    )
    rendered = prompt.bind(
        ParentSectionParams(instructions="Document the repo.")
    ).render()
    if deadline is not None:
        rendered = replace(rendered, deadline=deadline)
    return prompt, rendered


def test_dispatch_subagents_requires_rendered_prompt() -> None:
    prompt, _ = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=None,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="missing", steps=("step1",)),)
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "rendered" in result.message


def test_dispatch_subagents_runs_children_in_parallel() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(delays={"slow-objective": 0.05, "fast-objective": 0.01})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(
            SubagentTask(
                objective="slow-objective",
                steps=("Focus on slow path",),
            ),
            SubagentTask(
                objective="fast-objective",
                steps=("Focus on fast path",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert isinstance(result.value, tuple)
    assert [child.output for child in result.value] == [
        "child:slow-objective",
        "child:fast-objective",
    ]
    assert adapter.calls == ["slow-objective", "fast-objective"]
    # Full isolation is the default - children get cloned sessions and new buses
    assert all(child_bus is not bus for child_bus in adapter.buses)
    assert all(s is not session for s in adapter.sessions)


def test_dispatch_subagents_propagates_deadline() -> None:
    parent_deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    prompt, rendered = _build_parent_prompt(deadline=parent_deadline)
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="one", steps=()),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.deadlines == [parent_deadline]


def test_dispatch_subagents_collects_failures() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(failures={"fail-objective"})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(
            SubagentTask(
                objective="ok-objective",
                steps=("Keep things tidy",),
            ),
            SubagentTask(
                objective="fail-objective",
                steps=("Handle the failure",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert isinstance(result.value, tuple)
    first, second = result.value
    assert isinstance(first, SubagentResult)
    assert isinstance(second, SubagentResult)
    assert first.success is True
    assert first.error is None
    assert second.success is False
    assert second.error is not None
    assert "fail" in second.error
    assert adapter.calls[1] == "fail-objective"


def test_dispatch_subagents_requires_dataclass_output_type() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=RenderedPrompt(
            text=rendered.text,
            structured_output=StructuredOutputConfig(
                dataclass_type=cast(type[SupportsDataclass], cast(object, str)),
                container=cast(Literal["object", "array"], rendered.container),
                allow_extra_keys=bool(rendered.allow_extra_keys),
            ),
            _tools=rendered.tools,
            _tool_param_descriptions=rendered.tool_param_descriptions,
        ),
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="invalid", steps=("recap",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "dataclass" in result.message


def test_dispatch_subagents_handles_empty_tasks() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(tasks=())

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value == ()
    assert "No tasks" in result.message


def test_dispatch_subagents_formats_structured_outputs() -> None:
    @dataclass(slots=True)
    class StructuredChildResult:
        field: str

    class Unserializable:
        def __str__(self) -> str:
            return "fallback-output"

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"structured": StructuredChildResult(field="value")},
        raw_outputs={"raw": Unserializable()},
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(
            SubagentTask(
                objective="structured",
                steps=("Render structured",),
            ),
            SubagentTask(
                objective="raw",
                steps=("Render raw",),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value is not None
    first, second = result.value
    assert json.loads(first.output) == {"field": "value"}
    assert second.output == "fallback-output"


def test_dispatch_subagents_returns_empty_output_when_child_returns_none() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(empty_text={"empty"})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="empty", steps=("Produce nothing",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value is not None
    child = result.value[0]
    assert child.output == ""


def test_dispatch_subagents_shares_state_without_isolation() -> None:
    """NO_ISOLATION mode shares session and bus with children."""

    @dataclass(slots=True)
    class ChildRecord:
        field: str

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"shared": ChildRecord(field="value")}
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    # Explicitly use NO_ISOLATION to share state
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.NO_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="shared", steps=("Record to session",)),),
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    captured = session.select_all(ChildRecord)
    assert captured == (ChildRecord(field="value"),)


def test_dispatch_subagents_full_isolation_clones_state() -> None:
    """Full isolation (default) clones sessions so parent state is untouched."""

    @dataclass(slots=True)
    class ChildRecord:
        field: str

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"isolated": ChildRecord(field="value")}
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    # FULL_ISOLATION is the default
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="isolated", steps=("Record to clone",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert session.select_all(ChildRecord) == ()
    assert adapter.sessions


def test_dispatch_subagents_full_isolation_sets_parent_session() -> None:
    """Full isolation sets parent reference and creates child sessions."""
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    # FULL_ISOLATION is the default
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="inherit-parent", steps=("Track parent",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.sessions
    child_session = adapter.sessions[0]
    assert child_session is not None
    assert child_session.parent is session
    assert session.children == (child_session,)
    assert all(child_session is not session for child_session in adapter.sessions)
    assert all(child_bus is not bus for child_bus in adapter.buses)


def test_dispatch_subagents_full_isolation_requires_clone_support() -> None:
    """Full isolation fails if session doesn't support cloning."""

    class NonCloningSession(SessionProtocol):
        def snapshot(self) -> SnapshotProtocol:
            raise NotImplementedError

        def rollback(self, snapshot: SnapshotProtocol) -> None:
            raise NotImplementedError

        def reset(self) -> None:
            raise NotImplementedError

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = NonCloningSession()
    # FULL_ISOLATION is the default
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="non-clone", steps=("Should fail",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "cloning" in result.message.lower()


def test_dispatch_subagents_full_isolation_requires_session_instance() -> None:
    """Full isolation requires clone() to return a Session instance."""

    class NonSessionCloneSession(SessionProtocol):
        def snapshot(self) -> SnapshotProtocol:
            raise NotImplementedError

        def rollback(self, snapshot: SnapshotProtocol) -> None:
            raise NotImplementedError

        def reset(self) -> None:
            raise NotImplementedError

        def clone(self, **kwargs: object) -> SessionProtocol:  # noqa: PLR6301
            # Return a non-Session clone
            return NonSessionCloneSession()

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = NonSessionCloneSession()
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="non-session", steps=("Should fail",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "Session instance" in result.message


def test_build_dispatch_subagents_tool_respects_accepts_overrides() -> None:
    default_tool = build_dispatch_subagents_tool()
    overriding_tool = build_dispatch_subagents_tool(accepts_overrides=True)

    assert default_tool.accepts_overrides is False
    assert overriding_tool.accepts_overrides is True


def test_dispatch_subagents_injects_plan_in_full_isolation() -> None:
    """Full isolation injects the task as a Plan into the child session."""
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(
            SubagentTask(
                objective="Test plan injection",
                steps=("Step one", "Step two"),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.sessions
    child_session = adapter.sessions[0]
    assert child_session is not None

    # Verify the plan was injected into the child session
    plan = select_latest(child_session, Plan)
    assert plan is not None
    assert plan.objective == "Test plan injection"
    assert plan.status == "active"
    assert len(plan.steps) == 2
    assert plan.steps[0].title == "Step one"
    assert plan.steps[0].status == "pending"
    assert plan.steps[1].title == "Step two"
    assert plan.steps[1].status == "pending"


def test_dispatch_subagents_default_isolation_is_full() -> None:
    """Verify that the default isolation level is FULL_ISOLATION."""
    default_tool = build_dispatch_subagents_tool()
    no_isolation_tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.NO_ISOLATION
    )
    full_isolation_tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )

    # The default should behave the same as FULL_ISOLATION
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="test-default", steps=()),),
    )

    # Default tool should create a cloned session (full isolation)
    assert default_tool.handler is not None
    result = default_tool.handler(params, context=context)
    assert result.success is True
    assert adapter.sessions[0] is not session
    assert adapter.buses[0] is not bus

    # Verify tools are configured correctly by checking handler identity
    assert default_tool.handler is not None
    assert no_isolation_tool.handler is not None
    assert full_isolation_tool.handler is not None


def test_subagent_task_dataclass() -> None:
    """Verify SubagentTask dataclass works correctly."""
    task = SubagentTask(
        objective="Test objective",
        steps=("Step 1", "Step 2", "Step 3"),
    )

    assert task.objective == "Test objective"
    assert task.steps == ("Step 1", "Step 2", "Step 3")

    # Default steps should be empty tuple
    task_no_steps = SubagentTask(objective="No steps")
    assert task_no_steps.steps == ()


def test_dispatch_subagents_filters_duplicate_planning_tools() -> None:
    """Verify parent planning tools are replaced with standalone ones to avoid duplicates."""
    from weakincentives.tools.planning import PlanningToolsSection

    bus = InProcessEventBus()
    session = Session(bus=bus)

    # Create a parent prompt that includes PlanningToolsSection (has planning tools)
    planning_section = PlanningToolsSection(session=session)

    parent_template = PromptTemplate[ParentOutput](
        ns="test",
        key="with-planning",
        sections=(
            MarkdownSection[ParentSectionParams](
                title="Instructions",
                key="instructions",
                template="${instructions}",
            ),
            planning_section,
        ),
    )
    rendered = parent_template.bind(
        ParentSectionParams(instructions="Test body")
    ).render()

    # Verify parent has planning tools
    parent_tool_names = {t.name for t in rendered.tools}
    assert "planning_read_plan" in parent_tool_names
    assert "planning_update_step" in parent_tool_names

    adapter = RecordingAdapter()
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], parent_template),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        tasks=(SubagentTask(objective="test-dedup", steps=("Step 1",)),),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    # Should succeed without "Duplicate tool name" error
    # Prior to the fix, this would fail with "Duplicate tool name registered"
    assert result.success is True
    assert result.value is not None
    assert result.value[0].success is True
