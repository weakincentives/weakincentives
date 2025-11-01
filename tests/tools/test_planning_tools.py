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

"""Unit tests for the planning tool handlers."""

from __future__ import annotations

from typing import TypeVar, cast

import pytest

from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.prompt import SupportsDataclass
from weakincentives.prompt.tool import Tool, ToolResult
from weakincentives.session import Session, ToolData, select_all, select_latest
from weakincentives.tools import (
    AddStep,
    ClearPlan,
    MarkStep,
    NewPlanStep,
    Plan,
    PlanningToolsSection,
    PlanStep,
    ReadPlan,
    SetupPlan,
    ToolValidationError,
    UpdateStep,
)
from weakincentives.tools.planning import (
    _add_step_reducer,
    _clear_plan_reducer,
    _latest_plan,
    _mark_step_reducer,
    _next_step_index,
    _update_step_reducer,
)

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)
ResultT = TypeVar("ResultT", bound=SupportsDataclass)


def _find_tool(
    section: PlanningToolsSection, name: str
) -> Tool[SupportsDataclass, SupportsDataclass]:
    for tool in section.tools():
        if tool.name == name:
            assert tool.handler is not None
            return tool
    raise AssertionError(f"Tool {name} not found")


def _invoke_tool(
    bus: InProcessEventBus,
    tool: Tool[ParamsT, ResultT],
    params: ParamsT,
) -> ToolResult[ResultT]:
    handler = tool.handler
    assert handler is not None
    result = handler(params)
    event = ToolInvoked(
        prompt_name="test",
        adapter="adapter",
        name=tool.name,
        params=params,
        result=cast(ToolResult[object], result),
    )
    bus.publish(event)
    return result


def _make_tool_data(name: str, value: SupportsDataclass) -> ToolData[SupportsDataclass]:
    result = ToolResult(message="ok", value=value)
    event = ToolInvoked(
        prompt_name="test",
        adapter="adapter",
        name=name,
        params=value,
        result=cast(ToolResult[object], result),
    )
    return ToolData(value=value, source=event)


def test_setup_plan_normalizes_payloads() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")

    params = SetupPlan(
        objective="  refine onboarding flow  ",
        initial_steps=(NewPlanStep(title=" draft checklist ", details=" review doc "),),
    )
    _invoke_tool(bus, setup_tool, params)

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.objective == "refine onboarding flow"
    assert plan.status == "active"
    assert plan.steps == (
        PlanStep(
            step_id="S001",
            title="draft checklist",
            details="review doc",
            status="pending",
            notes=(),
        ),
    )


def test_setup_plan_normalizes_blank_details() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")

    params = SetupPlan(
        objective="ship",
        initial_steps=(NewPlanStep(title="draft", details="   "),),
    )
    _invoke_tool(bus, setup_tool, params)

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.steps[0].details is None


def test_setup_plan_rejects_invalid_objective() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")

    handler = setup_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(SetupPlan(objective="   "))

    long_objective = "x" * 241
    with pytest.raises(ToolValidationError):
        handler(SetupPlan(objective=long_objective))


def test_add_step_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    add_tool = _find_tool(section, "planning.add_step")

    with pytest.raises(ToolValidationError):
        handler = add_tool.handler
        assert handler is not None
        handler(AddStep(steps=(NewPlanStep(title="task"),)))


def test_add_step_appends_new_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    add_tool = _find_tool(section, "planning.add_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )
    _invoke_tool(
        bus,
        add_tool,
        AddStep(
            steps=(
                NewPlanStep(title="review"),
                NewPlanStep(title="release"),
            )
        ),
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert [step.step_id for step in plan.steps] == ["S001", "S002", "S003"]
    assert [step.title for step in plan.steps] == ["draft", "review", "release"]


def test_add_step_rejects_empty_payload() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    add_tool = _find_tool(section, "planning.add_step")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))

    handler = add_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(AddStep(steps=()))


def test_add_step_rejects_when_plan_not_active() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")
    add_tool = _find_tool(section, "planning.add_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )
    _invoke_tool(bus, mark_tool, MarkStep(step_id="S001", status="done"))

    handler = add_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(AddStep(steps=(NewPlanStep(title="later"),)))


def test_session_keeps_single_plan_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    add_tool = _find_tool(section, "planning.add_step")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))
    _invoke_tool(
        bus,
        add_tool,
        AddStep(steps=(NewPlanStep(title="draft"),)),
    )

    snapshots = select_all(session, Plan)
    assert len(snapshots) == 1


def test_update_step_rejects_empty_patch() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    update_tool = _find_tool(section, "planning.update_step")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))

    handler = update_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(UpdateStep(step_id="S001"))


def test_update_step_requires_step_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    update_tool = _find_tool(section, "planning.update_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )

    handler = update_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(UpdateStep(step_id="  ", title="rename"))


def test_update_step_rejects_unknown_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    update_tool = _find_tool(section, "planning.update_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )

    handler = update_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(UpdateStep(step_id="S999", title="rename"))


def test_mark_step_appends_note_and_updates_status() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=(NewPlanStep(title="draft"), NewPlanStep(title="review")),
        ),
    )
    _invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done", note="notes added"),
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    first, second = plan.steps
    assert first.status == "done"
    assert first.notes == ("notes added",)
    assert plan.status == "active"


def test_mark_step_rejects_abandoned_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")
    clear_tool = _find_tool(section, "planning.clear_plan")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )
    _invoke_tool(bus, clear_tool, ClearPlan())

    handler = mark_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(MarkStep(step_id="S001", status="done"))


def test_mark_step_requires_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )

    handler = mark_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(MarkStep(step_id="  ", status="done"))


def test_mark_step_rejects_empty_note() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )

    handler = mark_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(MarkStep(step_id="S001", status="done", note="   "))


def test_mark_step_rejects_overlong_note() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    mark_tool = _find_tool(section, "planning.mark_step")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
    )

    handler = mark_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(MarkStep(step_id="S001", status="done", note="x" * 513))


def test_clear_plan_marks_status_abandoned() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    clear_tool = _find_tool(section, "planning.clear_plan")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))
    _invoke_tool(bus, clear_tool, ClearPlan())

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "abandoned"
    assert plan.steps == ()


def test_clear_plan_rejects_when_already_abandoned() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    clear_tool = _find_tool(section, "planning.clear_plan")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))
    _invoke_tool(bus, clear_tool, ClearPlan())

    handler = clear_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(ClearPlan())


def test_read_plan_returns_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    read_tool = _find_tool(section, "planning.read_plan")

    _invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=(NewPlanStep(title="draft"), NewPlanStep(title="review")),
        ),
    )

    result = _invoke_tool(bus, read_tool, ReadPlan())
    assert isinstance(result.value, Plan)
    assert "Status: active" in result.message


def test_read_plan_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    read_tool = _find_tool(section, "planning.read_plan")

    handler = read_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(ReadPlan())


def test_read_plan_reports_empty_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = _find_tool(section, "planning.setup_plan")
    clear_tool = _find_tool(section, "planning.clear_plan")
    read_tool = _find_tool(section, "planning.read_plan")

    _invoke_tool(bus, setup_tool, SetupPlan(objective="ship", initial_steps=()))
    _invoke_tool(bus, clear_tool, ClearPlan())

    result = _invoke_tool(bus, read_tool, ReadPlan())
    assert "no steps recorded" in result.message


def test_reducers_ignore_events_without_plan() -> None:
    add_event = _make_tool_data(
        "planning.add_step",
        AddStep(steps=(NewPlanStep(title="draft"),)),
    )
    update_event = _make_tool_data(
        "planning.update_step",
        UpdateStep(step_id="S001", title="rename"),
    )
    mark_event = _make_tool_data(
        "planning.mark_step",
        MarkStep(step_id="S001", status="done"),
    )
    clear_event = _make_tool_data("planning.clear_plan", ClearPlan())

    assert _add_step_reducer((), add_event) == ()
    assert _update_step_reducer((), update_event) == ()
    assert _mark_step_reducer((), mark_event) == ()
    assert _clear_plan_reducer((), clear_event) == ()


def test_latest_plan_returns_none_when_empty() -> None:
    assert _latest_plan(()) is None


def test_next_step_index_skips_non_numeric_suffix() -> None:
    steps = (
        PlanStep(step_id="S001", title="a", details=None, status="pending"),
        PlanStep(step_id="Sbad", title="b", details=None, status="pending"),
    )
    assert _next_step_index(steps) == 1
