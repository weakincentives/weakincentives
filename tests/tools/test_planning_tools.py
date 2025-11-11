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

from datetime import UTC, datetime
from typing import cast

import pytest

from tests.tools.helpers import find_tool, invoke_tool
from weakincentives.adapters.core import SessionProtocol
from weakincentives.prompt import SupportsDataclass
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import (
    ReducerContext,
    Session,
    build_reducer_context,
    select_all,
    select_latest,
)
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


def _make_tool_event(name: str, value: SupportsDataclass) -> ToolInvoked:
    result = ToolResult(message="ok", value=value)
    return ToolInvoked(
        prompt_name="test",
        adapter="adapter",
        name=name,
        params=value,
        result=cast(ToolResult[object], result),
        session_id="session-example",
        created_at=datetime.now(UTC),
        duration_ms=0.0,
        value=value,
    )


def _make_reducer_context() -> ReducerContext:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    return build_reducer_context(session=session, event_bus=bus)


def test_setup_plan_normalizes_payloads() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")

    params = SetupPlan(
        objective="  refine onboarding flow  ",
        initial_steps=(NewPlanStep(title=" draft checklist ", details=" review doc "),),
    )
    invoke_tool(bus, setup_tool, params, session=session)

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
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")

    params = SetupPlan(
        objective="ship",
        initial_steps=(NewPlanStep(title="draft", details="   "),),
    )
    invoke_tool(bus, setup_tool, params, session=session)

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.steps[0].details is None


def test_setup_plan_rejects_invalid_objective() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, setup_tool, SetupPlan(objective="   "), session=session)

    long_objective = "x" * 241
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            setup_tool,
            SetupPlan(objective=long_objective),
            session=session,
        )


def test_setup_plan_requires_session_in_context() -> None:
    bus = InProcessEventBus()
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")

    params = SetupPlan(objective="ship", initial_steps=())

    handler = setup_tool.handler
    assert handler is not None
    context = ToolContext(
        prompt=None,
        rendered_prompt=None,
        adapter=None,
        session=cast(SessionProtocol, object()),
        event_bus=bus,
    )

    with pytest.raises(ToolValidationError):
        handler(params, context=context)


def test_add_step_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    add_tool = find_tool(section, "planning_add_step")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            add_tool,
            AddStep(steps=(NewPlanStep(title="task"),)),
            session=session,
        )


def test_add_step_appends_new_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )
    invoke_tool(
        bus,
        add_tool,
        AddStep(
            steps=(
                NewPlanStep(title="review"),
                NewPlanStep(title="release"),
            )
        ),
        session=session,
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert [step.step_id for step in plan.steps] == ["S001", "S002", "S003"]
    assert [step.title for step in plan.steps] == ["draft", "review", "release"]


def test_add_step_rejects_empty_payload() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, add_tool, AddStep(steps=()), session=session)


def test_add_step_rejects_when_plan_not_active() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            add_tool,
            AddStep(steps=(NewPlanStep(title="later"),)),
            session=session,
        )


def test_session_keeps_single_plan_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )
    invoke_tool(
        bus,
        add_tool,
        AddStep(steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    snapshots = select_all(session, Plan)
    assert len(snapshots) == 1


def test_update_step_rejects_empty_patch() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, update_tool, UpdateStep(step_id="S001"), session=session)


def test_update_step_updates_existing_step() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=(
                NewPlanStep(title="triage requests"),
                NewPlanStep(title="categorise follow-ups"),
            ),
        ),
        session=session,
    )

    invoke_tool(
        bus,
        update_tool,
        UpdateStep(step_id="S002", title="categorise replies"),
        session=session,
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert [step.title for step in plan.steps] == [
        "triage requests",
        "categorise replies",
    ]
    assert plan.steps[1].step_id == "S002"


def test_update_step_requires_step_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            update_tool,
            UpdateStep(step_id="  ", title="rename"),
            session=session,
        )


def test_update_step_rejects_unknown_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            update_tool,
            UpdateStep(step_id="S999", title="rename"),
            session=session,
        )


def test_mark_step_appends_note_and_updates_status() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=(NewPlanStep(title="draft"), NewPlanStep(title="review")),
        ),
        session=session,
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done", note="notes added"),
        session=session,
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    first, _second = plan.steps
    assert first.status == "done"
    assert first.notes == ("notes added",)
    assert plan.status == "active"


def test_mark_step_sets_plan_completed_when_all_done() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")
    mark_tool = find_tool(section, "planning_mark_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="resolve support backlog",
            initial_steps=(
                NewPlanStep(title="triage requests"),
                NewPlanStep(title="categorise follow-ups"),
            ),
        ),
        session=session,
    )
    invoke_tool(
        bus,
        add_tool,
        AddStep(steps=(NewPlanStep(title="draft update"),)),
        session=session,
    )

    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done", note="triage complete"),
        session=session,
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S002", status="done"),
        session=session,
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S003", status="done"),
        session=session,
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "completed"
    assert plan.steps[0].notes == ("triage complete",)
    assert all(step.status == "done" for step in plan.steps)


def test_mark_step_rejects_abandoned_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")
    clear_tool = find_tool(section, "planning_clear_plan")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )
    invoke_tool(bus, clear_tool, ClearPlan(), session=session)

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            mark_tool,
            MarkStep(step_id="S001", status="done"),
            session=session,
        )


def test_mark_step_requires_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            mark_tool,
            MarkStep(step_id="  ", status="done"),
            session=session,
        )


def test_mark_step_rejects_empty_note() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            mark_tool,
            MarkStep(step_id="S001", status="done", note="   "),
            session=session,
        )


def test_mark_step_rejects_overlong_note() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    mark_tool = find_tool(section, "planning_mark_step")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=(NewPlanStep(title="draft"),)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            mark_tool,
            MarkStep(step_id="S001", status="done", note="x" * 513),
            session=session,
        )


def test_clear_plan_marks_status_abandoned() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    clear_tool = find_tool(section, "planning_clear_plan")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )
    invoke_tool(bus, clear_tool, ClearPlan(), session=session)

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "abandoned"
    assert plan.steps == ()


def test_clear_plan_rejects_when_already_abandoned() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    clear_tool = find_tool(section, "planning_clear_plan")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )
    invoke_tool(bus, clear_tool, ClearPlan(), session=session)

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, clear_tool, ClearPlan(), session=session)


def test_read_plan_returns_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    read_tool = find_tool(section, "planning_read_plan")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=(NewPlanStep(title="draft"), NewPlanStep(title="review")),
        ),
        session=session,
    )

    result = invoke_tool(bus, read_tool, ReadPlan(), session=session)
    assert isinstance(result.value, Plan)
    assert result.message == "Retrieved the current plan with 2 steps."


def test_read_plan_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    read_tool = find_tool(section, "planning_read_plan")

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, read_tool, ReadPlan(), session=session)


def test_read_plan_reports_empty_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection()
    setup_tool = find_tool(section, "planning_setup_plan")
    clear_tool = find_tool(section, "planning_clear_plan")
    read_tool = find_tool(section, "planning_read_plan")

    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )
    invoke_tool(bus, clear_tool, ClearPlan(), session=session)

    result = invoke_tool(bus, read_tool, ReadPlan(), session=session)
    assert result.message == "Retrieved the current plan (no steps recorded)."


def test_reducers_ignore_events_without_plan() -> None:
    add_event = _make_tool_event(
        "planning_add_step",
        AddStep(steps=(NewPlanStep(title="draft"),)),
    )
    update_event = _make_tool_event(
        "planning_update_step",
        UpdateStep(step_id="S001", title="rename"),
    )
    mark_event = _make_tool_event(
        "planning_mark_step",
        MarkStep(step_id="S001", status="done"),
    )
    clear_event = _make_tool_event("planning_clear_plan", ClearPlan())

    context = _make_reducer_context()

    assert add_event.duration_ms == 0.0
    assert _add_step_reducer((), add_event, context=context) == ()
    assert _update_step_reducer((), update_event, context=context) == ()
    assert _mark_step_reducer((), mark_event, context=context) == ()
    assert _clear_plan_reducer((), clear_event, context=context) == ()


def test_latest_plan_returns_none_when_empty() -> None:
    assert _latest_plan(()) is None


def test_next_step_index_skips_non_numeric_suffix() -> None:
    steps = (
        PlanStep(step_id="S001", title="a", details=None, status="pending"),
        PlanStep(step_id="Sbad", title="b", details=None, status="pending"),
    )
    assert _next_step_index(steps) == 1


def test_planning_tools_section_disables_tool_overrides_by_default() -> None:
    section = PlanningToolsSection()

    assert section.accepts_overrides is False
    assert all(tool.accepts_overrides is False for tool in section.tools())


def test_planning_tools_section_allows_configuring_overrides() -> None:
    section = PlanningToolsSection(
        accepts_overrides=True,
    )

    add_tool = find_tool(section, "planning_add_step")
    read_tool = find_tool(section, "planning_read_plan")
    setup_tool = find_tool(section, "planning_setup_plan")

    assert section.accepts_overrides is True
    assert add_tool.accepts_overrides is True
    assert read_tool.accepts_overrides is True
    assert setup_tool.accepts_overrides is True
