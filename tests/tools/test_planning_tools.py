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
from uuid import uuid4

import pytest

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from tests.tools.helpers import build_tool_context, find_tool, invoke_tool
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    AddStep,
    Plan,
    PlanningToolsSection,
    PlanStep,
    ReadPlan,
    SetupPlan,
    UpdateStep,
)
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import Session
from weakincentives.types import SupportsDataclass


def _make_tool_event(name: str, value: SupportsDataclass) -> ToolInvoked:
    result = ToolResult(message="ok", value=value)
    return ToolInvoked(
        prompt_name="test",
        adapter=GENERIC_ADAPTER_NAME,
        name=name,
        params=value,
        result=cast(ToolResult[object], result),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
        rendered_output=result.render(),
    )


def test_plan_render_helpers() -> None:
    step = PlanStep(step_id=1, title="draft", status="pending")
    plan = Plan(objective="ship", status="active", steps=(step,))
    assert "1" in step.render()
    assert "Objective" in plan.render()


def test_plan_command_renders() -> None:
    setup = SetupPlan(objective="ship", initial_steps=("draft",))
    setup_no_steps = SetupPlan(objective="ship", initial_steps=())
    add = AddStep(steps=("refine",))
    empty_add = AddStep(steps=())
    update = UpdateStep(step_id=1, title="rename")
    status_update = UpdateStep(step_id=2, status="done")
    read = ReadPlan()

    assert "ship" in setup.render()
    assert "<none>" in setup_no_steps.render()
    assert "refine" in add.render()
    assert "no steps" in empty_add.render()
    assert "rename" in update.render()
    assert "done" in status_update.render()
    assert "snapshot" in read.render()


def test_planning_tools_reject_mismatched_context_session() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    tool = find_tool(section, "planning_setup_plan")
    handler = tool.handler
    assert handler is not None
    other_bus = InProcessEventBus()
    mismatched_session = Session(bus=other_bus)
    context = build_tool_context(mismatched_session)

    with pytest.raises(RuntimeError, match="session does not match"):
        handler(SetupPlan(objective="ship"), context=context)


def test_setup_plan_normalizes_payloads() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")

    params = SetupPlan(
        objective="  refine onboarding flow  ",
        initial_steps=(" draft checklist ",),
    )
    invoke_tool(setup_tool, params, session=session)

    plan = session[Plan].latest()
    assert plan is not None
    assert plan.objective == "refine onboarding flow"
    assert plan.status == "active"
    assert plan.steps == (
        PlanStep(step_id=1, title="draft checklist", status="pending"),
    )


def test_setup_plan_returns_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")

    params = SetupPlan(
        objective="ship feature",
        initial_steps=("draft", "review"),
    )
    result = invoke_tool(setup_tool, params, session=session)

    assert isinstance(result.value, Plan)
    assert result.value.objective == "ship feature"
    assert result.value.status == "active"
    assert len(result.value.steps) == 2
    assert result.value.steps[0].title == "draft"
    assert result.value.steps[1].title == "review"


def test_setup_plan_rejects_invalid_objective() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")

    with pytest.raises(ToolValidationError):
        invoke_tool(setup_tool, SetupPlan(objective="   "), session=session)

    long_objective = "x" * 501
    with pytest.raises(ToolValidationError):
        invoke_tool(
            setup_tool,
            SetupPlan(objective=long_objective),
            session=session,
        )


def test_add_step_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    add_tool = find_tool(section, "planning_add_step")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            add_tool,
            AddStep(steps=("task",)),
            session=session,
        )


def test_add_step_appends_new_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )
    invoke_tool(
        add_tool,
        AddStep(steps=("review", "release")),
        session=session,
    )

    plan = session[Plan].latest()
    assert plan is not None
    assert [step.step_id for step in plan.steps] == [1, 2, 3]
    assert [step.title for step in plan.steps] == ["draft", "review", "release"]


def test_add_step_returns_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )
    result = invoke_tool(
        add_tool,
        AddStep(steps=("review", "release")),
        session=session,
    )

    assert isinstance(result.value, Plan)
    assert result.value.objective == "ship"
    assert len(result.value.steps) == 3
    assert result.value.steps[1].title == "review"
    assert result.value.steps[2].title == "release"


def test_add_step_rejects_empty_payload() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(add_tool, AddStep(steps=()), session=session)


def test_add_step_rejects_when_plan_not_active() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )
    invoke_tool(
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            add_tool,
            AddStep(steps=("later",)),
            session=session,
        )


def test_session_keeps_single_plan_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )
    invoke_tool(
        add_tool,
        AddStep(steps=("draft",)),
        session=session,
    )

    snapshots = session[Plan].all()
    assert len(snapshots) == 1


def test_update_step_rejects_empty_patch() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(update_tool, UpdateStep(step_id=1), session=session)


def test_update_step_updates_existing_step_title() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=("triage requests", "categorise follow-ups"),
        ),
        session=session,
    )

    invoke_tool(
        update_tool,
        UpdateStep(step_id=2, title="categorise replies"),
        session=session,
    )

    plan = session[Plan].latest()
    assert plan is not None
    assert [step.title for step in plan.steps] == [
        "triage requests",
        "categorise replies",
    ]
    assert plan.steps[1].step_id == 2


def test_update_step_returns_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(
            objective="ship",
            initial_steps=("draft", "review"),
        ),
        session=session,
    )

    result = invoke_tool(
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )

    assert isinstance(result.value, Plan)
    assert result.value.objective == "ship"
    assert result.value.steps[0].status == "done"
    assert result.value.steps[1].status == "pending"


def test_update_step_updates_status() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft", "review")),
        session=session,
    )
    invoke_tool(
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )

    plan = session[Plan].latest()
    assert plan is not None
    assert plan.steps[0].status == "done"
    assert plan.status == "active"  # Not all steps done yet


def test_update_step_rejects_unknown_identifier() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            update_tool,
            UpdateStep(step_id=999, title="rename"),
            session=session,
        )


def test_update_step_sets_plan_completed_when_all_done() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    add_tool = find_tool(section, "planning_add_step")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(
            objective="resolve support backlog",
            initial_steps=("triage requests", "categorise follow-ups"),
        ),
        session=session,
    )
    invoke_tool(
        add_tool,
        AddStep(steps=("draft update",)),
        session=session,
    )

    invoke_tool(
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )
    invoke_tool(
        update_tool,
        UpdateStep(step_id=2, status="done"),
        session=session,
    )
    invoke_tool(
        update_tool,
        UpdateStep(step_id=3, status="done"),
        session=session,
    )

    plan = session[Plan].latest()
    assert plan is not None
    assert plan.status == "completed"
    assert all(step.status == "done" for step in plan.steps)


def test_update_step_rejects_completed_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft",)),
        session=session,
    )
    invoke_tool(
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            update_tool,
            UpdateStep(step_id=1, title="new title"),
            session=session,
        )


def test_read_plan_returns_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    read_tool = find_tool(section, "planning_read_plan")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=("draft", "review")),
        session=session,
    )

    result = invoke_tool(read_tool, ReadPlan(), session=session)
    assert isinstance(result.value, Plan)
    assert result.message == "Retrieved the current plan with 2 steps."


def test_read_plan_requires_existing_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    read_tool = find_tool(section, "planning_read_plan")

    with pytest.raises(ToolValidationError):
        invoke_tool(read_tool, ReadPlan(), session=session)


def test_read_plan_reports_empty_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    setup_tool = find_tool(section, "planning_setup_plan")
    read_tool = find_tool(section, "planning_read_plan")

    invoke_tool(
        setup_tool,
        SetupPlan(objective="ship", initial_steps=()),
        session=session,
    )

    result = invoke_tool(read_tool, ReadPlan(), session=session)
    assert result.message == "Retrieved the current plan (no steps recorded)."


def test_planning_tools_section_disables_tool_overrides_by_default() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)

    assert section.accepts_overrides is False
    assert all(tool.accepts_overrides is False for tool in section.tools())


def test_planning_tools_section_allows_configuring_overrides() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(
        session=session,
        accepts_overrides=True,
    )

    add_tool = find_tool(section, "planning_add_step")
    read_tool = find_tool(section, "planning_read_plan")
    setup_tool = find_tool(section, "planning_setup_plan")

    assert section.accepts_overrides is True
    assert add_tool.accepts_overrides is True
    assert read_tool.accepts_overrides is True
    assert setup_tool.accepts_overrides is True
