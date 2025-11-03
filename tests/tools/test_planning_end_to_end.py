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

"""End-to-end flow for the planning tool suite."""

from __future__ import annotations

from tests.tools.helpers import invoke_tool
from weakincentives.events import InProcessEventBus
from weakincentives.session import Session, select_latest
from weakincentives.tools import (
    AddStep,
    ClearPlan,
    MarkStep,
    NewPlanStep,
    Plan,
    PlanningToolsSection,
    ReadPlan,
    SetupPlan,
    UpdateStep,
)


def test_planning_end_to_end_flow() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    tools = {tool.name: tool for tool in section.tools()}

    setup_tool = tools["planning_setup_plan"]
    add_tool = tools["planning_add_step"]
    update_tool = tools["planning_update_step"]
    mark_tool = tools["planning_mark_step"]
    read_tool = tools["planning_read_plan"]
    clear_tool = tools["planning_clear_plan"]

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
    )
    invoke_tool(
        bus,
        add_tool,
        AddStep(steps=(NewPlanStep(title="draft update"),)),
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert [step.title for step in plan.steps] == [
        "triage requests",
        "categorise follow-ups",
        "draft update",
    ]

    invoke_tool(
        bus,
        update_tool,
        UpdateStep(step_id="S002", title="categorise replies"),
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done", note="triage complete"),
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S002", status="done"),
    )
    invoke_tool(
        bus,
        mark_tool,
        MarkStep(step_id="S003", status="done"),
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "completed"
    assert plan.steps[0].notes == ("triage complete",)

    result = invoke_tool(bus, read_tool, ReadPlan())
    assert result.message == "Retrieved the current plan with 3 steps."

    invoke_tool(bus, clear_tool, ClearPlan())

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "abandoned"
    assert plan.steps == ()
