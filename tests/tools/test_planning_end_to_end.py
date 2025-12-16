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

from typing import cast

from tests.tools.helpers import invoke_tool
from weakincentives.contrib.tools import (
    AddStep,
    Plan,
    PlanningToolsSection,
    ReadPlan,
    SetupPlan,
    UpdateStep,
)
from weakincentives.prompt.tool import Tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


def test_planning_end_to_end_flow() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    tools = {tool.name: tool for tool in section.tools()}

    setup_tool = cast(Tool[SetupPlan, SetupPlan], tools["planning_setup_plan"])
    add_tool = cast(Tool[AddStep, AddStep], tools["planning_add_step"])
    update_tool = cast(Tool[UpdateStep, UpdateStep], tools["planning_update_step"])
    read_tool = cast(Tool[ReadPlan, Plan], tools["planning_read_plan"])

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

    plan = session[Plan].latest()
    assert plan is not None
    assert [step.title for step in plan.steps] == [
        "triage requests",
        "categorise follow-ups",
        "draft update",
    ]

    invoke_tool(
        update_tool,
        UpdateStep(step_id=2, title="categorise replies"),
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

    result = invoke_tool(read_tool, ReadPlan(), session=session)
    assert result.message == "Retrieved the current plan with 3 steps."
