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

from typing import TypeVar, cast

from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.prompt import SupportsDataclass
from weakincentives.prompt.tool import Tool, ToolResult
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

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)
ResultT = TypeVar("ResultT", bound=SupportsDataclass)


def _invoke(
    bus: InProcessEventBus,
    tool: Tool[ParamsT, ResultT],
    params: ParamsT,
) -> ToolResult[ResultT]:
    handler = tool.handler
    assert handler is not None
    result = handler(params)
    bus.publish(
        ToolInvoked(
            prompt_name="test",
            adapter="adapter",
            name=tool.name,
            params=params,
            result=cast(ToolResult[object], result),
        )
    )
    return result


def test_planning_end_to_end_flow() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session)
    tools = {tool.name: tool for tool in section.tools()}

    setup_tool = tools["planning.setup_plan"]
    add_tool = tools["planning.add_step"]
    update_tool = tools["planning.update_step"]
    mark_tool = tools["planning.mark_step"]
    read_tool = tools["planning.read_plan"]
    clear_tool = tools["planning.clear_plan"]

    _invoke(
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
    _invoke(
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

    _invoke(
        bus,
        update_tool,
        UpdateStep(step_id="S002", title="categorise replies"),
    )
    _invoke(
        bus,
        mark_tool,
        MarkStep(step_id="S001", status="done", note="triage complete"),
    )
    _invoke(
        bus,
        mark_tool,
        MarkStep(step_id="S002", status="done"),
    )
    _invoke(
        bus,
        mark_tool,
        MarkStep(step_id="S003", status="done"),
    )

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "completed"
    assert plan.steps[0].notes == ("triage complete",)

    result = _invoke(bus, read_tool, ReadPlan())
    assert "Status: completed" in result.message

    _invoke(bus, clear_tool, ClearPlan())

    plan = select_latest(session, Plan)
    assert plan is not None
    assert plan.status == "abandoned"
    assert plan.steps == ()
