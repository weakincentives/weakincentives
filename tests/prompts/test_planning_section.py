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

"""Snapshot coverage for the planning tools section."""

from __future__ import annotations

import pytest

from weakincentives.prompt import PromptRenderError
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import PlanningStrategy, PlanningToolsSection


def _make_section(
    *,
    strategy: PlanningStrategy | None = None,
    accepts_overrides: bool = False,
) -> PlanningToolsSection:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    if strategy is None:
        return PlanningToolsSection(
            session=session, accepts_overrides=accepts_overrides
        )
    return PlanningToolsSection(
        session=session,
        strategy=strategy,
        accepts_overrides=accepts_overrides,
    )


def _render_section(strategy: PlanningStrategy | None = None) -> str:
    section = _make_section() if strategy is None else _make_section(strategy=strategy)

    params = section.default_params
    assert params is not None

    return section.render(params, depth=0, number="1")


def test_planning_section_renders_instructions() -> None:
    body = _render_section()

    assert "planning_setup_plan" in body
    assert "planning_read_plan" in body
    assert "multi-step" in body

    section = _make_section()
    tool_names = tuple(tool.name for tool in section.tools())
    assert tool_names == (
        "planning_setup_plan",
        "planning_add_step",
        "planning_update_step",
        "planning_read_plan",
    )


def test_planning_section_default_strategy_matches_react() -> None:
    default_body = _render_section()
    react_body = _render_section(PlanningStrategy.REACT)

    assert default_body == react_body


def test_plan_act_reflect_strategy_injects_guidance() -> None:
    body = _render_section(PlanningStrategy.PLAN_ACT_REFLECT)

    assert "plan-act-reflect" in body
    assert "update the step status" in body


def test_goal_decompose_route_synthesise_strategy_injects_guidance() -> None:
    body = _render_section(PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE)

    assert "restating the goal" in body
    assert "Break the goal into concrete sub-problems" in body
    assert "synthesise the results" in body


def test_planning_section_original_body_template_tracks_strategy() -> None:
    par_section = _make_section(strategy=PlanningStrategy.PLAN_ACT_REFLECT)
    assert "plan-act-reflect" in par_section.original_body_template()

    goal_section = _make_section(
        strategy=PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE,
    )
    assert "restating the goal" in goal_section.original_body_template()


def test_planning_section_rejects_missing_params() -> None:
    section = _make_section()

    with pytest.raises(PromptRenderError):
        section.render(None, depth=0, number="1")
