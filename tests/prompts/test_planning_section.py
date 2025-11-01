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

from weakincentives.session import Session
from weakincentives.tools import PlanningToolsSection


def test_planning_section_renders_instructions() -> None:
    session = Session()
    section = PlanningToolsSection(session=session)

    params = section.default_params
    assert params is not None

    body = section.render(params, depth=0)

    assert "planning_setup_plan" in body
    assert "planning_read_plan" in body
    assert "Stay brief" in body
    assert "multi-step" in body

    tool_names = tuple(tool.name for tool in section.tools())
    assert tool_names == (
        "planning_setup_plan",
        "planning_add_step",
        "planning_update_step",
        "planning_mark_step",
        "planning_clear_plan",
        "planning_read_plan",
    )
