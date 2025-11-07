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

"""Coverage tests for the SubagentsSection helper."""

from weakincentives.tools import SubagentsSection
from weakincentives.tools.subagents import (
    SubagentIsolationLevel,
    build_dispatch_subagents_tool,
    dispatch_subagents,
)


def test_subagents_section_render_mentions_tool() -> None:
    section = SubagentsSection()
    tools = section.tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == dispatch_subagents.name

    params = section.param_type()
    rendered = section.render(params, depth=0)

    assert "dispatch_subagents" in rendered
    assert "parallel" in rendered.lower()


def test_subagents_section_builds_isolated_tool() -> None:
    section = SubagentsSection(isolation_level=SubagentIsolationLevel.FULL_ISOLATION)
    tools = section.tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool is not dispatch_subagents

    expected = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    assert tool.name == expected.name
    assert tool.description == expected.description
