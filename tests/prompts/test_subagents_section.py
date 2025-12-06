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

import pytest

from weakincentives.prompt.errors import PromptRenderError
from weakincentives.tools import SubagentsSection
from weakincentives.tools.subagents import dispatch_subagents


def test_subagents_section_render_mentions_tool() -> None:
    section = SubagentsSection()
    tools = section.tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == dispatch_subagents.name

    params_type = section.param_type
    assert params_type is not None
    params = params_type()
    rendered = section.render(params, depth=0, number="1")

    assert "dispatch_subagents" in rendered
    assert "parallel" in rendered.lower()


def test_subagents_section_rejects_missing_params() -> None:
    section = SubagentsSection()

    with pytest.raises(PromptRenderError):
        section.render(None, depth=0, number="1")


def test_subagents_section_respects_accepts_overrides() -> None:
    section = SubagentsSection(accepts_overrides=True)
    tools = section.tools()
    assert len(tools) == 1
    tool = tools[0]

    assert section.accepts_overrides is True
    assert tool.accepts_overrides is True

    params_type = section.param_type
    assert params_type is not None
    rendered = section.render(params_type(), depth=1, number="1.1")

    assert rendered.startswith("### 1.1. Delegation")
