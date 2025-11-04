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

"""Tests for the SubagentsSection prompt integration."""

from __future__ import annotations

from weakincentives.prompt import Prompt
from weakincentives.prompt.subagents import SubagentsSection
from weakincentives.tools.subagents import dispatch_subagents


def test_section_exposes_dispatch_tool() -> None:
    section = SubagentsSection()

    tools = section.tools()

    assert tools == (dispatch_subagents,)


def test_section_renders_parallel_instruction() -> None:
    prompt = Prompt(
        ns="tests.subagents",
        key="section",
        sections=(SubagentsSection(),),
    )

    rendered = prompt.render()

    assert "dispatch_subagents" in rendered.text
    assert "MUST" in rendered.text
    assert "recap" in rendered.text.lower()
