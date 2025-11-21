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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from weakincentives.prompt.markdown import MarkdownSection
from weakincentives.prompt.section import Section
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult


@dataclass
class SectionParams:
    title: str = "placeholder"


@dataclass
class ToolParams:
    name: str


@dataclass
class ToolPayload:
    message: str


class _BareSection(Section[SectionParams]):
    def __init__(
        self,
        *,
        title: str,
        key: str,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
    ) -> None:
        super().__init__(
            title=title,
            key=key,
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    def render(self, params: SectionParams, depth: int, number: str) -> str:
        return ""


def _handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
    del context
    return ToolResult(message=params.name, value=ToolPayload(message=params.name))


def _build_tool(
    name: str, *, accepts_overrides: bool = True
) -> Tool[ToolParams, ToolPayload]:
    return Tool[ToolParams, ToolPayload](
        name=name,
        description="echo the provided name",
        handler=_handler,
        accepts_overrides=accepts_overrides,
    )


def test_sections_default_to_no_tools() -> None:
    section = _BareSection(title="Base", key="base")

    assert section.tools() == ()


def test_sections_reject_non_tool_entries() -> None:
    with pytest.raises(TypeError):
        _BareSection(title="Invalid", key="invalid", tools=["oops"])


def test_sections_expose_tools_in_order() -> None:
    first = _build_tool("first")
    second = _build_tool("second")
    section = _BareSection(title="With Tools", key="with-tools", tools=[first, second])

    tools = section.tools()

    assert tools == (first, second)
    assert tools[0] is first
    assert tools[1] is second


def test_text_section_accepts_tools() -> None:
    tool = _build_tool("text")
    section = MarkdownSection[SectionParams](
        title="Paragraph",
        template="Hello",
        key="paragraph",
        tools=[tool],
    )

    assert section.tools() == (tool,)


def test_section_accepts_overrides_flag_defaults_true() -> None:
    section = _BareSection(title="Defaults", key="defaults")

    assert section.accepts_overrides is True


def test_section_accepts_overrides_flag_can_be_disabled() -> None:
    section = _BareSection(
        title="Locked",
        key="locked",
        accepts_overrides=False,
    )

    assert section.accepts_overrides is False


def test_tool_accepts_overrides_flag_defaults_true() -> None:
    tool = _build_tool("defaults")

    assert tool.accepts_overrides is True


def test_tool_accepts_overrides_flag_can_be_disabled() -> None:
    tool = _build_tool("locked", accepts_overrides=False)

    assert tool.accepts_overrides is False
