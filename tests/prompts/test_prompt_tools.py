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

from dataclasses import dataclass
from typing import Any, cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    Section,
    SectionVisibility,
)
from weakincentives.prompt.errors import PromptValidationError
from weakincentives.prompt.tool import Tool


@dataclass
class GuidanceParams:
    primary_tool: str
    allow_tools: bool = True


@dataclass
class PrimarySectionParams:
    label: str = "primary_lookup"


@dataclass
class SecondaryToggleParams:
    enabled: bool = True


@dataclass
class PrimaryToolParams:
    message: str


@dataclass
class PrimaryToolPayload:
    message: str


@dataclass
class SecondaryToolParams:
    count: int


@dataclass
class SecondaryToolPayload:
    count: int


def _build_primary_tool() -> Tool[PrimaryToolParams, PrimaryToolPayload]:
    return Tool[PrimaryToolParams, PrimaryToolPayload](
        name="primary_lookup",
        description="Perform the primary lookup operation.",
        handler=None,
    )


def _build_secondary_tool() -> Tool[SecondaryToolParams, SecondaryToolPayload]:
    return Tool[SecondaryToolParams, SecondaryToolPayload](
        name="secondary_fetch",
        description="Perform the secondary fetch operation.",
        handler=None,
    )


def _build_prompt() -> tuple[
    PromptTemplate,
    Tool[PrimaryToolParams, PrimaryToolPayload],
    Tool[SecondaryToolParams, SecondaryToolPayload],
]:
    primary_tool = _build_primary_tool()
    secondary_tool = _build_secondary_tool()

    primary_section = MarkdownSection[PrimarySectionParams](
        title="Primary",
        template="",
        key="primary",
        tools=[primary_tool],
        default_params=PrimarySectionParams(),
    )
    guidance = MarkdownSection[GuidanceParams](
        title="Guidance",
        template=f"Use ${'{' + 'primary_tool' + '}'} when available.",
        key="guidance",
        enabled=lambda params: params.allow_tools,
        children=[primary_section],
    )
    secondary = MarkdownSection[SecondaryToggleParams](
        title="Secondary",
        template="",
        key="secondary",
        tools=[secondary_tool],
        default_params=SecondaryToggleParams(enabled=True),
        enabled=lambda params: params.enabled,
    )

    return (
        PromptTemplate(
            ns="tests/prompts",
            key="tools-basic",
            sections=[guidance, secondary],
        ),
        primary_tool,
        secondary_tool,
    )


def test_prompt_tools_depth_first_and_enablement() -> None:
    prompt, primary_tool, secondary_tool = _build_prompt()

    default_tools = prompt.render(GuidanceParams(primary_tool="primary_lookup")).tools

    assert default_tools == (primary_tool, secondary_tool)

    disabled_parent = prompt.render(
        GuidanceParams(primary_tool="primary_lookup", allow_tools=False)
    ).tools

    assert disabled_parent == (secondary_tool,)

    disabled_secondary = prompt.render(
        GuidanceParams(primary_tool="primary_lookup"),
        SecondaryToggleParams(enabled=False),
    ).tools

    assert disabled_secondary == (primary_tool,)


def test_prompt_tools_rejects_duplicate_tool_names() -> None:
    first_section = MarkdownSection[PrimarySectionParams](
        title="First Tools",
        template="",
        key="first-tools",
        tools=[_build_primary_tool()],
        default_params=PrimarySectionParams(),
    )
    second_section = MarkdownSection[SecondaryToggleParams](
        title="Second Tools",
        template="",
        key="second-tools",
        tools=[_build_primary_tool()],
        default_params=SecondaryToggleParams(),
    )

    with pytest.raises(PromptValidationError) as error_info:
        PromptTemplate(
            ns="tests/prompts",
            key="tools-duplicate",
            sections=[first_section, second_section],
        )

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("second-tools",)
    assert error.dataclass_type is PrimaryToolParams


def test_prompt_tools_allows_duplicate_tool_params_dataclass() -> None:
    primary_tool = _build_primary_tool()
    alternate_tool = Tool[PrimaryToolParams, PrimaryToolPayload](
        name="alternate_primary",
        description="Alternate primary operation.",
        handler=None,
    )

    first_section = MarkdownSection[PrimarySectionParams](
        title="Primary",
        template="",
        key="primary",
        tools=[primary_tool],
        default_params=PrimarySectionParams(),
    )
    second_section = MarkdownSection[SecondaryToggleParams](
        title="Alternate",
        template="",
        key="alternate",
        tools=[alternate_tool],
        default_params=SecondaryToggleParams(),
    )

    prompt = PromptTemplate(
        ns="tests/prompts",
        key="tools-duplicate-params",
        sections=[first_section, second_section],
    )

    tools = prompt.render().tools
    assert {tool.name for tool in tools} == {"primary_lookup", "alternate_primary"}
    assert all(tool.params_type is PrimaryToolParams for tool in tools)


class _InvalidToolSection(Section[GuidanceParams]):
    def render(
        self,
        params: GuidanceParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, depth, number, visibility
        _ = self.title
        return ""

    def tools(self) -> tuple[Any, ...]:
        _ = self.key
        return ("not-a-tool",)

    def clone(self, **kwargs: object) -> _InvalidToolSection:
        return _InvalidToolSection(title=self.title, key=self.key)


def test_prompt_tools_requires_tool_instances() -> None:
    invalid_section = _InvalidToolSection(title="Invalid", key="invalid")

    with pytest.raises(PromptValidationError) as error_info:
        PromptTemplate(
            ns="tests/prompts",
            key="tools-invalid-instance",
            sections=[invalid_section],
        )

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("invalid",)
    assert error.dataclass_type is GuidanceParams


def test_prompt_tools_rejects_tool_with_non_dataclass_params_type() -> None:
    tool = _build_primary_tool()
    tool.params_type = str

    section = MarkdownSection[PrimarySectionParams](
        title="Primary",
        template="",
        key="primary",
        tools=[tool],
        default_params=PrimarySectionParams(),
    )

    with pytest.raises(PromptValidationError) as error_info:
        PromptTemplate(
            ns="tests/prompts",
            key="tools-bad-params-type",
            sections=[section],
        )

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("primary",)
    assert error.dataclass_type is str
