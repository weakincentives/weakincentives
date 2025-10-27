from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from weakincentives.prompts import Prompt, Section, TextSection
from weakincentives.prompts.errors import PromptValidationError
from weakincentives.prompts.tool import Tool


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
    Prompt,
    Tool[PrimaryToolParams, PrimaryToolPayload],
    Tool[SecondaryToolParams, SecondaryToolPayload],
]:
    primary_tool = _build_primary_tool()
    secondary_tool = _build_secondary_tool()

    primary_section = TextSection[PrimarySectionParams](
        title="Primary",
        body="",
        tools=[primary_tool],
        defaults=PrimarySectionParams(),
    )
    guidance = TextSection[GuidanceParams](
        title="Guidance",
        body="Use ${primary_tool} when available.",
        enabled=lambda params: params.allow_tools,
        children=[primary_section],
    )
    secondary = TextSection[SecondaryToggleParams](
        title="Secondary",
        body="",
        tools=[secondary_tool],
        defaults=SecondaryToggleParams(enabled=True),
        enabled=lambda params: params.enabled,
    )

    return Prompt(sections=[guidance, secondary]), primary_tool, secondary_tool


def test_prompt_tools_depth_first_and_enablement() -> None:
    prompt, primary_tool, secondary_tool = _build_prompt()

    default_tools = prompt.tools(GuidanceParams(primary_tool="primary_lookup"))

    assert default_tools == (primary_tool, secondary_tool)

    disabled_parent = prompt.tools(
        GuidanceParams(primary_tool="primary_lookup", allow_tools=False)
    )

    assert disabled_parent == (secondary_tool,)

    disabled_secondary = prompt.tools(
        GuidanceParams(primary_tool="primary_lookup"),
        SecondaryToggleParams(enabled=False),
    )

    assert disabled_secondary == (primary_tool,)


def test_prompt_tools_rejects_duplicate_tool_names() -> None:
    first_section = TextSection[PrimarySectionParams](
        title="First Tools",
        body="",
        tools=[_build_primary_tool()],
        defaults=PrimarySectionParams(),
    )
    second_section = TextSection[SecondaryToggleParams](
        title="Second Tools",
        body="",
        tools=[_build_primary_tool()],
        defaults=SecondaryToggleParams(),
    )

    with pytest.raises(PromptValidationError) as error_info:
        Prompt(sections=[first_section, second_section])

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("Second Tools",)
    assert error.dataclass_type is PrimaryToolParams


def test_prompt_tools_allows_duplicate_tool_params_dataclass() -> None:
    primary_tool = _build_primary_tool()
    alternate_tool = Tool[PrimaryToolParams, PrimaryToolPayload](
        name="alternate_primary",
        description="Alternate primary operation.",
        handler=None,
    )

    first_section = TextSection[PrimarySectionParams](
        title="Primary",
        body="",
        tools=[primary_tool],
        defaults=PrimarySectionParams(),
    )
    second_section = TextSection[SecondaryToggleParams](
        title="Alternate",
        body="",
        tools=[alternate_tool],
        defaults=SecondaryToggleParams(),
    )

    prompt = Prompt(sections=[first_section, second_section])

    tools = prompt.tools()
    assert {tool.name for tool in tools} == {"primary_lookup", "alternate_primary"}
    assert all(tool.params_type is PrimaryToolParams for tool in tools)


class _InvalidToolSection(Section[GuidanceParams]):
    def render(self, params: GuidanceParams, depth: int) -> str:
        return ""

    def tools(self) -> tuple[Any, ...]:
        return ("not-a-tool",)


def test_prompt_tools_requires_tool_instances() -> None:
    invalid_section = _InvalidToolSection(title="Invalid")

    with pytest.raises(PromptValidationError) as error_info:
        Prompt(sections=[invalid_section])

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("Invalid",)
    assert error.dataclass_type is GuidanceParams


def test_prompt_tools_rejects_tool_with_non_dataclass_params_type() -> None:
    tool = _build_primary_tool()
    tool.params_type = str  # type: ignore[assignment]

    section = TextSection[PrimarySectionParams](
        title="Primary",
        body="",
        tools=[tool],
        defaults=PrimarySectionParams(),
    )

    with pytest.raises(PromptValidationError) as error_info:
        Prompt(sections=[section])

    error = cast(PromptValidationError, error_info.value)
    assert error.section_path == ("Primary",)
    assert error.dataclass_type is str
