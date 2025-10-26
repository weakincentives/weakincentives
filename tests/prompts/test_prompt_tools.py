from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompts import Prompt, TextSection
from weakincentives.prompts.tool import Tool, ToolsSection


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

    guidance = TextSection(
        title="Guidance",
        body="Use ${primary_tool} when available.",
        params=GuidanceParams,
        enabled=lambda params: params.allow_tools,
        children=[
            ToolsSection(
                title="Primary",
                tools=[primary_tool],
                params=PrimarySectionParams,
                defaults=PrimarySectionParams(),
            )
        ],
    )
    secondary = ToolsSection(
        title="Secondary",
        tools=[secondary_tool],
        params=SecondaryToggleParams,
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
