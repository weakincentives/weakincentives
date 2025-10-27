from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts.errors import PromptRenderError
from weakincentives.prompts.section import Section
from weakincentives.prompts.tool import Tool, ToolResult, ToolsSection


@dataclass
class ExampleParams:
    primary_tool: str


@dataclass
class MissingParams:
    other: str = "unused"


@dataclass
class ToolParams:
    value: str


@dataclass
class ToolResultPayload:
    value: str


class _BareSection(Section[ExampleParams]):
    def render(self, params: ExampleParams, depth: int) -> str:
        return ""


def _handler(params: ToolParams) -> ToolResult[ToolResultPayload]:
    return ToolResult(
        message=params.value, payload=ToolResultPayload(value=params.value)
    )


def _build_tool(name: str) -> Tool[ToolParams, ToolResultPayload]:
    return Tool[ToolParams, ToolResultPayload](
        name=name,
        description="echo the provided value",
        handler=_handler,
    )


def test_tools_section_inherits_section() -> None:
    assert issubclass(ToolsSection, Section)


def test_tools_section_renders_heading_and_description() -> None:
    example_tool = _build_tool("echo")
    section = ToolsSection[ExampleParams](
        title="Available Tools",
        tools=[example_tool],
        description="""
        Prefer ${primary_tool} when you need structured output.
        """,
    )

    rendered = section.render(ExampleParams(primary_tool="echo"), depth=1)

    assert rendered == (
        "### Available Tools\n\nPrefer echo when you need structured output."
    )


def test_section_tools_defaults_to_empty_tuple() -> None:
    section = _BareSection(title="Base")

    assert section.tools() == ()


def test_tools_section_exposes_tools_in_order() -> None:
    first = _build_tool("first")
    second = _build_tool("second")
    section = ToolsSection[ExampleParams](
        title="Order",
        tools=[first, second],
    )

    tools = section.tools()

    assert tools == (first, second)
    assert tools[0] is first
    assert tools[1] is second


def test_tools_section_renders_heading_without_description() -> None:
    section = ToolsSection[ExampleParams](
        title="Just Heading",
        tools=[_build_tool("first")],
    )

    rendered = section.render(ExampleParams(primary_tool="first"), depth=0)

    assert rendered == "## Just Heading"


def test_tools_section_ignores_empty_description_after_substitute() -> None:
    section = ToolsSection[ExampleParams](
        title="Empty",
        tools=[_build_tool("first")],
        description="${primary_tool}",
    )

    rendered = section.render(ExampleParams(primary_tool=""), depth=2)

    assert rendered == "#### Empty"


def test_tools_section_placeholder_names_handles_named_and_braced() -> None:
    section = ToolsSection[ExampleParams](
        title="Placeholders",
        tools=[_build_tool("echo")],
        description="""
        Prefer ${primary_tool} alongside $primary_tool.
        """,
    )

    placeholders = section.placeholder_names()

    assert placeholders == {"primary_tool"}


def test_tools_section_placeholder_names_returns_empty_for_missing_description() -> (
    None
):
    section = ToolsSection[ExampleParams](
        title="No Description",
        tools=[_build_tool("first")],
    )

    assert section.placeholder_names() == set()


def test_tools_section_raises_render_error_for_missing_placeholder() -> None:
    example_tool = _build_tool("echo")
    section = ToolsSection[ExampleParams](
        title="Available Tools",
        tools=[example_tool],
        description="Use ${primary_tool} wisely.",
    )

    with pytest.raises(PromptRenderError) as error_info:
        section.render(MissingParams(), depth=0)  # type: ignore[arg-type]

    error = error_info.value
    assert isinstance(error, PromptRenderError)
    assert error.placeholder == "primary_tool"


def test_tools_section_requires_tools() -> None:
    with pytest.raises(ValueError) as error_info:
        ToolsSection[ExampleParams](
            title="Empty",
            tools=[],
        )

    assert "Tool instance" in str(error_info.value)


def test_tools_section_rejects_non_tool_entries() -> None:
    with pytest.raises(TypeError) as error_info:
        ToolsSection[ExampleParams](
            title="Invalid",
            tools=["oops"],  # type: ignore[arg-type]
        )

    assert "Tool instances" in str(error_info.value)
