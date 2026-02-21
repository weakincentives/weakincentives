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

"""Tests for SUMMARY visibility behavior in rendering.

This module complements test_text_section.py and covers:
- Summary visibility exclusion of tools from rendered prompts
- Summary visibility skipping child sections and their tools
- Default summary visibility behavior
- Sibling section rendering after summarized sections
- render_override edge cases
- read_section vs open_sections tool injection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from weakincentives.prompt import (
    MarkdownSection,
    Section,
    SectionVisibility,
    Tool,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.rendering import PromptRenderer
from weakincentives.prompt.tool import ToolContext, ToolExample, ToolResult
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session, SetVisibilityOverride
from weakincentives.types import SupportsDataclass


@dataclass
class _ToolParams:
    query: str


@dataclass
class _ToolResult:
    answer: str


def _dummy_handler(
    params: _ToolParams, *, context: ToolContext
) -> ToolResult[_ToolResult]:
    del context
    return ToolResult.ok(_ToolResult(answer=params.query), message="ok")


@dataclass
class _SectionParams:
    title: str


def test_summary_visibility_excludes_tools_from_rendered_prompt() -> None:
    section = MarkdownSection[_SectionParams](
        title="Tools Section",
        template="Full content: ${title}",
        key="tools-section",
        summary="Summary: ${title}",
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="search_tool",
                description="Search for something.",
                handler=_dummy_handler,
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_SectionParams(title="Test"),))

    # Without visibility override, tools are included
    rendered_full = renderer.render(params_lookup)
    assert len(rendered_full.tools) == 1
    assert rendered_full.tools[0].name == "search_tool"

    # With SUMMARY visibility via session state, the section's own tools are excluded,
    # but the open_sections tool is injected for progressive disclosure
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=(section.key,), visibility=SectionVisibility.SUMMARY)
    )
    rendered_summary = renderer.render(params_lookup, session=session)
    assert len(rendered_summary.tools) == 1
    assert rendered_summary.tools[0].name == "open_sections"
    assert "Summary: Test" in rendered_summary.text


def test_summary_visibility_skips_child_sections() -> None:
    @dataclass
    class ChildParams:
        detail: str

    parent = MarkdownSection[_SectionParams](
        title="Parent",
        template="Parent full: ${title}",
        key="parent",
        summary="Parent summary: ${title}",
        children=[
            MarkdownSection[ChildParams](
                title="Child",
                template="Child content: ${detail}",
                key="child",
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup(
        (
            _SectionParams(title="Parent Title"),
            ChildParams(detail="Child Detail"),
        )
    )

    # Without visibility override, both parent and child are rendered
    rendered_full = renderer.render(params_lookup)
    assert "Parent full: Parent Title" in rendered_full.text
    assert "Child content: Child Detail" in rendered_full.text

    # With SUMMARY visibility on parent via session state, child is skipped
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=("parent",), visibility=SectionVisibility.SUMMARY)
    )
    rendered_summary = renderer.render(params_lookup, session=session)
    assert "Parent summary: Parent Title" in rendered_summary.text
    assert "Child content" not in rendered_summary.text
    assert "Child Detail" not in rendered_summary.text


def test_summary_visibility_skips_child_tools() -> None:
    @dataclass
    class ChildParams:
        query: str

    child_with_tool = MarkdownSection[ChildParams](
        title="Child With Tool",
        template="Child: ${query}",
        key="child-tool",
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="child_tool",
                description="Child tool description.",
                handler=_dummy_handler,
            )
        ],
    )

    parent = MarkdownSection[_SectionParams](
        title="Parent",
        template="Parent: ${title}",
        key="parent",
        summary="Summary: ${title}",
        children=[child_with_tool],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup(
        (
            _SectionParams(title="Parent"),
            ChildParams(query="test"),
        )
    )

    # Without visibility override, child tool is included
    rendered_full = renderer.render(params_lookup)
    assert len(rendered_full.tools) == 1
    assert rendered_full.tools[0].name == "child_tool"

    # With SUMMARY visibility on parent via session state, child (and its tools) are skipped.
    # The open_sections tool is injected for progressive disclosure.
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=("parent",), visibility=SectionVisibility.SUMMARY)
    )
    rendered_summary = renderer.render(params_lookup, session=session)
    assert len(rendered_summary.tools) == 1
    assert rendered_summary.tools[0].name == "open_sections"


def test_summary_visibility_default_excludes_tools() -> None:
    """Section configured with SUMMARY visibility by default excludes tools."""
    section = MarkdownSection[_SectionParams](
        title="Default Summary",
        template="Full: ${title}",
        key="default-summary",
        summary="Summary: ${title}",
        visibility=SectionVisibility.SUMMARY,  # Default to SUMMARY
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="excluded_tool",
                description="This tool should be excluded.",
                handler=_dummy_handler,
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_SectionParams(title="Test"),))

    # Section's own tools are excluded because default visibility is SUMMARY,
    # but open_sections tool is injected for progressive disclosure
    rendered = renderer.render(params_lookup)
    assert len(rendered.tools) == 1
    assert rendered.tools[0].name == "open_sections"
    assert "Summary: Test" in rendered.text

    # Override to FULL via session state includes tools
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=(section.key,), visibility=SectionVisibility.FULL)
    )
    rendered_full = renderer.render(params_lookup, session=session)
    assert len(rendered_full.tools) == 1
    assert "Full: Test" in rendered_full.text


def test_summary_visibility_sibling_after_summary_is_rendered() -> None:
    """Sibling sections after a SUMMARY section are rendered normally."""

    @dataclass
    class ChildParams:
        detail: str

    @dataclass
    class SiblingParams:
        info: str

    parent_with_child = MarkdownSection[_SectionParams](
        title="Parent",
        template="Parent full: ${title}",
        key="parent",
        summary="Parent summary: ${title}",
        children=[
            MarkdownSection[ChildParams](
                title="Child",
                template="Child content: ${detail}",
                key="child",
            )
        ],
    )

    sibling = MarkdownSection[SiblingParams](
        title="Sibling",
        template="Sibling content: ${info}",
        key="sibling",
    )

    registry = PromptRegistry()
    registry.register_sections(
        (
            cast(Section[SupportsDataclass], parent_with_child),
            cast(Section[SupportsDataclass], sibling),
        )
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup(
        (
            _SectionParams(title="Parent Title"),
            ChildParams(detail="Child Detail"),
            SiblingParams(info="Sibling Info"),
        )
    )

    # With SUMMARY visibility on parent via session state, child is skipped but sibling is rendered
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=("parent",), visibility=SectionVisibility.SUMMARY)
    )
    rendered = renderer.render(params_lookup, session=session)
    assert "Parent summary: Parent Title" in rendered.text
    assert "Child content" not in rendered.text  # Child is skipped
    assert "Sibling content: Sibling Info" in rendered.text  # Sibling is rendered


def test_markdown_section_render_override_with_empty_body_and_tools() -> None:
    """render_override handles empty body with tools that have examples."""

    section = MarkdownSection[_SectionParams](
        title="ToolsOnly",
        template="Default: ${title}",
        key="tools-only",
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="override_tool",
                description="Test tool for override.",
                handler=_dummy_handler,
                examples=(
                    ToolExample[_ToolParams, _ToolResult](
                        description="Example query",
                        input=_ToolParams(query="test"),
                        output=_ToolResult(answer="result"),
                    ),
                ),
            )
        ],
    )

    # Override with empty template results in empty body but tools still render
    rendered = section.render_override("", _SectionParams(title="test"), 0, "1")

    assert rendered.startswith("## 1. ToolsOnly")
    assert "override_tool" in rendered


def test_markdown_section_render_override_with_empty_body_no_tools() -> None:
    """render_override with empty body and no tools returns heading only."""

    section = MarkdownSection[_SectionParams](
        title="Empty",
        template="Default: ${title}",
        key="empty",
    )

    rendered = section.render_override("", _SectionParams(title="test"), 0, "1")

    assert rendered == "## 1. Empty"


def test_summary_visibility_without_tools_injects_read_section() -> None:
    """Summarized section without tools gets read_section tool instead of open_sections."""
    section = MarkdownSection[_SectionParams](
        title="NoTools",
        template="Full: ${title}",
        key="no-tools-section",
        summary="Summary: ${title}",
        visibility=SectionVisibility.SUMMARY,
        # No tools attached
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_SectionParams(title="Test"),))
    rendered = renderer.render(params_lookup)

    # read_section tool should be injected (not open_sections)
    assert len(rendered.tools) == 1
    assert rendered.tools[0].name == "read_section"
    # Summary suffix should point to read_section
    assert "read_section" in rendered.text
    assert "open_sections" not in rendered.text


def test_summary_visibility_with_tools_injects_open_sections() -> None:
    """Summarized section with tools gets open_sections tool."""
    section = MarkdownSection[_SectionParams](
        title="WithTools",
        template="Full: ${title}",
        key="with-tools-section",
        summary="Summary: ${title}",
        visibility=SectionVisibility.SUMMARY,
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="section_tool",
                description="Tool in section.",
                handler=_dummy_handler,
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_SectionParams(title="Test"),))
    rendered = renderer.render(params_lookup)

    # open_sections tool should be injected (not read_section)
    assert len(rendered.tools) == 1
    assert rendered.tools[0].name == "open_sections"
    # Summary suffix should point to open_sections
    assert "open_sections" in rendered.text


def test_summary_visibility_mixed_injects_both_tools() -> None:
    """Prompt with both tooled and tool-less summarized sections gets both tools."""
    section_with_tools = MarkdownSection[_SectionParams](
        title="WithTools",
        template="Full with tools: ${title}",
        key="with-tools",
        summary="Summary with tools: ${title}",
        visibility=SectionVisibility.SUMMARY,
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="tooled_section_tool",
                description="Tool in section.",
                handler=_dummy_handler,
            )
        ],
    )

    section_without_tools = MarkdownSection[_SectionParams](
        title="NoTools",
        template="Full no tools: ${title}",
        key="no-tools",
        summary="Summary no tools: ${title}",
        visibility=SectionVisibility.SUMMARY,
        # No tools attached
    )

    registry = PromptRegistry()
    registry.register_sections(
        (
            cast(Section[SupportsDataclass], section_with_tools),
            cast(Section[SupportsDataclass], section_without_tools),
        )
    )
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_SectionParams(title="Test"),))
    rendered = renderer.render(params_lookup)

    # Both tools should be injected
    tool_names = {tool.name for tool in rendered.tools}
    assert tool_names == {"open_sections", "read_section"}
