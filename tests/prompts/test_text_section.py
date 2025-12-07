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
from types import SimpleNamespace
from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptRenderError,
    Section,
    SectionVisibility,
    SupportsDataclass,
    Tool,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.rendering import PromptRenderer
from weakincentives.prompt.tool import ToolContext, ToolResult


@dataclass
class GreetingParams:
    greeting: str


def test_text_section_renders_heading_and_body() -> None:
    section = MarkdownSection[GreetingParams](
        title="Greeting",
        template="""
            Greeting:
            ${greeting}
        """,
        key="greeting",
    )

    output = section.render(GreetingParams(greeting="hello"), depth=0, number="1")

    assert output == "## 1. Greeting\n\nGreeting:\nhello"


def test_text_section_performs_strict_substitution() -> None:
    @dataclass
    class PlaceholderParams:
        value: str

    section = MarkdownSection[PlaceholderParams](
        title="Placeholder Demo",
        template="Value: ${value}",
        key="placeholder-demo",
    )

    output = section.render(PlaceholderParams(value="42"), depth=1, number="1.1")

    assert output == "### 1.1. Placeholder Demo\n\nValue: 42"


def test_text_section_supports_slotted_dataclass_params() -> None:
    @dataclass(slots=True)
    class SlottedParams:
        value: str

    section = MarkdownSection[SlottedParams](
        title="Slots",
        template="Slot value: ${value}",
        key="slots",
    )

    output = section.render(SlottedParams(value="ok"), depth=0, number="1")

    assert output == "## 1. Slots\n\nSlot value: ok"


def test_text_section_rejects_non_dataclass_params() -> None:
    section = MarkdownSection[SimpleNamespace](
        title="Reject",
        template="Value: ${value}",
        key="reject",
    )

    with pytest.raises(PromptRenderError) as error_info:
        section.render(SimpleNamespace(value="nope"), depth=0, number="1")

    error = error_info.value
    assert isinstance(error, PromptRenderError)
    assert error.dataclass_type is SimpleNamespace


def test_text_section_with_summary_renders_summary_when_visibility_is_summary() -> None:
    @dataclass
    class ContentParams:
        topic: str

    section = MarkdownSection[ContentParams](
        title="Content",
        template="Full content about ${topic}. This is a detailed explanation.",
        key="content",
        summary="Brief: ${topic}",
    )

    output = section.render(
        ContentParams(topic="testing"),
        depth=0,
        number="1",
        visibility=SectionVisibility.SUMMARY,
    )

    assert output == "## 1. Content\n\nBrief: testing"


def test_text_section_with_summary_renders_full_by_default() -> None:
    @dataclass
    class ContentParams:
        topic: str

    section = MarkdownSection[ContentParams](
        title="Content",
        template="Full content about ${topic}.",
        key="content",
        summary="Brief: ${topic}",
    )

    output = section.render(ContentParams(topic="testing"), depth=0, number="1")

    assert output == "## 1. Content\n\nFull content about testing."


def test_text_section_without_summary_falls_back_to_full() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="NoSummary",
        template="Full content: ${value}",
        key="no-summary",
    )

    # Even with SUMMARY visibility, falls back to full when no summary is set
    output = section.render(
        ContentParams(value="test"),
        depth=0,
        number="1",
        visibility=SectionVisibility.SUMMARY,
    )

    assert output == "## 1. NoSummary\n\nFull content: test"


def test_text_section_visibility_override_takes_precedence() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Override",
        template="Full: ${value}",
        key="override",
        summary="Summary: ${value}",
        visibility=SectionVisibility.FULL,  # Default visibility is FULL
    )

    # Override to SUMMARY at render time
    output = section.render(
        ContentParams(value="test"),
        depth=0,
        number="1",
        visibility=SectionVisibility.SUMMARY,
    )

    assert output == "## 1. Override\n\nSummary: test"


def test_text_section_default_visibility_is_full() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Default",
        template="Full: ${value}",
        key="default",
        summary="Summary: ${value}",
    )

    assert section.visibility == SectionVisibility.FULL


def test_text_section_configurable_default_visibility() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Configured",
        template="Full: ${value}",
        key="configured",
        summary="Summary: ${value}",
        visibility=SectionVisibility.SUMMARY,
    )

    assert section.visibility == SectionVisibility.SUMMARY
    output = section.render(ContentParams(value="test"), depth=0, number="1")

    assert output == "## 1. Configured\n\nSummary: test"


def test_text_section_effective_visibility_returns_override() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Test",
        template="Full: ${value}",
        key="test",
        summary="Summary: ${value}",
        visibility=SectionVisibility.FULL,
    )

    assert section.effective_visibility() == SectionVisibility.FULL
    assert (
        section.effective_visibility(override=SectionVisibility.SUMMARY)
        == SectionVisibility.SUMMARY
    )


def test_text_section_effective_visibility_fallback_to_full_without_summary() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="NoSummary",
        template="Full: ${value}",
        key="no-summary",
    )

    assert section.summary is None
    assert section.effective_visibility() == SectionVisibility.FULL
    # Falls back to FULL when requesting SUMMARY but no summary is set
    assert (
        section.effective_visibility(override=SectionVisibility.SUMMARY)
        == SectionVisibility.FULL
    )


def test_text_section_original_summary_template_returns_summary() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Test",
        template="Full: ${value}",
        key="test",
        summary="Summary: ${value}",
    )

    assert section.original_summary_template() == "Summary: ${value}"


def test_text_section_original_summary_template_returns_none_when_not_set() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Test",
        template="Full: ${value}",
        key="test",
    )

    assert section.original_summary_template() is None


def test_text_section_clone_preserves_summary_and_visibility() -> None:
    @dataclass
    class ContentParams:
        value: str

    section = MarkdownSection[ContentParams](
        title="Original",
        template="Full: ${value}",
        key="original",
        summary="Summary: ${value}",
        visibility=SectionVisibility.SUMMARY,
    )

    cloned = section.clone()

    assert cloned.summary == section.summary
    assert cloned.visibility == section.visibility


# --- Tests for SUMMARY visibility behavior in rendering ---


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
    return ToolResult(message="ok", value=_ToolResult(answer=params.query))


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

    # With SUMMARY visibility, the section's own tools are excluded,
    # but the open_sections tool is injected for progressive disclosure
    rendered_summary = renderer.render(
        params_lookup,
        visibility_overrides={(section.key,): SectionVisibility.SUMMARY},
    )
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

    # With SUMMARY visibility on parent, child is skipped
    rendered_summary = renderer.render(
        params_lookup,
        visibility_overrides={("parent",): SectionVisibility.SUMMARY},
    )
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

    # With SUMMARY visibility on parent, child (and its tools) are skipped.
    # The open_sections tool is injected for progressive disclosure.
    rendered_summary = renderer.render(
        params_lookup,
        visibility_overrides={("parent",): SectionVisibility.SUMMARY},
    )
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

    # Override to FULL includes tools
    rendered_full = renderer.render(
        params_lookup,
        visibility_overrides={(section.key,): SectionVisibility.FULL},
    )
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

    # With SUMMARY visibility on parent, child is skipped but sibling is rendered
    rendered = renderer.render(
        params_lookup,
        visibility_overrides={("parent",): SectionVisibility.SUMMARY},
    )
    assert "Parent summary: Parent Title" in rendered.text
    assert "Child content" not in rendered.text  # Child is skipped
    assert "Sibling content: Sibling Info" in rendered.text  # Sibling is rendered
