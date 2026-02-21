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

"""Tests for MarkdownSection basic rendering, visibility, and clone."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptRenderError,
    PromptValidationError,
    SectionVisibility,
)


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


def test_text_section_effective_visibility_raises_when_summary_without_template() -> (
    None
):
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

    # Raises when requesting SUMMARY but no summary is set
    with pytest.raises(PromptValidationError) as excinfo:
        section.effective_visibility(override=SectionVisibility.SUMMARY)

    assert "SUMMARY visibility requested" in str(excinfo.value)
    assert "no-summary" in str(excinfo.value)


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


def test_text_section_clone_with_children() -> None:
    """Clone method properly clones child sections."""
    child = MarkdownSection(
        title="Child",
        template="Child content",
        key="child",
    )
    parent = MarkdownSection(
        title="Parent",
        template="Parent content",
        key="parent",
        children=[child],
    )

    cloned = parent.clone()

    # Verify parent was cloned
    assert cloned is not parent
    assert cloned.title == parent.title

    # Verify children were cloned
    assert len(cloned.children) == 1
    cloned_child = cloned.children[0]
    assert cloned_child is not child
    assert cloned_child.title == child.title
    assert cloned_child.key == child.key
