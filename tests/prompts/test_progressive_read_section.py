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

"""Tests for the read_section handler and related progressive disclosure logic.

This module complements test_progressive_disclosure.py and covers:
- read_section handler behavior
- Nested/hierarchical section rendering
- Disabled child sections
- _collect_child_keys_for_node helper
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptValidationError,
    SectionVisibility,
)
from weakincentives.prompt.progressive_disclosure import (
    ReadSectionParams,
    ReadSectionResult,
    _collect_child_keys_for_node,
    compute_current_visibility,
    create_read_section_handler,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.section import Section
from weakincentives.prompt.tool import ToolContext
from weakincentives.types.dataclass import SupportsDataclass


@dataclass
class _TestParams:
    name: str = "test"


def _make_section(
    *,
    key: str = "test-section",
    summary: str | None = None,
    visibility: SectionVisibility
    | Callable[[_TestParams], SectionVisibility]
    | Callable[[], SectionVisibility] = SectionVisibility.FULL,
) -> MarkdownSection[_TestParams]:
    return MarkdownSection[_TestParams](
        title="Test Section",
        template="Content: ${name}",
        key=key,
        summary=summary,
        visibility=visibility,
        default_params=_TestParams(),
    )


def _make_registry(
    sections: tuple[MarkdownSection[_TestParams], ...],
) -> PromptRegistry:
    registry = PromptRegistry()
    for section in sections:
        registry.register_section(
            cast(Section[SupportsDataclass], section),
            path=(section.key,),
            depth=0,
        )
    return registry


def _make_tool_context() -> ToolContext:
    """Create a mock ToolContext."""
    return ToolContext(
        prompt=MagicMock(),
        rendered_prompt=None,
        adapter=MagicMock(),
        session=MagicMock(),
    )


# Tests for read_section handler


def test_read_section_returns_rendered_content() -> None:
    """Handler returns rendered section content."""
    section = _make_section(
        key="summarized",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("summarized",): SectionVisibility.SUMMARY}

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="hello")},
    )

    params = ReadSectionParams(section_key="summarized")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    assert "Content: hello" in result.value.content
    assert "Test Section" in result.value.content  # Title should be in output


def test_read_section_result_render() -> None:
    """ReadSectionResult.render returns the content field."""
    result = ReadSectionResult(content="# Section\n\nHello world")
    assert result.render() == "# Section\n\nHello world"


def test_read_section_rejects_nonexistent_section() -> None:
    """Handler rejects keys for sections that don't exist."""
    section = _make_section(key="exists")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = compute_current_visibility(snapshot)

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = ReadSectionParams(section_key="does-not-exist")

    with pytest.raises(PromptValidationError, match="does not exist"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_read_section_rejects_non_summarized_section() -> None:
    """Handler rejects keys for sections that are not summarized."""
    section = _make_section(
        key="expanded",
        visibility=SectionVisibility.FULL,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("expanded",): SectionVisibility.FULL}

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = ReadSectionParams(section_key="expanded")

    with pytest.raises(PromptValidationError, match="is not summarized"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_read_section_nested_key_parsing() -> None:
    """Handler correctly parses dot-notation section keys."""
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child",
                template="Child: ${name}",
                key="child",
                summary="Child summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(name="child-value"),
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.FULL,
        ("parent", "child"): SectionVisibility.SUMMARY,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    params = ReadSectionParams(section_key="parent.child")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    assert "Child" in result.value.content


def test_read_section_renders_children_with_their_visibility() -> None:
    """Handler renders children with their current visibility."""
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent content: ${name}",
        key="parent",
        summary="Parent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child",
                template="Child content: ${name}",
                key="child",
                summary="Child summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(),
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.SUMMARY,
        ("parent", "child"): SectionVisibility.SUMMARY,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    params = ReadSectionParams(section_key="parent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Parent content should be rendered with FULL visibility
    assert "Parent content: test" in result.value.content
    # Child should still show as summarized
    assert "Child summary" in result.value.content


def test_read_section_with_sibling_section() -> None:
    """Handler stops rendering at sibling sections (same depth as parent)."""

    @dataclass
    class OtherParams:
        other: str = "other"

    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        summary="Parent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child",
                template="Child: ${name}",
                key="child",
                default_params=_TestParams(),
            )
        ],
    )

    sibling = MarkdownSection[OtherParams](
        title="Sibling",
        template="Sibling: ${other}",
        key="sibling",
        default_params=OtherParams(),
    )

    registry = PromptRegistry()
    registry.register_sections(
        (
            cast(Section[SupportsDataclass], parent),
            cast(Section[SupportsDataclass], sibling),
        )
    )
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.SUMMARY,
        ("parent", "child"): SectionVisibility.FULL,
        ("sibling",): SectionVisibility.FULL,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={
            _TestParams: _TestParams(name="test"),
            OtherParams: OtherParams(),
        },
    )

    params = ReadSectionParams(section_key="parent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Parent and child content should be present
    assert "Parent: test" in result.value.content
    assert "Child: test" in result.value.content
    # Sibling should NOT be in the output (not a child of parent)
    assert "Sibling" not in result.value.content


def test_read_section_with_disabled_child() -> None:
    """Handler skips disabled child sections."""

    @dataclass
    class EnabledParams:
        show_child: bool = False

    parent = MarkdownSection[EnabledParams](
        title="Parent",
        template="Parent content",
        key="parent",
        summary="Parent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=EnabledParams(),
        children=[
            MarkdownSection[EnabledParams](
                title="Conditional Child",
                template="Child content",
                key="conditional-child",
                enabled=lambda p: p.show_child,
                default_params=EnabledParams(),
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.SUMMARY,
        ("parent", "conditional-child"): SectionVisibility.FULL,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={EnabledParams: EnabledParams(show_child=False)},
    )

    params = ReadSectionParams(section_key="parent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Parent content should be present
    assert "Parent content" in result.value.content
    # Child should be skipped because it's disabled
    assert "Child content" not in result.value.content


def test_read_section_with_nested_summarized_children() -> None:
    """Handler handles nested summarized children correctly."""
    grandparent = MarkdownSection[_TestParams](
        title="Grandparent",
        template="Grandparent: ${name}",
        key="grandparent",
        summary="Grandparent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Parent",
                template="Parent: ${name}",
                key="parent",
                summary="Parent summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(),
                children=[
                    MarkdownSection[_TestParams](
                        title="Child",
                        template="Child: ${name}",
                        key="child",
                        default_params=_TestParams(),
                    )
                ],
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], grandparent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("grandparent",): SectionVisibility.SUMMARY,
        ("grandparent", "parent"): SectionVisibility.SUMMARY,
        ("grandparent", "parent", "child"): SectionVisibility.FULL,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    params = ReadSectionParams(section_key="grandparent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Grandparent content should be full
    assert "Grandparent: test" in result.value.content
    # Parent should be rendered as summary (still summarized)
    assert "Parent summary" in result.value.content
    # Child should be skipped (parent is summarized, so children are hidden)
    assert "Child:" not in result.value.content


def test_read_section_collect_child_keys_with_siblings() -> None:
    """Handler correctly collects child keys when there are sibling sections."""
    # This tests the break condition in _collect_child_keys_for_node (lines 373-374)
    grandparent = MarkdownSection[_TestParams](
        title="Grandparent",
        template="Grandparent: ${name}",
        key="grandparent",
        summary="Grandparent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="First Parent",
                template="First Parent: ${name}",
                key="first-parent",
                summary="First parent summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(),
                children=[
                    MarkdownSection[_TestParams](
                        title="Child of First",
                        template="Child of First: ${name}",
                        key="child-of-first",
                        default_params=_TestParams(),
                    )
                ],
            ),
            MarkdownSection[_TestParams](
                title="Second Parent",
                template="Second Parent: ${name}",
                key="second-parent",
                default_params=_TestParams(),
            ),
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], grandparent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("grandparent",): SectionVisibility.SUMMARY,
        ("grandparent", "first-parent"): SectionVisibility.SUMMARY,
        ("grandparent", "first-parent", "child-of-first"): SectionVisibility.FULL,
        ("grandparent", "second-parent"): SectionVisibility.FULL,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    params = ReadSectionParams(section_key="grandparent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Grandparent content should be full
    assert "Grandparent: test" in result.value.content
    # First parent should be rendered as summary (still summarized)
    assert "First parent summary" in result.value.content
    # Second parent should be rendered with full content
    assert "Second Parent: test" in result.value.content
    # Child of first should be skipped (first parent is summarized)
    assert "Child of First:" not in result.value.content


def test_read_section_skip_depth_reset() -> None:
    """Handler correctly resets skip depth when encountering sibling of summarized section."""
    # This tests line 314: skip_depth = None
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        summary="Parent summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Summarized Child",
                template="Summarized Child: ${name}",
                key="summarized-child",
                summary="Summarized child summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(),
                children=[
                    MarkdownSection[_TestParams](
                        title="Grandchild",
                        template="Grandchild: ${name}",
                        key="grandchild",
                        default_params=_TestParams(),
                    )
                ],
            ),
            MarkdownSection[_TestParams](
                title="Sibling Child",
                template="Sibling Child: ${name}",
                key="sibling-child",
                default_params=_TestParams(),
            ),
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.SUMMARY,
        ("parent", "summarized-child"): SectionVisibility.SUMMARY,
        ("parent", "summarized-child", "grandchild"): SectionVisibility.FULL,
        ("parent", "sibling-child"): SectionVisibility.FULL,
    }

    tool = create_read_section_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    params = ReadSectionParams(section_key="parent")
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is True
    # Parent content should be full
    assert "Parent: test" in result.value.content
    # Summarized child should be rendered as summary
    assert "Summarized child summary" in result.value.content
    # Grandchild should be skipped (summarized child hides it)
    assert "Grandchild:" not in result.value.content
    # Sibling child should be rendered (skip_depth should be reset)
    assert "Sibling Child: test" in result.value.content


def test_collect_child_keys_skips_grandchildren_and_continues() -> None:
    """Test branch 406->398: skipping grandchildren continues to find siblings."""
    # Structure: parent -> [child1 -> [grandchild], child2]
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child1",
                template="Child1: ${name}",
                key="child1",
                default_params=_TestParams(),
                children=[
                    MarkdownSection[_TestParams](
                        title="Grandchild",
                        template="Grandchild: ${name}",
                        key="grandchild",
                        default_params=_TestParams(),
                    )
                ],
            ),
            MarkdownSection[_TestParams](
                title="Child2",
                template="Child2: ${name}",
                key="child2",
                default_params=_TestParams(),
            ),
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()

    # Find parent node in registry
    parent_node = None
    for node in snapshot.sections:
        if node.section.key == "parent":
            parent_node = node
            break

    assert parent_node is not None

    # Collect child keys for parent - should skip grandchild but find child2
    child_keys = _collect_child_keys_for_node(parent_node, snapshot)

    # Should contain both direct children but not grandchild
    assert child_keys == ("child1", "child2")
