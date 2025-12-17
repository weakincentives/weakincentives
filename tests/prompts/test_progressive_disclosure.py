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

"""Tests for progressive disclosure functionality."""

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
    SetVisibilityOverride,
    VisibilityExpansionRequired,
)
from weakincentives.prompt.progressive_disclosure import (
    CONTEXT_FOLDER,
    OpenSectionsParams,
    OpenSectionsResult,
    _find_section_node,
    _is_descendant,
    _write_section_context,
    build_summary_suffix,
    compute_current_visibility,
    create_open_sections_handler,
    has_summarized_sections,
    section_has_tools,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.section import Section
from weakincentives.prompt.tool import Tool, ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.types.dataclass import SupportsDataclass


@dataclass
class _TestParams:
    name: str = "test"


@dataclass
class _ToolParams:
    """Params for test tool."""

    query: str = ""


@dataclass
class _ToolResult:
    """Result for test tool."""

    answer: str = ""


class _MockFilesystem:
    """Mock filesystem for testing context file writing."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.directories: set[str] = set()

    def mkdir(
        self, path: str, *, parents: bool = False, exist_ok: bool = False
    ) -> None:
        """Create a directory."""
        self.directories.add(path)

    def write(self, path: str, content: str, *, mode: str = "overwrite") -> None:
        """Write content to a file."""
        self.files[path] = content


def _make_test_tool() -> Tool[_ToolParams, _ToolResult]:
    """Create a test tool for sections."""

    def handler(
        params: _ToolParams, *, context: ToolContext
    ) -> ToolResult[_ToolResult]:
        return ToolResult(
            message="Done",
            value=_ToolResult(answer="test"),
            success=True,
        )

    return Tool[_ToolParams, _ToolResult](
        name="test_tool",
        description="A test tool",
        handler=handler,
    )


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


def _make_tool_context(
    filesystem: _MockFilesystem | None = None,
) -> ToolContext:
    """Create a mock ToolContext."""
    return ToolContext(
        prompt=MagicMock(),
        rendered_prompt=None,
        adapter=MagicMock(),
        session=MagicMock(),
        filesystem=filesystem,  # type: ignore[arg-type]
    )


# Tests for build_summary_suffix


def test_build_summary_suffix_without_children() -> None:
    """Suffix without child sections."""
    suffix = build_summary_suffix("my-section", ())
    assert 'key "my-section"' in suffix
    assert "subsections" not in suffix
    assert "---" in suffix


def test_build_summary_suffix_with_children() -> None:
    """Suffix includes child section names."""
    suffix = build_summary_suffix("parent", ("child1", "child2"))
    assert 'key "parent"' in suffix
    assert "subsections: child1, child2" in suffix


# Tests for has_summarized_sections


def test_has_summarized_sections_no_summarized() -> None:
    """Returns False when all sections are FULL visibility."""
    section = _make_section(visibility=SectionVisibility.FULL)
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is False


def test_has_summarized_sections_without_summary_text() -> None:
    """Raises when SUMMARY visibility but no summary text."""
    section = _make_section(
        visibility=SectionVisibility.SUMMARY,
        summary=None,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    with pytest.raises(PromptValidationError) as excinfo:
        has_summarized_sections(snapshot)

    assert "SUMMARY visibility requested" in str(excinfo.value)


def test_has_summarized_sections_with_summary_text() -> None:
    """Returns True when SUMMARY visibility with summary text."""
    section = _make_section(
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is True


def test_has_summarized_sections_with_overrides() -> None:
    """Visibility overrides via session state can expand summarized sections."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    # Without override, it's summarized
    assert has_summarized_sections(snapshot) is True

    # With override to FULL via session state, no summarized sections
    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.broadcast(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    assert has_summarized_sections(snapshot, session=session) is False


# Tests for compute_current_visibility


def test_compute_current_visibility_default() -> None:
    """Returns default visibility when no overrides."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    result = compute_current_visibility(snapshot)
    assert result["sec",] == SectionVisibility.SUMMARY


def test_compute_current_visibility_with_overrides() -> None:
    """Applies visibility overrides from session state."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.broadcast(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    result = compute_current_visibility(snapshot, session=session)
    assert result["sec",] == SectionVisibility.FULL


def test_compute_current_visibility_uses_params_for_callable() -> None:
    """Visibility selectors can depend on section parameters."""

    def selector(params: _TestParams) -> SectionVisibility:
        return (
            SectionVisibility.SUMMARY
            if params.name == "summarize"
            else SectionVisibility.FULL
        )

    section = _make_section(
        key="sec",
        summary="Summary",
        visibility=selector,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    summarized = compute_current_visibility(
        snapshot, param_lookup={_TestParams: _TestParams(name="summarize")}
    )
    assert summarized["sec",] == SectionVisibility.SUMMARY

    expanded = compute_current_visibility(
        snapshot, param_lookup={_TestParams: _TestParams(name="full")}
    )
    assert expanded["sec",] == SectionVisibility.FULL


# Tests for open_sections handler


def test_open_sections_raises_visibility_expansion_required() -> None:
    """Handler raises VisibilityExpansionRequired for tool-bearing sections."""
    # Create a section WITH tools to test the exception path
    section = MarkdownSection[_TestParams](
        title="With Tools",
        template="Content: ${name}",
        key="summarized",
        summary="Summary text",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        tools=[_make_test_tool()],
    )
    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section),
        path=(section.key,),
        depth=0,
    )
    snapshot = registry.snapshot()
    current_visibility = {("summarized",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("summarized",),
        reason="Need details",
    )

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert exc.section_keys == ("summarized",)
    assert exc.reason == "Need details"
    assert exc.requested_overrides["summarized",] == SectionVisibility.FULL


def test_open_sections_rejects_empty_keys() -> None:
    """Handler rejects empty section_keys."""
    section = _make_section(key="sec")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = compute_current_visibility(snapshot)

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=(),
        reason="Empty",
    )

    with pytest.raises(PromptValidationError, match="At least one section key"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_nonexistent_section() -> None:
    """Handler rejects keys for sections that don't exist."""
    section = _make_section(key="exists")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = compute_current_visibility(snapshot)

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("does-not-exist",),
        reason="Invalid",
    )

    with pytest.raises(PromptValidationError, match="does not exist"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_already_expanded() -> None:
    """Handler rejects keys for sections already at FULL visibility."""
    section = _make_section(
        key="expanded",
        visibility=SectionVisibility.FULL,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("expanded",): SectionVisibility.FULL}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("expanded",),
        reason="Already expanded",
    )

    with pytest.raises(PromptValidationError, match="already expanded"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_nested_key_parsing() -> None:
    """Handler correctly parses dot-notation section keys."""
    # Create parent section with child that has tools (to test exception path)
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
                default_params=_TestParams(),
                tools=[_make_test_tool()],  # Add tools to trigger exception path
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

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("parent.child",),
        reason="Need child details",
    )

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert ("parent", "child") in exc.requested_overrides


def test_open_sections_nested_key_content_only() -> None:
    """Handler writes context file for content-only nested sections."""
    # Create parent section with content-only child
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
                default_params=_TestParams(),
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

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    filesystem = _MockFilesystem()
    params = OpenSectionsParams(
        section_keys=("parent.child",),
        reason="Need child details",
    )

    result = tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]

    assert result.success is True
    assert result.value is not None
    assert "context/parent.child.md" in result.value.written_files


# Tests for VisibilityExpansionRequired exception


def test_visibility_expansion_required_str_representation() -> None:
    """Exception has informative string representation."""
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides={("sec1",): SectionVisibility.FULL},
        reason="Need more info",
        section_keys=("sec1",),
    )

    str_repr = str(exc)
    assert "sec1" in str_repr
    assert "Need more info" in str_repr


def test_visibility_expansion_required_attributes() -> None:
    """Exception stores provided attributes."""
    overrides = {
        ("a",): SectionVisibility.FULL,
        ("b", "c"): SectionVisibility.FULL,
    }
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides=overrides,
        reason="Testing",
        section_keys=("a", "b.c"),
    )

    assert exc.requested_overrides == overrides
    assert exc.reason == "Testing"
    assert exc.section_keys == ("a", "b.c")


# Tests for _is_descendant helper


def test_is_descendant_returns_true_for_child() -> None:
    """Child path is detected as descendant."""
    assert _is_descendant(("parent", "child"), ("parent",)) is True


def test_is_descendant_returns_true_for_grandchild() -> None:
    """Grandchild path is detected as descendant."""
    assert _is_descendant(("a", "b", "c"), ("a",)) is True


def test_is_descendant_returns_false_for_same_path() -> None:
    """Same path is not a descendant."""
    assert _is_descendant(("parent",), ("parent",)) is False


def test_is_descendant_returns_false_for_sibling() -> None:
    """Sibling path is not a descendant."""
    assert _is_descendant(("sibling",), ("parent",)) is False


def test_is_descendant_returns_false_for_parent() -> None:
    """Parent path is not a descendant of child."""
    assert _is_descendant(("parent",), ("parent", "child")) is False


# Tests for section_has_tools helper


def test_section_has_tools_returns_true_for_section_with_tools() -> None:
    """Section with tools is detected."""
    section = MarkdownSection[_TestParams](
        title="With Tools",
        template="Content: ${name}",
        key="with-tools",
        default_params=_TestParams(),
        tools=[_make_test_tool()],
    )
    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section),
        path=(section.key,),
        depth=0,
    )
    snapshot = registry.snapshot()

    assert section_has_tools(("with-tools",), snapshot) is True


def test_section_has_tools_returns_false_for_content_only() -> None:
    """Content-only section is correctly identified."""
    section = _make_section(key="content-only")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert section_has_tools(("content-only",), snapshot) is False


def test_section_has_tools_returns_true_for_nested_tools() -> None:
    """Section with child that has tools is detected."""
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child with Tools",
                template="Child: ${name}",
                key="child",
                default_params=_TestParams(),
                tools=[_make_test_tool()],
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()

    # Parent has tools because child has tools
    assert section_has_tools(("parent",), snapshot) is True
    # Child directly has tools
    assert section_has_tools(("parent", "child"), snapshot) is True


# Tests for OpenSectionsResult


def test_open_sections_result_stores_written_files() -> None:
    """OpenSectionsResult stores file paths."""
    result = OpenSectionsResult(written_files=("context/a.md", "context/b.md"))
    assert result.written_files == ("context/a.md", "context/b.md")


# Tests for build_summary_suffix with has_tools parameter


def test_build_summary_suffix_content_only_without_children() -> None:
    """Suffix for content-only section mentions context file."""
    suffix = build_summary_suffix("my-section", (), has_tools=False)
    assert 'key "my-section"' in suffix
    assert f"{CONTEXT_FOLDER}/my-section.md" in suffix


def test_build_summary_suffix_content_only_with_children() -> None:
    """Suffix for content-only section with children mentions context file."""
    suffix = build_summary_suffix("parent", ("child1", "child2"), has_tools=False)
    assert 'key "parent"' in suffix
    assert "child1, child2" in suffix
    assert f"{CONTEXT_FOLDER}/parent.md" in suffix


def test_build_summary_suffix_with_tools_without_children() -> None:
    """Suffix for tool-bearing section mentions additional tools."""
    suffix = build_summary_suffix("my-section", (), has_tools=True)
    assert 'key "my-section"' in suffix
    assert "additional tools" in suffix


def test_build_summary_suffix_with_tools_and_children() -> None:
    """Suffix for tool-bearing section with children mentions tools may become available."""
    suffix = build_summary_suffix("parent", ("child1", "child2"), has_tools=True)
    assert 'key "parent"' in suffix
    assert "child1, child2" in suffix
    assert "tools may become available" in suffix


# Tests for content-only section handling


def test_open_sections_writes_context_file_for_content_only() -> None:
    """Handler writes context file for content-only sections."""
    section = _make_section(
        key="content-only",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("content-only",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    filesystem = _MockFilesystem()
    params = OpenSectionsParams(
        section_keys=("content-only",),
        reason="Need details",
    )

    result = tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]

    assert result.success is True
    assert result.value is not None
    assert "context/content-only.md" in result.value.written_files
    assert "context/content-only.md" in filesystem.files
    assert CONTEXT_FOLDER in filesystem.directories


def test_open_sections_fails_without_filesystem() -> None:
    """Handler returns failure when filesystem is unavailable."""
    section = _make_section(
        key="content-only",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("content-only",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("content-only",),
        reason="Need details",
    )

    # No filesystem provided
    result = tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    assert result.success is False
    assert result.value is None
    assert "no filesystem available" in result.message


def test_open_sections_raises_for_tool_bearing_section() -> None:
    """Handler raises exception for sections with tools."""
    section = MarkdownSection[_TestParams](
        title="With Tools",
        template="Content: ${name}",
        key="with-tools",
        summary="Summary text",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        tools=[_make_test_tool()],
    )
    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section),
        path=(section.key,),
        depth=0,
    )
    snapshot = registry.snapshot()
    current_visibility = {("with-tools",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    filesystem = _MockFilesystem()
    params = OpenSectionsParams(
        section_keys=("with-tools",),
        reason="Need details",
    )

    with pytest.raises(VisibilityExpansionRequired):
        tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]


def test_open_sections_raises_for_mixed_request() -> None:
    """Handler raises for all sections if any has tools."""
    # Create one content-only section and one with tools
    content_section = _make_section(
        key="content-only",
        summary="Content summary",
        visibility=SectionVisibility.SUMMARY,
    )
    tool_section = MarkdownSection[_TestParams](
        title="With Tools",
        template="Content: ${name}",
        key="with-tools",
        summary="Tool summary",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(),
        tools=[_make_test_tool()],
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], content_section),
        path=(content_section.key,),
        depth=0,
    )
    registry.register_section(
        cast(Section[SupportsDataclass], tool_section),
        path=(tool_section.key,),
        depth=0,
    )
    snapshot = registry.snapshot()
    current_visibility = {
        ("content-only",): SectionVisibility.SUMMARY,
        ("with-tools",): SectionVisibility.SUMMARY,
    }

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    filesystem = _MockFilesystem()
    params = OpenSectionsParams(
        section_keys=("content-only", "with-tools"),
        reason="Need both",
    )

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    # Both sections should be in the overrides
    assert ("content-only",) in exc.requested_overrides
    assert ("with-tools",) in exc.requested_overrides
    # No files should have been written
    assert len(filesystem.files) == 0


def test_open_sections_writes_multiple_content_files() -> None:
    """Handler writes multiple context files for multiple content-only sections."""
    section_a = _make_section(
        key="section-a",
        summary="Summary A",
        visibility=SectionVisibility.SUMMARY,
    )
    section_b = _make_section(
        key="section-b",
        summary="Summary B",
        visibility=SectionVisibility.SUMMARY,
    )

    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section_a),
        path=(section_a.key,),
        depth=0,
    )
    registry.register_section(
        cast(Section[SupportsDataclass], section_b),
        path=(section_b.key,),
        depth=0,
    )
    snapshot = registry.snapshot()
    current_visibility = {
        ("section-a",): SectionVisibility.SUMMARY,
        ("section-b",): SectionVisibility.SUMMARY,
    }

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    filesystem = _MockFilesystem()
    params = OpenSectionsParams(
        section_keys=("section-a", "section-b"),
        reason="Need both",
    )

    result = tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]

    assert result.success is True
    assert result.value is not None
    assert len(result.value.written_files) == 2
    assert "context/section-a.md" in result.value.written_files
    assert "context/section-b.md" in result.value.written_files
    assert len(filesystem.files) == 2


class _FailingFilesystem(_MockFilesystem):
    """Mock filesystem that fails on write."""

    def write(self, path: str, content: str, *, mode: str = "overwrite") -> None:
        """Raise an exception to test error handling."""
        raise OSError("Disk full")


def test_open_sections_handles_write_failure() -> None:
    """Handler returns failure when filesystem write fails."""
    section = _make_section(
        key="content-only",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("content-only",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
        param_lookup={_TestParams: _TestParams(name="test")},
    )

    filesystem = _FailingFilesystem()
    params = OpenSectionsParams(
        section_keys=("content-only",),
        reason="Need details",
    )

    result = tool.handler(params, context=_make_tool_context(filesystem))  # type: ignore[arg-type]

    assert result.success is False
    assert result.value is None
    assert "Failed to write context" in result.message
    assert "Disk full" in result.message


# Tests for internal helper functions


def test_find_section_node_returns_none_for_nonexistent() -> None:
    """_find_section_node returns None for nonexistent path."""
    section = _make_section(key="exists")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    # Looking for a path that doesn't exist
    result = _find_section_node(("does-not-exist",), snapshot)
    assert result is None


def test_write_section_context_raises_for_nonexistent() -> None:
    """_write_section_context raises for nonexistent section."""
    section = _make_section(key="exists")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    filesystem = _MockFilesystem()

    with pytest.raises(PromptValidationError, match="not found"):
        _write_section_context(
            ("does-not-exist",),
            snapshot,
            {_TestParams: _TestParams(name="test")},
            filesystem,  # type: ignore[arg-type]
        )
