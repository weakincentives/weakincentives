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

"""Unit tests covering prompt rendering, validation, and error handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptRenderError,
    PromptTemplate,
    PromptValidationError,
    SectionVisibility,
    SupportsDataclass,
)
from weakincentives.prompt.prompt import (
    RenderedPrompt,
    _format_specialization_argument,
)


@dataclass
class IntroParams:
    title: str


@dataclass
class DetailsParams:
    body: str


@dataclass
class OutroParams:
    footer: str


def build_prompt() -> PromptTemplate:
    intro = MarkdownSection[IntroParams](
        title="Intro",
        template="Intro: ${title}",
        key="intro",
    )
    details = MarkdownSection[DetailsParams](
        title="Details",
        template="Details: ${body}",
        key="details",
    )
    outro = MarkdownSection[OutroParams](
        title="Outro",
        template="Outro: ${footer}",
        key="outro",
        default_params=OutroParams(footer="bye"),
    )
    return PromptTemplate(
        ns="tests/prompts",
        key="render-basic",
        sections=[intro, details, outro],
    )


@dataclass
class ParentToggleParams:
    heading: str
    include_children: bool


@dataclass
class ChildNestedParams:
    detail: str


@dataclass
class LeafParams:
    note: str


@dataclass
class SummaryParams:
    summary: str


def test_prompt_renders_section_without_params() -> None:
    static_section = MarkdownSection(
        title="Static", key="static", template="Static content."
    )
    template = PromptTemplate(
        ns="tests.prompts",
        key="paramless-section",
        sections=(static_section,),
    )

    rendered = Prompt(template).render()

    assert rendered.text.strip().endswith("Static content.")


def test_prompt_rejects_placeholders_for_paramless_section() -> None:
    section = MarkdownSection(title="Bad", key="bad", template="${value}")

    with pytest.raises(PromptValidationError):
        PromptTemplate(ns="tests.prompts", key="bad-section", sections=(section,))


def build_nested_prompt() -> PromptTemplate:
    leaf = MarkdownSection[LeafParams](
        title="Leaf",
        template="Leaf: ${note}",
        key="leaf",
    )
    child = MarkdownSection[ChildNestedParams](
        title="Child",
        template="Child detail: ${detail}",
        key="child",
        children=[leaf],
    )
    parent = MarkdownSection[ParentToggleParams](
        title="Parent",
        template="Parent: ${heading}",
        key="parent",
        children=[child],
        enabled=lambda params: params.include_children,
    )
    summary = MarkdownSection[SummaryParams](
        title="Summary",
        template="Summary: ${summary}",
        key="summary",
    )
    return PromptTemplate(
        ns="tests/prompts",
        key="render-nested",
        sections=[parent, summary],
    )


def test_prompt_render_merges_defaults_and_overrides() -> None:
    template = build_prompt()

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="hello"),
            DetailsParams(body="world"),
        )
        .render()
    )

    assert rendered.text == "\n\n".join(
        [
            "## 1. Intro (intro)\n\nIntro: hello",
            "## 2. Details (details)\n\nDetails: world",
            "## 3. Outro (outro)\n\nOutro: bye",
        ]
    )


def test_prompt_render_accepts_unordered_inputs() -> None:
    template = build_prompt()

    rendered = (
        Prompt(template)
        .bind(
            DetailsParams(body="unordered"),
            IntroParams(title="still works"),
        )
        .render()
    )

    assert "still works" in rendered.text
    assert "unordered" in rendered.text


def test_prompt_render_requires_parameter_values() -> None:
    template = build_prompt()

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(IntroParams(title="missing detail")).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.dataclass_type is DetailsParams


def test_prompt_render_requires_dataclass_instances() -> None:
    template = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        Prompt(template).bind(cast(SupportsDataclass, IntroParams)).render()

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_rejects_duplicate_param_instances() -> None:
    template = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        Prompt(template).bind(
            IntroParams(title="first"), IntroParams(title="second")
        ).render()

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_renders_nested_sections_and_depth() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Main Heading", include_children=True),
            ChildNestedParams(detail="Child detail"),
            LeafParams(note="Deep note"),
            SummaryParams(summary="All done"),
        )
        .render()
    )

    assert rendered.text == "\n\n".join(
        [
            "## 1. Parent (parent)\n\nParent: Main Heading",
            "### 1.1. Child (parent.child)\n\nChild detail: Child detail",
            "#### 1.1.1. Leaf (parent.child.leaf)\n\nLeaf: Deep note",
            "## 2. Summary (summary)\n\nSummary: All done",
        ]
    )


def test_prompt_render_skips_disabled_parent_and_children() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Unused", include_children=False),
            SummaryParams(summary="Visible"),
        )
        .render()
    )

    assert rendered.text == "## 2. Summary (summary)\n\nSummary: Visible"
    assert "Parent" not in rendered.text
    assert "Child" not in rendered.text
    assert "Leaf" not in rendered.text


def test_prompt_render_wraps_template_errors_with_context() -> None:
    @dataclass
    class ErrorParams:
        value: str

    class ExplodingSection(MarkdownSection[ErrorParams]):
        def render(
            self,
            params: ErrorParams,
            depth: int,
            number: str,
            *,
            path: tuple[str, ...] = (),
            visibility: SectionVisibility | None = None,
        ) -> str:
            del params, depth, number, path, visibility
            raise ValueError(f"boom:{self.title}")

    section = ExplodingSection(
        title="Explode",
        template="unused",
        key="explode",
    )
    template = PromptTemplate(
        ns="tests/prompts", key="render-error", sections=[section]
    )

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(ErrorParams(value="x")).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("explode",)
    assert exc.value.dataclass_type is ErrorParams


def test_prompt_render_propagates_enabled_errors() -> None:
    @dataclass
    class ToggleParams:
        flag: bool

    def raising_enabled(params: ToggleParams) -> bool:
        raise RuntimeError("enabled failure")

    section = MarkdownSection[ToggleParams](
        title="Guard",
        template="Guard: ${flag}",
        key="guard",
        enabled=cast(Callable[[SupportsDataclass], bool], raising_enabled),
    )
    template = PromptTemplate(
        ns="tests/prompts",
        key="render-enabled-error",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(ToggleParams(flag=True)).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("guard",)
    assert exc.value.dataclass_type is ToggleParams


def test_rendered_prompt_str_returns_text() -> None:
    rendered = RenderedPrompt(text="Rendered output")

    assert str(rendered) == "Rendered output"


def test_prompt_bind_mutates_and_replaces_params() -> None:
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="bind-mutation",
        sections=[
            MarkdownSection[IntroParams](title="Intro", template="", key="intro")
        ],
    )
    bound = Prompt(prompt)

    assert bound.bind() is bound  # no-op, identity
    assert bound.params == ()

    first = IntroParams(title="v1")
    second = IntroParams(title="v2")

    assert bound.bind(first) is bound
    assert bound.params == (first,)

    assert bound.bind(second) is bound
    assert bound.params == (second,)


def test_format_specialization_argument_variants() -> None:
    assert _format_specialization_argument(None) == "?"
    assert _format_specialization_argument(int) == "int"
    assert _format_specialization_argument({"id": 1}) == "{'id': 1}"


def test_markdown_section_missing_placeholder_raises_prompt_error() -> None:
    section = MarkdownSection[IntroParams](
        title="Intro",
        template="Hello ${name}",
        key="intro",
    )

    with pytest.raises(PromptRenderError) as exc:
        section.render(IntroParams(title="ignored"), depth=0, number="1")

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.placeholder == "name"


# Tests for auto-rendering tool-free SUMMARY sections to VFS


@dataclass
class _RefParams:
    topic: str = "default"


def test_summary_section_without_tools_auto_renders_to_vfs() -> None:
    """Section with SUMMARY visibility and no tools is auto-rendered to VFS."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    reference_section = MarkdownSection[_RefParams](
        title="Reference",
        template="Full reference content about ${topic}.",
        key="reference",
        summary="Reference docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="testing"),
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="vfs-test",
        sections=[reference_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check that VFS suffix is used instead of open_sections
    assert "/context/reference.md" in rendered.text
    assert "open_sections" not in rendered.text

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None
    file_paths = [f.path.segments for f in vfs_snapshot.files]
    assert ("context", "reference.md") in file_paths

    # Verify the file content
    context_file = next(
        f for f in vfs_snapshot.files if f.path.segments == ("context", "reference.md")
    )
    assert "Full reference content about testing" in context_file.content


def test_summary_section_with_tools_uses_open_sections() -> None:
    """Section with SUMMARY visibility and tools uses open_sections."""
    from weakincentives.prompt import Tool
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    @dataclass
    class _ToolParams:
        value: str = "test"

    @dataclass
    class _ToolResult:
        result: str = ""

    tool = Tool[_ToolParams, _ToolResult](
        name="ref_tool",
        description="Reference tool",
        handler=None,
    )

    reference_section = MarkdownSection[_RefParams](
        title="Reference",
        template="Full reference with tool: ${topic}.",
        key="reference",
        summary="Reference docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="testing"),
        tools=(tool,),
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="tools-test",
        sections=[reference_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check that open_sections suffix is used
    assert "open_sections" in rendered.text
    assert "/context/reference.md" not in rendered.text

    # Verify open_sections tool is in the rendered tools
    tool_names = [t.name for t in rendered.tools]
    assert "open_sections" in tool_names


def test_summary_section_without_workspace_falls_back_to_open_sections() -> None:
    """Section with SUMMARY visibility falls back to open_sections without workspace."""
    reference_section = MarkdownSection[_RefParams](
        title="Reference",
        template="Full reference: ${topic}.",
        key="reference",
        summary="Reference docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="testing"),
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="no-vfs-test",
        sections=[reference_section],
    )

    rendered = Prompt(prompt).render()

    # Check that open_sections suffix is used
    assert "open_sections" in rendered.text
    assert "/context/reference.md" not in rendered.text


def test_summary_section_with_children_renders_full_subtree_to_vfs() -> None:
    """Section with children renders entire subtree to VFS."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    child_section = MarkdownSection[_RefParams](
        title="Child",
        template="Child content: ${topic}.",
        key="child",
        default_params=_RefParams(topic="child-topic"),
    )

    parent_section = MarkdownSection[_RefParams](
        title="Parent",
        template="Parent content: ${topic}.",
        key="parent",
        summary="Parent docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="parent-topic"),
        children=[child_section],
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="subtree-test",
        sections=[parent_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check VFS suffix is used
    assert "/context/parent.md" in rendered.text

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    # Find the context file
    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Verify both parent and child content are in the file
    assert "Parent content: parent-topic" in context_file.content
    assert "Child content: child-topic" in context_file.content


def test_summary_section_with_disabled_child_skips_child_in_vfs() -> None:
    """Disabled child sections are skipped when rendering to VFS."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    # Child section that is disabled
    disabled_child = MarkdownSection[_RefParams](
        title="Disabled Child",
        template="Disabled content: ${topic}.",
        key="disabled-child",
        default_params=_RefParams(topic="disabled"),
        enabled=lambda _: False,
    )

    # Enabled child section
    enabled_child = MarkdownSection[_RefParams](
        title="Enabled Child",
        template="Enabled content: ${topic}.",
        key="enabled-child",
        default_params=_RefParams(topic="enabled"),
    )

    parent_section = MarkdownSection[_RefParams](
        title="Parent",
        template="Parent content: ${topic}.",
        key="parent",
        summary="Parent docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="parent-topic"),
        children=[disabled_child, enabled_child],
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="disabled-child-test",
        sections=[parent_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check VFS suffix is used
    assert "/context/parent.md" in rendered.text

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    # Find the context file
    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Verify parent and enabled child content are in the file
    assert "Parent content: parent-topic" in context_file.content
    assert "Enabled content: enabled" in context_file.content
    # Disabled child should NOT be in the file
    assert "Disabled content" not in context_file.content


def test_summary_section_with_parameterless_child() -> None:
    """Child sections without params type are rendered correctly."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    # Child section without params
    static_child = MarkdownSection(
        title="Static Child",
        template="Static content without params.",
        key="static-child",
    )

    parent_section = MarkdownSection[_RefParams](
        title="Parent",
        template="Parent content: ${topic}.",
        key="parent",
        summary="Parent docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="parent-topic"),
        children=[static_child],
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="parameterless-child-test",
        sections=[parent_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check VFS suffix is used
    assert "/context/parent.md" in rendered.text

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    # Find the context file
    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Verify both parent and static child content are in the file
    assert "Parent content: parent-topic" in context_file.content
    assert "Static content without params" in context_file.content


def test_summary_section_child_uses_params_from_lookup() -> None:
    """Child sections use params from lookup when available."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    @dataclass
    class _ChildParams:
        value: str

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    child_section = MarkdownSection[_ChildParams](
        title="Child",
        template="Child value: ${value}.",
        key="child",
    )

    parent_section = MarkdownSection[_RefParams](
        title="Parent",
        template="Parent content: ${topic}.",
        key="parent",
        summary="Parent docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="parent-topic"),
        children=[child_section],
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="params-lookup-test",
        sections=[parent_section, vfs_section],
    )

    Prompt(prompt).bind(_ChildParams(value="from-lookup")).render()

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    # Find the context file
    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Verify child used params from lookup
    assert "Child value: from-lookup" in context_file.content


def test_summary_section_with_child_constructable_without_args() -> None:
    """Child sections whose params can be constructed without args work correctly."""
    from weakincentives.runtime import InProcessEventBus, Session
    from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem

    @dataclass
    class _DefaultableParams:
        value: str = "default-value"

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    # Child section with params that have defaults
    child_section = MarkdownSection[_DefaultableParams](
        title="Child",
        template="Child value: ${value}.",
        key="child",
    )

    parent_section = MarkdownSection[_RefParams](
        title="Parent",
        template="Parent content: ${topic}.",
        key="parent",
        summary="Parent docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="parent-topic"),
        children=[child_section],
    )

    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="constructable-params-test",
        sections=[parent_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Check VFS suffix is used
    assert "/context/parent.md" in rendered.text

    # Check that content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    # Find the context file
    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Verify child was rendered with default params
    assert "Child value: default-value" in context_file.content


class _MockWorkspaceWithNullSession:
    """Mock workspace section that returns None for session."""

    @property
    def session(self) -> None:
        return None


def test_auto_render_skipped_when_session_is_none() -> None:
    """Auto-rendering is skipped when workspace section's session is None."""
    from weakincentives.prompt.progressive_disclosure import WorkspaceSection
    from weakincentives.prompt.registry import SectionNode
    from weakincentives.prompt.rendering import PromptRenderer

    # Create mock workspace that matches protocol but returns None session
    mock = _MockWorkspaceWithNullSession()
    # The mock matches WorkspaceSection protocol due to having 'session' property
    assert isinstance(mock, WorkspaceSection)

    # Create a simple section node for testing
    section = MarkdownSection[_RefParams](
        title="Test",
        template="Test: ${topic}.",
        key="test",
        summary="Test docs available.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_RefParams(topic="test-topic"),
    )

    # Test that _auto_render_to_vfs handles None session gracefully
    prompt = PromptTemplate(
        ns="tests/auto-render",
        key="null-session-test",
        sections=[section],
    )
    assert prompt._snapshot is not None  # Type narrowing
    renderer = PromptRenderer(
        registry=prompt._snapshot,
        structured_output=None,
    )

    # Create a minimal node
    node = cast(
        SectionNode[SupportsDataclass],
        SectionNode(section=section, path=("test",), depth=0, number="1"),
    )

    # This should return early without error when session is None
    renderer._auto_render_to_vfs(
        node,
        _RefParams(topic="test-topic"),
        mock,  # workspace with None session
        {},
    )
    # If we get here without error, the None session check worked
