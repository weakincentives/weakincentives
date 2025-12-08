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

"""Integration tests for auto-rendering SUMMARY sections to VFS/Podman."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt._visibility import SectionVisibility
from weakincentives.prompt.progressive_disclosure import (
    WorkspaceSection,
    build_vfs_summary_suffix,
    compute_vfs_context_path,
    find_workspace_section,
    section_subtree_has_tools,
)
from weakincentives.prompt.tool import Tool, ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools.vfs import VfsToolsSection, VirtualFileSystem


@dataclass(slots=True, frozen=True)
class _TestParams:
    """Test parameters for sections."""

    content: str = "test-content"


@dataclass(slots=True, frozen=True)
class _ToolResult:
    """Result type for test tools."""

    value: str = ""


def _create_test_tool() -> Tool[_TestParams, _ToolResult]:
    """Create a simple test tool."""

    def handler(
        params: _TestParams, *, context: ToolContext
    ) -> ToolResult[_ToolResult]:
        return ToolResult(message="done", value=_ToolResult(value=params.content))

    return Tool[_TestParams, _ToolResult](
        name="test_tool",
        description="A test tool",
        handler=handler,
    )


# ==============================================================================
# VFS Integration Tests
# ==============================================================================


def test_vfs_summary_section_without_tools_renders_to_vfs() -> None:
    """A SUMMARY section without tools should auto-render to VFS."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    doc_section = MarkdownSection[_TestParams](
        title="Documentation",
        template="Full documentation: ${content}.",
        key="docs",
        summary="Documentation overview.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="detailed docs"),
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="vfs-auto-render",
        sections=[doc_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Should use VFS suffix instead of open_sections
    assert "/context/docs.md" in rendered.text
    assert "open_sections" not in rendered.text

    # Verify content was written to VFS
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "docs.md")),
        None,
    )
    assert context_file is not None
    assert "Full documentation: detailed docs" in context_file.content


def test_vfs_summary_section_with_tools_uses_open_sections() -> None:
    """A SUMMARY section with tools should use open_sections, not VFS."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    tool = _create_test_tool()

    doc_section = MarkdownSection[_TestParams](
        title="Interactive Docs",
        template="Interactive documentation: ${content}.",
        key="interactive-docs",
        summary="Interactive docs overview.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="interactive content"),
        tools=(tool,),
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="vfs-with-tools",
        sections=[doc_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Should use open_sections suffix, not VFS
    assert "open_sections" in rendered.text
    assert "/context/interactive-docs.md" not in rendered.text

    # VFS should NOT have the context file
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    if vfs_snapshot is not None:
        context_file = next(
            (
                f
                for f in vfs_snapshot.files
                if f.path.segments == ("context", "interactive-docs.md")
            ),
            None,
        )
        assert context_file is None


def test_vfs_nested_summary_sections_render_complete_subtree() -> None:
    """Nested SUMMARY sections should render the complete subtree to VFS."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    child1 = MarkdownSection[_TestParams](
        title="Child One",
        template="Child one content: ${content}.",
        key="child1",
        default_params=_TestParams(content="child-1-data"),
    )

    child2 = MarkdownSection[_TestParams](
        title="Child Two",
        template="Child two content: ${content}.",
        key="child2",
        default_params=_TestParams(content="child-2-data"),
    )

    parent = MarkdownSection[_TestParams](
        title="Parent Section",
        template="Parent content: ${content}.",
        key="parent",
        summary="Parent overview with children.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="parent-data"),
        children=[child1, child2],
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="nested-vfs",
        sections=[parent, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # VFS suffix should be used
    assert "/context/parent.md" in rendered.text

    # Verify complete subtree was written
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # All sections should be in the rendered file
    assert "Parent content: parent-data" in context_file.content
    assert "Child one content: child-1-data" in context_file.content
    assert "Child two content: child-2-data" in context_file.content


def test_vfs_multiple_summary_sections_render_to_separate_files() -> None:
    """Multiple SUMMARY sections should each render to separate VFS files."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    section1 = MarkdownSection[_TestParams](
        title="Section One",
        template="Content one: ${content}.",
        key="section1",
        summary="Overview one.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="data-1"),
    )

    section2 = MarkdownSection[_TestParams](
        title="Section Two",
        template="Content two: ${content}.",
        key="section2",
        summary="Overview two.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="data-2"),
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="multi-vfs",
        sections=[section1, section2, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Both VFS suffixes should be present
    assert "/context/section1.md" in rendered.text
    assert "/context/section2.md" in rendered.text

    # Verify both files were created
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    file1 = next(
        (
            f
            for f in vfs_snapshot.files
            if f.path.segments == ("context", "section1.md")
        ),
        None,
    )
    file2 = next(
        (
            f
            for f in vfs_snapshot.files
            if f.path.segments == ("context", "section2.md")
        ),
        None,
    )

    assert file1 is not None
    assert file2 is not None
    assert "Content one: data-1" in file1.content
    assert "Content two: data-2" in file2.content


def test_vfs_section_is_detected_as_workspace_section() -> None:
    """VfsToolsSection should be detected as a WorkspaceSection."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    assert isinstance(vfs_section, WorkspaceSection)
    assert vfs_section.session is session


def test_find_workspace_section_finds_vfs_section() -> None:
    """find_workspace_section should find VfsToolsSection in registry."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    other_section = MarkdownSection(
        title="Other",
        template="Other content.",
        key="other",
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="find-vfs",
        sections=[other_section, vfs_section],
    )

    assert prompt._snapshot is not None
    found = find_workspace_section(prompt._snapshot)
    assert found is not None
    assert found.session is session


# ==============================================================================
# Podman Integration Tests
# ==============================================================================


class _MockPodmanSection:
    """Mock PodmanSandboxSection for testing without container runtime."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        return self._session

    @property
    def key(self) -> str:
        return "podman"

    @property
    def title(self) -> str:
        return "Podman Sandbox"


def test_podman_mock_is_detected_as_workspace_section() -> None:
    """Mock Podman section should be detected as WorkspaceSection."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    mock_podman = _MockPodmanSection(session=session)

    assert isinstance(mock_podman, WorkspaceSection)
    assert mock_podman.session is session


def test_podman_section_has_session_property() -> None:
    """Real PodmanSandboxSection should implement WorkspaceSection protocol."""
    from inspect import signature

    from weakincentives.tools.podman import PodmanSandboxSection

    # Check if PodmanSandboxSection has the session property (protocol check)
    # We can't instantiate without a Podman connection, so we check the class
    assert hasattr(PodmanSandboxSection, "session")

    # Verify the class would satisfy the protocol by checking it has session param
    # This is a static check since we can't instantiate without Podman
    init_sig = signature(PodmanSandboxSection.__init__)
    assert "session" in init_sig.parameters


# ==============================================================================
# Helper Function Tests
# ==============================================================================


def test_compute_vfs_context_path_formats_correctly() -> None:
    """compute_vfs_context_path should format paths correctly."""
    assert compute_vfs_context_path(("docs",)) == "/context/docs.md"
    assert compute_vfs_context_path(("api", "v1")) == "/context/api.v1.md"
    assert (
        compute_vfs_context_path(("guides", "getting-started"))
        == "/context/guides.getting-started.md"
    )


def test_build_vfs_summary_suffix_includes_path() -> None:
    """build_vfs_summary_suffix should include the VFS path."""
    suffix = build_vfs_summary_suffix(("docs",))
    assert "/context/docs.md" in suffix
    assert "summarized" in suffix.lower()


def test_section_subtree_has_tools_detects_direct_tools() -> None:
    """section_subtree_has_tools should detect direct tools."""
    tool = _create_test_tool()

    section_with_tools = MarkdownSection(
        title="With Tools",
        template="Content.",
        key="with-tools",
        tools=(tool,),
    )

    section_without_tools = MarkdownSection(
        title="Without Tools",
        template="Content.",
        key="without-tools",
    )

    assert section_subtree_has_tools(section_with_tools) is True
    assert section_subtree_has_tools(section_without_tools) is False


def test_section_subtree_has_tools_detects_nested_tools() -> None:
    """section_subtree_has_tools should detect tools in children."""
    tool = _create_test_tool()

    child_with_tools = MarkdownSection(
        title="Child",
        template="Child content.",
        key="child",
        tools=(tool,),
    )

    parent = MarkdownSection(
        title="Parent",
        template="Parent content.",
        key="parent",
        children=[child_with_tools],
    )

    assert section_subtree_has_tools(parent) is True


def test_section_subtree_has_tools_no_tools_in_tree() -> None:
    """section_subtree_has_tools should return False when no tools in tree."""
    child = MarkdownSection(
        title="Child",
        template="Child content.",
        key="child",
    )

    parent = MarkdownSection(
        title="Parent",
        template="Parent content.",
        key="parent",
        children=[child],
    )

    assert section_subtree_has_tools(parent) is False


# ==============================================================================
# Edge Cases
# ==============================================================================


def test_no_workspace_section_falls_back_to_open_sections() -> None:
    """Without a workspace section, should fall back to open_sections."""
    doc_section = MarkdownSection[_TestParams](
        title="Documentation",
        template="Full documentation: ${content}.",
        key="docs",
        summary="Documentation overview.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="detailed docs"),
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="no-workspace",
        sections=[doc_section],  # No VFS or Podman section
    )

    rendered = Prompt(prompt).render()

    # Should use open_sections since no workspace is available
    assert "open_sections" in rendered.text
    assert "/context/docs.md" not in rendered.text


def test_child_tools_prevent_auto_render() -> None:
    """Tools in child sections should prevent auto-rendering."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    tool = _create_test_tool()

    child_with_tools = MarkdownSection(
        title="Child",
        template="Child content.",
        key="child",
        tools=(tool,),
    )

    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent content: ${content}.",
        key="parent",
        summary="Parent overview.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="parent-data"),
        children=[child_with_tools],
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="child-tools",
        sections=[parent, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Should use open_sections due to child tools
    assert "open_sections" in rendered.text
    assert "/context/parent.md" not in rendered.text


def test_disabled_children_not_rendered_to_vfs() -> None:
    """Disabled child sections should not be rendered to VFS."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    enabled_child = MarkdownSection[_TestParams](
        title="Enabled",
        template="Enabled: ${content}.",
        key="enabled",
        default_params=_TestParams(content="visible"),
    )

    disabled_child = MarkdownSection[_TestParams](
        title="Disabled",
        template="Disabled: ${content}.",
        key="disabled",
        default_params=_TestParams(content="hidden"),
        enabled=lambda _: False,
    )

    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${content}.",
        key="parent",
        summary="Parent overview.",
        visibility=SectionVisibility.SUMMARY,
        default_params=_TestParams(content="parent-data"),
        children=[enabled_child, disabled_child],
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="disabled-children",
        sections=[parent, vfs_section],
    )

    Prompt(prompt).render()

    vfs_snapshot = session.query(VirtualFileSystem).latest()
    assert vfs_snapshot is not None

    context_file = next(
        (f for f in vfs_snapshot.files if f.path.segments == ("context", "parent.md")),
        None,
    )
    assert context_file is not None

    # Enabled child should be present
    assert "Enabled: visible" in context_file.content
    # Disabled child should NOT be present
    assert "Disabled: hidden" not in context_file.content


def test_full_visibility_does_not_auto_render() -> None:
    """Sections with FULL visibility should not auto-render to VFS."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)

    doc_section = MarkdownSection[_TestParams](
        title="Documentation",
        template="Full documentation: ${content}.",
        key="docs",
        visibility=SectionVisibility.FULL,  # FULL, not SUMMARY
        default_params=_TestParams(content="detailed docs"),
    )

    prompt = PromptTemplate(
        ns="tests/integration",
        key="full-visibility",
        sections=[doc_section, vfs_section],
    )

    rendered = Prompt(prompt).render()

    # Full content should be inline, not in VFS
    assert "Full documentation: detailed docs" in rendered.text
    assert "/context/docs.md" not in rendered.text

    # VFS should not have the context file
    vfs_snapshot = session.query(VirtualFileSystem).latest()
    if vfs_snapshot is not None:
        context_file = next(
            (
                f
                for f in vfs_snapshot.files
                if f.path.segments == ("context", "docs.md")
            ),
            None,
        )
        assert context_file is None
