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

"""Tests for prompt protocol conformance.

Verifies that all implementations of ToolSuiteSection and WorkspaceSection
protocols conform to the expected interfaces.
"""

from __future__ import annotations

import pytest

from weakincentives.prompt.protocols import ToolSuiteSection, WorkspaceSection
from weakincentives.runtime import InProcessDispatcher, Session


@pytest.fixture
def session() -> Session:
    """Create a session for testing."""
    return Session(bus=InProcessDispatcher())


class TestToolSuiteSectionProtocol:
    """Tests for ToolSuiteSection protocol conformance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """ToolSuiteSection can be used with isinstance."""
        # Protocol should be usable with isinstance
        assert hasattr(ToolSuiteSection, "__protocol_attrs__") or hasattr(
            ToolSuiteSection, "_is_runtime_protocol"
        )

    def test_vfs_tools_section_conforms(self, session: Session) -> None:
        """VfsToolsSection implements ToolSuiteSection."""
        from weakincentives.contrib.tools import VfsToolsSection

        section = VfsToolsSection(session=session)

        # Verify protocol attributes exist
        assert hasattr(section, "session")
        assert hasattr(section, "accepts_overrides")
        assert hasattr(section, "clone")

        # Verify runtime check works
        assert isinstance(section, ToolSuiteSection)

        # Verify attribute types
        assert section.session is session
        assert isinstance(section.accepts_overrides, bool)

    def test_planning_tools_section_conforms(self, session: Session) -> None:
        """PlanningToolsSection implements ToolSuiteSection."""
        from weakincentives.contrib.tools import PlanningToolsSection

        section = PlanningToolsSection(session=session)

        assert isinstance(section, ToolSuiteSection)
        assert section.session is session
        assert isinstance(section.accepts_overrides, bool)
        assert callable(section.clone)

    def test_asteval_section_conforms(self, session: Session) -> None:
        """AstevalSection implements ToolSuiteSection."""
        from weakincentives.contrib.tools import AstevalSection

        section = AstevalSection(session=session)

        assert isinstance(section, ToolSuiteSection)
        assert section.session is session
        assert isinstance(section.accepts_overrides, bool)
        assert callable(section.clone)

    def test_workspace_digest_section_does_not_conform(self, session: Session) -> None:
        """WorkspaceDigestSection does NOT implement ToolSuiteSection.

        WorkspaceDigestSection is a Section subclass that stores session internally
        but does not expose a public `session` property required by the protocol.
        """
        from weakincentives.contrib.tools import WorkspaceDigestSection

        section = WorkspaceDigestSection(session=session)

        # WorkspaceDigestSection does not expose session as a property
        assert not isinstance(section, ToolSuiteSection)

    def test_clone_returns_new_instance(self, session: Session) -> None:
        """Clone creates a new section with the new session."""
        from weakincentives.contrib.tools import PlanningToolsSection

        section = PlanningToolsSection(session=session)
        new_session = Session(bus=InProcessDispatcher())

        cloned = section.clone(session=new_session)

        assert cloned is not section
        assert cloned.session is new_session
        assert isinstance(cloned, ToolSuiteSection)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        """Classes not implementing the protocol fail isinstance check."""

        class NotAToolSuite:
            pass

        obj = NotAToolSuite()
        assert not isinstance(obj, ToolSuiteSection)

    def test_partial_implementation_fails(self) -> None:
        """Classes with only some attributes fail isinstance check."""

        class PartialImpl:
            @property
            def session(self) -> Session:
                return Session(bus=InProcessDispatcher())

            # Missing accepts_overrides and clone

        obj = PartialImpl()
        assert not isinstance(obj, ToolSuiteSection)


class TestWorkspaceSectionProtocol:
    """Tests for WorkspaceSection protocol conformance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """WorkspaceSection can be used with isinstance."""
        assert hasattr(WorkspaceSection, "__protocol_attrs__") or hasattr(
            WorkspaceSection, "_is_runtime_protocol"
        )

    def test_workspace_section_extends_tool_suite(self) -> None:
        """WorkspaceSection inherits from ToolSuiteSection.

        Note: We can't use issubclass() because protocols with non-method
        members (properties) don't support it. We verify inheritance through
        the MRO instead.
        """
        # Check inheritance via MRO (Method Resolution Order)
        assert ToolSuiteSection in WorkspaceSection.__mro__

    def test_vfs_tools_section_conforms_to_workspace(self, session: Session) -> None:
        """VfsToolsSection implements WorkspaceSection."""
        from weakincentives.contrib.tools import VfsToolsSection

        section = VfsToolsSection(session=session)

        # Verify WorkspaceSection protocol
        assert isinstance(section, WorkspaceSection)
        assert hasattr(section, "filesystem")

        # Verify filesystem property returns correct type
        from weakincentives.filesystem import Filesystem

        assert isinstance(section.filesystem, Filesystem)

    def test_claude_agent_workspace_section_conforms(self, session: Session) -> None:
        """ClaudeAgentWorkspaceSection implements WorkspaceSection."""
        from weakincentives.adapters.claude_agent_sdk import ClaudeAgentWorkspaceSection

        section = ClaudeAgentWorkspaceSection(session=session)

        try:
            assert isinstance(section, WorkspaceSection)
            assert hasattr(section, "filesystem")

            from weakincentives.filesystem import Filesystem

            assert isinstance(section.filesystem, Filesystem)
        finally:
            # Clean up temp directory created by section
            section.cleanup()

    def test_workspace_section_has_all_tool_suite_attrs(self, session: Session) -> None:
        """WorkspaceSection implementations have all ToolSuiteSection attributes."""
        from weakincentives.contrib.tools import VfsToolsSection

        section = VfsToolsSection(session=session)

        # ToolSuiteSection attributes
        assert hasattr(section, "session")
        assert hasattr(section, "accepts_overrides")
        assert hasattr(section, "clone")

        # WorkspaceSection attributes
        assert hasattr(section, "filesystem")

    def test_clone_preserves_workspace_protocol(self, session: Session) -> None:
        """Cloned workspace sections still implement WorkspaceSection."""
        from weakincentives.contrib.tools import VfsToolsSection

        section = VfsToolsSection(session=session)
        new_session = Session(bus=InProcessDispatcher())

        cloned = section.clone(session=new_session)

        assert isinstance(cloned, WorkspaceSection)
        assert hasattr(cloned, "filesystem")

    def test_non_workspace_tool_suite_fails_isinstance(self, session: Session) -> None:
        """ToolSuiteSection implementations without filesystem fail WorkspaceSection check."""
        from weakincentives.contrib.tools import PlanningToolsSection

        section = PlanningToolsSection(session=session)

        # Should pass ToolSuiteSection
        assert isinstance(section, ToolSuiteSection)
        # Should fail WorkspaceSection (no filesystem)
        assert not isinstance(section, WorkspaceSection)


class TestProtocolExports:
    """Tests for protocol module exports."""

    def test_protocols_exported_from_prompt(self) -> None:
        """ToolSuiteSection and WorkspaceSection exported from prompt module."""
        from weakincentives.prompt import (
            ToolSuiteSection as ExportedToolSuite,
            WorkspaceSection as ExportedWorkspace,
        )

        assert ExportedToolSuite is ToolSuiteSection
        assert ExportedWorkspace is WorkspaceSection

    def test_protocols_exported_from_contrib_tools(self) -> None:
        """Protocols re-exported from contrib.tools for convenience."""
        from weakincentives.contrib.tools import (
            ToolSuiteSection as ContribToolSuite,
            WorkspaceSection as ContribWorkspace,
        )

        assert ContribToolSuite is ToolSuiteSection
        assert ContribWorkspace is WorkspaceSection

    def test_protocols_in_module_all(self) -> None:
        """Protocols listed in __all__ exports."""
        from weakincentives import prompt

        assert "ToolSuiteSection" in prompt.__all__
        assert "WorkspaceSection" in prompt.__all__


class TestDuckTypingConformance:
    """Tests for duck-typing protocol conformance."""

    def test_custom_implementation_passes_isinstance(self) -> None:
        """Custom class implementing protocol passes isinstance."""
        from weakincentives.contrib.tools import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        class CustomWorkspaceSection:
            def __init__(self, session: Session) -> None:
                self._session = session
                self._fs = InMemoryFilesystem()

            @property
            def session(self) -> Session:
                return self._session

            @property
            def accepts_overrides(self) -> bool:
                return False

            @property
            def filesystem(self) -> Filesystem:
                return self._fs

            def clone(self, **kwargs: object) -> CustomWorkspaceSection:
                new_session = kwargs.get("session", self._session)
                assert isinstance(new_session, Session)
                return CustomWorkspaceSection(new_session)

        session = Session(bus=InProcessDispatcher())
        custom = CustomWorkspaceSection(session)

        # Should pass both protocol checks
        assert isinstance(custom, ToolSuiteSection)
        assert isinstance(custom, WorkspaceSection)

    def test_method_signature_mismatch_still_passes_runtime_check(self) -> None:
        """Runtime check only verifies attribute existence, not signatures.

        This is a known limitation of @runtime_checkable protocols.
        Type checkers will catch signature mismatches at static analysis time.
        """

        class QuirkySection:
            @property
            def session(self) -> str:  # Wrong return type
                return "not a session"

            @property
            def accepts_overrides(self) -> int:  # Wrong return type
                return 42

            def clone(self) -> None:  # Wrong signature
                pass

        quirky = QuirkySection()
        # Runtime check only verifies attribute existence
        assert isinstance(quirky, ToolSuiteSection)
