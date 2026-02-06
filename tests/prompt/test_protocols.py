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
    return Session(dispatcher=InProcessDispatcher())


class TestToolSuiteSectionProtocol:
    """Tests for ToolSuiteSection protocol conformance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """ToolSuiteSection can be used with isinstance."""
        # Protocol should be usable with isinstance
        assert hasattr(ToolSuiteSection, "__protocol_attrs__") or hasattr(
            ToolSuiteSection, "_is_runtime_protocol"
        )

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
                return Session(dispatcher=InProcessDispatcher())

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

        session = Session(dispatcher=InProcessDispatcher())
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
