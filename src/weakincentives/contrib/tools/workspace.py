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

"""Workspace and tool suite section protocols.

This module provides protocols that all capability sections should implement:

- **ToolSuiteSection**: Base protocol for any section exposing tools. Provides
  session binding and standardized accepts_overrides handling.

- **WorkspaceSection**: Extended protocol for sections that manage a filesystem.
  Used by the WorkspaceDigestOptimizer to identify valid workspace sections.

Both protocols are ``@runtime_checkable``, so use ``isinstance(section,
ToolSuiteSection)`` or ``isinstance(section, WorkspaceSection)`` to test
whether a section implements the required functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

if TYPE_CHECKING:
    from weakincentives.filesystem import Filesystem

    from ...runtime.session import Session

__all__ = ["ToolSuiteSection", "WorkspaceSection"]


@runtime_checkable
class ToolSuiteSection(Protocol):
    """Protocol for sections that expose tool suites.

    All tool suite sections (VfsToolsSection, AstevalSection,
    PlanningToolsSection, PodmanSandboxSection, WorkspaceDigestSection)
    should implement this protocol. This enables consistent handling
    of session binding and cloning across all capability sections.

    The protocol requires:

    - **session**: Property returning the associated Session
    - **accepts_overrides**: Property indicating if overrides are accepted
    - **clone()**: Method for creating copies with a new session

    Example::

        from weakincentives.contrib.tools import VfsToolsSection, VfsConfig

        vfs = VfsToolsSection(session=session)
        assert vfs.accepts_overrides is False  # Default
    """

    @property
    def session(self) -> Session:
        """Return the session associated with this tool suite section."""
        ...

    @property
    def accepts_overrides(self) -> bool:
        """Return True if this section accepts parameter overrides."""
        ...

    def clone(self, **kwargs: object) -> ToolSuiteSection:
        """Clone the section with new session.

        Args:
            **kwargs: Must include ``session`` with the new Session instance.

        Returns:
            A new section instance bound to the provided session.
        """
        ...


@runtime_checkable
class WorkspaceSection(ToolSuiteSection, Protocol):
    """Protocol for workspace sections that manage a filesystem.

    Extends ToolSuiteSection with a filesystem property. All workspace
    sections (VfsToolsSection, PodmanSandboxSection, ClaudeAgentWorkspaceSection)
    should implement this protocol. This enables the WorkspaceDigestOptimizer
    to identify valid workspace sections without importing adapter-specific code.

    Additional requirement beyond ToolSuiteSection:

    - **filesystem**: Property returning the Filesystem managed by this section
    """

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this workspace section."""
        ...

    @override
    def clone(self, **kwargs: object) -> WorkspaceSection:
        """Clone the section with new session/bus."""
        ...
