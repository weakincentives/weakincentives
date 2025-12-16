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

"""Workspace section protocol for workspace digest optimization.

This module provides a protocol that all workspace sections
(VFS, Podman, Claude Agent SDK) should implement. The optimizer
checks for this protocol to identify valid workspace sections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...runtime.session import Session
    from .filesystem import Filesystem

__all__ = ["WorkspaceSection"]


@runtime_checkable
class WorkspaceSection(Protocol):
    """Protocol for workspace sections.

    All workspace sections (VfsToolsSection, PodmanSandboxSection,
    ClaudeAgentWorkspaceSection) should implement this protocol.
    This enables the WorkspaceDigestOptimizer to identify valid
    workspace sections without importing adapter-specific code.

    The protocol requires:
    - A `_is_workspace_section` class attribute set to True (marker)
    - A session property returning the associated Session
    - A filesystem property returning the Filesystem managed by this section
    - A clone() method for creating copies with a new session
    """

    _is_workspace_section: bool

    @property
    def session(self) -> Session:
        """Return the session associated with this workspace section."""
        ...

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this workspace section."""
        ...

    def clone(self, **kwargs: object) -> WorkspaceSection:
        """Clone the section with new session/bus."""
        ...
