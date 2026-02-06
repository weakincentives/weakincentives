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

"""Domain-specific tool utilities for LLM agent prompts.

This package provides utilities that extend the core weakincentives primitives.

Tool Categories
---------------

**Workspace Digest** (``WorkspaceDigestSection``)
    Caching layer for workspace summaries. Renders cached digest content
    from session state, with dynamic visibility (summary vs full body).

**In-Memory Filesystem** (``InMemoryFilesystem``)
    Session-scoped in-memory filesystem implementation for testing and
    evaluation scenarios. Implements the Filesystem protocol.

Public Exports
--------------

Workspace Digest
~~~~~~~~~~~~~~~~

WorkspaceDigestSection
    Prompt section that renders cached workspace digests from session
    state. Supports SUMMARY/FULL visibility modes.

WorkspaceDigest
    State object containing digest summary and body.

set_workspace_digest, clear_workspace_digest, latest_workspace_digest
    Session state management functions for digests.

In-Memory Filesystem
~~~~~~~~~~~~~~~~~~~~

InMemoryFilesystem
    In-memory implementation of the Filesystem protocol.

Example Usage
-------------

Workspace digest caching::

    from weakincentives.contrib.tools import (
        WorkspaceDigestSection,
        set_workspace_digest,
    )

    digest_section = WorkspaceDigestSection(session=session)

    # Populate digest
    set_workspace_digest(
        session,
        section_key="workspace-digest",
        body="Full project analysis...",
        summary="Python web app with FastAPI backend.",
    )

    # Section now renders the cached digest

In-memory filesystem for testing::

    from weakincentives.contrib.tools import InMemoryFilesystem

    fs = InMemoryFilesystem()
    fs.write("test.txt", "Hello, world!")
    result = fs.read("test.txt")
    print(result.content)  # "Hello, world!"
"""

from __future__ import annotations

from .digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
    clear_workspace_digest,
    latest_workspace_digest,
    set_workspace_digest,
)
from .filesystem_memory import InMemoryFilesystem

__all__ = [
    "InMemoryFilesystem",
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "clear_workspace_digest",
    "latest_workspace_digest",
    "set_workspace_digest",
]


def __dir__() -> list[str]:
    """Return sorted list of public symbols."""
    return sorted(__all__)
