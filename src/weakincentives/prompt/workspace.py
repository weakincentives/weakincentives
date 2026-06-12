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

"""Workspace preview for sandbox-backed prompts.

When a :class:`~weakincentives.prompt.PromptTemplate` declares environment
intent via ``sandbox=SandboxConfig(...)``, the template carries a workspace
preview section. The preview renders from the *opened* sandbox:
``prompt.render(session=..., sandbox=sandbox)`` resolves
:class:`WorkspacePreviewParams` (built via :func:`workspace_preview_params`
from ``sandbox.filesystem.list``) at render time, so the prompt always
describes the environment the agent actually acts on and no run state is
ever stored on the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ..dataclasses import FrozenDataclass
from .markdown import MarkdownSection

if TYPE_CHECKING:
    from ..filesystem import Filesystem

__all__ = [
    "WORKSPACE_PREVIEW_KEY",
    "WorkspacePreviewParams",
    "workspace_preview_params",
    "workspace_preview_section",
]

WORKSPACE_PREVIEW_KEY: Final = "workspace"
"""Section key used by the auto-appended workspace preview section."""

_PREVIEW_ENTRY_LIMIT: Final = 50
_PENDING_LISTING: Final = "(workspace not yet materialized)"
_EMPTY_LISTING: Final = "(empty workspace)"

_PREVIEW_TEMPLATE: Final = """
Your working directory is a sandboxed workspace materialized from the
prompt's sandbox configuration. Use the tools available to explore and
work with its contents.

${listing}
"""


@FrozenDataclass()
class WorkspacePreviewParams:
    """Listing of the opened sandbox root rendered into the prompt."""

    listing: str = _PENDING_LISTING


def workspace_preview_section(
    *, key: str = WORKSPACE_PREVIEW_KEY
) -> MarkdownSection[WorkspacePreviewParams]:
    """Build the workspace preview section appended to sandbox-backed templates."""
    return MarkdownSection[WorkspacePreviewParams](
        title="Workspace",
        key=key,
        template=_PREVIEW_TEMPLATE,
        default_params=WorkspacePreviewParams(),
        accepts_overrides=False,
    )


def workspace_preview_params(
    filesystem: Filesystem, *, limit: int = _PREVIEW_ENTRY_LIMIT
) -> WorkspacePreviewParams:
    """Render the top-level listing of an opened sandbox filesystem.

    Args:
        filesystem: The opened sandbox's filesystem facet.
        limit: Maximum number of entries included in the listing.

    Returns:
        Params carrying a markdown listing of the sandbox root.
    """
    entries = filesystem.list(".")
    lines = [
        f"- {entry.name}/" if entry.is_directory else f"- {entry.name}"
        for entry in entries[:limit]
    ]
    remaining = len(entries) - limit
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    listing = "\n".join(lines) if lines else _EMPTY_LISTING
    return WorkspacePreviewParams(listing=listing)
