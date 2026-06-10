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

"""Generic workspace section for prompt templates.

Provides a provider-agnostic workspace section that manages a temporary
directory with host file mounts.  The same ``WorkspaceSection`` instance
works with any adapter (Claude Agent SDK, Codex App Server, etc.),
making prompt templates portable across providers.

The mount machinery (:class:`HostMount`, materialization, fingerprints)
lives in :mod:`weakincentives.sandbox` and is re-exported here for the
section API.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, override

from ..clock import SYSTEM_CLOCK
from ..filesystem import Filesystem, HostBackend
from ..resources import ResourceRegistry
from ..runtime.session import Session
from ..sandbox import (
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    compute_workspace_fingerprint,
    materialize_mounts,
)
from .markdown import MarkdownSection

__all__ = [
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSection",
    "WorkspaceSecurityError",
    "compute_workspace_fingerprint",
]

_TEMPLATE_PREVIEW_LIMIT: Final[int] = 10


def _utcnow() -> datetime:
    return SYSTEM_CLOCK.utcnow()


def _render_workspace_template(previews: tuple[HostMountPreview, ...]) -> str:
    """Render the workspace section template from mount previews."""
    lines = [
        (
            "The workspace has been populated with the following mounted content. "
            "Use the tools available to explore and work with these files."
        )
    ]

    if not previews:
        lines.append("\n- (no host mounts configured)")
    else:
        for preview in previews:
            kind = "directory" if preview.is_directory else "file"
            lines.append(f"\n**{preview.mount_path}** ({kind}):")
            if preview.entries:
                visible_entries = preview.entries[:_TEMPLATE_PREVIEW_LIMIT]
                lines.extend(f"  - {entry}" for entry in visible_entries)
                remaining = len(preview.entries) - _TEMPLATE_PREVIEW_LIMIT
                if remaining > 0:
                    lines.append(f"  - ... and {remaining} more")
            lines.append(f"  - Total: {preview.bytes_copied:,} bytes")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WorkspaceSection
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _WorkspaceSectionParams:
    """Default params for WorkspaceSection (empty placeholder)."""


class WorkspaceSection(MarkdownSection[_WorkspaceSectionParams]):
    """Provider-agnostic prompt section managing a temporary workspace.

    This section creates a temporary directory, copies host files into it
    according to the supplied ``HostMount`` configurations, and renders a
    markdown summary of the mounted content.  The same ``WorkspaceSection``
    works with any adapter—no adapter-specific subclass required.

    Attributes:
        temp_dir: Path to the temporary directory.
        mount_previews: Summaries of each materialized mount.
        created_at: UTC timestamp when the workspace was created.
    """

    def __init__(
        self,
        *,
        session: Session,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        accepts_overrides: bool = False,
        key: str = "workspace",
        _temp_dir: Path | None = None,
        _mount_previews: tuple[HostMountPreview, ...] | None = None,
        _created_at: datetime | None = None,
        _filesystem: Filesystem | None = None,
        _ref_lock: threading.Lock | None = None,
        _ref_count: list[int] | None = None,
    ) -> None:
        """Initialize the workspace section.

        Args:
            session: Session for state management.
            mounts: Host mount configurations.
            allowed_host_roots: Security boundary for host paths.
            accepts_overrides: Whether section accepts prompt overrides.
            key: Section key (default ``"workspace"``).
            _temp_dir: Internal - pre-existing temp directory (for cloning).
            _mount_previews: Internal - pre-existing mount previews (for cloning).
            _created_at: Internal - pre-existing creation timestamp (for cloning).
            _filesystem: Internal - pre-existing filesystem (for cloning).
            _ref_lock: Internal - shared lock for reference counting (for cloning).
            _ref_count: Internal - shared reference count (for cloning).
        """
        self._session = session
        self._mounts = tuple(mounts)
        self._allowed_host_roots = tuple(Path(r) for r in allowed_host_roots)
        self._accepts_overrides = accepts_overrides
        self._key = key

        if _temp_dir is not None and _mount_previews is not None:
            self._temp_dir = _temp_dir
            self._mount_previews = _mount_previews
            self._created_at = _created_at or _utcnow()
            self._filesystem: Filesystem = (
                _filesystem
                if _filesystem is not None
                else Filesystem.host(self._temp_dir)
            )
        elif mounts:
            self._temp_dir, self._mount_previews = materialize_mounts(
                mounts, allowed_host_roots=self._allowed_host_roots
            )
            self._created_at = _utcnow()
            self._filesystem = Filesystem.host(self._temp_dir)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="wink-workspace-"))
            self._mount_previews = ()
            self._created_at = _utcnow()
            self._filesystem = Filesystem.host(self._temp_dir)

        self._ref_lock = _ref_lock if _ref_lock is not None else threading.Lock()
        self._ref_count = _ref_count if _ref_count is not None else [1]

        template = _render_workspace_template(self._mount_previews)

        super().__init__(
            title="Workspace",
            key=key,
            template=template,
            default_params=_WorkspaceSectionParams(),
            tools=(),
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """Return the session associated with this section."""
        return self._session

    @property
    def temp_dir(self) -> Path:
        """Return the path to the temporary workspace directory."""
        return self._temp_dir

    @property
    def mount_previews(self) -> tuple[HostMountPreview, ...]:
        """Return summaries of each materialized mount."""
        return self._mount_previews

    @property
    def created_at(self) -> datetime:
        """Return the UTC timestamp when the workspace was created."""
        return self._created_at

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this workspace section."""
        return self._filesystem

    @property
    def workspace_fingerprint(self) -> str:
        """Return a deterministic fingerprint for reuse detection."""
        return compute_workspace_fingerprint(self._mounts)

    @override
    def cleanup(self) -> None:
        """Remove the temporary workspace directory and associated resources."""
        with self._ref_lock:
            self._ref_count[0] -= 1
            if self._ref_count[0] > 0:
                return
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        backend = self._filesystem.backend
        if isinstance(backend, HostBackend):  # pragma: no branch
            backend.cleanup()

    @override
    def resources(self) -> ResourceRegistry:
        """Return resources contributed by this workspace section."""
        return ResourceRegistry.build({Filesystem: self._filesystem})

    @override
    def clone(self, **kwargs: Any) -> WorkspaceSection:
        """Clone the section with a new session.

        Args:
            **kwargs: Must include 'session' key with a Session value.

        Returns:
            New WorkspaceSection with the same workspace config.

        Raises:
            TypeError: If session is not provided or dispatcher doesn't match.
        """
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone WorkspaceSection."
            raise TypeError(msg)
        provided_dispatcher = kwargs.get("dispatcher")
        if (
            provided_dispatcher is not None
            and provided_dispatcher is not session_obj.dispatcher
        ):
            msg = "Provided dispatcher must match the target session's dispatcher."
            raise TypeError(msg)
        with self._ref_lock:
            self._ref_count[0] += 1
        return WorkspaceSection(
            session=session_obj,
            mounts=self._mounts,
            allowed_host_roots=self._allowed_host_roots,
            accepts_overrides=self._accepts_overrides,
            key=self._key,
            _temp_dir=self._temp_dir,
            _mount_previews=self._mount_previews,
            _created_at=self._created_at,
            _filesystem=self._filesystem,
            _ref_lock=self._ref_lock,
            _ref_count=self._ref_count,
        )
