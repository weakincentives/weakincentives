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

"""Workspace management for Codex App Server execution.

Reuses mount/copy machinery from the Claude Agent SDK workspace but renders
a provider-agnostic section description.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, override

from ...dataclasses import FrozenDataclass
from ...errors import WinkError
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import MarkdownSection
from ...resources import ResourceRegistry
from ...runtime.session import Session

__all__ = [
    "CodexWorkspaceSection",
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
]

_PREVIEW_ENTRY_LIMIT: Final[int] = 20
_TEMPLATE_PREVIEW_LIMIT: Final[int] = 10


class WorkspaceBudgetExceededError(WinkError):
    """Raised when a workspace mount exceeds its byte budget."""


class WorkspaceSecurityError(WinkError):
    """Raised when a workspace mount violates security constraints."""


@FrozenDataclass()
class HostMount:
    """Configuration for mounting host files into the workspace.

    Attributes:
        host_path: Absolute or relative path to the host file or directory.
        mount_path: Relative path within the temp directory. Defaults to the
            basename of host_path.
        include_glob: Glob patterns to include (empty = include all).
        exclude_glob: Glob patterns to exclude.
        max_bytes: Maximum total bytes to copy. None means unlimited.
        follow_symlinks: Whether to follow symbolic links when copying.
    """

    host_path: str
    mount_path: str | None = None
    include_glob: tuple[str, ...] = ()
    exclude_glob: tuple[str, ...] = ()
    max_bytes: int | None = None
    follow_symlinks: bool = False


@FrozenDataclass()
class HostMountPreview:
    """Summary of a materialized host mount.

    Attributes:
        host_path: Original host path from mount configuration.
        resolved_host: Resolved absolute path on the host.
        mount_path: Relative path within the temp directory.
        entries: Preview of copied entries (limited to first 20).
        is_directory: Whether the source was a directory.
        bytes_copied: Total bytes copied for this mount.
    """

    host_path: str
    resolved_host: Path
    mount_path: str
    entries: tuple[str, ...]
    is_directory: bool
    bytes_copied: int


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _resolve_mount_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
    """Resolve and validate a host path against allowed roots."""
    resolved = Path(host_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Host path does not exist: {host_path}")

    if allowed_roots:
        for root in allowed_roots:
            try:
                _ = resolved.relative_to(root.resolve())
                break
            except ValueError:
                continue
        else:
            roots_str = str([str(r) for r in allowed_roots])
            msg = f"Host path '{host_path}' is outside allowed roots: {roots_str}"
            raise WorkspaceSecurityError(msg)

    return resolved


def _matches_globs(
    rel_path: str,
    include_glob: tuple[str, ...],
    exclude_glob: tuple[str, ...],
) -> bool:
    """Check if a path matches the include/exclude glob patterns."""
    if exclude_glob:
        for pattern in exclude_glob:
            if fnmatch.fnmatch(rel_path, pattern):
                return False

    if include_glob:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in include_glob)

    return True


def _copy_mount_to_temp(
    source: Path, target: Path, mount: HostMount
) -> HostMountPreview:
    """Copy files from host to temp directory with filtering."""
    entries: list[str] = []
    bytes_copied = 0

    if source.is_file():
        _ = target.parent.mkdir(parents=True, exist_ok=True)
        file_bytes = source.stat().st_size

        if mount.max_bytes and file_bytes > mount.max_bytes:
            raise WorkspaceBudgetExceededError(
                f"File exceeds byte budget: {file_bytes} > {mount.max_bytes}"
            )

        _ = shutil.copy2(source, target)
        bytes_copied = file_bytes
        entries.append(source.name)

    else:
        for root, _dirs, files in os.walk(source, followlinks=mount.follow_symlinks):
            root_path = Path(root)
            rel_root = root_path.relative_to(source)

            for file_name in files:
                rel_path = rel_root / file_name

                if not _matches_globs(
                    str(rel_path), mount.include_glob, mount.exclude_glob
                ):
                    continue

                file_path = root_path / file_name
                file_bytes = file_path.stat().st_size

                if mount.max_bytes and bytes_copied + file_bytes > mount.max_bytes:
                    total = bytes_copied + file_bytes
                    msg = f"Mount exceeds byte budget: {total} > {mount.max_bytes}"
                    raise WorkspaceBudgetExceededError(msg)

                dest = target / rel_path
                _ = dest.parent.mkdir(parents=True, exist_ok=True)
                _ = shutil.copy2(file_path, dest)

                bytes_copied += file_bytes
                entries.append(str(rel_path))

    return HostMountPreview(
        host_path=mount.host_path,
        resolved_host=source,
        mount_path=target.name,
        entries=tuple(entries[:_PREVIEW_ENTRY_LIMIT]),
        is_directory=source.is_dir(),
        bytes_copied=bytes_copied,
    )


def _create_workspace(
    mounts: Sequence[HostMount],
    *,
    allowed_host_roots: Sequence[Path],
    temp_dir_prefix: str = "wink-codex-",
) -> tuple[Path, tuple[HostMountPreview, ...]]:
    """Create a temporary workspace with host files copied in."""
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_prefix))
    previews: list[HostMountPreview] = []

    try:
        for mount in mounts:
            resolved = _resolve_mount_path(mount.host_path, list(allowed_host_roots))
            mount_path = mount.mount_path or Path(mount.host_path).name
            target = temp_dir / mount_path
            preview = _copy_mount_to_temp(source=resolved, target=target, mount=mount)
            previews.append(preview)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return temp_dir, tuple(previews)


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


def compute_workspace_fingerprint(mounts: tuple[HostMount, ...]) -> str:
    """Compute a deterministic fingerprint from mount configurations."""
    config_data = [
        {
            "host_path": m.host_path,
            "mount_path": m.mount_path,
            "include_glob": list(m.include_glob),
            "exclude_glob": list(m.exclude_glob),
            "max_bytes": m.max_bytes,
        }
        for m in mounts
    ]
    json_str = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass(slots=True, frozen=True)
class _CodexWorkspaceSectionParams:
    """Default params for CodexWorkspaceSection (empty placeholder)."""


class CodexWorkspaceSection(MarkdownSection[_CodexWorkspaceSectionParams]):
    """Prompt section describing the Codex App Server workspace.

    Manages a temporary workspace directory with host file mounts.
    Provider-agnostic: does not reference Claude-specific tooling.
    """

    def __init__(
        self,
        *,
        session: Session,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        accepts_overrides: bool = False,
        _temp_dir: Path | None = None,
        _mount_previews: tuple[HostMountPreview, ...] | None = None,
        _created_at: datetime | None = None,
        _filesystem: Filesystem | None = None,
    ) -> None:
        self._session = session
        self._mounts = tuple(mounts)
        self._allowed_host_roots = tuple(Path(r) for r in allowed_host_roots)
        self._accepts_overrides = accepts_overrides

        if _temp_dir is not None and _mount_previews is not None:
            self._temp_dir = _temp_dir
            self._mount_previews = _mount_previews
            self._created_at = _created_at or _utcnow()
            self._filesystem: Filesystem = (
                _filesystem
                if _filesystem is not None
                else HostFilesystem(_root=str(self._temp_dir))
            )
        elif mounts:
            self._temp_dir, self._mount_previews = _create_workspace(
                mounts, allowed_host_roots=self._allowed_host_roots
            )
            self._created_at = _utcnow()
            self._filesystem = HostFilesystem(_root=str(self._temp_dir))
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="wink-codex-"))
            self._mount_previews = ()
            self._created_at = _utcnow()
            self._filesystem = HostFilesystem(_root=str(self._temp_dir))

        template = _render_workspace_template(self._mount_previews)

        super().__init__(
            title="Workspace",
            key="codex-workspace",
            template=template,
            default_params=_CodexWorkspaceSectionParams(),
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
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        if isinstance(self._filesystem, HostFilesystem):  # pragma: no branch
            self._filesystem.cleanup()

    @override
    def resources(self) -> ResourceRegistry:
        """Return resources contributed by this workspace section."""
        return ResourceRegistry.build({Filesystem: self._filesystem})

    @override
    def clone(self, **kwargs: Any) -> CodexWorkspaceSection:
        """Clone the section with a new session."""
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone CodexWorkspaceSection."
            raise TypeError(msg)
        provided_dispatcher = kwargs.get("dispatcher")
        if (
            provided_dispatcher is not None
            and provided_dispatcher is not session_obj.dispatcher
        ):
            msg = "Provided dispatcher must match the target session's dispatcher."
            raise TypeError(msg)
        return CodexWorkspaceSection(
            session=session_obj,
            mounts=self._mounts,
            allowed_host_roots=self._allowed_host_roots,
            accepts_overrides=self._accepts_overrides,
            _temp_dir=self._temp_dir,
            _mount_previews=self._mount_previews,
            _created_at=self._created_at,
            _filesystem=self._filesystem,
        )
