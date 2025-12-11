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

"""Workspace management for Claude Agent SDK execution."""

from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ...dataclasses import FrozenDataclass
from ...errors import WinkError

__all__ = [
    "ClaudeAgentWorkspace",
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "cleanup_workspace",
    "create_workspace",
]


class WorkspaceBudgetExceededError(WinkError, RuntimeError):
    """Raised when a mount exceeds its byte budget."""


class WorkspaceSecurityError(WinkError, RuntimeError):
    """Raised when host path resolution violates security boundaries."""


@FrozenDataclass()
class HostMount:
    """Configuration for mounting host files into the workspace.

    Attributes:
        host_path: Path on the host filesystem to copy from.
        mount_path: Relative path in temp_dir. Defaults to host_path basename.
        include_glob: Glob patterns for files to include. Empty means all.
        exclude_glob: Glob patterns for files to exclude.
        max_bytes: Maximum bytes to copy. None means unlimited.
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
        host_path: Original host_path from the mount configuration.
        resolved_host: Absolute resolved path on host filesystem.
        mount_path: Relative path within the temp directory.
        entries: Preview of copied entries (up to 20).
        is_directory: Whether the source was a directory.
        bytes_copied: Total bytes copied for this mount.
    """

    host_path: str
    resolved_host: Path
    mount_path: str
    entries: tuple[str, ...]
    is_directory: bool
    bytes_copied: int


@FrozenDataclass()
class ClaudeAgentWorkspace:
    """Workspace state for Claude Agent SDK execution.

    Attributes:
        temp_dir: Path to the temporary directory.
        mount_previews: Summaries of each materialized mount.
        created_at: When the workspace was created.
    """

    temp_dir: Path
    mount_previews: tuple[HostMountPreview, ...]
    created_at: datetime


def create_workspace(
    mounts: Sequence[HostMount],
    *,
    allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
    temp_dir_prefix: str = "wink-sdk-",
) -> ClaudeAgentWorkspace:
    """Create a temporary workspace with host files copied in.

    Args:
        mounts: Host paths to copy into temp directory.
        allowed_host_roots: Security boundary for host path resolution.
            If non-empty, all host_paths must resolve to within one of these roots.
        temp_dir_prefix: Prefix for temporary directory name.

    Returns:
        Workspace with temp_dir ready for SDK use.

    Raises:
        WorkspaceSecurityError: If host path violates security boundaries.
        WorkspaceBudgetExceededError: If a mount exceeds its byte budget.
    """
    allowed_roots = tuple(Path(root).resolve() for root in allowed_host_roots)
    return _materialize_workspace(
        mounts=mounts,
        allowed_roots=allowed_roots,
        temp_dir_prefix=temp_dir_prefix,
    )


def cleanup_workspace(workspace: ClaudeAgentWorkspace) -> None:
    """Remove temporary directory.

    Args:
        workspace: The workspace to clean up.
    """
    if workspace.temp_dir.exists():
        shutil.rmtree(workspace.temp_dir, ignore_errors=True)


def _materialize_workspace(
    mounts: Sequence[HostMount],
    allowed_roots: tuple[Path, ...],
    temp_dir_prefix: str,
) -> ClaudeAgentWorkspace:
    """Create temp directory and copy host files."""
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_prefix))
    previews: list[HostMountPreview] = []

    try:
        for mount in mounts:
            resolved = _resolve_mount_path(mount.host_path, allowed_roots)
            mount_path = mount.mount_path or Path(mount.host_path).name
            target = temp_dir / mount_path

            preview = _copy_mount_to_temp(
                source=resolved,
                target=target,
                mount=mount,
            )
            previews.append(preview)

        return ClaudeAgentWorkspace(
            temp_dir=temp_dir,
            mount_previews=tuple(previews),
            created_at=datetime.now(UTC),
        )
    except Exception:
        # Clean up on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _resolve_mount_path(
    host_path: str,
    allowed_roots: tuple[Path, ...],
) -> Path:
    """Resolve host path and validate against security boundaries."""
    resolved = Path(host_path).resolve()

    if not resolved.exists():
        msg = f"Host path does not exist: {host_path}"
        raise WorkspaceSecurityError(msg)

    if allowed_roots:
        is_within_allowed = any(
            _is_path_within(resolved, root) for root in allowed_roots
        )
        if not is_within_allowed:
            msg = (
                f"Host path '{host_path}' resolves to '{resolved}' "
                f"which is outside allowed roots: {[str(r) for r in allowed_roots]}"
            )
            raise WorkspaceSecurityError(msg)

    return resolved


def _is_path_within(path: Path, root: Path) -> bool:
    """Check if path is within root directory."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    else:
        return True


def _copy_mount_to_temp(
    source: Path,
    target: Path,
    mount: HostMount,
) -> HostMountPreview:
    """Copy files from host to temp directory with filtering."""
    entries: list[str] = []
    bytes_copied = 0

    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        bytes_copied = source.stat().st_size
        entries.append(source.name)
    else:
        bytes_copied, entries = _copy_directory(
            source=source,
            target=target,
            mount=mount,
        )

    return HostMountPreview(
        host_path=mount.host_path,
        resolved_host=source,
        mount_path=target.name,
        entries=tuple(entries[:20]),  # Preview limit
        is_directory=source.is_dir(),
        bytes_copied=bytes_copied,
    )


def _copy_directory(
    source: Path,
    target: Path,
    mount: HostMount,
) -> tuple[int, list[str]]:
    """Copy directory contents with glob filtering."""
    entries: list[str] = []
    bytes_copied = 0

    for root, _dirs, files in source.walk(follow_symlinks=mount.follow_symlinks):
        rel_root = root.relative_to(source)

        for file_name in files:
            rel_path = rel_root / file_name

            if not _matches_globs(
                str(rel_path),
                mount.include_glob,
                mount.exclude_glob,
            ):
                continue

            file_path = root / file_name
            file_bytes = file_path.stat().st_size

            if (
                mount.max_bytes is not None
                and bytes_copied + file_bytes > mount.max_bytes
            ):
                msg = (
                    f"Mount exceeds byte budget: "
                    f"{bytes_copied + file_bytes} > {mount.max_bytes}"
                )
                raise WorkspaceBudgetExceededError(msg)

            dest = target / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)

            bytes_copied += file_bytes
            entries.append(str(rel_path))

    return bytes_copied, entries


def _matches_globs(
    path: str,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> bool:
    """Check if path matches include/exclude glob patterns."""
    # If include_globs is empty, include all files
    if include_globs:
        included = any(fnmatch.fnmatch(path, pattern) for pattern in include_globs)
        if not included:
            return False

    # Check exclusions
    if exclude_globs:
        excluded = any(fnmatch.fnmatch(path, pattern) for pattern in exclude_globs)
        if excluded:
            return False

    return True
