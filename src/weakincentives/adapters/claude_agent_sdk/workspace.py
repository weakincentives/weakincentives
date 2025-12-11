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

_PREVIEW_ENTRY_LIMIT = 20


class WorkspaceBudgetExceededError(WinkError):
    """Raised when a workspace mount exceeds its byte budget."""

    pass


class WorkspaceSecurityError(WinkError):
    """Raised when a workspace mount violates security constraints."""

    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


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


@FrozenDataclass()
class ClaudeAgentWorkspace:
    """Workspace state for Claude Agent SDK execution.

    Attributes:
        temp_dir: Path to the temporary directory.
        mount_previews: Summaries of each materialized mount.
        created_at: UTC timestamp when the workspace was created.
    """

    temp_dir: Path
    mount_previews: tuple[HostMountPreview, ...]
    created_at: datetime


def _resolve_mount_path(
    host_path: str,
    allowed_roots: Sequence[Path],
) -> Path:
    """Resolve and validate a host path against allowed roots.

    Args:
        host_path: Path to resolve.
        allowed_roots: Security boundary for host path resolution.

    Returns:
        Resolved absolute path.

    Raises:
        WorkspaceSecurityError: If path is outside allowed roots.
        FileNotFoundError: If path does not exist.
    """
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
            raise WorkspaceSecurityError(
                f"Host path '{host_path}' is outside allowed roots: "
                f"{[str(r) for r in allowed_roots]}"
            )

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
    source: Path,
    target: Path,
    mount: HostMount,
) -> HostMountPreview:
    """Copy files from host to temp directory with filtering.

    Args:
        source: Resolved source path on the host.
        target: Destination path in the temp directory.
        mount: Mount configuration with globs and byte limits.

    Returns:
        Preview of the copied mount.

    Raises:
        WorkspaceBudgetExceededError: If byte budget is exceeded.
    """
    entries: list[str] = []
    bytes_copied = 0

    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        file_bytes = source.stat().st_size

        if mount.max_bytes and file_bytes > mount.max_bytes:
            raise WorkspaceBudgetExceededError(
                f"File exceeds byte budget: {file_bytes} > {mount.max_bytes}"
            )

        shutil.copy2(source, target)
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
                    raise WorkspaceBudgetExceededError(
                        f"Mount exceeds byte budget: "
                        f"{bytes_copied + file_bytes} > {mount.max_bytes}"
                    )

                dest = target / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest)

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


def create_workspace(
    mounts: Sequence[HostMount],
    *,
    allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
    temp_dir_prefix: str = "wink-sdk-",
) -> ClaudeAgentWorkspace:
    """Create a temporary workspace with host files copied in.

    Args:
        mounts: Host paths to copy into the temp directory.
        allowed_host_roots: Security boundary for host path resolution.
            Empty means allow any path.
        temp_dir_prefix: Prefix for the temporary directory name.

    Returns:
        Workspace with temp_dir ready for SDK use.

    Raises:
        WorkspaceSecurityError: If a mount path is outside allowed roots.
        WorkspaceBudgetExceededError: If a mount exceeds its byte budget.
        FileNotFoundError: If a host path does not exist.
    """
    allowed_roots = [Path(r) for r in allowed_host_roots]
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

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return ClaudeAgentWorkspace(
        temp_dir=temp_dir,
        mount_previews=tuple(previews),
        created_at=_utcnow(),
    )


def cleanup_workspace(workspace: ClaudeAgentWorkspace) -> None:
    """Remove temporary directory.

    Args:
        workspace: Workspace to clean up.
    """
    if workspace.temp_dir.exists():
        shutil.rmtree(workspace.temp_dir, ignore_errors=True)
