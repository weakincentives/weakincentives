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

"""Host mount declarations and materialization.

Moved verbatim from ``prompt/workspace.py``: :class:`HostMount` declares
what host content a sandbox starts from, :func:`materialize_mounts` copies
it into a fresh temp directory with symlink and byte-budget guards, and
:func:`compute_workspace_fingerprint` derives a deterministic key for
mount-config reuse. ``prompt.workspace`` re-exports these for the
workspace section until it dissolves into ``WorkspaceConfig``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ..dataclasses import FrozenDataclass
from ..errors import WinkError

__all__ = [
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "compute_workspace_fingerprint",
    "materialize_mounts",
]

_PREVIEW_ENTRY_LIMIT: Final[int] = 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspaceBudgetExceededError(WinkError):
    """Raised when a workspace mount exceeds its byte budget."""


class WorkspaceSecurityError(WinkError):
    """Raised when a workspace mount violates security constraints."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _should_copy_mount_file(
    file_path: Path, *, resolved_source: Path, follow_symlinks: bool
) -> bool:
    """Return whether a mount file is allowed to be copied."""
    if not follow_symlinks and file_path.is_symlink():
        return False
    return not follow_symlinks or file_path.resolve().is_relative_to(resolved_source)


def _check_single_file_symlink(source: Path, follow_symlinks: bool) -> None:
    """Validate a single-file symlink mount."""
    if not source.is_symlink():
        return
    if not follow_symlinks:
        msg = f"Symlink mount rejected (follow_symlinks=False): {source}"
        raise WorkspaceSecurityError(msg)
    resolved = source.resolve()
    if not resolved.is_relative_to(source.parent.resolve()):
        msg = f"Symlink escapes parent directory: {source} -> {resolved}"
        raise WorkspaceSecurityError(msg)


def _copy_mount_to_temp(
    source: Path, target: Path, mount: HostMount
) -> HostMountPreview:
    """Copy files from host to temp directory with filtering."""
    entries: list[str] = []
    bytes_copied = 0

    if source.is_file():
        _check_single_file_symlink(source, mount.follow_symlinks)

        _ = target.parent.mkdir(parents=True, exist_ok=True)
        file_bytes = source.stat().st_size

        if mount.max_bytes is not None and file_bytes > mount.max_bytes:
            raise WorkspaceBudgetExceededError(
                f"File exceeds byte budget: {file_bytes} > {mount.max_bytes}"
            )

        _ = shutil.copy2(source, target, follow_symlinks=mount.follow_symlinks)
        bytes_copied = file_bytes
        entries.append(source.name)

    else:
        resolved_source = source.resolve()
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

                if not _should_copy_mount_file(
                    file_path,
                    resolved_source=resolved_source,
                    follow_symlinks=mount.follow_symlinks,
                ):
                    continue

                file_bytes = file_path.stat().st_size

                if (
                    mount.max_bytes is not None
                    and bytes_copied + file_bytes > mount.max_bytes
                ):
                    total = bytes_copied + file_bytes
                    msg = f"Mount exceeds byte budget: {total} > {mount.max_bytes}"
                    raise WorkspaceBudgetExceededError(msg)

                dest = target / rel_path
                _ = dest.parent.mkdir(parents=True, exist_ok=True)
                _ = shutil.copy2(file_path, dest, follow_symlinks=mount.follow_symlinks)

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


def _validate_mount_target(target: Path, temp_dir: Path, mount_path: str) -> None:
    """Raise if mount target escapes the workspace directory."""
    if not target.resolve().is_relative_to(temp_dir.resolve()):
        msg = f"Mount path '{mount_path}' escapes workspace directory"
        raise WorkspaceSecurityError(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def materialize_mounts(
    mounts: Sequence[HostMount],
    *,
    allowed_host_roots: Sequence[Path],
    temp_dir_prefix: str = "wink-workspace-",
) -> tuple[Path, tuple[HostMountPreview, ...]]:
    """Create a temporary directory with host files copied in.

    Every mount is validated against ``allowed_host_roots``, its target is
    confined to the new directory, and symlink/byte-budget guards apply
    during the copy. On any failure the partially-built directory is
    removed before the error propagates.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_prefix))
    previews: list[HostMountPreview] = []

    try:
        for mount in mounts:
            resolved = _resolve_mount_path(mount.host_path, list(allowed_host_roots))
            mount_path = mount.mount_path or Path(mount.host_path).name
            target = temp_dir / mount_path
            _validate_mount_target(target, temp_dir, mount_path)
            preview = _copy_mount_to_temp(source=resolved, target=target, mount=mount)
            previews.append(preview)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return temp_dir, tuple(previews)


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
