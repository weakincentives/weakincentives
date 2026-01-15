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

"""Podman container lifecycle management.

This module handles Podman container creation, teardown, and workspace
management including overlay directories and host mount hydration.

Example usage::

    from weakincentives.contrib.tools.podman_lifecycle import (
        PodmanWorkspace,
        WorkspaceHandle,
        create_container,
    )
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...runtime.logging import StructuredLogger, get_logger
from . import vfs as vfs_module
from .vfs import HostMount, VfsPath

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "tools.podman.lifecycle"}
)

_DEFAULT_IMAGE: Final[str] = "python:3.12-bookworm"
_DEFAULT_WORKDIR: Final[str] = "/workspace"
_DEFAULT_USER: Final[str] = "65534:65534"
_TMPFS_SIZE: Final[int] = 268_435_456
_TMPFS_TARGET: Final[str] = tempfile.gettempdir()
_CPU_PERIOD: Final[int] = 100_000
_CPU_QUOTA: Final[int] = 100_000
_MEMORY_LIMIT: Final[str] = "1g"
_CACHE_ENV: Final[str] = "WEAKINCENTIVES_CACHE"
_MISSING_DEPENDENCY_MESSAGE: Final[str] = (
    "Install weakincentives[podman] to enable the Podman tool suite."
)


@runtime_checkable
class PodmanClient(Protocol):
    """Subset of :class:`podman.PodmanClient` used by the section."""

    containers: Any
    images: Any

    def close(self) -> None: ...


type ClientFactory = Callable[[], PodmanClient]


@FrozenDataclass()
class PodmanWorkspace:
    """Active Podman container backing the session."""

    container_id: str
    container_name: str
    image: str
    workdir: str
    overlay_path: str
    env: tuple[tuple[str, str], ...]
    started_at: datetime
    last_used_at: datetime


@dataclass(slots=True)
class WorkspaceHandle:
    """Internal handle for workspace management."""

    descriptor: PodmanWorkspace
    overlay_path: Path


@FrozenDataclass()
class ResolvedHostMount:
    """Resolved host mount with all paths validated."""

    source_label: str
    resolved_host: Path
    mount_path: VfsPath
    include_glob: tuple[str, ...]
    exclude_glob: tuple[str, ...]
    max_bytes: int | None
    follow_symlinks: bool
    preview: vfs_module.HostMountPreview


def default_cache_root() -> Path:
    """Get the default cache root directory for overlays.

    Returns:
        Path to cache directory, respecting WEAKINCENTIVES_CACHE env var.
    """
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "weakincentives" / "podman"


def build_client_factory(
    *,
    base_url: str | None,
    identity: str | None,
) -> ClientFactory:
    """Build a factory function for creating Podman clients.

    Args:
        base_url: Podman API base URL.
        identity: SSH identity file path.

    Returns:
        Factory function that creates PodmanClient instances.
    """

    def _factory() -> PodmanClient:
        try:
            from podman import PodmanClient as RealPodmanClient
        except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
            raise RuntimeError(_MISSING_DEPENDENCY_MESSAGE) from error

        kwargs: dict[str, str] = {}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if identity is not None:
            kwargs["identity"] = identity
        return RealPodmanClient(**kwargs)

    return _factory


def resolve_podman_host_mounts(
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> tuple[tuple[ResolvedHostMount, ...], tuple[vfs_module.HostMountPreview, ...]]:
    """Resolve and validate host mounts.

    Args:
        mounts: Host mount specifications.
        allowed_roots: Allowed root directories for mounts.

    Returns:
        Tuple of (resolved mounts, mount previews).
    """
    if not mounts:
        return (), ()
    resolved: list[ResolvedHostMount] = []
    previews: list[vfs_module.HostMountPreview] = []
    for mount in mounts:
        spec = resolve_single_host_mount(mount, allowed_roots)
        resolved.append(spec)
        previews.append(spec.preview)
    return tuple(resolved), tuple(previews)


def resolve_single_host_mount(
    mount: HostMount,
    allowed_roots: Sequence[Path],
) -> ResolvedHostMount:
    """Resolve a single host mount specification.

    Args:
        mount: Host mount to resolve.
        allowed_roots: Allowed root directories.

    Returns:
        Resolved mount with validated paths.

    Raises:
        ToolValidationError: If mount path is invalid or outside allowed roots.
    """
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    vfs_module.ensure_ascii(host_path, "host path")
    resolved_host = resolve_host_path(host_path, allowed_roots)
    include_glob = normalize_mount_globs(mount.include_glob, "include_glob")
    exclude_glob = normalize_mount_globs(mount.exclude_glob, "exclude_glob")
    mount_path = (
        vfs_module.normalize_path(mount.mount_path)
        if mount.mount_path is not None
        else VfsPath(())
    )
    preview_entries = preview_mount_entries(resolved_host)
    preview = vfs_module.HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        entries=preview_entries,
        is_directory=resolved_host.is_dir(),
    )
    return ResolvedHostMount(
        source_label=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
        max_bytes=mount.max_bytes,
        follow_symlinks=mount.follow_symlinks,
        preview=preview,
    )


def resolve_host_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
    """Resolve a host path within allowed roots.

    Args:
        host_path: Relative path from allowed root.
        allowed_roots: Allowed root directories.

    Returns:
        Resolved absolute path.

    Raises:
        ToolValidationError: If path is outside allowed roots or missing.
    """
    if not allowed_roots:
        raise ToolValidationError("No allowed host roots configured for mounts.")
    for root in allowed_roots:
        candidate = (root / host_path).expanduser().resolve()
        try:
            _ = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    raise ToolValidationError("Host path is outside the allowed roots or missing.")


def normalize_mount_globs(patterns: Sequence[str], field: str) -> tuple[str, ...]:
    """Normalize glob patterns for mount filtering.

    Args:
        patterns: Glob patterns to normalize.
        field: Field name for error messages.

    Returns:
        Normalized pattern tuple.
    """
    normalized: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        vfs_module.ensure_ascii(stripped, field)
        normalized.append(stripped)
    return tuple(normalized)


def preview_mount_entries(root: Path) -> tuple[str, ...]:
    """Generate preview entries for a mount point.

    Args:
        root: Root path to preview.

    Returns:
        Tuple of entry labels (with / suffix for directories).

    Raises:
        ToolValidationError: If inspection fails.
    """
    if root.is_file():
        return (root.name,)
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError as error:
        raise ToolValidationError(f"Failed to inspect host mount {root}.") from error
    labels: list[str] = []
    for child in children:
        suffix = "/" if child.is_dir() else ""
        labels.append(f"{child.name}{suffix}")
    return tuple(labels)


def iter_host_mount_files(root: Path, follow_symlinks: bool) -> Iterator[Path]:
    """Iterate over files in a mount source.

    Args:
        root: Root path to iterate.
        follow_symlinks: Whether to follow symlinks.

    Yields:
        File paths within the root.
    """
    if root.is_file():
        yield root
        return
    for current, _dirnames, filenames in root.walk(
        follow_symlinks=follow_symlinks,
    ):
        for name in filenames:
            yield current / name


def copy_mount_into_overlay(
    *,
    overlay: Path,
    mount: ResolvedHostMount,
) -> None:
    """Copy mount contents into overlay directory.

    Args:
        overlay: Target overlay directory.
        mount: Resolved mount specification.

    Raises:
        ToolValidationError: If copy fails or exceeds byte budget.
    """
    base_target = host_path_for(overlay, mount.mount_path)
    consumed_bytes = 0
    source = mount.resolved_host
    for file_path in iter_host_mount_files(source, mount.follow_symlinks):
        relative = (
            Path(file_path.name) if source.is_file() else file_path.relative_to(source)
        )
        relative_label = relative.as_posix()
        if mount.include_glob and not any(
            fnmatch.fnmatchcase(relative_label, pattern)
            for pattern in mount.include_glob
        ):
            continue
        if any(
            fnmatch.fnmatchcase(relative_label, pattern)
            for pattern in mount.exclude_glob
        ):
            continue
        target = base_target / relative
        assert_within_overlay(overlay, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = file_path.stat().st_size
        except OSError as error:
            raise ToolValidationError(
                f"Failed to stat mounted file {file_path}."
            ) from error
        if mount.max_bytes is not None and consumed_bytes + size > mount.max_bytes:
            raise ToolValidationError("Host mount exceeded the configured byte budget.")
        consumed_bytes += size
        try:
            _ = shutil.copy2(file_path, target)
        except OSError as error:
            raise ToolValidationError(
                "Failed to materialize host mounts inside the Podman workspace."
            ) from error


def host_path_for(root: Path, path: VfsPath) -> Path:
    """Convert VfsPath to host path within root.

    Args:
        root: Root directory.
        path: VfsPath to convert.

    Returns:
        Host path.
    """
    host = root
    for segment in path.segments:
        host /= segment
    return host


def container_path_for(path: VfsPath, workdir: str = _DEFAULT_WORKDIR) -> str:
    """Convert VfsPath to container path.

    Args:
        path: VfsPath to convert.
        workdir: Container working directory.

    Returns:
        Absolute container path.
    """
    if not path.segments:
        return workdir
    return posixpath.join(workdir, *path.segments)


def assert_within_overlay(root: Path, candidate: Path) -> None:
    """Assert that a path is within the overlay boundary.

    Args:
        root: Overlay root directory.
        candidate: Path to check.

    Raises:
        ToolValidationError: If path escapes the boundary.
    """
    try:
        resolved = candidate.resolve()
    except FileNotFoundError:
        try:
            resolved = candidate.parent.resolve()
        except FileNotFoundError as error:  # pragma: no cover - defensive guard
            raise ToolValidationError("Workspace path is unavailable.") from error
    try:
        _ = resolved.relative_to(root)
    except ValueError as error:
        raise ToolValidationError("Path escapes the workspace boundary.") from error


def normalize_base_env(env: Mapping[str, str]) -> dict[str, str]:
    """Normalize environment variables for container.

    Args:
        env: Environment mapping.

    Returns:
        Normalized environment dictionary.
    """
    from .podman_shell import normalize_env

    return normalize_env(env)


def create_container(  # noqa: PLR0913
    *,
    client: PodmanClient,
    session_id: str,
    image: str,
    overlay: Path,
    base_env: tuple[tuple[str, str], ...],
    clock: Callable[[], datetime],
) -> WorkspaceHandle:
    """Create and start a Podman container.

    Args:
        client: Podman client instance.
        session_id: Session identifier for container naming.
        image: Container image to use.
        overlay: Host overlay directory to mount.
        base_env: Base environment variables.
        clock: Clock function for timestamps.

    Returns:
        WorkspaceHandle for the created container.

    Raises:
        ToolValidationError: If container creation or readiness check fails.
    """
    _LOGGER.info(
        "Creating Podman workspace",
        event="podman.workspace.create",
        context={"overlay": str(overlay), "image": image},
    )
    _ = client.images.pull(image)
    env = normalize_base_env(dict(base_env))
    name = f"wink-{session_id}"
    container = client.containers.create(
        image=image,
        command=["sleep", "infinity"],
        name=name,
        workdir=_DEFAULT_WORKDIR,
        user=_DEFAULT_USER,
        mem_limit=_MEMORY_LIMIT,
        memswap_limit=_MEMORY_LIMIT,
        cpu_period=_CPU_PERIOD,
        cpu_quota=_CPU_QUOTA,
        environment=env,
        network_mode="none",
        mounts=[
            {
                "Target": _TMPFS_TARGET,
                "Type": "tmpfs",
                "TmpfsOptions": {"SizeBytes": _TMPFS_SIZE},
            },
            {
                "Target": _DEFAULT_WORKDIR,
                "Source": str(overlay),
                "Type": "bind",
                "Options": ["rbind", "rw"],
            },
        ],
        remove=True,
    )
    container.start()
    exit_code, _output = container.exec_run(
        ["test", "-d", _DEFAULT_WORKDIR],
        stdout=True,
        stderr=True,
        demux=True,
    )
    if exit_code != 0:
        raise ToolValidationError("Podman workspace failed readiness checks.")
    now = clock().astimezone(UTC)
    descriptor = PodmanWorkspace(
        container_id=container.id,
        container_name=name,
        image=image,
        workdir=_DEFAULT_WORKDIR,
        overlay_path=str(overlay),
        env=tuple(sorted(env.items())),
        started_at=now,
        last_used_at=now,
    )
    return WorkspaceHandle(descriptor=descriptor, overlay_path=overlay)


def teardown_container(
    *,
    client_factory: ClientFactory,
    container_id: str,
) -> None:
    """Stop and remove a Podman container.

    Args:
        client_factory: Factory for creating client.
        container_id: Container ID to teardown.
    """
    try:
        client = client_factory()
    except Exception:
        return
    try:
        try:
            container = client.containers.get(container_id)
        except Exception:
            return
        with suppress(Exception):
            container.stop(timeout=1)
        with suppress(Exception):
            container.remove(force=True)
    finally:
        with suppress(Exception):
            client.close()


__all__ = [
    "ClientFactory",
    "PodmanClient",
    "PodmanWorkspace",
    "ResolvedHostMount",
    "WorkspaceHandle",
    "assert_within_overlay",
    "build_client_factory",
    "container_path_for",
    "copy_mount_into_overlay",
    "create_container",
    "default_cache_root",
    "host_path_for",
    "iter_host_mount_files",
    "normalize_mount_globs",
    "preview_mount_entries",
    "resolve_host_path",
    "resolve_podman_host_mounts",
    "resolve_single_host_mount",
    "teardown_container",
]
