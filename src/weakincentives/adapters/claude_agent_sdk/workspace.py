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
    "ClaudeAgentWorkspaceSection",
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
]

_PREVIEW_ENTRY_LIMIT: Final[int] = 20
_TEMPLATE_PREVIEW_LIMIT: Final[int] = 10


class WorkspaceBudgetExceededError(WinkError):
    """Raised when a workspace mount exceeds its byte budget.

    This error occurs during workspace creation when the total bytes of files
    being copied from a host mount exceeds the `max_bytes` limit specified in
    the HostMount configuration. The workspace creation is rolled back (temp
    directory removed) when this error is raised.

    Example:
        A mount configured with `max_bytes=1_000_000` will raise this error
        if the files to be copied total more than 1MB.
    """

    pass


class WorkspaceSecurityError(WinkError):
    """Raised when a workspace mount violates security constraints.

    This error occurs when a HostMount's `host_path` resolves to a location
    outside the configured `allowed_host_roots`. This prevents accidental or
    malicious access to sensitive system files.

    Example:
        If `allowed_host_roots=[Path("/project")]` and a mount tries to access
        `/etc/passwd`, this error is raised because `/etc` is outside `/project`.

    Note:
        When no `allowed_host_roots` are specified, all paths are allowed.
        Always configure allowed roots in production to enforce security boundaries.
    """

    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


@FrozenDataclass()
class HostMount:
    """Configuration for mounting host files into the Claude Agent SDK workspace.

    Use HostMount to specify which files or directories from the host filesystem
    should be copied into the temporary workspace where the Claude agent operates.
    Files are copied (not linked), so agent modifications don't affect originals.

    Attributes:
        host_path: Absolute or relative path to the host file or directory.
            Relative paths are resolved from the current working directory.
        mount_path: Relative path within the temp directory where files will be
            placed. Defaults to the basename of host_path (e.g., "/foo/bar" -> "bar").
        include_glob: Glob patterns to include (e.g., ("*.py", "*.json")).
            Empty tuple means include all files. Patterns match against relative
            paths from the source directory.
        exclude_glob: Glob patterns to exclude (e.g., ("*.pyc", "__pycache__/*")).
            Exclusions are checked before inclusions.
        max_bytes: Maximum total bytes to copy for this mount. Raises
            WorkspaceBudgetExceededError if exceeded. None means unlimited.
        follow_symlinks: Whether to follow symbolic links when copying.
            Defaults to False for security.

    Example:
        >>> mount = HostMount(
        ...     host_path="/path/to/project",
        ...     mount_path="project",
        ...     include_glob=("*.py", "*.md"),
        ...     exclude_glob=("*_test.py",),
        ...     max_bytes=10_000_000,  # 10MB limit
        ... )
    """

    host_path: str
    mount_path: str | None = None
    include_glob: tuple[str, ...] = ()
    exclude_glob: tuple[str, ...] = ()
    max_bytes: int | None = None
    follow_symlinks: bool = False


@FrozenDataclass()
class HostMountPreview:
    """Summary of a materialized host mount after workspace creation.

    HostMountPreview provides visibility into what was actually copied during
    workspace creation. Use this to verify mounts were resolved correctly and
    to display workspace contents to users or in logs.

    Attributes:
        host_path: Original host path from the HostMount configuration.
        resolved_host: Resolved absolute path on the host after symlink resolution.
        mount_path: Relative path within the temp directory where files were placed.
        entries: Preview of copied file entries (limited to first 20 entries).
            For large mounts, this provides a sample without overwhelming output.
        is_directory: True if the source was a directory, False if a single file.
        bytes_copied: Total bytes copied for this mount. Useful for monitoring
            workspace size and debugging budget issues.

    Note:
        The `entries` field is truncated for large directories. Check the actual
        workspace directory if you need the complete file list.
    """

    host_path: str
    resolved_host: Path
    mount_path: str
    entries: tuple[str, ...]
    is_directory: bool
    bytes_copied: int


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


def _create_workspace(
    mounts: Sequence[HostMount],
    *,
    allowed_host_roots: Sequence[Path],
    temp_dir_prefix: str = "wink-sdk-",
) -> tuple[Path, tuple[HostMountPreview, ...]]:
    """Create a temporary workspace with host files copied in.

    Args:
        mounts: Host paths to copy into the temp directory.
        allowed_host_roots: Security boundary for host path resolution.
        temp_dir_prefix: Prefix for the temporary directory name.

    Returns:
        Tuple of (temp_dir, mount_previews).

    Raises:
        WorkspaceSecurityError: If a mount path is outside allowed roots.
        WorkspaceBudgetExceededError: If a mount exceeds its byte budget.
        FileNotFoundError: If a host path does not exist.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_prefix))
    previews: list[HostMountPreview] = []

    try:
        for mount in mounts:
            resolved = _resolve_mount_path(mount.host_path, list(allowed_host_roots))
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

    return temp_dir, tuple(previews)


@dataclass(slots=True, frozen=True)
class _ClaudeAgentWorkspaceSectionParams:
    """Default params for ClaudeAgentWorkspaceSection (empty placeholder)."""

    pass


def _render_workspace_template(previews: tuple[HostMountPreview, ...]) -> str:
    """Render the workspace section template from mount previews."""
    lines = [
        (
            "Claude Code provides direct access to the workspace via its native tools "
            "(Read, Write, Edit, Glob, Grep, Bash). The workspace has the following "
            "mounted content:"
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

    lines.append(
        "\n\nUse Claude Code's native tools to explore and modify the workspace. "
        "Focus on understanding the project structure before making changes."
    )

    return "\n".join(lines)


class ClaudeAgentWorkspaceSection(MarkdownSection[_ClaudeAgentWorkspaceSectionParams]):
    """Prompt section that manages the Claude Agent SDK workspace filesystem.

    This section creates and manages a temporary directory workspace where the
    Claude agent can read, write, and modify files. Host files are copied into
    the workspace via HostMount configurations, and the agent uses the SDK's
    native tools (Read, Write, Edit, Glob, Grep, Bash) to interact with them.

    The section renders a description of the workspace contents into the prompt,
    helping the agent understand what files are available. It also provides the
    `Filesystem` resource for tools that need filesystem access.

    Attributes:
        temp_dir: Path to the temporary workspace directory.
        mount_previews: Summaries of each materialized host mount.
        created_at: UTC timestamp when the workspace was created.
        filesystem: HostFilesystem instance rooted at the temp directory.

    Lifecycle:
        1. Create the section with HostMount configurations
        2. Workspace is materialized immediately (files copied to temp dir)
        3. Use in a PromptTemplate to provide workspace context to the agent
        4. Call `cleanup()` when done to remove the temp directory

    Example:
        >>> workspace = ClaudeAgentWorkspaceSection(
        ...     session=session,
        ...     mounts=[
        ...         HostMount(host_path="/path/to/code", mount_path="src"),
        ...     ],
        ...     allowed_host_roots=[Path("/path/to")],
        ... )
        >>> try:
        ...     # Use workspace in prompt template
        ...     prompt = MyPrompt(sections=[workspace])
        ... finally:
        ...     workspace.cleanup()  # Always clean up temp directory

    Warning:
        Always call `cleanup()` when finished to avoid leaving temporary
        directories on disk. Consider using try/finally or a context manager.
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
        """Initialize the workspace section and materialize host mounts.

        Creates a temporary directory and copies files from each HostMount
        into it. The workspace is ready for agent use immediately after
        initialization.

        Args:
            session: Session instance for state management and event dispatch.
            mounts: Sequence of HostMount configurations specifying which host
                files to copy into the workspace. Empty sequence creates an
                empty workspace.
            allowed_host_roots: Security boundary paths. If non-empty, all mount
                host_paths must resolve to locations under one of these roots.
                Empty sequence allows all paths (not recommended for production).
            accepts_overrides: If True, allows prompt parameters to override
                section content at render time.
            _temp_dir: Internal parameter for cloning - do not use directly.
            _mount_previews: Internal parameter for cloning - do not use directly.
            _created_at: Internal parameter for cloning - do not use directly.
            _filesystem: Internal parameter for cloning - do not use directly.

        Raises:
            WorkspaceSecurityError: If a mount path is outside allowed_host_roots.
            WorkspaceBudgetExceededError: If a mount exceeds its max_bytes limit.
            FileNotFoundError: If a mount's host_path does not exist.

        Note:
            The workspace is created synchronously during __init__. For large
            mounts, this may take noticeable time. Consider the performance
            implications when mounting large directories.
        """
        self._session = session
        self._mounts = tuple(mounts)
        self._allowed_host_roots = tuple(Path(r) for r in allowed_host_roots)
        self._accepts_overrides = accepts_overrides

        if _temp_dir is not None and _mount_previews is not None:
            # Cloning path - reuse existing workspace state
            self._temp_dir = _temp_dir
            self._mount_previews = _mount_previews
            self._created_at = _created_at or _utcnow()
            # Use provided filesystem or create new one (for backward compatibility)
            self._filesystem: Filesystem = (
                _filesystem
                if _filesystem is not None
                else HostFilesystem(_root=str(self._temp_dir))
            )
        elif mounts:
            # Create workspace from mounts
            self._temp_dir, self._mount_previews = _create_workspace(
                mounts,
                allowed_host_roots=self._allowed_host_roots,
            )
            self._created_at = _utcnow()
            self._filesystem = HostFilesystem(_root=str(self._temp_dir))
        else:
            # Empty workspace
            self._temp_dir = Path(tempfile.mkdtemp(prefix="wink-sdk-"))
            self._mount_previews = ()
            self._created_at = _utcnow()
            self._filesystem = HostFilesystem(_root=str(self._temp_dir))

        template = _render_workspace_template(self._mount_previews)

        super().__init__(
            title="Workspace",
            key="claude-agent-workspace",
            template=template,
            default_params=_ClaudeAgentWorkspaceSectionParams(),
            tools=(),
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """Return the session associated with this workspace section.

        The session manages state and event dispatch for the workspace.
        """
        return self._session

    @property
    def temp_dir(self) -> Path:
        """Return the path to the temporary workspace directory.

        This is the root directory where all mounted files are placed.
        Use this path to access workspace files directly or to configure
        external tools that need the workspace location.

        Returns:
            Absolute Path to the temporary workspace directory.
        """
        return self._temp_dir

    @property
    def mount_previews(self) -> tuple[HostMountPreview, ...]:
        """Return summaries of each materialized mount.

        Provides visibility into what files were copied during workspace
        creation. Useful for debugging, logging, and verifying mount
        configuration.

        Returns:
            Tuple of HostMountPreview objects, one per configured mount.
        """
        return self._mount_previews

    @property
    def created_at(self) -> datetime:
        """Return the UTC timestamp when the workspace was created.

        Returns:
            Timezone-aware datetime in UTC.
        """
        return self._created_at

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem interface for this workspace.

        Provides a HostFilesystem instance rooted at the temporary workspace
        directory. Tools and other components can use this to perform
        file operations within the workspace.

        Returns:
            Filesystem instance (HostFilesystem) for workspace file operations.

        Note:
            This filesystem is contributed to the prompt's ResourceRegistry
            via the `resources()` method, making it available to tools.
        """
        return self._filesystem

    def cleanup(self) -> None:
        """Remove the temporary workspace directory and associated resources.

        Deletes the temp directory and all its contents, and cleans up any
        external resources used by the filesystem (e.g., git snapshot directories).

        This method is idempotent - calling it multiple times is safe.
        Errors during deletion are silently ignored to ensure cleanup completes.

        Warning:
            Always call this method when finished with the workspace to avoid
            leaving temporary files on disk. Use try/finally to ensure cleanup
            happens even if an exception occurs.

        Example:
            >>> workspace = ClaudeAgentWorkspaceSection(session=session, mounts=[...])
            >>> try:
            ...     # Use workspace
            ...     pass
            ... finally:
            ...     workspace.cleanup()
        """
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        # HostFilesystem.cleanup() removes external git directories used for snapshots
        if isinstance(self._filesystem, HostFilesystem):
            self._filesystem.cleanup()

    @override
    def resources(self) -> ResourceRegistry:
        """Return the resource registry for this workspace section.

        Contributes a Filesystem binding to the prompt's resource registry,
        allowing tools to access the workspace filesystem via dependency
        injection.

        Returns:
            ResourceRegistry containing a Filesystem binding to this
            workspace's HostFilesystem instance.

        Note:
            When this section is included in a PromptTemplate, tools can
            request `Filesystem` as a dependency and receive the workspace
            filesystem automatically.
        """
        return ResourceRegistry.build({Filesystem: self._filesystem})

    @override
    def clone(self, **kwargs: Any) -> ClaudeAgentWorkspaceSection:
        """Clone the section with a new session, sharing the same workspace.

        Creates a new ClaudeAgentWorkspaceSection that shares the same underlying
        temporary directory and filesystem. This is used when creating derived
        prompts that need to operate on the same workspace.

        Args:
            **kwargs: Keyword arguments for cloning. Required keys:
                - session: Session instance for the cloned section.
                Optional keys:
                - dispatcher: If provided, must match the session's dispatcher.

        Returns:
            New ClaudeAgentWorkspaceSection sharing the same temp directory,
            mount previews, and filesystem as the original.

        Raises:
            TypeError: If 'session' is not provided or is not a Session instance.
            TypeError: If 'dispatcher' is provided but doesn't match the session's
                dispatcher.

        Note:
            The cloned section shares the same workspace state. Changes made
            through either section affect the same files. Call cleanup() only
            once (on either the original or clone) to avoid double-cleanup.
        """
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone ClaudeAgentWorkspaceSection."
            raise TypeError(msg)
        provided_dispatcher = kwargs.get("dispatcher")
        if (
            provided_dispatcher is not None
            and provided_dispatcher is not session_obj.dispatcher
        ):
            msg = "Provided dispatcher must match the target session's dispatcher."
            raise TypeError(msg)
        return ClaudeAgentWorkspaceSection(
            session=session_obj,
            mounts=self._mounts,
            allowed_host_roots=self._allowed_host_roots,
            accepts_overrides=self._accepts_overrides,
            _temp_dir=self._temp_dir,
            _mount_previews=self._mount_previews,
            _created_at=self._created_at,
            _filesystem=self._filesystem,
        )
