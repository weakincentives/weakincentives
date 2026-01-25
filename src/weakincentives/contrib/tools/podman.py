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

"""Podman-backed sandbox providing isolated shell and filesystem tools.

This module provides :class:`PodmanSandboxSection`, a prompt section that exposes
an isolated Linux container as a tool surface. Agents can execute shell commands,
read and write files, search with glob/grep, and run Python scripts inside the
container without affecting the host system.

Key features:

- **Isolation**: Containers run with no network access, restricted CPU/memory,
  and a non-root user.
- **Filesystem tools**: ``ls``, ``read_file``, ``write_file``, ``edit_file``,
  ``glob``, ``grep``, and ``rm`` operate on ``/workspace``.
- **Shell execution**: ``shell_execute`` runs arbitrary commands with timeout
  enforcement.
- **Python evaluation**: ``evaluate_python`` runs short scripts via ``python3 -c``.
- **Host mounts**: Optionally copy host files into the container at startup.

Example usage::

    from weakincentives.contrib.tools import PodmanSandboxConfig, PodmanSandboxSection
    from weakincentives.runtime.session import Session

    session = Session()
    config = PodmanSandboxConfig(
        image="python:3.12-bookworm",
        mounts=(HostMount(host_path="src"),),
        allowed_host_roots=(Path.cwd(),),
    )
    section = PodmanSandboxSection(session=session, config=config)

    # Add to prompt and use tools...

    section.close()  # Clean up when done

Requirements:

- The ``podman`` CLI must be installed and accessible.
- For the Python API, install ``weakincentives[podman]``.
"""

from __future__ import annotations

import fnmatch
import math
import os
import posixpath
import shutil
import subprocess  # nosec: B404
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast, override, runtime_checkable

from weakincentives.filesystem import Filesystem, HostFilesystem

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.markdown import MarkdownSection
from ...prompt.policy import ReadBeforeWritePolicy
from ...prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session, replace_latest
from . import vfs as vfs_module
from ._context import ensure_context_uses_session
from .asteval import (
    EvalParams,
    EvalResult,
)
from .podman_connection import (
    resolve_connection_settings,
    resolve_podman_connection,
)
from .podman_eval import PodmanEvalSuite
from .vfs import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    FilesystemToolHandlers,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    WriteFile,
    WriteFileParams,
)

_LOGGER: StructuredLogger = get_logger(__name__, context={"component": "tools.podman"})

_DEFAULT_IMAGE: Final[str] = "python:3.12-bookworm"
_DEFAULT_WORKDIR: Final[str] = "/workspace"
_DEFAULT_USER: Final[str] = "65534:65534"
_TMPFS_SIZE: Final[int] = 268_435_456
_TMPFS_TARGET: Final[str] = tempfile.gettempdir()
_MAX_STDIO_CHARS: Final[int] = 32 * 1024
_MAX_COMMAND_LENGTH: Final[int] = 4_096
_MAX_ENV_LENGTH: Final[int] = 512
_MAX_ENV_VARS: Final[int] = 64
_MAX_TIMEOUT: Final[float] = 120.0
_MIN_TIMEOUT: Final[float] = 1.0
_DEFAULT_TIMEOUT: Final[float] = 30.0
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_PATH_SEGMENT: Final[int] = 80
_ASCII: Final[str] = "ascii"
_CAPTURE_DISABLED: Final[str] = "capture disabled"
_CACHE_ENV: Final[str] = "WEAKINCENTIVES_CACHE"
_CPU_PERIOD: Final[int] = 100_000
_CPU_QUOTA: Final[int] = 100_000
_MEMORY_LIMIT: Final[str] = "1g"
_MISSING_DEPENDENCY_MESSAGE: Final[str] = (
    "Install weakincentives[podman] to enable the Podman tool suite."
)
_MAX_MATCH_RESULTS: Final[int] = 2_000
_REMOVE_PATH_SCRIPT: Final[str] = """
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1])
if not target.exists():
    raise SystemExit(3)
if target.is_symlink():
    target.unlink()
elif target.is_dir():
    shutil.rmtree(target)
else:
    target.unlink()
"""
_PODMAN_TEMPLATE: Final[str] = """\
You have access to an isolated Linux container powered by Podman. The `ls`,
`read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
filesystem interface but operate on `/workspace` inside the container. The
`evaluate_python` tool is a thin wrapper around `python3 -c` (≤5 seconds); read
and edit files directly from the workspace. `shell_execute` runs short commands
(≤120 seconds) in the shared environment. No network access or privileged
operations are available. Do not assume files outside `/workspace` exist."""


@runtime_checkable
class _PodmanClient(Protocol):
    """Subset of :class:`podman.PodmanClient` used by the section."""

    containers: Any
    images: Any

    def close(self) -> None: ...


type _ClientFactory = Callable[[], _PodmanClient]


@runtime_checkable
class _ExecRunner(Protocol):
    def __call__(
        self,
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002 - matches subprocess API
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


@FrozenDataclass()
class _ExecConfig:
    command: Sequence[str]
    stdin: str | None = None
    cwd: str | None = None
    environment: Mapping[str, str] | None = None
    timeout: float | None = None
    capture_output: bool = True


@FrozenDataclass()
class _PodmanSectionParams:
    image: str = _DEFAULT_IMAGE
    workspace_root: str = _DEFAULT_WORKDIR


@FrozenDataclass()
class PodmanSandboxConfig:
    """Configuration for :class:`PodmanSandboxSection`.

    This dataclass consolidates all settings needed to create an isolated Podman
    container workspace. Common use cases include mounting host directories,
    setting environment variables, and configuring remote Podman connections.

    Attributes:
        image: Container image to use. Defaults to ``python:3.12-bookworm``.
        mounts: Sequence of :class:`HostMount` specifications describing files
            or directories to copy into the container's ``/workspace``.
        allowed_host_roots: Paths on the host that mounts may reference. Mounts
            with ``host_path`` values outside these roots will be rejected.
        base_url: Podman socket URL (e.g., ``unix:///run/podman/podman.sock``).
            If omitted, the default connection is used.
        identity: SSH key path for remote Podman connections. Ignored for local
            sockets.
        base_environment: Environment variables injected into every command
            executed inside the container.
        cache_dir: Directory for overlay storage. Defaults to
            ``~/.cache/weakincentives/podman``.
        client_factory: Callable returning a Podman client. Primarily for
            testing; leave ``None`` for production use.
        clock: Callable returning the current UTC datetime. Defaults to
            ``datetime.now(UTC)``.
        connection_name: Named Podman connection from ``podman system connection``.
            Takes precedence over ``base_url`` and ``identity``.
        exec_runner: Callable to execute subprocess commands. For testing only.
        accepts_overrides: If ``True``, tools accept runtime parameter overrides.

    Example::

        from weakincentives.contrib.tools import PodmanSandboxConfig, PodmanSandboxSection

        config = PodmanSandboxConfig(
            mounts=(HostMount(host_path="src"),),
            allowed_host_roots=(Path.cwd(),),
            base_environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        section = PodmanSandboxSection(session=session, config=config)
    """

    image: str = _DEFAULT_IMAGE
    mounts: Sequence[HostMount] = ()
    allowed_host_roots: Sequence[os.PathLike[str] | str] = ()
    base_url: str | None = None
    identity: str | os.PathLike[str] | None = None
    base_environment: Mapping[str, str] | None = None
    cache_dir: os.PathLike[str] | str | None = None
    client_factory: _ClientFactory | None = None
    clock: Callable[[], datetime] | None = None
    connection_name: str | None = None
    exec_runner: _ExecRunner | None = None
    accepts_overrides: bool = False


@FrozenDataclass()
class PodmanShellParams:
    """Parameter payload accepted by the ``shell_execute`` tool.

    Use this dataclass to invoke shell commands inside the Podman container.
    Commands run in an isolated environment with no network access.

    Attributes:
        command: Tuple of command arguments. The first element is the executable;
            subsequent elements are arguments. Example: ``("ls", "-la", "/workspace")``.
            Total length must not exceed 4,096 characters.
        cwd: Working directory relative to ``/workspace``. Must not start with
            ``/`` or contain ``.`` or ``..`` segments. Defaults to ``/workspace``.
        env: Environment variables merged with the container's base environment.
            Keys are uppercased automatically. Maximum 64 variables; each value
            limited to 512 characters.
        stdin: Text piped to the command's standard input. Must be ASCII.
        timeout_seconds: Maximum execution time in seconds. Clamped to the range
            [1.0, 120.0]. Defaults to 30 seconds.
        capture_output: If ``True`` (default), stdout and stderr are captured and
            returned. Set to ``False`` for commands with large output.
    """

    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    stdin: str | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT
    capture_output: bool = True


@FrozenDataclass()
class PodmanShellResult:
    """Structured result returned by the ``shell_execute`` tool.

    Instances are immutable and capture the complete outcome of a command,
    including output streams, timing, and exit status.

    Attributes:
        command: The normalized command tuple that was executed.
        cwd: Absolute path of the working directory inside the container.
        exit_code: Process exit code. Zero indicates success; 124 indicates
            the command was terminated due to timeout.
        stdout: Captured standard output (truncated to 32 KB). Contains
            ``"capture disabled"`` if ``capture_output`` was ``False``.
        stderr: Captured standard error (truncated to 32 KB). Contains
            ``"capture disabled"`` if ``capture_output`` was ``False``.
        duration_ms: Wall-clock execution time in milliseconds.
        timed_out: ``True`` if the command exceeded ``timeout_seconds``.
    """

    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

    def render(self) -> str:
        """Format the result as a human-readable multi-line string.

        Returns:
            A string containing the command, exit code, timing, and output
            streams, suitable for display or logging.
        """
        command_str = " ".join(self.command)
        lines = [
            "Shell command result:",
            f"Command: {command_str}",
            f"CWD: {self.cwd}",
            f"Exit code: {self.exit_code}",
            f"Timed out: {self.timed_out}",
            f"Duration: {self.duration_ms} ms",
            "STDOUT:",
            self.stdout or "<empty>",
            "STDERR:",
            self.stderr or "<empty>",
        ]
        return "\n".join(lines)


@FrozenDataclass()
class PodmanWorkspace:
    """Descriptor for an active Podman container backing a session.

    This immutable dataclass is stored in the session slice and updated
    whenever the container is used. It provides metadata for introspection
    and debugging but should not be modified directly.

    Attributes:
        container_id: Podman container ID (full SHA256 hash).
        container_name: Human-readable container name (e.g., ``wink-<session_id>``).
        image: Container image used to create the workspace.
        workdir: Default working directory inside the container (``/workspace``).
        overlay_path: Host filesystem path where the overlay is mounted.
        env: Tuple of ``(key, value)`` pairs representing environment variables.
        started_at: UTC timestamp when the container was started.
        last_used_at: UTC timestamp of the most recent command execution.
    """

    container_id: str
    container_name: str
    image: str
    workdir: str
    overlay_path: str
    env: tuple[tuple[str, str], ...]
    started_at: datetime
    last_used_at: datetime


@dataclass(slots=True)
class _WorkspaceHandle:
    descriptor: PodmanWorkspace
    overlay_path: Path


@FrozenDataclass()
class _ResolvedHostMount:
    source_label: str
    resolved_host: Path
    mount_path: VfsPath
    include_glob: tuple[str, ...]
    exclude_glob: tuple[str, ...]
    max_bytes: int | None
    follow_symlinks: bool
    preview: vfs_module.HostMountPreview


def _default_cache_root() -> Path:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "weakincentives" / "podman"


def _resolve_podman_host_mounts(
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> tuple[tuple[_ResolvedHostMount, ...], tuple[vfs_module.HostMountPreview, ...]]:
    if not mounts:
        return (), ()
    resolved: list[_ResolvedHostMount] = []
    previews: list[vfs_module.HostMountPreview] = []
    for mount in mounts:
        spec = _resolve_single_host_mount(mount, allowed_roots)
        resolved.append(spec)
        previews.append(spec.preview)
    return tuple(resolved), tuple(previews)


def _resolve_single_host_mount(
    mount: HostMount,
    allowed_roots: Sequence[Path],
) -> _ResolvedHostMount:
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    vfs_module.ensure_ascii(host_path, "host path")
    resolved_host = _resolve_host_path(host_path, allowed_roots)
    include_glob = _normalize_mount_globs(mount.include_glob, "include_glob")
    exclude_glob = _normalize_mount_globs(mount.exclude_glob, "exclude_glob")
    mount_path = (
        vfs_module.normalize_path(mount.mount_path)
        if mount.mount_path is not None
        else VfsPath(())
    )
    preview_entries = _preview_mount_entries(resolved_host)
    preview = vfs_module.HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        entries=preview_entries,
        is_directory=resolved_host.is_dir(),
    )
    return _ResolvedHostMount(
        source_label=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
        max_bytes=mount.max_bytes,
        follow_symlinks=mount.follow_symlinks,
        preview=preview,
    )


def _resolve_host_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
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


def _normalize_mount_globs(patterns: Sequence[str], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        vfs_module.ensure_ascii(stripped, field)
        normalized.append(stripped)
    return tuple(normalized)


def _preview_mount_entries(root: Path) -> tuple[str, ...]:
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


def _iter_host_mount_files(root: Path, follow_symlinks: bool) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for current, _dirnames, filenames in root.walk(
        follow_symlinks=follow_symlinks,
    ):
        for name in filenames:
            yield current / name


def _default_exec_runner(
    cmd: list[str],
    *,
    input: str | None = None,  # noqa: A002 - matches subprocess API
    text: bool | None = None,
    capture_output: bool | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(  # nosec: B603
        cmd,
        input=input,
        text=True if text is None else text,
        capture_output=True if capture_output is None else capture_output,
        timeout=timeout,
    )
    return cast(subprocess.CompletedProcess[str], completed)


def _build_client_factory(
    *,
    base_url: str | None,
    identity: str | None,
) -> _ClientFactory:
    def _factory() -> _PodmanClient:
        try:
            from podman import PodmanClient
        except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
            raise RuntimeError(_MISSING_DEPENDENCY_MESSAGE) from error

        kwargs: dict[str, str] = {}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if identity is not None:
            kwargs["identity"] = identity
        return PodmanClient(**kwargs)

    return _factory


def _ensure_ascii(value: str, *, field: str) -> str:
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:
        raise ToolValidationError(f"{field} must be ASCII.") from error
    return value


def _normalize_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if not command:
        raise ToolValidationError("command must contain at least one entry.")
    total_length = 0
    normalized: list[str] = []
    for index, entry in enumerate(command):
        if not entry:
            raise ToolValidationError(f"command[{index}] must not be empty.")
        normalized_entry = _ensure_ascii(entry, field="command")
        total_length += len(normalized_entry)
        if total_length > _MAX_COMMAND_LENGTH:
            raise ToolValidationError("command is too long (limit 4,096 characters).")
        normalized.append(normalized_entry)
    return tuple(normalized)


def _normalize_env(env: Mapping[str, str]) -> dict[str, str]:
    if len(env) > _MAX_ENV_VARS:
        raise ToolValidationError("env contains too many entries (max 64).")
    normalized: dict[str, str] = {}
    for key, value in env.items():
        normalized_key = _ensure_ascii(key, field="env key").upper()
        if not normalized_key:
            raise ToolValidationError("env keys must not be empty.")
        if len(normalized_key) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"env key {normalized_key!r} is longer than {_MAX_PATH_SEGMENT} characters."
            )
        normalized_value = _ensure_ascii(value, field="env value")
        if len(normalized_value) > _MAX_ENV_LENGTH:
            raise ToolValidationError(
                f"env value for {normalized_key!r} exceeds {_MAX_ENV_LENGTH} characters."
            )
        normalized[normalized_key] = normalized_value
    return normalized


def _normalize_timeout(timeout_seconds: float) -> float:
    if math.isnan(timeout_seconds):
        raise ToolValidationError("timeout_seconds must be a real number.")
    return max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, timeout_seconds))


def _normalize_cwd(path: str | None) -> str:
    if path is None or path == "":
        return _DEFAULT_WORKDIR
    stripped = path.strip()
    if stripped.startswith("/"):
        raise ToolValidationError("cwd must be relative to /workspace.")
    parts = [segment for segment in stripped.split("/") if segment]
    if len(parts) > _MAX_PATH_DEPTH:
        raise ToolValidationError("cwd exceeds maximum depth of 16 segments.")
    normalized_segments: list[str] = []
    for segment in parts:
        if segment in {".", ".."}:
            raise ToolValidationError("cwd must not contain '.' or '..' segments.")
        if len(segment) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"cwd segment {segment!r} exceeds {_MAX_PATH_SEGMENT} characters."
            )
        normalized_segment = _ensure_ascii(segment, field="cwd")
        normalized_segments.append(normalized_segment)
    if not normalized_segments:
        return _DEFAULT_WORKDIR
    return posixpath.join(_DEFAULT_WORKDIR, *normalized_segments)


def _truncate_stream(value: str) -> str:
    if len(value) <= _MAX_STDIO_CHARS:
        return value
    truncated = value[: _MAX_STDIO_CHARS - len("[truncated]")]
    return f"{truncated}[truncated]"


class PodmanSandboxSection(MarkdownSection[_PodmanSectionParams]):
    """Prompt section exposing an isolated Podman container as a tool surface.

    This section provides filesystem tools (``ls``, ``read_file``, ``write_file``,
    ``edit_file``, ``glob``, ``grep``, ``rm``), a ``shell_execute`` tool for
    running arbitrary commands, and an ``evaluate_python`` tool for quick
    Python expressions. All operations execute inside a sandboxed container
    with no network access and strict resource limits.

    The container is started lazily on first tool invocation and persists for
    the lifetime of the section. Files are stored in a host overlay directory,
    allowing inspection and persistence across restarts.

    Typical usage::

        from weakincentives.contrib.tools import PodmanSandboxConfig, PodmanSandboxSection

        config = PodmanSandboxConfig(
            mounts=(HostMount(host_path="src"),),
            allowed_host_roots=(Path.cwd(),),
        )
        section = PodmanSandboxSection(session=session, config=config)
        prompt.add_section(section)

        # Later, clean up the container
        section.close()

    Notes:
        - The ``podman`` CLI must be installed and accessible.
        - For remote Podman hosts, configure ``connection_name`` or ``base_url``.
        - Call :meth:`close` to stop the container when finished.

    Args:
        session: The session instance for slice storage and event dispatch.
        config: Optional configuration; defaults are used if omitted.
    """

    def __init__(
        self,
        *,
        session: Session,
        config: PodmanSandboxConfig | None = None,
        _overlay_path: Path | None = None,
        _filesystem: Filesystem | None = None,
    ) -> None:
        config = config or PodmanSandboxConfig()
        self._session = session
        self._image = config.image
        self._mounts = tuple(config.mounts)
        base_url, identity_str, connection_name = resolve_connection_settings(
            base_url=config.base_url,
            identity=config.identity,
            connection_name=config.connection_name,
        )
        self._client_factory = config.client_factory or _build_client_factory(
            base_url=base_url,
            identity=identity_str,
        )
        self._base_url = base_url
        self._identity = identity_str
        self._base_env = tuple(
            sorted((config.base_environment or {}).items(), key=lambda item: item[0])
        )
        self._overlay_root = (
            Path(config.cache_dir).expanduser()
            if config.cache_dir is not None
            else _default_cache_root()
        )
        self._overlay_root.mkdir(parents=True, exist_ok=True)
        allowed_roots = tuple(
            vfs_module.normalize_host_root(path) for path in config.allowed_host_roots
        )
        self._allowed_roots = allowed_roots
        (
            self._resolved_mounts,
            self._mount_previews,
        ) = _resolve_podman_host_mounts(self._mounts, self._allowed_roots)
        # Create overlay directory eagerly so filesystem is available immediately
        # Use provided overlay path (for cloning) or create new one based on session_id
        if _overlay_path is not None and _filesystem is not None:
            # Cloning path - reuse existing overlay and filesystem
            self._overlay_path = _overlay_path
            self._filesystem: Filesystem = _filesystem
        else:
            # Fresh initialization - create overlay and hydrate mounts eagerly
            # so filesystem operations work before a container is started
            self._overlay_path = self._overlay_root / str(self._session.session_id)
            self._overlay_path.mkdir(parents=True, exist_ok=True)
            for mount in self._resolved_mounts:
                self._copy_mount_into_overlay(overlay=self._overlay_path, mount=mount)
            self._filesystem = HostFilesystem(_root=str(self._overlay_path))
        self._clock = config.clock or (lambda: datetime.now(UTC))
        self._workspace_handle: _WorkspaceHandle | None = None
        self._lock = threading.RLock()
        self._connection_name = connection_name
        self._exec_runner: _ExecRunner = config.exec_runner or _default_exec_runner
        self._finalizer = weakref.finalize(
            self,  # ty: ignore[invalid-argument-type]  # ty: Self vs concrete type
            PodmanSandboxSection._cleanup_from_finalizer,
            weakref.ref(self),
        )
        self._config = PodmanSandboxConfig(
            image=self._image,
            mounts=self._mounts,
            allowed_host_roots=self._allowed_roots,
            base_url=self._base_url,
            identity=self._identity,
            base_environment=dict(self._base_env),
            cache_dir=self._overlay_root,
            client_factory=self._client_factory,
            clock=self._clock,
            connection_name=self._connection_name,
            exec_runner=self._exec_runner,
            accepts_overrides=config.accepts_overrides,
        )

        session[PodmanWorkspace].register(PodmanWorkspace, replace_latest)

        # Use /workspace as the mount point so paths like /workspace/file.txt
        # are correctly interpreted as file.txt in the overlay directory
        self._fs_handlers = FilesystemToolHandlers(
            clock=self._clock, mount_point=_DEFAULT_WORKDIR
        )
        self._shell_suite = _PodmanShellSuite(section=self)
        self._eval_suite = PodmanEvalSuite(section=self)
        accepts_overrides = config.accepts_overrides
        tools = (
            Tool[ListDirectoryParams, tuple[FileInfo, ...]](
                name="ls",
                description="List directory entries under a relative path.",
                handler=self._fs_handlers.list_directory,
                examples=(
                    ToolExample[ListDirectoryParams, tuple[FileInfo, ...]](
                        description="List the workspace root",
                        input=ListDirectoryParams(path="/workspace"),
                        output=(
                            FileInfo(
                                path=VfsPath(("workspace", "README.md")),
                                kind="file",
                                size_bytes=4_096,
                                version=1,
                                updated_at=datetime(2024, 1, 1, tzinfo=UTC),
                            ),
                            FileInfo(
                                path=VfsPath(("workspace", "src")),
                                kind="directory",
                                size_bytes=None,
                                version=None,
                                updated_at=None,
                            ),
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadFileParams, ReadFileResult](
                name="read_file",
                description="Read UTF-8 file contents with pagination support.",
                handler=self._fs_handlers.read_file,
                examples=(
                    ToolExample[ReadFileParams, ReadFileResult](
                        description="Read the top of README.md",
                        input=ReadFileParams(
                            file_path="/workspace/README.md", offset=0, limit=3
                        ),
                        output=ReadFileResult(
                            path=VfsPath(("workspace", "README.md")),
                            content=(
                                "   1 | # weakincentives\n"
                                "   2 | Open source automation harness\n"
                                "   3 | for safe agents"
                            ),
                            offset=0,
                            limit=3,
                            total_lines=120,
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[WriteFileParams, WriteFile](
                name="write_file",
                description="Create a new UTF-8 text file.",
                handler=self._fs_handlers.write_file,
                examples=(
                    ToolExample[WriteFileParams, WriteFile](
                        description="Create a notes file in the container",
                        input=WriteFileParams(
                            file_path="/workspace/notes.txt",
                            content="Remember to run make check",
                        ),
                        output=WriteFile(
                            path=VfsPath(("workspace", "notes.txt")),
                            content="Remember to run make check",
                            mode="create",
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[EditFileParams, WriteFile](
                name="edit_file",
                description="Replace occurrences of a string within a file.",
                handler=self._fs_handlers.edit_file,
                examples=(
                    ToolExample[EditFileParams, WriteFile](
                        description="Update a TODO entry",
                        input=EditFileParams(
                            file_path="/workspace/notes.txt",
                            old_string="TODO: add tests",
                            new_string="TODO: add integration tests",
                            replace_all=False,
                        ),
                        output=WriteFile(
                            path=VfsPath(("workspace", "notes.txt")),
                            content="Completed: scaffold\nTODO: add integration tests",
                            mode="overwrite",
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[GlobParams, tuple[GlobMatch, ...]](
                name="glob",
                description="Match files beneath a directory using shell patterns.",
                handler=self._fs_handlers.glob,
                examples=(
                    ToolExample[GlobParams, tuple[GlobMatch, ...]](
                        description="Find Python files under src",
                        input=GlobParams(pattern="**/*.py", path="/workspace/src"),
                        output=(
                            GlobMatch(
                                path=VfsPath(("workspace", "src", "__init__.py")),
                                size_bytes=128,
                                version=1,
                                updated_at=datetime(2024, 1, 1, tzinfo=UTC),
                            ),
                            GlobMatch(
                                path=VfsPath(
                                    (
                                        "workspace",
                                        "src",
                                        "weakincentives",
                                        "__init__.py",
                                    )
                                ),
                                size_bytes=256,
                                version=2,
                                updated_at=datetime(2024, 1, 2, tzinfo=UTC),
                            ),
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[GrepParams, tuple[GrepMatch, ...]](
                name="grep",
                description="Search files for a regular expression pattern.",
                handler=self._fs_handlers.grep,
                examples=(
                    ToolExample[GrepParams, tuple[GrepMatch, ...]](
                        description="Search for TODO comments",
                        input=GrepParams(
                            pattern="TODO", path="/workspace/src", glob="**/*.py"
                        ),
                        output=(
                            GrepMatch(
                                path=VfsPath(
                                    (
                                        "workspace",
                                        "src",
                                        "weakincentives",
                                        "tools",
                                        "podman.py",
                                    )
                                ),
                                line_number=42,
                                line="# TODO: improve sandbox docs",
                            ),
                            GrepMatch(
                                path=VfsPath(
                                    (
                                        "workspace",
                                        "src",
                                        "weakincentives",
                                        "runtime",
                                        "__init__.py",
                                    )
                                ),
                                line_number=10,
                                line="TODO: replace placeholder logger",
                            ),
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[RemoveParams, DeleteEntry](
                name="rm",
                description="Remove files or directories recursively.",
                handler=self._fs_handlers.remove,
                examples=(
                    ToolExample[RemoveParams, DeleteEntry](
                        description="Delete a stale build artifact",
                        input=RemoveParams(path="/workspace/build/output"),
                        output=DeleteEntry(
                            path=VfsPath(("workspace", "build", "output"))
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[PodmanShellParams, PodmanShellResult](
                name="shell_execute",
                description="Run a short command inside the Podman workspace.",
                handler=self._shell_suite.run_shell,
                examples=(
                    ToolExample[PodmanShellParams, PodmanShellResult](
                        description="Check the current working directory",
                        input=PodmanShellParams(command=("pwd",), cwd=None),
                        output=PodmanShellResult(
                            command=("pwd",),
                            cwd="/workspace",
                            exit_code=0,
                            stdout="/workspace",
                            stderr="",
                            duration_ms=12,
                            timed_out=False,
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[EvalParams, EvalResult](
                name="evaluate_python",
                description=(
                    "Run a short Python script via `python3 -c` inside the Podman workspace. "
                    "Captures stdout/stderr and reports the exit code."
                ),
                handler=self._eval_suite.evaluate_python,
                examples=(
                    ToolExample[EvalParams, EvalResult](
                        description="Run a small calculation",
                        input=EvalParams(code="print(3 * 7)"),
                        output=EvalResult(
                            value_repr=None,
                            stdout="21\n",
                            stderr="",
                            globals={},
                            reads=(),
                            writes=(),
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
        )
        template = _PODMAN_TEMPLATE
        mounts_block = vfs_module.render_host_mounts_block(self._mount_previews)
        if mounts_block:
            template = f"{_PODMAN_TEMPLATE}\n\n{mounts_block}"

        # Default policy: must read file before overwriting
        # Pass mount_point so policy normalizes paths like handlers do
        default_policies = (ReadBeforeWritePolicy(mount_point=_DEFAULT_WORKDIR),)

        super().__init__(
            title="Podman Workspace",
            key="podman.shell",
            template=template,
            default_params=_PodmanSectionParams(
                image=self._image, workspace_root=_DEFAULT_WORKDIR
            ),
            tools=tools,
            policies=default_policies,
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """Return the session associated with this sandbox section."""
        return self._session

    @override
    def clone(self, **kwargs: object) -> PodmanSandboxSection:  # pragma: no cover
        """Create a copy of this section bound to a different session.

        The cloned section shares the same overlay directory and filesystem,
        allowing multiple sessions to operate on the same workspace.

        Args:
            **kwargs: Must include ``session`` (a :class:`Session` instance).
                An optional ``dispatcher`` may be provided but must match the
                session's dispatcher.

        Returns:
            A new :class:`PodmanSandboxSection` using the provided session.

        Raises:
            TypeError: If ``session`` is missing or ``dispatcher`` does not
                match the session's dispatcher.
        """
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone PodmanSandboxSection."
            raise TypeError(msg)
        provided_dispatcher = kwargs.get("dispatcher")
        if (
            provided_dispatcher is not None
            and provided_dispatcher is not session.dispatcher
        ):
            msg = "Provided dispatcher must match the target session's dispatcher."
            raise TypeError(msg)
        return PodmanSandboxSection(
            session=session,
            config=self._config,
            _overlay_path=self._overlay_path,
            _filesystem=self._filesystem,
        )

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this section."""
        return self._filesystem

    @staticmethod
    def resolve_connection(
        connection_name: str | None = None,
    ) -> dict[str, str | None] | None:
        """Resolve Podman connection settings from the system configuration.

        Queries ``podman system connection list`` to find connection details.
        Useful for validating configuration before creating a sandbox section.

        Args:
            connection_name: Specific connection name to look up. If ``None``,
                the default connection is returned.

        Returns:
            A dictionary with keys ``base_url``, ``identity``, and
            ``connection_name``, or ``None`` if no connection is found.

        Example::

            conn = PodmanSandboxSection.resolve_connection("my-remote")
            if conn:
                config = PodmanSandboxConfig(
                    base_url=conn["base_url"],
                    identity=conn["identity"],
                )
        """
        resolved = resolve_podman_connection(preferred_name=connection_name)
        if resolved is None:
            return None
        return {
            "base_url": resolved.base_url,
            "identity": resolved.identity,
            "connection_name": resolved.connection_name,
        }

    def _ensure_workspace(self) -> _WorkspaceHandle:
        with self._lock:
            if self._workspace_handle is not None:
                return self._workspace_handle
            handle = self._create_workspace()
            self._workspace_handle = handle
            self._session[PodmanWorkspace].seed(handle.descriptor)
            return handle

    def _create_workspace(self) -> _WorkspaceHandle:
        client = self._client_factory()
        overlay = self._workspace_overlay_path()
        overlay.mkdir(parents=True, exist_ok=True)
        self._hydrate_overlay_mounts(overlay)
        _LOGGER.info(
            "Creating Podman workspace",
            event="podman.workspace.create",
            context={"overlay": str(overlay), "image": self._image},
        )
        _ = client.images.pull(self._image)
        env = _normalize_env(dict(self._base_env))
        name = f"wink-{self._session.session_id}"
        container = client.containers.create(
            image=self._image,
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
        now = self._clock().astimezone(UTC)
        descriptor = PodmanWorkspace(
            container_id=container.id,
            container_name=name,
            image=self._image,
            workdir=_DEFAULT_WORKDIR,
            overlay_path=str(overlay),
            env=tuple(sorted(env.items())),
            started_at=now,
            last_used_at=now,
        )
        return _WorkspaceHandle(descriptor=descriptor, overlay_path=overlay)

    def _workspace_overlay_path(self) -> Path:
        return self._overlay_path

    def _hydrate_overlay_mounts(self, overlay: Path) -> None:
        """Hydrate mounts into the overlay if it's empty.

        Note: Mounts are now hydrated eagerly during __init__, so this method
        is effectively a no-op in normal operation. It exists as a safety net
        in case the overlay was somehow cleared between init and container
        creation.
        """
        if not self._resolved_mounts:
            return
        iterator = overlay.iterdir()
        try:
            _ = next(iterator)
        except StopIteration:  # pragma: no cover
            pass  # pragma: no cover
        else:
            return
        for mount in self._resolved_mounts:  # pragma: no cover
            self._copy_mount_into_overlay(
                overlay=overlay, mount=mount
            )  # pragma: no cover

    def _workspace_env(self) -> dict[str, str]:
        return (
            dict(self._workspace_handle.descriptor.env)
            if self._workspace_handle
            else dict(self._base_env)
        )

    @staticmethod
    def _copy_mount_into_overlay(
        *,
        overlay: Path,
        mount: _ResolvedHostMount,
    ) -> None:
        base_target = _host_path_for(overlay, mount.mount_path)
        consumed_bytes = 0
        source = mount.resolved_host
        for file_path in _iter_host_mount_files(source, mount.follow_symlinks):
            relative = (
                Path(file_path.name)
                if source.is_file()
                else file_path.relative_to(source)
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
            _assert_within_overlay(overlay, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = file_path.stat().st_size
            except OSError as error:
                raise ToolValidationError(
                    f"Failed to stat mounted file {file_path}."
                ) from error
            if mount.max_bytes is not None and consumed_bytes + size > mount.max_bytes:
                raise ToolValidationError(
                    "Host mount exceeded the configured byte budget."
                )
            consumed_bytes += size
            try:
                _ = shutil.copy2(file_path, target)
            except OSError as error:
                raise ToolValidationError(
                    "Failed to materialize host mounts inside the Podman workspace."
                ) from error

    def _touch_workspace(self) -> None:
        with self._lock:
            handle = self._workspace_handle
            if handle is None:
                return
            now = self._clock().astimezone(UTC)
            updated_descriptor = replace(handle.descriptor, last_used_at=now)
            self._workspace_handle = _WorkspaceHandle(
                descriptor=updated_descriptor,
                overlay_path=handle.overlay_path,
            )
            self._session[PodmanWorkspace].seed(updated_descriptor)

    def _teardown_workspace(self) -> None:
        with self._lock:
            handle = self._workspace_handle
            self._workspace_handle = None
        if handle is None:
            return
        try:
            client = self._client_factory()
        except Exception:
            return
        try:
            try:
                container = client.containers.get(handle.descriptor.container_id)
            except Exception:
                return
            with suppress(Exception):
                container.stop(timeout=1)
            with suppress(Exception):
                container.remove(force=True)
        finally:
            with suppress(Exception):
                client.close()

    def ensure_workspace(self) -> _WorkspaceHandle:
        """Start the container if not already running and return its handle.

        This method is idempotent: calling it multiple times returns the same
        workspace. The container is created with resource limits (1 GB memory,
        1 CPU) and no network access.

        Returns:
            An internal handle containing the workspace descriptor and overlay
            path. Most callers should use the tool methods rather than this
            handle directly.

        Raises:
            ToolValidationError: If the container fails to start or pass
                readiness checks.
        """
        return self._ensure_workspace()

    def workspace_environment(self) -> dict[str, str]:
        """Return the current environment variables for the workspace.

        If the workspace is running, returns the container's environment.
        Otherwise, returns the base environment from configuration.

        Returns:
            A mutable dictionary of environment variable key-value pairs.
        """
        return self._workspace_env()

    def touch_workspace(self) -> None:
        """Update the workspace's ``last_used_at`` timestamp to now.

        Call this after executing commands to keep the workspace metadata
        current. This is done automatically by the shell and eval tools.
        """
        self._touch_workspace()

    def run_cli_exec(self, *, config: _ExecConfig) -> subprocess.CompletedProcess[str]:
        """Execute a command inside the container using the Podman CLI.

        This low-level method builds a ``podman exec`` command and runs it via
        subprocess. It handles stdin, environment variables, and timeouts.

        Args:
            config: Execution configuration specifying the command, stdin,
                working directory, environment overrides, timeout, and whether
                to capture output.

        Returns:
            A :class:`subprocess.CompletedProcess` with return code and
            captured stdout/stderr (if capture was enabled).

        Raises:
            subprocess.TimeoutExpired: If the command exceeds the timeout.
            FileNotFoundError: If the ``podman`` CLI is not installed.
        """
        handle = self.ensure_workspace()
        env = self.workspace_environment()
        if config.environment:
            env.update(config.environment)

        exec_cmd: list[str] = ["podman"]
        if self._connection_name:
            exec_cmd.extend(["--connection", self._connection_name])
        exec_cmd.append("exec")
        if config.stdin is not None:
            exec_cmd.append("--interactive")
        exec_cmd.extend(["--workdir", config.cwd or _DEFAULT_WORKDIR])
        for key, value in env.items():
            exec_cmd.extend(["--env", f"{key}={value}"])
        exec_cmd.append(handle.descriptor.container_name)
        exec_cmd.extend(config.command)
        runner = self._exec_runner
        return runner(
            exec_cmd,
            input=config.stdin,
            text=True,
            capture_output=config.capture_output,
            timeout=config.timeout,
        )

    def run_cli_cp(
        self,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Copy files between the host and container using ``podman cp``.

        Args:
            source: Source path. Use ``<container>:<path>`` for container paths.
            destination: Destination path. Use ``<container>:<path>`` for
                container paths.
            timeout: Optional timeout in seconds for the copy operation.

        Returns:
            A :class:`subprocess.CompletedProcess` with return code and output.

        Example::

            # Copy from host to container
            section.run_cli_cp(
                source="/tmp/script.py",
                destination="my-container:/workspace/script.py",
            )
        """
        cmd: list[str] = ["podman"]
        if self._connection_name:
            cmd.extend(["--connection", self._connection_name])
        cmd.extend(["cp", source, destination])
        runner = self._exec_runner
        return runner(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def run_python_script(
        self,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a Python script inside the container via ``python3 -c``.

        This is a convenience wrapper around :meth:`run_cli_exec` for running
        inline Python code.

        Args:
            script: Python source code to execute.
            args: Command-line arguments passed to the script (accessible via
                ``sys.argv[1:]``).
            timeout: Optional timeout in seconds. Defaults to no timeout.

        Returns:
            A :class:`subprocess.CompletedProcess` with return code and
            captured stdout/stderr.

        Example::

            result = section.run_python_script(
                script="import sys; print(sys.argv[1])",
                args=["hello"],
            )
            assert result.stdout.strip() == "hello"
        """
        return self.run_cli_exec(
            config=_ExecConfig(
                command=["python3", "-c", script, *args], timeout=timeout
            ),
        )

    def write_via_container(
        self,
        *,
        path: VfsPath,
        content: str,
        mode: str,
    ) -> None:
        """Write content to a file inside the container workspace.

        Creates parent directories as needed. For append mode, reads the
        existing file content and appends the new content.

        Args:
            path: Virtual filesystem path relative to ``/workspace``.
            content: UTF-8 text content to write.
            mode: Write mode - ``"create"`` for new files, ``"overwrite"`` to
                replace existing content, or ``"append"`` to add to the end.

        Raises:
            ToolValidationError: If the file is not valid UTF-8 (append mode),
                the parent directory cannot be created, or the copy fails.
        """
        handle = self.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        payload = content
        if mode == "append" and host_path.exists():
            try:
                existing = host_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ToolValidationError("File is not valid UTF-8.") from error
            except OSError as error:
                raise ToolValidationError("Failed to read existing file.") from error
            payload = f"{existing}{content}"
        container_path = _container_path_for(path)
        parent = posixpath.dirname(container_path)
        if parent and parent != "/":
            mkdir_result = self.run_cli_exec(
                config=_ExecConfig(command=["mkdir", "-p", parent], cwd="/"),
            )
            if mkdir_result.returncode != 0:
                message = (
                    mkdir_result.stderr.strip()
                    or mkdir_result.stdout.strip()
                    or "Failed to create target directory."
                )
                raise ToolValidationError(message)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            _ = tmp.write(payload)
            temp_path = tmp.name
        destination = f"{handle.descriptor.container_name}:{container_path}"
        try:
            completed = self.run_cli_cp(
                source=temp_path,
                destination=destination,
            )
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute filesystem commands."
            ) from error
        finally:
            with suppress(OSError):
                Path(temp_path).unlink()
        if completed.returncode != 0:
            message = (
                completed.stderr.strip() or completed.stdout.strip() or "Write failed."
            )
            raise ToolValidationError(message)

    def new_client(self) -> _PodmanClient:
        """Create a new Podman client using the configured connection settings.

        Each call returns a fresh client instance. The caller is responsible
        for calling ``close()`` on the returned client when finished.

        Returns:
            A Podman client instance implementing the ``_PodmanClient`` protocol.

        Raises:
            RuntimeError: If the ``podman`` package is not installed.
        """
        return self._client_factory()

    @property
    def connection_name(self) -> str | None:
        """Return the Podman connection name, if configured."""
        return self._connection_name

    @property
    def exec_runner(self) -> _ExecRunner:
        """Return the subprocess execution function used for CLI commands."""
        return self._exec_runner

    def close(self) -> None:
        """Stop and remove the container, releasing resources.

        This method is idempotent: calling it multiple times has no additional
        effect. The overlay directory is preserved on disk for inspection.

        After calling ``close()``, the section cannot be used to run commands
        until a new workspace is created (which happens automatically on the
        next tool invocation).
        """
        finalizer = self._finalizer
        if finalizer.alive:
            _ = finalizer()

    @staticmethod
    def _cleanup_from_finalizer(
        section_ref: weakref.ReferenceType[PodmanSandboxSection],
    ) -> None:
        section = section_ref()
        if section is not None:
            section._teardown_workspace()


def _host_path_for(root: Path, path: VfsPath) -> Path:
    host = root
    for segment in path.segments:
        host /= segment
    return host


def _container_path_for(path: VfsPath) -> str:
    if not path.segments:
        return _DEFAULT_WORKDIR
    return posixpath.join(_DEFAULT_WORKDIR, *path.segments)


def _assert_within_overlay(root: Path, candidate: Path) -> None:
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


class _PodmanShellSuite:
    """Handler collection bound to a :class:`PodmanSandboxSection`."""

    def __init__(self, *, section: PodmanSandboxSection) -> None:
        super().__init__()
        self._section = section

    def run_shell(
        self, params: PodmanShellParams, *, context: ToolContext
    ) -> ToolResult[PodmanShellResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        command = _normalize_command(params.command)
        cwd = _normalize_cwd(params.cwd)
        env_overrides = _normalize_env(params.env)
        timeout_seconds = _normalize_timeout(params.timeout_seconds)
        if params.stdin:
            _ = _ensure_ascii(params.stdin, field="stdin")
        _ = self._section.ensure_workspace()

        return self._run_shell_via_cli(
            params=params,
            command=command,
            cwd=cwd,
            environment=env_overrides,
            timeout_seconds=timeout_seconds,
        )

    def _run_shell_via_cli(
        self,
        *,
        params: PodmanShellParams,
        command: tuple[str, ...],
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ToolResult[PodmanShellResult]:
        exec_cmd = list(command)
        start = time.perf_counter()
        try:
            completed = self._section.run_cli_exec(
                config=_ExecConfig(
                    command=exec_cmd,
                    stdin=params.stdin if params.stdin else None,
                    cwd=cwd,
                    environment=environment,
                    timeout=timeout_seconds,
                    capture_output=params.capture_output,
                )
            )
            timed_out = False
            exit_code = completed.returncode
            stdout_text = completed.stdout
            stderr_text = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_code = 124
            stdout_text = str(error.stdout or "")
            stderr_text = str(error.stderr or "")
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute commands over SSH connections."
            ) from error
        duration_ms = int((time.perf_counter() - start) * 1_000)
        self._section.touch_workspace()
        stdout_text_clean = str(stdout_text or "").rstrip()
        stderr_text_clean = str(stderr_text or "").rstrip()
        if not params.capture_output:
            stdout_text_final = _CAPTURE_DISABLED
            stderr_text_final = _CAPTURE_DISABLED
        else:
            stdout_text_final = _truncate_stream(stdout_text_clean)
            stderr_text_final = _truncate_stream(stderr_text_clean)
        result = PodmanShellResult(
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=stdout_text_final,
            stderr=stderr_text_final,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
        message = f"`shell_execute` exited with {exit_code}."
        if timed_out:
            message = "`shell_execute` exceeded the configured timeout."
        return ToolResult(message=message, value=result)


__all__ = [
    "PodmanSandboxSection",
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanWorkspace",
]
