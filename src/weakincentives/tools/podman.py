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

"""Podman-backed tool surface exposing the ``shell_execute`` command."""

from __future__ import annotations

import base64
import fnmatch
import json
import os
import posixpath
import re
import subprocess  # nosec: B404
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast, runtime_checkable

from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session import Session, replace_latest, select_latest
from . import vfs as vfs_module
from ._context import ensure_context_uses_session
from .errors import ToolValidationError
from .vfs import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    ListDirectoryParams,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    VirtualFileSystem,
    WriteFile,
    WriteFileParams,
)

_LOGGER: StructuredLogger = get_logger(__name__, context={"component": "tools.podman"})

_DEFAULT_IMAGE: Final[str] = "python:3.12-bookworm"
_DEFAULT_WORKDIR: Final[str] = "/workspace"
_DEFAULT_USER: Final[str] = "65534:65534"
_TMPFS_SIZE: Final[int] = 268_435_456
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
_PODMAN_BASE_URL_ENV: Final[str] = "PODMAN_BASE_URL"
_PODMAN_IDENTITY_ENV: Final[str] = "PODMAN_IDENTITY"
_PODMAN_CONNECTION_ENV: Final[str] = "PODMAN_CONNECTION"
_CPU_PERIOD: Final[int] = 100_000
_CPU_QUOTA: Final[int] = 100_000
_MEMORY_LIMIT: Final[str] = "1g"
_MISSING_DEPENDENCY_MESSAGE: Final[str] = (
    "Install weakincentives[podman] to enable the Podman tool suite."
)
_MAX_MATCH_RESULTS: Final[int] = 2_000
_WRITE_FILE_SCRIPT: Final[str] = """
import base64
import sys
from pathlib import Path

mode = sys.argv[1]
target = Path(sys.argv[2])
payload = base64.b64decode(sys.argv[3])
content = payload.decode("utf-8")
target.parent.mkdir(parents=True, exist_ok=True)
if mode == "create":
    open_mode = "x"
elif mode == "overwrite":
    open_mode = "w"
elif mode == "append":
    open_mode = "a"
else:
    raise SystemExit(2)
with target.open(open_mode, encoding="utf-8") as handle:
    handle.write(content)
"""
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
Podman Workspace
----------------
You have access to an isolated Linux container powered by Podman. The `ls`,
`read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
filesystem interface but operate on `/workspace` inside the container. The
`shell_execute` tool runs short commands (≤120 seconds) in the same environment.
No network access or privileged operations are available. Do not assume files
outside `/workspace` exist."""


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


@dataclass(slots=True, frozen=True)
class _PodmanSectionParams:
    image: str = _DEFAULT_IMAGE
    workspace_root: str = _DEFAULT_WORKDIR


@dataclass(slots=True, frozen=True)
class PodmanShellParams:
    """Parameter payload accepted by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    stdin: str | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT
    capture_output: bool = True


@dataclass(slots=True, frozen=True)
class PodmanShellResult:
    """Structured command summary returned by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool


@dataclass(slots=True, frozen=True)
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
class _WorkspaceHandle:
    descriptor: PodmanWorkspace
    overlay_path: Path


@dataclass(slots=True, frozen=True)
class _PodmanConnectionInfo:
    base_url: str | None
    identity: str | None
    connection_name: str | None


def _default_cache_root() -> Path:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "weakincentives" / "podman"


def _resolve_podman_connection(
    *,
    preferred_name: str | None = None,
) -> _PodmanConnectionInfo | None:
    env_base_url = os.environ.get(_PODMAN_BASE_URL_ENV)
    env_identity = os.environ.get(_PODMAN_IDENTITY_ENV)
    env_connection = os.environ.get(_PODMAN_CONNECTION_ENV)
    if env_base_url or env_identity:
        return _PodmanConnectionInfo(
            base_url=env_base_url,
            identity=env_identity,
            connection_name=preferred_name or env_connection,
        )
    resolved_name = preferred_name or env_connection
    return _connection_from_cli(resolved_name)


def _connection_from_cli(
    connection_name: str | None,
) -> _PodmanConnectionInfo | None:
    try:
        result = subprocess.run(  # nosec B603 B607
            ["podman", "system", "connection", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        connections = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    candidate: dict[str, Any] | None = None
    if connection_name:
        for entry in connections:
            if entry.get("Name") == connection_name:
                candidate = entry
                break
    else:
        for entry in connections:
            if entry.get("Default"):
                candidate = entry
                break
        if candidate is None and connections:
            candidate = connections[0]
    if candidate is None:
        return None
    return _PodmanConnectionInfo(
        base_url=candidate.get("URI"),
        identity=candidate.get("Identity"),
        connection_name=candidate.get("Name"),
    )


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
    if timeout_seconds != timeout_seconds:  # NaN guard
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


class PodmanToolsSection(MarkdownSection[_PodmanSectionParams]):
    """Prompt section exposing the Podman ``shell_execute`` tool."""

    def __init__(
        self,
        *,
        session: Session,
        image: str = _DEFAULT_IMAGE,
        base_url: str | None = None,
        identity: str | os.PathLike[str] | None = None,
        base_environment: Mapping[str, str] | None = None,
        cache_dir: os.PathLike[str] | str | None = None,
        client_factory: _ClientFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        connection_name: str | None = None,
        exec_runner: _ExecRunner | None = None,
        accepts_overrides: bool = False,
    ) -> None:
        self._session = session
        self._image = image
        env_connection = os.environ.get(_PODMAN_CONNECTION_ENV)
        preferred_connection = connection_name or env_connection
        resolved_connection: _PodmanConnectionInfo | None = None
        if base_url is None or identity is None:
            resolved_connection = _resolve_podman_connection(
                preferred_name=preferred_connection
            )
        if base_url is None and resolved_connection is not None:
            base_url = resolved_connection.base_url
        if identity is None and resolved_connection is not None:
            identity = resolved_connection.identity
        if connection_name is None:
            if resolved_connection is not None:
                connection_name = resolved_connection.connection_name
            else:
                connection_name = env_connection
        if base_url is None:
            message = (
                "Podman connection could not be resolved. Configure `podman system connection` {}"
            ).format("or set PODMAN_BASE_URL/PODMAN_IDENTITY.")
            raise ToolValidationError(message)
        identity_str = str(identity) if identity is not None else None
        self._client_factory = client_factory or _build_client_factory(
            base_url=base_url,
            identity=identity_str,
        )
        self._base_env = tuple(
            sorted((base_environment or {}).items(), key=lambda item: item[0])
        )
        self._overlay_root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else _default_cache_root()
        )
        self._overlay_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._workspace_handle: _WorkspaceHandle | None = None
        self._lock = threading.RLock()
        self._connection_name = connection_name
        self._exec_runner: _ExecRunner = exec_runner or _default_exec_runner
        self._finalizer = weakref.finalize(
            self, PodmanToolsSection._cleanup_from_finalizer, weakref.ref(self)
        )

        session.register_reducer(PodmanWorkspace, replace_latest)
        self._initialize_vfs_state(session)

        vfs_suite = _PodmanVfsSuite(section=self)
        shell_suite = _PodmanShellSuite(section=self)
        tools = (
            Tool[ListDirectoryParams, tuple[FileInfo, ...]](
                name="ls",
                description="List directory entries under a relative path.",
                handler=vfs_suite.list_directory,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadFileParams, ReadFileResult](
                name="read_file",
                description="Read UTF-8 file contents with pagination support.",
                handler=vfs_suite.read_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[WriteFileParams, WriteFile](
                name="write_file",
                description="Create a new UTF-8 text file.",
                handler=vfs_suite.write_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[EditFileParams, WriteFile](
                name="edit_file",
                description="Replace occurrences of a string within a file.",
                handler=vfs_suite.edit_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[GlobParams, tuple[GlobMatch, ...]](
                name="glob",
                description="Match files beneath a directory using shell patterns.",
                handler=vfs_suite.glob,
                accepts_overrides=accepts_overrides,
            ),
            Tool[GrepParams, tuple[GrepMatch, ...]](
                name="grep",
                description="Search files for a regular expression pattern.",
                handler=vfs_suite.grep,
                accepts_overrides=accepts_overrides,
            ),
            Tool[RemoveParams, DeleteEntry](
                name="rm",
                description="Remove files or directories recursively.",
                handler=vfs_suite.remove,
                accepts_overrides=accepts_overrides,
            ),
            Tool[PodmanShellParams, PodmanShellResult](
                name="shell_execute",
                description="Run a short command inside the Podman workspace.",
                handler=shell_suite.run_shell,
                accepts_overrides=accepts_overrides,
            ),
        )
        super().__init__(
            title="Podman Workspace",
            key="podman.shell",
            template=_PODMAN_TEMPLATE,
            default_params=_PodmanSectionParams(
                image=image, workspace_root=_DEFAULT_WORKDIR
            ),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session

    def _initialize_vfs_state(self, session: Session) -> None:
        session.register_reducer(VirtualFileSystem, replace_latest)
        session.seed_slice(VirtualFileSystem, (VirtualFileSystem(),))
        session.register_reducer(
            WriteFile,
            vfs_module.make_write_reducer(),
            slice_type=VirtualFileSystem,
        )
        session.register_reducer(
            DeleteEntry,
            vfs_module.make_delete_reducer(),
            slice_type=VirtualFileSystem,
        )

    def latest_snapshot(self) -> VirtualFileSystem:
        snapshot = select_latest(self._session, VirtualFileSystem)
        return snapshot or VirtualFileSystem()

    @staticmethod
    def resolve_connection(
        connection_name: str | None = None,
    ) -> dict[str, str | None] | None:
        resolved = _resolve_podman_connection(preferred_name=connection_name)
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
            self._session.seed_slice(PodmanWorkspace, (handle.descriptor,))
            return handle

    def _create_workspace(self) -> _WorkspaceHandle:
        client = self._client_factory()
        overlay = self._overlay_root / str(self._session.session_id)
        overlay.mkdir(parents=True, exist_ok=True)
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
            mounts=[
                {
                    "Target": "/tmp",  # nosec: B108
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

    def _workspace_env(self) -> dict[str, str]:
        return (
            dict(self._workspace_handle.descriptor.env)
            if self._workspace_handle
            else dict(self._base_env)
        )

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
            self._session.seed_slice(PodmanWorkspace, (updated_descriptor,))

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
        return self._ensure_workspace()

    def workspace_environment(self) -> dict[str, str]:
        return self._workspace_env()

    def touch_workspace(self) -> None:
        self._touch_workspace()

    def run_cli_exec(
        self,
        *,
        command: Sequence[str],
        stdin: str | None = None,
        cwd: str | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        handle = self.ensure_workspace()
        env = self.workspace_environment()
        if environment:
            env.update(environment)

        exec_cmd: list[str] = ["podman"]
        if self._connection_name:
            exec_cmd.extend(["--connection", self._connection_name])
        exec_cmd.append("exec")
        if stdin is not None:
            exec_cmd.append("--interactive")
        exec_cmd.extend(["--workdir", cwd or _DEFAULT_WORKDIR])
        for key, value in env.items():
            exec_cmd.extend(["--env", f"{key}={value}"])
        exec_cmd.append(handle.descriptor.container_name)
        exec_cmd.extend(command)
        runner = self._exec_runner
        return runner(
            exec_cmd,
            input=stdin,
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
        return self.run_cli_exec(
            command=["python3", "-c", script, *args],
            timeout=timeout,
        )

    def new_client(self) -> _PodmanClient:
        return self._client_factory()

    @property
    def connection_name(self) -> str | None:
        return self._connection_name

    @property
    def exec_runner(self) -> _ExecRunner:
        return self._exec_runner

    def close(self) -> None:
        finalizer = self._finalizer
        if finalizer.alive:
            _ = finalizer()

    @staticmethod
    def _cleanup_from_finalizer(
        section_ref: weakref.ReferenceType[PodmanToolsSection],
    ) -> None:
        section = section_ref()
        if section is not None:
            section._teardown_workspace()


def _host_path_for(root: Path, path: VfsPath) -> Path:
    host = root
    for segment in path.segments:
        host = host / segment
    return host


def _container_path_for(path: VfsPath) -> str:
    if not path.segments:
        return _DEFAULT_WORKDIR
    return posixpath.join(_DEFAULT_WORKDIR, *path.segments)


def _encode_content(content: str) -> str:
    payload = base64.b64encode(content.encode("utf-8"))
    return payload.decode("ascii")


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


def _compose_child_path(base: VfsPath, name: str) -> VfsPath | None:
    candidate = VfsPath((*base.segments, name))
    try:
        return vfs_module.normalize_path(candidate)
    except ToolValidationError:
        return None


def _compose_relative_path(base: VfsPath, relative: Path) -> VfsPath | None:
    segments = (*base.segments, *relative.parts)
    candidate = VfsPath(segments)
    try:
        return vfs_module.normalize_path(candidate)
    except ToolValidationError:
        return None


def _iter_workspace_files(base: Path) -> Iterator[Path]:
    if not base.exists():
        return
    for dirpath, _, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        for name in filenames:
            yield current / name


def _stat_file(path: Path) -> tuple[int, datetime]:
    try:
        stat_result = path.stat()
    except OSError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError("Failed to stat workspace file.") from error
    size = stat_result.st_size
    updated_at = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)
    return size, updated_at


def _format_remove_message(path: VfsPath, count: int) -> str:
    path_label = "/".join(path.segments) or "/"
    label = "entry" if count == 1 else "entries"
    return f"Deleted {count} {label} under {path_label}."


def _format_read_message(path: VfsPath, start: int, end: int) -> str:
    path_label = "/".join(path.segments) or "/"
    if start == end:
        return f"Read file {path_label} (no lines returned)."
    return f"Read file {path_label} (lines {start + 1}-{end})."


class _PodmanVfsSuite:
    """Filesystem tool handlers bound to a :class:`PodmanToolsSection`."""

    def __init__(self, *, section: PodmanToolsSection) -> None:
        super().__init__()
        self._section = section

    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(
            params.path, allow_empty=True, field="path"
        )
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        _assert_within_overlay(handle.overlay_path, host_path)
        if host_path.exists() and host_path.is_file():
            raise ToolValidationError("Cannot list a file path; provide a directory.")
        snapshot = self._section.latest_snapshot()
        entries = self._build_directory_entries(
            base=path,
            host_path=host_path,
            snapshot=snapshot,
            overlay_root=handle.overlay_path,
        )
        message = vfs_module.format_directory_message(path, entries)
        self._section.touch_workspace()
        return ToolResult(message=message, value=tuple(entries))

    def read_file(
        self, params: ReadFileParams, *, context: ToolContext
    ) -> ToolResult[ReadFileResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        offset = vfs_module.normalize_offset(params.offset)
        limit = vfs_module.normalize_limit(params.limit)
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        if not host_path.exists() or not host_path.is_file():
            raise ToolValidationError("File does not exist in the workspace.")
        _assert_within_overlay(handle.overlay_path, host_path)
        try:
            content = host_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolValidationError("File is not valid UTF-8.") from error
        except OSError as error:  # pragma: no cover - defensive guard
            raise ToolValidationError("Failed to read file contents.") from error
        lines = content.splitlines()
        total_lines = len(lines)
        start = min(offset, total_lines)
        end = min(start + limit, total_lines)
        numbered = [
            f"{index + 1:>4} | {line}"
            for index, line in enumerate(lines[start:end], start=start)
        ]
        formatted = "\n".join(numbered)
        message = _format_read_message(path, start, end)
        self._section.touch_workspace()
        return ToolResult(
            message=message,
            value=ReadFileResult(
                path=path,
                content=formatted,
                offset=start,
                limit=end - start,
                total_lines=total_lines,
            ),
        )

    def write_file(
        self, params: WriteFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        content = vfs_module.normalize_content(params.content)
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        if host_path.exists():
            raise ToolValidationError(
                "File already exists; use edit_file to modify existing content."
            )
        _assert_within_overlay(handle.overlay_path, host_path)
        self._write_via_container(path=path, content=content, mode="create")
        self._section.touch_workspace()
        message = vfs_module.format_write_file_message(path, content, "create")
        return ToolResult(
            message=message,
            value=WriteFile(path=path, content=content, mode="create"),
        )

    def edit_file(
        self, params: EditFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        if len(params.old_string) > vfs_module.MAX_WRITE_LENGTH:
            raise ToolValidationError("old_string exceeds the 48,000 character limit.")
        if len(params.new_string) > vfs_module.MAX_WRITE_LENGTH:
            raise ToolValidationError("new_string exceeds the 48,000 character limit.")
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        if not host_path.exists() or not host_path.is_file():
            raise ToolValidationError("File does not exist in the workspace.")
        _assert_within_overlay(handle.overlay_path, host_path)
        try:
            existing = host_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolValidationError("File is not valid UTF-8.") from error
        occurrences = existing.count(params.old_string)
        if occurrences == 0:
            raise ToolValidationError("old_string not found in the target file.")
        if not params.replace_all and occurrences != 1:
            raise ToolValidationError(
                "old_string must match exactly once unless replace_all is true."
            )
        if params.replace_all:
            replacements = occurrences
            updated = existing.replace(params.old_string, params.new_string)
        else:
            replacements = 1
            updated = existing.replace(params.old_string, params.new_string, 1)
        normalized = vfs_module.normalize_content(updated)
        self._write_via_container(path=path, content=normalized, mode="overwrite")
        self._section.touch_workspace()
        message = vfs_module.format_edit_message(path, replacements)
        return ToolResult(
            message=message,
            value=WriteFile(path=path, content=normalized, mode="overwrite"),
        )

    def glob(
        self, params: GlobParams, *, context: ToolContext
    ) -> ToolResult[tuple[GlobMatch, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        base = vfs_module.normalize_string_path(
            params.path, allow_empty=True, field="path"
        )
        pattern = params.pattern.strip()
        if not pattern:
            raise ToolValidationError("Pattern must not be empty.")
        _ = vfs_module.ensure_ascii(pattern, "pattern")
        handle = self._section.ensure_workspace()
        host_base = _host_path_for(handle.overlay_path, base)
        _assert_within_overlay(handle.overlay_path, host_base)
        matches: list[GlobMatch] = []
        snapshot = self._section.latest_snapshot()
        for file_path in _iter_workspace_files(host_base):
            try:
                relative = file_path.relative_to(host_base)
            except ValueError:
                continue
            candidate_path = _compose_relative_path(base, relative)
            if candidate_path is None:
                continue
            relative_label = relative.as_posix()
            if not fnmatch.fnmatchcase(relative_label, pattern):
                continue
            try:
                match = self._build_glob_match(
                    target=candidate_path,
                    host_path=file_path,
                    snapshot=snapshot,
                    overlay_root=handle.overlay_path,
                )
            except ToolValidationError:
                continue
            matches.append(match)
            if len(matches) >= _MAX_MATCH_RESULTS:
                break
        matches.sort(key=lambda match: match.path.segments)
        message = vfs_module.format_glob_message(base, pattern, matches)
        self._section.touch_workspace()
        return ToolResult(message=message, value=tuple(matches))

    def grep(
        self, params: GrepParams, *, context: ToolContext
    ) -> ToolResult[tuple[GrepMatch, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        try:
            pattern = re.compile(params.pattern)
        except re.error as error:
            return ToolResult(
                message=f"Invalid regular expression: {error}",
                value=None,
                success=False,
            )
        base_path: VfsPath | None = None
        if params.path is not None:
            base_path = vfs_module.normalize_string_path(
                params.path, allow_empty=True, field="path"
            )
        glob_pattern = params.glob.strip() if params.glob is not None else None
        if glob_pattern:
            _ = vfs_module.ensure_ascii(glob_pattern, "glob")
        handle = self._section.ensure_workspace()
        host_base = _host_path_for(handle.overlay_path, base_path or VfsPath(()))
        _assert_within_overlay(handle.overlay_path, host_base)
        matches: list[GrepMatch] = []
        for file_path in _iter_workspace_files(host_base):
            try:
                relative = file_path.relative_to(host_base)
            except ValueError:
                continue
            relative_label = relative.as_posix()
            if glob_pattern and not fnmatch.fnmatchcase(relative_label, glob_pattern):
                continue
            target_path = _compose_relative_path(base_path or VfsPath(()), relative)
            if target_path is None:
                continue
            try:
                _assert_within_overlay(handle.overlay_path, file_path)
            except ToolValidationError:
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError:
                continue
            for index, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(
                        GrepMatch(
                            path=target_path,
                            line_number=index,
                            line=line,
                        )
                    )
                    if len(matches) >= _MAX_MATCH_RESULTS:
                        break
            if len(matches) >= _MAX_MATCH_RESULTS:
                break
        message = vfs_module.format_grep_message(params.pattern, matches)
        self._section.touch_workspace()
        return ToolResult(message=message, value=tuple(matches))

    def remove(
        self, params: RemoveParams, *, context: ToolContext
    ) -> ToolResult[DeleteEntry]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.path, field="path")
        if not path.segments:
            raise ToolValidationError("Cannot remove the workspace root.")
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.overlay_path, path)
        if not host_path.exists():
            raise ToolValidationError("No files matched the provided path.")
        _assert_within_overlay(handle.overlay_path, host_path)
        removed_entries = sum(1 for _ in _iter_workspace_files(host_path))
        removed_entries = 1 if host_path.is_file() else max(removed_entries, 1)
        args = (_container_path_for(path),)
        try:
            completed = self._section.run_python_script(
                script=_REMOVE_PATH_SCRIPT,
                args=args,
            )
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute filesystem commands."
            ) from error
        if completed.returncode != 0:
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "Removal failed."
            )
            raise ToolValidationError(message)
        self._section.touch_workspace()
        message = _format_remove_message(path, removed_entries)
        return ToolResult(
            message=message,
            value=DeleteEntry(path=path),
        )

    def _build_directory_entries(
        self,
        *,
        base: VfsPath,
        host_path: Path,
        snapshot: VirtualFileSystem,
        overlay_root: Path,
    ) -> list[FileInfo]:
        entries: list[FileInfo] = []
        if not host_path.exists():
            return entries
        try:
            children = sorted(host_path.iterdir(), key=lambda child: child.name.lower())
        except OSError as error:
            raise ToolValidationError(
                "Failed to inspect directory contents."
            ) from error
        for child in children:
            entry_path = _compose_child_path(base, child.name)
            if entry_path is None:
                continue
            if child.is_dir() and not child.is_symlink():
                entries.append(
                    FileInfo(
                        path=entry_path,
                        kind="directory",
                        size_bytes=None,
                        version=None,
                        updated_at=None,
                    )
                )
                continue
            try:
                info = self._build_file_info(
                    path=entry_path,
                    host_file=child,
                    snapshot=snapshot,
                    overlay_root=overlay_root,
                )
            except ToolValidationError:
                continue
            entries.append(info)
        entries.sort(key=lambda entry: entry.path.segments)
        return entries[:_MAX_MATCH_RESULTS]

    def _build_file_info(
        self,
        *,
        path: VfsPath,
        host_file: Path,
        snapshot: VirtualFileSystem,
        overlay_root: Path,
    ) -> FileInfo:
        _assert_within_overlay(overlay_root, host_file)
        snapshot_entry = vfs_module.find_file(snapshot.files, path)
        size_bytes, updated_at = _stat_file(host_file)
        version = snapshot_entry.version if snapshot_entry else None
        updated = snapshot_entry.updated_at if snapshot_entry else updated_at
        return FileInfo(
            path=path,
            kind="file",
            size_bytes=size_bytes,
            version=version,
            updated_at=updated,
        )

    def _build_glob_match(
        self,
        *,
        target: VfsPath,
        host_path: Path,
        snapshot: VirtualFileSystem,
        overlay_root: Path,
    ) -> GlobMatch:
        _assert_within_overlay(overlay_root, host_path)
        snapshot_entry = vfs_module.find_file(snapshot.files, target)
        size_bytes, updated_at = _stat_file(host_path)
        if snapshot_entry is None:
            return GlobMatch(
                path=target,
                size_bytes=size_bytes,
                version=1,
                updated_at=updated_at,
            )
        return GlobMatch(
            path=target,
            size_bytes=size_bytes,
            version=snapshot_entry.version,
            updated_at=snapshot_entry.updated_at,
        )

    def _write_via_container(
        self,
        *,
        path: VfsPath,
        content: str,
        mode: str,
    ) -> None:
        encoded = _encode_content(content)
        args = (mode, _container_path_for(path), encoded)
        try:
            completed = self._section.run_python_script(
                script=_WRITE_FILE_SCRIPT,
                args=args,
            )
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute filesystem commands."
            ) from error
        if completed.returncode != 0:
            message = (
                completed.stderr.strip() or completed.stdout.strip() or "Write failed."
            )
            raise ToolValidationError(message)


class _PodmanShellSuite:
    """Handler collection bound to a :class:`PodmanToolsSection`."""

    def __init__(self, *, section: PodmanToolsSection) -> None:
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
                command=exec_cmd,
                stdin=params.stdin if params.stdin else None,
                cwd=cwd,
                environment=environment,
                timeout=timeout_seconds,
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
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanToolsSection",
    "PodmanWorkspace",
]
