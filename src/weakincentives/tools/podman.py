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

import json
import os
import posixpath
import subprocess  # nosec: B404
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast, runtime_checkable

from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session import Session, replace_latest
from ._context import ensure_context_uses_session
from .errors import ToolValidationError

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
_PODMAN_TEMPLATE: Final[str] = """\
Podman Workspace
----------------
You have access to an isolated Linux container powered by Podman. Phase 1 only
exposes the `shell_execute` tool so you can run short commands (≤120 seconds).
The container shares `/workspace` with upcoming VFS tools but no filesystem
operations are available yet. No network access or privileged operations are
available. Do not assume files outside `/workspace` exist."""


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

        suite = _PodmanShellSuite(section=self)
        tools = (
            Tool[PodmanShellParams, PodmanShellResult](
                name="shell_execute",
                description="Run a short command inside the Podman workspace.",
                handler=suite.run_shell,
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
        handle = self._section.ensure_workspace()
        environment = self._section.workspace_environment()
        environment.update(env_overrides)

        return self._run_shell_via_cli(
            params=params,
            command=command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            handle=handle,
        )

    def _run_shell_via_cli(
        self,
        *,
        params: PodmanShellParams,
        command: tuple[str, ...],
        cwd: str,
        environment: dict[str, str],
        timeout_seconds: float,
        handle: _WorkspaceHandle,
    ) -> ToolResult[PodmanShellResult]:
        container_name = handle.descriptor.container_name
        base_cmd = ["podman"]
        connection = self._section.connection_name
        if connection:
            base_cmd.extend(["--connection", connection])
        exec_cmd = [*base_cmd, "exec"]
        if params.stdin:
            exec_cmd.append("--interactive")
        exec_cmd.extend(["--workdir", cwd])
        for key, value in environment.items():
            exec_cmd.extend(["--env", f"{key}={value}"])
        exec_cmd.append(container_name)
        exec_cmd.extend(command)
        runner = self._section.exec_runner
        start = time.perf_counter()
        try:
            completed = runner(  # nosec: B603
                exec_cmd,
                input=params.stdin if params.stdin else None,
                text=True,
                capture_output=True,
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
