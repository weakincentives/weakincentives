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

"""Podman-backed tool surface exposing the ``shell_execute`` command.

This module provides the ``PodmanSandboxSection`` which manages an isolated
Linux container for shell commands and file operations. It delegates to
specialized modules for connection handling, shell execution, lifecycle
management, evaluation, and tool definitions.

Example usage::

    from weakincentives.contrib.tools import PodmanSandboxSection, PodmanSandboxConfig
    from weakincentives.runtime import Session, InProcessDispatcher

    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    config = PodmanSandboxConfig(mounts=(HostMount(host_path="src"),))
    section = PodmanSandboxSection(session=session, config=config)
"""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess  # nosec: B404
import tempfile
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, cast, override, runtime_checkable

from weakincentives.filesystem import Filesystem, HostFilesystem

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.markdown import MarkdownSection
from ...prompt.policy import ReadBeforeWritePolicy
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session, replace_latest
from . import vfs as vfs_module
from .podman_connection import (
    resolve_connection_settings,
    resolve_podman_connection,
)
from .podman_eval import PodmanEvalSuite
from .podman_lifecycle import (
    ClientFactory as _ClientFactory,
    PodmanClient as _PodmanClient,
    PodmanWorkspace,
    ResolvedHostMount as _ResolvedHostMount,
    WorkspaceHandle as _WorkspaceHandle,
    build_client_factory as _build_client_factory,
    container_path_for as _container_path_for,
    copy_mount_into_overlay,
    default_cache_root as _default_cache_root,
    host_path_for as _host_path_for,
    resolve_podman_host_mounts as _resolve_podman_host_mounts,
)
from .podman_shell import (
    PodmanShellParams,
    PodmanShellResult,
    PodmanShellSuite as _PodmanShellSuite,
    ShellExecConfig as _ExecConfig,
)
from .podman_tools import build_podman_tools
from .vfs import (
    FilesystemToolHandlers,
    HostMount,
    VfsPath,
)

_LOGGER: StructuredLogger = get_logger(__name__, context={"component": "tools.podman"})

_DEFAULT_IMAGE: Final[str] = "python:3.12-bookworm"
_DEFAULT_WORKDIR: Final[str] = "/workspace"
_PODMAN_TEMPLATE: Final[str] = """\
You have access to an isolated Linux container powered by Podman. The `ls`,
`read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
filesystem interface but operate on `/workspace` inside the container. The
`evaluate_python` tool is a thin wrapper around `python3 -c` (≤5 seconds); read
and edit files directly from the workspace. `shell_execute` runs short commands
(≤120 seconds) in the shared environment. No network access or privileged
operations are available. Do not assume files outside `/workspace` exist."""


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
class _PodmanSectionParams:
    image: str = _DEFAULT_IMAGE
    workspace_root: str = _DEFAULT_WORKDIR


@FrozenDataclass()
class PodmanSandboxConfig:
    """Configuration for :class:`PodmanSandboxSection`.

    Example::

        from weakincentives.contrib.tools import PodmanSandboxConfig, PodmanSandboxSection

        config = PodmanSandboxConfig(mounts=(HostMount(host_path="src"),))
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


class PodmanSandboxSection(MarkdownSection[_PodmanSectionParams]):
    """Prompt section exposing the Podman ``shell_execute`` tool.

    Use :class:`PodmanSandboxConfig` to consolidate configuration::

        config = PodmanSandboxConfig(mounts=(HostMount(host_path="src"),))
        section = PodmanSandboxSection(session=session, config=config)
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

        tools = build_podman_tools(
            fs_handlers=self._fs_handlers,
            shell_suite=self._shell_suite,
            eval_suite=self._eval_suite,
            accepts_overrides=accepts_overrides,
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
        return self._session

    @override
    def clone(self, **kwargs: object) -> PodmanSandboxSection:  # pragma: no cover
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
        resolved = resolve_podman_connection(preferred_name=connection_name)
        if resolved is None:
            return None
        return {
            "base_url": resolved.base_url,
            "identity": resolved.identity,
            "connection_name": resolved.connection_name,
        }

    def _ensure_workspace(self) -> _WorkspaceHandle:
        from .podman_lifecycle import create_container

        with self._lock:
            if self._workspace_handle is not None:
                return self._workspace_handle
            handle = self._create_workspace(create_container)
            self._workspace_handle = handle
            self._session[PodmanWorkspace].seed(handle.descriptor)
            return handle

    def _create_workspace(
        self, create_fn: Callable[..., _WorkspaceHandle]
    ) -> _WorkspaceHandle:
        client = self._client_factory()
        overlay = self._workspace_overlay_path()
        overlay.mkdir(parents=True, exist_ok=True)
        self._hydrate_overlay_mounts(overlay)
        return create_fn(
            client=client,
            session_id=str(self._session.session_id),
            image=self._image,
            overlay=overlay,
            base_env=self._base_env,
            clock=self._clock,
        )

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
        copy_mount_into_overlay(overlay=overlay, mount=mount)

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
        from .podman_lifecycle import teardown_container

        with self._lock:
            handle = self._workspace_handle
            self._workspace_handle = None
        if handle is None:
            return
        teardown_container(
            client_factory=self._client_factory,
            container_id=handle.descriptor.container_id,
        )

    def ensure_workspace(self) -> _WorkspaceHandle:
        return self._ensure_workspace()

    def workspace_environment(self) -> dict[str, str]:
        return self._workspace_env()

    def touch_workspace(self) -> None:
        self._touch_workspace()

    def run_cli_exec(self, *, config: _ExecConfig) -> subprocess.CompletedProcess[str]:
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
        section_ref: weakref.ReferenceType[PodmanSandboxSection],
    ) -> None:
        section = section_ref()
        if section is not None:
            section._teardown_workspace()


# Re-export shutil for tests that patch copy operations
shutil = shutil


__all__ = [
    "PodmanSandboxConfig",
    "PodmanSandboxSection",
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanWorkspace",
]
