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

"""Shared test helpers for Podman tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from uuid import uuid4

import weakincentives.contrib.tools.podman as podman_module
import weakincentives.contrib.tools.vfs as vfs_module
from weakincentives.contrib.tools import (
    HostMount,
    PodmanSandboxConfig,
    PodmanSandboxSection,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True)
class ExecCall:
    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    stdin: bool


@dataclass(slots=True)
class ExecResponse:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class FakeContainer:
    def __init__(
        self,
        *,
        container_id: str,
        queue: list[ExecResponse],
        readiness_exit_code: int,
        return_bytes: bool,
    ) -> None:
        self.id = container_id
        self.name = container_id
        self.started = False
        self.exec_calls: list[ExecCall] = []
        self._queue = queue
        self._readiness_exit_code = readiness_exit_code
        self._return_bytes = return_bytes
        self.stop_calls = 0
        self.remove_calls = 0

    def start(self, **_: object) -> None:
        self.started = True

    def exec_run(
        self,
        command: list[str],
        *,
        stdout: bool,
        stderr: bool,
        demux: bool,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
        stdin: bool = False,
    ) -> tuple[int | None, bytes | tuple[bytes, bytes]]:
        del stdout, stderr, demux
        if command == ["test", "-d", "/workspace"]:
            return self._readiness_exit_code, (b"", b"")
        self.exec_calls.append(
            ExecCall(
                command=tuple(command),
                env=dict(environment or {}),
                workdir=workdir or "/workspace",
                stdin=stdin,
            )
        )
        if not self._queue:
            raise AssertionError("No queued exec response for command.")
        response = self._queue.pop(0)
        if self._return_bytes:
            return response.exit_code, response.stdout.encode()
        return response.exit_code, (response.stdout.encode(), response.stderr.encode())

    def stop(self, timeout: object | None = None) -> None:
        self.stop_calls += 1

    def remove(self, force: object | None = None) -> None:
        self.remove_calls += 1


class FakeContainerCollection:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self._containers: dict[str, FakeContainer] = {}
        self._queue: list[ExecResponse] = []
        self._next_readiness_exit_code = 0
        self._return_bytes = False

    def create(self, **kwargs: object) -> FakeContainer:
        container_id = f"container-{uuid4().hex}"
        container = FakeContainer(
            container_id=container_id,
            queue=self._queue,
            readiness_exit_code=self._next_readiness_exit_code,
            return_bytes=self._return_bytes,
        )
        self._next_readiness_exit_code = 0
        self._containers[container_id] = container
        self.created.append(kwargs)
        return container

    def get(self, container_id: str) -> FakeContainer:
        return self._containers[container_id]

    def queue_response(self, response: ExecResponse) -> None:
        self._queue.append(response)

    def set_readiness_exit_code(self, exit_code: int) -> None:
        self._next_readiness_exit_code = exit_code

    def set_return_bytes(self, enabled: bool) -> None:
        self._return_bytes = enabled


class FakeImageCollection:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    def pull(self, image: str) -> None:
        self.pulled.append(image)


class FakePodmanClient:
    def __init__(self) -> None:
        self.containers = FakeContainerCollection()
        self.images = FakeImageCollection()
        self._closed = False

    def close(self) -> None:
        self._closed = True


class FakeCliRunner:
    def __init__(self, responses: Sequence[ExecResponse] | None = None) -> None:
        self._responses: list[ExecResponse] = list(
            responses or (ExecResponse(exit_code=0),)
        )
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        self.calls.append(list(cmd))
        self.kwargs.append(
            {
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        response = self._responses.pop(0) if self._responses else ExecResponse(0)
        return CompletedProcess(
            cmd,
            response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
        )


def make_section(
    *,
    session: Session,
    client: FakePodmanClient,
    cache_dir: Path,
    connection_name: str | None = None,
    runner: Callable[..., CompletedProcess[str]] | None = None,
    auto_connect: bool = False,
    mounts: Sequence[HostMount] = (),
    allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
) -> PodmanSandboxSection:
    exec_runner = runner if runner is not None else FakeCliRunner()
    allowed_roots = tuple(allowed_host_roots)
    base_config = PodmanSandboxConfig(
        client_factory=lambda: client,
        cache_dir=cache_dir,
        base_environment={"PATH": "/usr/bin"},
        connection_name=connection_name,
        exec_runner=exec_runner,
        mounts=mounts,
        allowed_host_roots=allowed_roots,
    )
    if auto_connect:
        return PodmanSandboxSection(session=session, config=base_config)
    configured = replace(
        base_config,
        base_url="ssh://example",
        identity="/tmp/identity",
    )
    return PodmanSandboxSection(session=session, config=configured)


def setup_host_mount(
    tmp_path: Path, *, content: str = "hello world"
) -> tuple[Path, HostMount, Path]:
    host_root = tmp_path / "host-root"
    repo = host_root / "sunfish"
    repo.mkdir(parents=True, exist_ok=True)
    file_path = repo / "README.md"
    file_path.write_text(content, encoding="utf-8")
    mount = HostMount(
        host_path="sunfish",
        mount_path=vfs_module.VfsPath(("sunfish",)),
    )
    return host_root, mount, file_path


def prepare_resolved_mount(
    tmp_path: Path,
    *,
    include_glob: tuple[str, ...] = (),
    exclude_glob: tuple[str, ...] = (),
    max_bytes: int | None = None,
) -> tuple[podman_module._ResolvedHostMount, Path, Path]:
    host_root = tmp_path / "resolved-root"
    repo = host_root / "sunfish"
    repo.mkdir(parents=True, exist_ok=True)
    file_path = repo / "payload.txt"
    file_path.write_text("payload", encoding="utf-8")
    mount = HostMount(
        host_path="sunfish",
        mount_path=vfs_module.VfsPath(("sunfish",)),
        include_glob=include_glob,
        exclude_glob=exclude_glob,
        max_bytes=max_bytes,
    )
    resolved = podman_module._resolve_single_host_mount(mount, (host_root,))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    return resolved, file_path, overlay
