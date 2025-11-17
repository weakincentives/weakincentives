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

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

import weakincentives.tools.podman as podman_module
import weakincentives.tools.vfs as vfs_module
from tests.tools.helpers import build_tool_context, find_tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import (
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    PodmanShellParams,
    PodmanShellResult,
    PodmanToolsSection,
    PodmanWorkspace,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    WriteFileParams,
)
from weakincentives.tools.errors import ToolValidationError


@dataclass(slots=True)
class _ExecCall:
    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    stdin: bool


@dataclass(slots=True)
class _ExecResponse:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class _FakeContainer:
    def __init__(
        self,
        *,
        container_id: str,
        queue: list[_ExecResponse],
        readiness_exit_code: int,
        return_bytes: bool,
    ) -> None:
        self.id = container_id
        self.name = container_id
        self.started = False
        self.exec_calls: list[_ExecCall] = []
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
            _ExecCall(
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


class _FakeContainerCollection:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self._containers: dict[str, _FakeContainer] = {}
        self._queue: list[_ExecResponse] = []
        self._next_readiness_exit_code = 0
        self._return_bytes = False

    def create(self, **kwargs: object) -> _FakeContainer:
        container_id = f"container-{uuid4().hex}"
        container = _FakeContainer(
            container_id=container_id,
            queue=self._queue,
            readiness_exit_code=self._next_readiness_exit_code,
            return_bytes=self._return_bytes,
        )
        self._next_readiness_exit_code = 0
        self._containers[container_id] = container
        self.created.append(kwargs)
        return container

    def get(self, container_id: str) -> _FakeContainer:
        return self._containers[container_id]

    def queue_response(self, response: _ExecResponse) -> None:
        self._queue.append(response)

    def set_readiness_exit_code(self, exit_code: int) -> None:
        self._next_readiness_exit_code = exit_code

    def set_return_bytes(self, enabled: bool) -> None:
        self._return_bytes = enabled


class _FakeImageCollection:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    def pull(self, image: str) -> None:
        self.pulled.append(image)


class _FakePodmanClient:
    def __init__(self) -> None:
        self.containers = _FakeContainerCollection()
        self.images = _FakeImageCollection()
        self._closed = False

    def close(self) -> None:
        self._closed = True


class _FakeCliRunner:
    def __init__(self, responses: Sequence[_ExecResponse] | None = None) -> None:
        self._responses: list[_ExecResponse] = list(
            responses or (_ExecResponse(exit_code=0),)
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
        response = self._responses.pop(0) if self._responses else _ExecResponse(0)
        return CompletedProcess(
            cmd,
            response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
        )


def _make_section(
    *,
    session: Session,
    client: _FakePodmanClient,
    cache_dir: Path,
    connection_name: str | None = None,
    runner: Callable[..., CompletedProcess[str]] | None = None,
    auto_connect: bool = False,
    mounts: Sequence[HostMount] = (),
    allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
) -> PodmanToolsSection:
    exec_runner = runner if runner is not None else _FakeCliRunner()
    allowed_roots = tuple(allowed_host_roots)
    if auto_connect:
        return PodmanToolsSection(
            session=session,
            client_factory=lambda: client,
            cache_dir=cache_dir,
            base_environment={"PATH": "/usr/bin"},
            connection_name=connection_name,
            exec_runner=exec_runner,
            mounts=mounts,
            allowed_host_roots=allowed_roots,
        )
    return PodmanToolsSection(
        session=session,
        client_factory=lambda: client,
        cache_dir=cache_dir,
        base_environment={"PATH": "/usr/bin"},
        connection_name=connection_name,
        exec_runner=exec_runner,
        mounts=mounts,
        allowed_host_roots=allowed_roots,
        base_url="ssh://example",
        identity="/tmp/identity",
    )


def _setup_host_mount(
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


@pytest.fixture()
def session_and_bus() -> tuple[Session, InProcessEventBus]:
    bus = InProcessEventBus()
    return Session(bus=bus), bus


def test_section_registers_shell_tool(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "shell_execute")
    assert tool.description.startswith("Run a short command")


def test_section_registers_vfs_tool(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "ls")
    assert tool.description.startswith("List directory entries")


def test_host_mount_seeded_into_snapshot(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, file_path = _setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )

    snapshot = section.latest_snapshot()

    assert len(snapshot.files) == 1
    file = snapshot.files[0]
    assert file.path.segments == ("sunfish", file_path.name)
    assert file.content == file_path.read_text(encoding="utf-8")


def test_host_mount_populates_prompt_copy(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, _file_path = _setup_host_mount(tmp_path)
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path / "cache",
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )

    assert "Configured host mounts:" in section.template
    assert str(host_root / "sunfish") in section.template


def test_host_mount_materializes_overlay(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, file_path = _setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )

    handle = section._workspace_handle
    assert handle is not None
    mounted = handle.overlay_path / "sunfish" / file_path.name
    assert mounted.read_text(encoding="utf-8") == file_path.read_text(encoding="utf-8")


def test_host_mount_hydration_skips_existing_overlay(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, file_path = _setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )
    overlay = section._workspace_overlay_path()
    overlay.mkdir(parents=True, exist_ok=True)
    placeholder = overlay / "existing.txt"
    placeholder.write_text("keep", encoding="utf-8")

    section._hydrate_overlay_mounts(overlay)

    mounted = overlay / "sunfish" / file_path.name
    assert not mounted.exists()
    assert placeholder.read_text(encoding="utf-8") == "keep"


def test_host_mount_hydration_raises_on_write_error(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, file_path = _setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )
    overlay = section._workspace_overlay_path()
    overlay.mkdir(parents=True, exist_ok=True)
    target = overlay / "sunfish" / file_path.name
    original_write_text = Path.write_text

    def _fail_on_target(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path == target:
            raise OSError("boom")
        return original_write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", _fail_on_target)

    with pytest.raises(ToolValidationError):
        section._hydrate_overlay_mounts(overlay)


def test_section_exposes_new_client(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)

    assert section.new_client() is client


def test_close_stops_and_removes_container(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )
    handle = section._workspace_handle
    assert handle is not None
    container = client.containers.get(handle.descriptor.container_id)

    section.close()

    assert container.stop_calls == 1
    assert container.remove_calls == 1
    assert section._workspace_handle is None


def test_close_is_idempotent(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )
    handle = section._workspace_handle
    assert handle is not None
    container = client.containers.get(handle.descriptor.container_id)

    section.close()
    section.close()

    assert container.stop_calls == 1
    assert container.remove_calls == 1


def test_close_without_workspace(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)

    section.close()


def test_close_handles_client_factory_failure(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )

    def _raise() -> _FakePodmanClient:
        raise RuntimeError("boom")

    section._client_factory = _raise

    section.close()


def test_close_handles_missing_container(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )

    class _BrokenContainers:
        def get(self, container_id: str) -> _FakeContainer:
            raise RuntimeError("missing container")

    class _BrokenClient:
        def __init__(self) -> None:
            self.containers = _BrokenContainers()
            self.images = _FakeImageCollection()

        def close(self) -> None:
            return None

    section._client_factory = lambda: _BrokenClient()

    section.close()


def test_section_auto_resolves_connection(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0, stdout="auto")])
    resolved = podman_module._PodmanConnectionInfo(
        base_url="ssh://detected",
        identity="/tmp/key",
        connection_name="auto-conn",
    )

    def _fake_resolve(
        *, preferred_name: str | None = None
    ) -> podman_module._PodmanConnectionInfo | None:
        return resolved

    monkeypatch.setattr(
        podman_module,
        "_resolve_podman_connection",
        _fake_resolve,
    )

    section = PodmanToolsSection(
        session=session,
        cache_dir=tmp_path,
        client_factory=lambda: client,
        base_environment={"PATH": "/usr/bin"},
        exec_runner=cli_runner,
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(bus, session),
    )

    assert section.connection_name == "auto-conn"


def test_connection_resolution_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODMAN_BASE_URL", "ssh://env")
    monkeypatch.setenv("PODMAN_IDENTITY", "/tmp/env")
    result = podman_module._resolve_podman_connection(preferred_name="custom")
    assert result is not None
    assert result.base_url == "ssh://env"
    assert result.identity == "/tmp/env"
    assert result.connection_name == "custom"


def test_connection_resolution_uses_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)
    connections = [
        {
            "Name": "first",
            "URI": "ssh://first",
            "Identity": "/tmp/first",
            "Default": False,
        },
        {
            "Name": "desired",
            "URI": "ssh://desired",
            "Identity": "/tmp/desired",
            "Default": False,
        },
    ]

    def _fake_run(
        *_: object,
        **__: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps(connections))

    monkeypatch.setattr(podman_module.subprocess, "run", _fake_run)

    result = podman_module._resolve_podman_connection(preferred_name="desired")
    assert result is not None
    assert result.base_url == "ssh://desired"
    assert result.identity == "/tmp/desired"
    assert result.connection_name == "desired"


def test_connection_resolution_falls_back_to_first_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)
    connections = [
        {
            "Name": "first",
            "URI": "ssh://first",
            "Identity": "/tmp/first",
            "Default": False,
        },
        {
            "Name": "second",
            "URI": "ssh://second",
            "Identity": "/tmp/second",
            "Default": False,
        },
    ]

    def _fake_run(
        *_: object,
        **__: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps(connections))

    monkeypatch.setattr(podman_module.subprocess, "run", _fake_run)

    result = podman_module._resolve_podman_connection()
    assert result is not None
    assert result.base_url == "ssh://first"
    assert result.connection_name == "first"


def test_connection_resolution_handles_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)

    def _fail(*_: object, **__: object) -> None:
        raise FileNotFoundError("podman")

    monkeypatch.setattr(podman_module.subprocess, "run", _fail)

    assert podman_module._resolve_podman_connection() is None


def test_connection_resolution_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)

    def _fake_run(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="not json")

    monkeypatch.setattr(podman_module.subprocess, "run", _fake_run)

    assert podman_module._resolve_podman_connection() is None


def test_connection_resolution_missing_requested_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)
    connections = [
        {
            "Name": "first",
            "URI": "ssh://first",
            "Identity": "/tmp/first",
            "Default": False,
        },
    ]

    def _fake_run(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps(connections))

    monkeypatch.setattr(podman_module.subprocess, "run", _fake_run)

    assert podman_module._resolve_podman_connection(preferred_name="missing") is None


def test_resolve_connection_static_handles_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        podman_module,
        "_resolve_podman_connection",
        lambda *, preferred_name=None: None,
    )
    assert PodmanToolsSection.resolve_connection() is None


def test_resolve_connection_static_returns_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = podman_module._PodmanConnectionInfo(
        base_url="ssh://detected",
        identity="/tmp/key",
        connection_name="resolved",
    )
    monkeypatch.setattr(
        podman_module,
        "_resolve_podman_connection",
        lambda *, preferred_name=None: resolved,
    )
    result = PodmanToolsSection.resolve_connection(connection_name="preferred")
    assert result == {
        "base_url": "ssh://detected",
        "identity": "/tmp/key",
        "connection_name": "resolved",
    }


def test_section_requires_connection_when_detection_fails(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    monkeypatch.setattr(
        podman_module,
        "_resolve_podman_connection",
        lambda *, preferred_name=None: None,
    )
    with pytest.raises(ToolValidationError):
        PodmanToolsSection(
            session=session,
            cache_dir=tmp_path,
            client_factory=lambda: client,
            base_environment={"PATH": "/usr/bin"},
            exec_runner=_FakeCliRunner(),
        )


def test_shell_execute_runs_commands_and_stores_workspace(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [_ExecResponse(exit_code=0, stdout="hello world\n", stderr="")]
    )
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "hello world"))
    result = handler(params, context=build_tool_context(bus, session))

    assert isinstance(result.value, PodmanShellResult)
    assert result.value.exit_code == 0
    assert "hello world" in result.value.stdout

    workspace = session.select_all(PodmanWorkspace)
    assert workspace
    assert workspace[-1].image == "python:3.12-bookworm"
    handle = section._workspace_handle
    assert handle is not None
    assert cli_runner.calls[-1][-2:] == ["echo", "hello world"]


def test_shell_execute_validates_command(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=_FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None
    params = PodmanShellParams(command=())

    with pytest.raises(ToolValidationError):
        handler(params, context=build_tool_context(bus, session))


def test_shell_execute_merges_environment(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("printenv",), env={"custom": "value"})
    handler(params, context=build_tool_context(bus, session))

    call = " ".join(cli_runner.calls[-1])
    assert "PATH=/usr/bin" in call
    assert "CUSTOM=value" in call


def test_shell_execute_respects_capture_flag(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [_ExecResponse(exit_code=0, stdout="x" * 40_000, stderr="")]
    )
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("cat",), capture_output=False)
    result = handler(params, context=build_tool_context(bus, session))
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "capture disabled"
    assert cli_runner.kwargs[-1]["capture_output"] is False


def test_shell_execute_captures_output_by_default(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [_ExecResponse(exit_code=0, stdout="normal output", stderr="")]
    )
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "hi"))
    result = handler(params, context=build_tool_context(bus, session))
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "normal output"
    assert cli_runner.kwargs[-1]["capture_output"] is True


def test_shell_execute_normalizes_cwd(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("pwd",), cwd="src/docs")
    handler(params, context=build_tool_context(bus, session))

    call = cli_runner.calls[-1]
    idx = call.index("--workdir")
    assert call[idx + 1] == "/workspace/src/docs"


def test_shell_execute_rejects_non_ascii_stdin(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=_FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("cat",), stdin="ümlaut")
    with pytest.raises(ToolValidationError):
        handler(params, context=build_tool_context(bus, session))


def test_shell_execute_rejects_mismatched_session(tmp_path: Path) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    other_session = Session(bus=bus)
    client = _FakePodmanClient()
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=_FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(RuntimeError, match="session does not match"):
        handler(
            PodmanShellParams(command=("true",)),
            context=build_tool_context(bus, other_session),
        )


def test_shell_execute_cli_fallback(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner([_ExecResponse(exit_code=0, stdout="cli output")])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        connection_name="podman-machine-default",
        runner=cli_runner,
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "cli"), stdin="payload")
    result = handler(params, context=build_tool_context(bus, session))

    call = cli_runner.calls[-1]
    assert call[:3] == ["podman", "--connection", "podman-machine-default"]
    assert "--interactive" in call
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "cli output"


def test_shell_execute_cli_capture_disabled(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [_ExecResponse(exit_code=0, stdout="cli output", stderr="cli err")]
    )
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        connection_name="podman-machine-default",
        runner=cli_runner,
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "cli"), capture_output=False)
    result = handler(params, context=build_tool_context(bus, session))
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "capture disabled"
    assert value.stderr == "capture disabled"


def test_shell_execute_cli_timeout(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()

    def _timeout_runner(
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=["podman"], timeout=1.0, output="partial", stderr="error"
        )

    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        connection_name="podman-machine-default",
        runner=_timeout_runner,
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("sleep", "1"))
    result = handler(params, context=build_tool_context(bus, session))
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.timed_out
    assert value.stdout == "partial"
    assert value.stderr == "error"


def test_shell_execute_cli_missing_binary(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()

    def _missing_runner(
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise FileNotFoundError("podman not found")

    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        connection_name="podman-machine-default",
        runner=_missing_runner,
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            PodmanShellParams(command=("true",)),
            context=build_tool_context(bus, session),
        )


def test_default_cache_root_respects_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WEAKINCENTIVES_CACHE", str(tmp_path))
    assert podman_module._default_cache_root() == tmp_path


def test_default_cache_root_uses_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("WEAKINCENTIVES_CACHE", raising=False)
    monkeypatch.setattr(podman_module.Path, "home", lambda: tmp_path)
    expected = tmp_path / ".cache" / "weakincentives" / "podman"
    assert podman_module._default_cache_root() == expected


def test_client_factory_creates_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            created["kwargs"] = kwargs

        def close(self) -> None:
            created["closed"] = True

    fake_module = SimpleNamespace(PodmanClient=_StubClient)
    monkeypatch.setitem(sys.modules, "podman", fake_module)

    factory = podman_module._build_client_factory(base_url=None, identity=None)
    client = factory()
    assert created["kwargs"] == {}
    client.close()
    assert created["closed"] is True


def test_default_exec_runner_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def _fake_run(
        args: list[str],
        *,
        input: str | None = None,  # noqa: A002
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        recorded["args"] = args
        recorded["input"] = input
        recorded["text"] = text
        recorded["capture_output"] = capture_output
        recorded["timeout"] = timeout
        return CompletedProcess(args, 0, stdout="ok", stderr="err")

    monkeypatch.setattr(podman_module.subprocess, "run", _fake_run)

    result = podman_module._default_exec_runner(
        ["echo", "hi"],
        input="payload",
        text=True,
        capture_output=True,
        timeout=1.5,
    )

    assert recorded["args"] == ["echo", "hi"]
    assert recorded["input"] == "payload"
    assert recorded["text"] is True
    assert recorded["capture_output"] is True
    assert recorded["timeout"] == 1.5
    assert result.stdout == "ok"


def test_client_factory_uses_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, dict[str, str]] = {}

    class _DummyClient:
        def __init__(self, **kwargs: str) -> None:
            created["kwargs"] = kwargs

        def close(self) -> None:
            return None

    fake_module = SimpleNamespace(PodmanClient=_DummyClient)
    monkeypatch.setitem(sys.modules, "podman", fake_module)
    factory = podman_module._build_client_factory(
        base_url="unix:///tmp/podman.sock",
        identity="identity.pem",
    )
    client = factory()
    try:
        assert created["kwargs"] == {
            "base_url": "unix:///tmp/podman.sock",
            "identity": "identity.pem",
        }
    finally:
        client.close()


def test_command_validation_rejects_blank_entry() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._normalize_command(("",))


def test_command_validation_rejects_long_entry() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._normalize_command(("x" * 5_000,))


def test_env_validation_guards_limits() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._normalize_env({str(index): "x" for index in range(70)})
    with pytest.raises(ToolValidationError):
        podman_module._normalize_env({"": "value"})
    with pytest.raises(ToolValidationError):
        podman_module._normalize_env({"k" * 100: "value"})
    with pytest.raises(ToolValidationError):
        podman_module._normalize_env({"KEY": "v" * 600})


def test_timeout_validation_rejects_nan() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._normalize_timeout(float("nan"))


def test_cwd_validation_guards_paths() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._normalize_cwd("/tmp")
    with pytest.raises(ToolValidationError):
        podman_module._normalize_cwd("/".join(str(index) for index in range(20)))
    with pytest.raises(ToolValidationError):
        podman_module._normalize_cwd("a/../b")
    with pytest.raises(ToolValidationError):
        podman_module._normalize_cwd("x" * 90)
    assert podman_module._normalize_cwd("   ") == "/workspace"


def test_truncate_stream_marks_output() -> None:
    truncated = podman_module._truncate_stream("a" * (35_000))
    assert truncated.endswith("[truncated]")


def test_workspace_reuse_between_calls(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [_ExecResponse(exit_code=0), _ExecResponse(exit_code=0)]
    )
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)), context=build_tool_context(bus, session)
    )
    handler(
        PodmanShellParams(command=("true",)), context=build_tool_context(bus, session)
    )

    assert len(client.containers._containers) == 1


def test_touch_workspace_handles_missing_handle(
    tmp_path: Path,
) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = _make_section(
        session=session, client=_FakePodmanClient(), cache_dir=tmp_path
    )
    section._touch_workspace()


def test_readiness_failure_raises(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    client.containers.set_readiness_exit_code(1)
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            PodmanShellParams(command=("true",)),
            context=build_tool_context(bus, session),
        )


def test_shell_execute_truncates_output(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    cli_runner = _FakeCliRunner(
        [
            _ExecResponse(
                exit_code=0,
                stdout="a" * 40_000,
                stderr="b" * 40_000,
            )
        ]
    )
    section = _make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    result = handler(
        PodmanShellParams(command=("true",)), context=build_tool_context(bus, session)
    )
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout.endswith("[truncated]")
    assert value.stderr.endswith("[truncated]")


def test_ls_lists_workspace_files(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    docs = handle.overlay_path / "docs"
    docs.mkdir(parents=True)
    (docs / "README.md").write_text("hello world", encoding="utf-8")

    tool = find_tool(section, "ls")
    handler = tool.handler
    assert handler is not None
    result = handler(
        ListDirectoryParams(path="docs"),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    entries = cast(tuple[FileInfo, ...], result.value)
    assert any(entry.path.segments == ("docs", "README.md") for entry in entries)


def test_read_file_returns_numbered_lines(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "notes.txt"
    target.write_text("first\nsecond\nthird", encoding="utf-8")

    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None
    result = handler(
        ReadFileParams(file_path="notes.txt", limit=2),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    read_result = cast(ReadFileResult, result.value)
    assert "1 | first" in read_result.content
    assert "2 | second" in read_result.content


def test_write_file_updates_snapshot(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    tool = find_tool(section, "write_file")
    handler = tool.handler
    assert handler is not None
    result = handler(
        WriteFileParams(file_path="src/app.py", content="print('hi')"),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    call = runner.calls[-1]
    assert "python3" in call


def test_edit_file_invokes_cli(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("hello", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None
    handler(
        EditFileParams(
            file_path="file.txt",
            old_string="hello",
            new_string="hi",
            replace_all=False,
        ),
        context=build_tool_context(bus, session),
    )
    call = runner.calls[-1]
    assert "overwrite" in call


def test_glob_matches_files(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    src = handle.overlay_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")
    (src / "README.md").write_text("details", encoding="utf-8")
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None
    result = handler(
        GlobParams(pattern="*.py", path="src"),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    matches = cast(tuple[GlobMatch, ...], result.value)
    assert any(match.path.segments == ("src", "main.py") for match in matches)


def test_grep_finds_pattern(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "log.txt"
    target.write_text("first\nmatch line\nlast", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None
    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    grep_matches = cast(tuple[GrepMatch, ...], result.value)
    assert any(match.line_number == 2 for match in grep_matches)


def test_rm_executes_cli(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "temp.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "rm")
    handler = tool.handler
    assert handler is not None
    handler(
        RemoveParams(path="temp.txt"),
        context=build_tool_context(bus, session),
    )
    call = runner.calls[-1]
    assert "python3" in call


def test_exec_runner_property_exposes_runner(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner()
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )

    assert section.exec_runner is runner


def test_container_path_for_root() -> None:
    assert podman_module._container_path_for(vfs_module.VfsPath(())) == "/workspace"


def test_assert_within_overlay_raises_for_outside(tmp_path: Path) -> None:
    root = tmp_path / "overlay"
    root.mkdir()
    outside = root.parent / "other"
    outside.mkdir()
    with pytest.raises(ToolValidationError):
        podman_module._assert_within_overlay(root, outside)


def test_assert_within_overlay_handles_missing_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "overlay"
    root.mkdir()
    missing = root / "missing" / "child"
    original = Path.resolve

    def _fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == missing:
            raise FileNotFoundError("missing")
        return original(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _fake_resolve)
    podman_module._assert_within_overlay(root, missing)


def test_compose_child_path_rejects_invalid_segment() -> None:
    base = vfs_module.VfsPath(("src",))
    assert podman_module._compose_child_path(base, "..") is None


def test_compose_relative_path_rejects_invalid_segments() -> None:
    base = vfs_module.VfsPath(("src",))
    relative = Path("..")
    assert podman_module._compose_relative_path(base, relative) is None


def test_iter_workspace_files_handles_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert list(podman_module._iter_workspace_files(missing)) == []


def test_format_read_message_handles_empty_slice() -> None:
    path = vfs_module.VfsPath(("docs", "notes.txt"))
    message = podman_module._format_read_message(path, 0, 0)
    assert "no lines" in message


def test_ls_rejects_file_path(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "ls")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ListDirectoryParams(path="file.txt"),
            context=build_tool_context(bus, session),
        )


def test_read_file_missing_path(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ReadFileParams(file_path="missing.txt"),
            context=build_tool_context(bus, session),
        )


def test_read_file_rejects_invalid_encoding(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "binary.bin"
    target.write_bytes(b"\xff")
    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ReadFileParams(file_path="binary.bin"),
            context=build_tool_context(bus, session),
        )


def test_write_file_rejects_existing_file(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "write_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            WriteFileParams(file_path="file.txt", content="other"),
            context=build_tool_context(bus, session),
        )


def test_edit_file_rejects_long_strings(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None
    long_text = "x" * (vfs_module.MAX_WRITE_LENGTH + 1)

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string=long_text,
                new_string="short",
            ),
            context=build_tool_context(bus, session),
        )

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="short",
                new_string=long_text,
            ),
            context=build_tool_context(bus, session),
        )


def test_edit_file_missing_path(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="a",
                new_string="b",
            ),
            context=build_tool_context(bus, session),
        )


def test_edit_file_rejects_invalid_encoding(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_bytes(b"\xff")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="a",
                new_string="b",
            ),
            context=build_tool_context(bus, session),
        )


def test_edit_file_requires_occurrence(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("hello", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="missing",
                new_string="new",
            ),
            context=build_tool_context(bus, session),
        )


def test_edit_file_requires_unique_match(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("foo foo", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="foo",
                new_string="bar",
            ),
            context=build_tool_context(bus, session),
        )


def test_edit_file_replace_all_branch(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner([_ExecResponse(exit_code=0)])
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("foo foo", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    handler(
        EditFileParams(
            file_path="file.txt",
            old_string="foo",
            new_string="bar",
            replace_all=True,
        ),
        context=build_tool_context(bus, session),
    )


def test_glob_rejects_empty_pattern(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            GlobParams(pattern="   ", path="/"),
            context=build_tool_context(bus, session),
        )


def test_glob_skips_invalid_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    outside = section.ensure_workspace().overlay_path.parent / "outside.txt"

    def _fake_iter(_: Path) -> Iterator[Path]:
        yield outside

    monkeypatch.setattr(podman_module, "_iter_workspace_files", _fake_iter)
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GlobParams(pattern="*", path="/"),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_glob_skips_invalid_composed_path(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")

    monkeypatch.setattr(
        podman_module,
        "_compose_relative_path",
        lambda *_: None,
    )
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GlobParams(pattern="*", path="/"),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_glob_skips_invalid_file_info(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")

    def _raise(self: object, *args: object, **kwargs: object) -> GlobMatch:
        raise ToolValidationError("boom")

    monkeypatch.setattr(podman_module._PodmanVfsSuite, "_build_glob_match", _raise)
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GlobParams(pattern="*", path="/"),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_glob_honors_result_limit(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "first.txt").write_text("a", encoding="utf-8")
    (handle.overlay_path / "second.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(podman_module, "_MAX_MATCH_RESULTS", 1)
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GlobParams(pattern="*.txt", path="/"),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    matches = cast(tuple[GlobMatch, ...], result.value)
    assert len(matches) == 1


def test_grep_rejects_invalid_regex(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="[", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert not result.success


def test_grep_honors_glob_argument(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.py").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob="*.py"),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches[0].path.segments == ("main.py",)


def test_grep_respects_glob_filter(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob="*.py"),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_skips_invalid_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    outside = section.ensure_workspace().overlay_path.parent / "outside.txt"

    def _fake_iter(_: Path) -> Iterator[Path]:
        yield outside

    monkeypatch.setattr(podman_module, "_iter_workspace_files", _fake_iter)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_skips_invalid_composed_path(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
    monkeypatch.setattr(
        podman_module,
        "_compose_relative_path",
        lambda *_: None,
    )
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_skips_overlay_errors(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.txt").write_text("match", encoding="utf-8")

    original = podman_module._assert_within_overlay
    call_count = {"value": 0}

    def _wrapped(root: Path, candidate: Path) -> None:
        call_count["value"] += 1
        if call_count["value"] > 1:
            raise ToolValidationError("boom")
        original(root, candidate)

    monkeypatch.setattr(podman_module, "_assert_within_overlay", _wrapped)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_skips_invalid_file_encoding(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "main.txt"
    target.write_text("match", encoding="utf-8")

    def _fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "main.txt"
    target.write_text("match", encoding="utf-8")

    def _fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value == ()


def test_grep_honors_result_limit(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "one.txt").write_text("match", encoding="utf-8")
    (handle.overlay_path / "two.txt").write_text("match", encoding="utf-8")
    monkeypatch.setattr(podman_module, "_MAX_MATCH_RESULTS", 1)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    grep_matches = cast(tuple[GrepMatch, ...], result.value)
    assert len(grep_matches) == 1


def test_remove_rejects_root_and_missing(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "rm")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="/"),
            context=build_tool_context(bus, session),
        )

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="missing.txt"),
            context=build_tool_context(bus, session),
        )

    original = vfs_module.normalize_string_path

    def _fake(path: str, *, field: str) -> vfs_module.VfsPath:
        if path == "trigger":
            return vfs_module.VfsPath(())
        return original(path, field=field)

    monkeypatch.setattr(vfs_module, "normalize_string_path", _fake)
    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="trigger"),
            context=build_tool_context(bus, session),
        )


def test_remove_handles_cli_errors(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "temp.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "rm")
    handler = tool.handler
    assert handler is not None

    def _raise(*_: object, **__: object) -> None:
        raise FileNotFoundError("podman")

    monkeypatch.setattr(section, "run_python_script", _raise)

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="temp.txt"),
            context=build_tool_context(bus, session),
        )

    class _Response:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(section, "run_python_script", lambda **_: _Response())

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="temp.txt"),
            context=build_tool_context(bus, session),
        )


def test_build_directory_entries_handles_missing_dir(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    result = suite._build_directory_entries(
        base=vfs_module.VfsPath(()),
        host_path=tmp_path / "missing",
        snapshot=vfs_module.VirtualFileSystem(),
        overlay_root=tmp_path,
    )
    assert result == []


def test_build_directory_entries_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    host = tmp_path / "dir"
    host.mkdir()

    def _raise(*_: object, **__: object) -> Iterator[Path]:
        raise OSError("boom")

    monkeypatch.setattr(Path, "iterdir", _raise)
    with pytest.raises(ToolValidationError):
        suite._build_directory_entries(
            base=vfs_module.VfsPath(()),
            host_path=host,
            snapshot=vfs_module.VirtualFileSystem(),
            overlay_root=tmp_path,
        )


def test_build_directory_entries_skips_invalid_entries(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    host = tmp_path / "dir"
    host.mkdir()
    (host / "file.txt").write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        podman_module,
        "_compose_child_path",
        lambda *_: None,
    )
    result = suite._build_directory_entries(
        base=vfs_module.VfsPath(()),
        host_path=host,
        snapshot=vfs_module.VirtualFileSystem(),
        overlay_root=tmp_path,
    )
    assert result == []


def test_build_directory_entries_handles_directories(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    host = tmp_path / "dir"
    host.mkdir()
    (host / "nested").mkdir()
    result = suite._build_directory_entries(
        base=vfs_module.VfsPath(()),
        host_path=host,
        snapshot=vfs_module.VirtualFileSystem(),
        overlay_root=tmp_path,
    )
    assert any(entry.kind == "directory" for entry in result)


def test_build_directory_entries_skips_failed_file_info(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    host = tmp_path / "dir"
    host.mkdir()
    (host / "file.txt").write_text("data", encoding="utf-8")

    def _raise(*_: object, **__: object) -> FileInfo:
        raise ToolValidationError("boom")

    monkeypatch.setattr(suite, "_build_file_info", _raise)
    result = suite._build_directory_entries(
        base=vfs_module.VfsPath(()),
        host_path=host,
        snapshot=vfs_module.VirtualFileSystem(),
        overlay_root=tmp_path,
    )
    assert result == []


def test_build_glob_match_returns_existing_metadata(tmp_path: Path) -> None:
    suite = podman_module._PodmanVfsSuite(
        section=_make_section(
            session=Session(bus=InProcessEventBus()),
            client=_FakePodmanClient(),
            cache_dir=tmp_path,
        )
    )
    path = vfs_module.VfsPath(("src", "main.py"))
    file = vfs_module.VfsFile(
        path=path,
        content="print()",
        encoding="utf-8",
        size_bytes=10,
        version=2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    snapshot = vfs_module.VirtualFileSystem(files=(file,))
    host = Path(__file__)
    result = suite._build_glob_match(
        target=path,
        host_path=host,
        snapshot=snapshot,
        overlay_root=host.parent,
    )
    assert result.version == file.version


def test_write_via_container_handles_cli_failures(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    suite = podman_module._PodmanVfsSuite(section=section)
    path = vfs_module.VfsPath(("file.txt",))

    def _raise(*_: object, **__: object) -> None:
        raise FileNotFoundError("podman")

    monkeypatch.setattr(section, "run_python_script", _raise)
    with pytest.raises(ToolValidationError):
        suite._write_via_container(path=path, content="data", mode="create")

    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(section, "run_python_script", lambda **_: _Failed())
    with pytest.raises(ToolValidationError):
        suite._write_via_container(path=path, content="data", mode="create")


def test_tools_module_missing_attr_raises() -> None:
    import weakincentives.tools as tools

    with pytest.raises(AttributeError):
        _ = tools.TOTALLY_UNKNOWN
