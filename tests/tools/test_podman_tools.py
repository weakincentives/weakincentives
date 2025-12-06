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
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from types import MethodType, SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

import weakincentives.tools.podman as podman_module
import weakincentives.tools.vfs as vfs_module
from tests.tools.helpers import build_tool_context, find_tool, invoke_tool
from weakincentives.prompt.tool import Tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import (
    EditFileParams,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    PodmanSandboxConfig,
    PodmanSandboxSection,
    PodmanShellParams,
    PodmanShellResult,
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
) -> PodmanSandboxSection:
    exec_runner = runner if runner is not None else _FakeCliRunner()
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


def _prepare_resolved_mount(
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


def test_truncate_eval_stream_limits_length() -> None:
    short = podman_module._truncate_eval_stream("hello")
    assert short == "hello"
    long_text = "x" * (podman_module._EVAL_MAX_STREAM_LENGTH + 5)
    truncated = podman_module._truncate_eval_stream(long_text)
    assert truncated.endswith("...")
    assert len(truncated) == podman_module._EVAL_MAX_STREAM_LENGTH


def test_section_registers_eval_tool(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "evaluate_python")
    assert tool.description.startswith("Run a short Python")


def test_host_mount_snapshot_starts_empty(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    host_root, mount, _file_path = _setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )

    snapshot = section.latest_snapshot()

    assert snapshot.files == ()


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


def test_host_mount_resolver_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ToolValidationError):
        podman_module._resolve_single_host_mount(
            HostMount(host_path=""),
            (tmp_path,),
        )


def test_resolve_host_path_requires_allowed_roots() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._resolve_host_path("docs", ())


def test_resolve_host_path_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ToolValidationError):
        podman_module._resolve_host_path("../outside", (root,))


def test_normalize_mount_globs_discards_empty_entries() -> None:
    result = podman_module._normalize_mount_globs(
        (" *.py ", " ", "*.md"),
        "include_glob",
    )
    assert result == ("*.py", "*.md")


def test_preview_mount_entries_handles_file(tmp_path: Path) -> None:
    file_path = tmp_path / "item.txt"
    file_path.write_text("payload", encoding="utf-8")
    result = podman_module._preview_mount_entries(file_path)
    assert result == ("item.txt",)


def test_preview_mount_entries_raises_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "items"
    directory.mkdir()
    original_iterdir = Path.iterdir

    def _raise(self: Path) -> Iterator[Path]:
        if self == directory:
            raise OSError("boom")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _raise)
    with pytest.raises(ToolValidationError):
        podman_module._preview_mount_entries(directory)


def test_iter_host_mount_files_handles_file(tmp_path: Path) -> None:
    file_path = tmp_path / "item.txt"
    file_path.write_text("payload", encoding="utf-8")
    entries = tuple(podman_module._iter_host_mount_files(file_path, False))
    assert entries == (file_path,)


def test_host_mount_allows_binary_files(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    host_root = tmp_path / "host-root"
    repo = host_root / "sunfish"
    repo.mkdir(parents=True, exist_ok=True)
    file_path = repo / "payload.bin"
    payload = b"\x00\xffbinary\x01"
    file_path.write_bytes(payload)
    cache_dir = tmp_path / "cache"
    section = _make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(
            HostMount(host_path="sunfish", mount_path=vfs_module.VfsPath(("sunfish",))),
        ),
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
    assert mounted.read_bytes() == payload


def test_copy_mount_respects_include_glob(tmp_path: Path) -> None:
    resolved, file_path, overlay = _prepare_resolved_mount(
        tmp_path, include_glob=("*.py",)
    )
    podman_module.PodmanSandboxSection._copy_mount_into_overlay(
        overlay=overlay,
        mount=resolved,
    )
    target = overlay / "sunfish" / file_path.name
    assert not target.exists()


def test_copy_mount_respects_exclude_glob(tmp_path: Path) -> None:
    resolved, file_path, overlay = _prepare_resolved_mount(
        tmp_path, exclude_glob=("*.txt",)
    )
    podman_module.PodmanSandboxSection._copy_mount_into_overlay(
        overlay=overlay,
        mount=resolved,
    )
    target = overlay / "sunfish" / file_path.name
    assert not target.exists()


def test_copy_mount_stat_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved, file_path, overlay = _prepare_resolved_mount(tmp_path)
    original_stat = Path.stat

    def _raise(self: Path) -> os.stat_result:
        if self == file_path:
            raise OSError("boom")
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", _raise)
    with pytest.raises(ToolValidationError):
        podman_module.PodmanSandboxSection._copy_mount_into_overlay(
            overlay=overlay,
            mount=resolved,
        )


def test_copy_mount_max_bytes_guard(tmp_path: Path) -> None:
    resolved, _file_path, overlay = _prepare_resolved_mount(tmp_path, max_bytes=1)
    with pytest.raises(ToolValidationError):
        podman_module.PodmanSandboxSection._copy_mount_into_overlay(
            overlay=overlay,
            mount=resolved,
        )


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
    original_copy = podman_module.shutil.copy2

    def _fail_on_target(
        src: Path,
        dst: Path,
        *,
        follow_symlinks: bool = True,
    ) -> object:
        if dst == target:
            raise OSError("boom")
        return original_copy(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(podman_module.shutil, "copy2", _fail_on_target)

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
            self.last_requested = container_id
            raise RuntimeError("missing container")

    class _BrokenClient:
        def __init__(self) -> None:
            self.containers = _BrokenContainers()
            self.images = _FakeImageCollection()

        def close(self) -> None:
            self.closed = True

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

    section = PodmanSandboxSection(
        session=session,
        config=PodmanSandboxConfig(
            cache_dir=tmp_path,
            client_factory=lambda: client,
            base_environment={"PATH": "/usr/bin"},
            exec_runner=cli_runner,
        ),
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


def test_run_cli_cp_honors_connection_flag(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    runner = _FakeCliRunner()
    section = _make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
        connection_name="remote",
    )

    section.run_cli_cp(source="/tmp/src", destination="/tmp/dst")

    call = runner.calls[-1]
    assert call[:4] == ["podman", "--connection", "remote", "cp"]


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


def test_connection_resolution_prefers_default_entry(
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
            "Name": "preferred",
            "URI": "ssh://preferred",
            "Identity": "/tmp/preferred",
            "Default": True,
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
    assert result.base_url == "ssh://preferred"
    assert result.connection_name == "preferred"


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
    assert PodmanSandboxSection.resolve_connection() is None


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
    result = PodmanSandboxSection.resolve_connection(connection_name="preferred")
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
        PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(
                cache_dir=tmp_path,
                client_factory=lambda: client,
                base_environment={"PATH": "/usr/bin"},
                exec_runner=_FakeCliRunner(),
            ),
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
            self.was_closed = True

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
        check: bool = False,
    ) -> CompletedProcess[str]:
        recorded["args"] = args
        recorded["input"] = input
        recorded["text"] = text
        recorded["capture_output"] = capture_output
        recorded["timeout"] = timeout
        recorded["check"] = check
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
            self.closed = True

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


def test_podman_shell_result_renders_human_readable() -> None:
    result = PodmanShellResult(
        command=("echo", "hello"),
        cwd="/workspace",
        exit_code=0,
        stdout="hello",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )

    rendered = result.render()

    assert "echo hello" in rendered
    assert "Exit code: 0" in rendered
    assert "STDOUT:" in rendered
    assert "hello" in rendered
    assert "STDERR:" in rendered


def test_evaluate_python_runs_script_passthrough(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    captured: dict[str, object] = {}

    def _run_script(
        self: PodmanSandboxSection,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        captured["script"] = script
        captured["args"] = tuple(args)
        captured["timeout"] = timeout
        return CompletedProcess(
            ["python3"],
            0,
            stdout="hello",
            stderr="",
        )

    monkeypatch.setattr(
        section,
        "run_python_script",
        MethodType(_run_script, section),
        raising=False,
    )

    result = invoke_tool(bus, tool, EvalParams(code="print('ok')"), session=session)

    assert result.success
    assert result.message == "Evaluation succeeded (exit code 0)."
    payload = cast(EvalResult, result.value)
    assert payload.stdout == "hello"
    assert payload.stderr == ""
    assert payload.value_repr is None
    assert payload.reads == ()
    assert payload.writes == ()
    assert payload.globals == {}
    assert captured["script"] == "print('ok')"
    assert captured["args"] == ()
    assert captured["timeout"] == podman_module._EVAL_TIMEOUT_SECONDS


def test_evaluate_python_accepts_large_scripts(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))
    code = "\n".join("print('line')" for _ in range(600))

    captured: dict[str, object] = {}

    def _run_script(
        self: PodmanSandboxSection,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        captured["script"] = script
        captured["args"] = tuple(args)
        captured["timeout"] = timeout
        return CompletedProcess(["python3"], 0, stdout="", stderr="")

    monkeypatch.setattr(
        section,
        "run_python_script",
        MethodType(_run_script, section),
        raising=False,
    )

    result = invoke_tool(bus, tool, EvalParams(code=code), session=session)

    assert result.success
    assert captured["script"] == code
    assert len(code) > 2_000


def test_evaluate_python_rejects_control_characters(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    with pytest.raises(ToolValidationError, match="unsupported control characters"):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="print('ok')\x01"),
            session=session,
        )


def test_evaluate_python_marks_failure_on_nonzero_exit(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    def _run_script(
        self: PodmanSandboxSection,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del self, script, args, timeout
        return CompletedProcess(["python3"], 2, stdout="", stderr="boom")

    monkeypatch.setattr(
        section,
        "run_python_script",
        MethodType(_run_script, section),
        raising=False,
    )

    result = invoke_tool(bus, tool, EvalParams(code="fail"), session=session)

    assert not result.success
    assert result.message == "Evaluation failed (exit code 2)."
    payload = cast(EvalResult, result.value)
    assert payload.stderr == "boom"


def test_evaluate_python_reports_timeout(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    def _raise_timeout(
        self: PodmanSandboxSection,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del self, script, args
        raise subprocess.TimeoutExpired(cmd="python3", timeout=timeout or 0.0)

    monkeypatch.setattr(
        section,
        "run_python_script",
        MethodType(_raise_timeout, section),
        raising=False,
    )

    result = invoke_tool(
        bus, tool, EvalParams(code="while True: pass"), session=session
    )

    assert not result.success
    assert result.message == "Evaluation timed out."
    payload = cast(EvalResult, result.value)
    assert payload.stderr == "Execution timed out."


def test_evaluate_python_missing_cli_raises(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    def _fail(*_: object, **__: object) -> CompletedProcess[str]:
        raise FileNotFoundError("missing podman")

    monkeypatch.setattr(section, "run_python_script", _fail)

    with pytest.raises(ToolValidationError, match="Podman CLI is required"):
        invoke_tool(bus, tool, EvalParams(code="0"), session=session)


def test_evaluate_python_truncates_streams(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    def _run_script(
        self: PodmanSandboxSection,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del self, script, args, timeout
        return CompletedProcess(
            ["python3"],
            0,
            stdout="o" * 5_000,
            stderr="e" * 5_000,
        )

    monkeypatch.setattr(
        section,
        "run_python_script",
        MethodType(_run_script, section),
        raising=False,
    )

    result = invoke_tool(bus, tool, EvalParams(code="0"), session=session)

    payload = cast(EvalResult, result.value)
    assert payload.stdout.endswith("...")
    assert payload.stderr.endswith("...")


def test_evaluate_python_rejects_reads_writes_and_globals(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    read = EvalFileRead(path=vfs_module.VfsPath(("docs", "notes.txt")))
    write = EvalFileWrite(
        path=vfs_module.VfsPath(("reports", "out.txt")),
        content="value",
        mode="create",
    )

    with pytest.raises(ToolValidationError, match="reads are not supported"):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", reads=(read,)),
            session=session,
        )

    with pytest.raises(ToolValidationError, match="writes are not supported"):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
        )

    with pytest.raises(ToolValidationError, match="globals are not supported"):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", globals={"value": "1"}),
            session=session,
        )


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
    assert "cp" in call


def test_write_via_container_appends_existing_content(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "script.sh"
    target.write_text("base", encoding="utf-8")
    exec_calls: list[list[str]] = []
    cp_payloads: list[str] = []

    def _fake_exec(
        self: PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        del self
        _ = (
            config.stdin,
            config.environment,
            config.timeout,
            config.capture_output,
        )
        exec_calls.append(list(config.command))
        return CompletedProcess(config.command, 0, stdout="", stderr="")

    def _fake_cp(
        self: PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del destination, timeout
        cp_payloads.append(Path(source).read_text(encoding="utf-8"))
        return CompletedProcess(["podman", "cp"], 0, stdout="", stderr="")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_fake_exec, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_fake_cp, section))

    section.write_via_container(
        path=vfs_module.VfsPath(("script.sh",)),
        content="+",
        mode="append",
    )

    assert exec_calls[-1] == ["mkdir", "-p", "/workspace"]
    assert cp_payloads[-1] == "base+"


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
    assert "cp" in call


def test_write_via_container_reports_mkdir_failure(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    section.ensure_workspace()
    cp_calls = 0

    def _fail_exec(
        self: PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        del self
        _ = (
            config.stdin,
            config.environment,
            config.timeout,
            config.capture_output,
        )
        return CompletedProcess(list(config.command), 1, stdout="", stderr="boom")

    def _fake_cp(
        self: PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        nonlocal cp_calls
        cp_calls += 1
        return CompletedProcess(["podman", "cp"], 0, stdout="", stderr="")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_fail_exec, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_fake_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("nested", "file.txt")),
            content="data",
            mode="create",
        )

    assert cp_calls == 0


def test_write_via_container_rejects_non_utf8_append(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "data.bin"
    target.write_bytes(b"\xff\xff")

    def _unexpected(
        self: PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_exec should not be called")

    def _unexpected_cp(
        self: PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_cp should not be called")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_unexpected, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_unexpected_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("data.bin",)),
            content="payload",
            mode="append",
        )


def test_write_via_container_propagates_read_oserror(
    session_and_bus: tuple[Session, InProcessEventBus],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "data.txt"
    target.write_text("payload", encoding="utf-8")
    original_read_text = Path.read_text

    def _raise(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self == target:
            raise OSError("boom")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", _raise)

    def _unexpected(
        self: PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_exec should not be called")

    def _unexpected_cp(
        self: PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_cp should not be called")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_unexpected, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_unexpected_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("data.txt",)),
            content="payload",
            mode="append",
        )


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
    assert result.value is None
    assert "Invalid regular expression" in result.message


def test_grep_honors_glob_argument(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.py").write_text("match", encoding="utf-8")
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
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
    assert len(matches) == 1


def test_grep_supports_default_path(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path=None, glob=None),
        context=build_tool_context(bus, session),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("main.txt",)), line_number=1, line="match"),
    )


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


def test_grep_ignores_blank_glob(
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
        GrepParams(pattern="match", path="/", glob="  "),
        context=build_tool_context(bus, session),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("notes.txt",)), line_number=1, line="match"),
    )


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

    def _fake_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        raise UnicodeDecodeError(encoding or "utf-8", b"x", 0, 1, "bad")

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
    secondary = handle.overlay_path / "other.txt"
    secondary.write_text("match", encoding="utf-8")
    original = Path.read_text

    def _fake_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self == target:
            raise OSError("boom")
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )
    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("other.txt",)), line_number=1, line="match"),
    )


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


def test_grep_skips_binary_and_collects_match(
    session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
) -> None:
    session, bus = session_and_bus
    client = _FakePodmanClient()
    section = _make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "binary.bin").write_bytes(b"\xff\xfe\x00")
    (handle.overlay_path / "valid.txt").write_text("line with match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(bus, session),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(
            path=vfs_module.VfsPath(("valid.txt",)),
            line_number=1,
            line="line with match",
        ),
    )


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
    path = vfs_module.VfsPath(("file.txt",))

    def _raise(*_: object, **__: object) -> None:
        raise FileNotFoundError("podman")

    monkeypatch.setattr(section, "run_cli_cp", _raise)
    with pytest.raises(ToolValidationError):
        section.write_via_container(path=path, content="data", mode="create")

    failure = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(section, "run_cli_cp", lambda **__: failure)
    with pytest.raises(ToolValidationError):
        section.write_via_container(path=path, content="data", mode="create")


def test_tools_module_missing_attr_raises() -> None:
    from weakincentives import tools

    with pytest.raises(AttributeError):
        _ = tools.TOTALLY_UNKNOWN
