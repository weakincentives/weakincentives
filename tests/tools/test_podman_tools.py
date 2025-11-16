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

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

import weakincentives.tools.podman as podman_module
from tests.tools.helpers import build_tool_context, find_tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import (
    PodmanShellParams,
    PodmanShellResult,
    PodmanToolsSection,
    PodmanWorkspace,
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
) -> PodmanToolsSection:
    return PodmanToolsSection(
        session=session,
        client_factory=lambda: client,
        cache_dir=cache_dir,
        base_environment={"PATH": "/usr/bin"},
        connection_name=connection_name,
        exec_runner=runner,
    )


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


def test_client_factory_creates_client() -> None:
    factory = podman_module._build_client_factory(base_url=None, identity=None)
    client = factory()
    try:
        assert hasattr(client, "containers")
    finally:
        client.close()


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
