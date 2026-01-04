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

"""Shell execution tests for PodmanSandboxSection."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess
from types import MethodType
from typing import cast

import pytest

import weakincentives.contrib.tools.podman as podman_module
import weakincentives.contrib.tools.podman_eval as podman_eval_module
import weakincentives.contrib.tools.vfs as vfs_module
from tests.tools.helpers import build_tool_context, find_tool, invoke_tool
from tests.tools.podman_test_helpers import (
    ExecResponse,
    FakeCliRunner,
    FakePodmanClient,
    make_section,
)
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
    PodmanSandboxSection,
    PodmanShellParams,
    PodmanShellResult,
    PodmanWorkspace,
)
from weakincentives.prompt.tool import Tool
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


def test_section_registers_shell_tool(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "shell_execute")
    assert tool.description.startswith("Run a short command")


def test_section_registers_eval_tool(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "evaluate_python")
    assert tool.description.startswith("Run a short Python")


def test_truncate_eval_stream_limits_length() -> None:
    short = podman_eval_module.truncate_eval_stream("hello")
    assert short == "hello"
    long_text = "x" * (podman_eval_module._EVAL_MAX_STREAM_LENGTH + 5)
    truncated = podman_eval_module.truncate_eval_stream(long_text)
    assert truncated.endswith("...")
    assert len(truncated) == podman_eval_module._EVAL_MAX_STREAM_LENGTH


def test_shell_execute_runs_commands_and_stores_workspace(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [ExecResponse(exit_code=0, stdout="hello world\n", stderr="")]
    )
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "hello world"))
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )

    assert isinstance(result.value, PodmanShellResult)
    assert result.value.exit_code == 0
    assert "hello world" in result.value.stdout

    workspace = session[PodmanWorkspace].all()
    assert workspace
    assert workspace[-1].image == "python:3.12-bookworm"
    handle = section._workspace_handle
    assert handle is not None
    assert cli_runner.calls[-1][-2:] == ["echo", "hello world"]


def test_shell_execute_validates_command(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None
    params = PodmanShellParams(command=())

    with pytest.raises(ToolValidationError):
        handler(
            params, context=build_tool_context(session, filesystem=section.filesystem)
        )


def test_shell_execute_merges_environment(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("printenv",), env={"custom": "value"})
    handler(params, context=build_tool_context(session, filesystem=section.filesystem))

    call = " ".join(cli_runner.calls[-1])
    assert "PATH=/usr/bin" in call
    assert "CUSTOM=value" in call


def test_shell_execute_respects_capture_flag(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [ExecResponse(exit_code=0, stdout="x" * 40_000, stderr="")]
    )
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("cat",), capture_output=False)
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "capture disabled"
    assert cli_runner.kwargs[-1]["capture_output"] is False


def test_shell_execute_captures_output_by_default(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [ExecResponse(exit_code=0, stdout="normal output", stderr="")]
    )
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("echo", "hi"))
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "normal output"
    assert cli_runner.kwargs[-1]["capture_output"] is True


def test_shell_execute_normalizes_cwd(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("pwd",), cwd="src/docs")
    handler(params, context=build_tool_context(session, filesystem=section.filesystem))

    call = cli_runner.calls[-1]
    idx = call.index("--workdir")
    assert call[idx + 1] == "/workspace/src/docs"


def test_shell_execute_rejects_non_ascii_stdin(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(command=("cat",), stdin="ümlaut")
    with pytest.raises(ToolValidationError):
        handler(
            params, context=build_tool_context(session, filesystem=section.filesystem)
        )


def test_shell_execute_rejects_mismatched_session(tmp_path: Path) -> None:
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    other_bus = InProcessDispatcher()
    other_session = Session(bus=other_bus)
    client = FakePodmanClient()
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=FakeCliRunner()
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(RuntimeError, match="session does not match"):
        handler(
            PodmanShellParams(command=("true",)),
            context=build_tool_context(other_session),
        )


def test_shell_execute_cli_fallback(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0, stdout="cli output")])
    section = make_section(
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
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )

    call = cli_runner.calls[-1]
    assert call[:3] == ["podman", "--connection", "podman-machine-default"]
    assert "--interactive" in call
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "cli output"


def test_shell_execute_cli_capture_disabled(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [ExecResponse(exit_code=0, stdout="cli output", stderr="cli err")]
    )
    section = make_section(
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
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.stdout == "capture disabled"
    assert value.stderr == "capture disabled"


def test_shell_execute_cli_timeout(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()

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

    section = make_section(
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
    result = handler(
        params, context=build_tool_context(session, filesystem=section.filesystem)
    )
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert value.timed_out
    assert value.stdout == "partial"
    assert value.stderr == "error"


def test_shell_execute_cli_missing_binary(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()

    def _missing_runner(
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002
        text: bool | None = None,
        capture_output: bool | None = None,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise FileNotFoundError("podman not found")

    section = make_section(
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
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_shell_execute_truncates_output(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [
            ExecResponse(
                exit_code=0,
                stdout="a" * 40_000,
                stderr="b" * 40_000,
            )
        ]
    )
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    result = handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
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


def test_run_python_script_delegates_to_run_cli_exec(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    _ = section.ensure_workspace()

    result = section.run_python_script(
        script="print('hello')",
        args=["arg1", "arg2"],
        timeout=30.0,
    )

    assert result.returncode == 0
    # Verify the command was constructed correctly
    call = runner.calls[-1]
    assert "python3" in call
    assert "-c" in call
    assert "print('hello')" in call
    assert "arg1" in call
    assert "arg2" in call


def test_evaluate_python_runs_script_passthrough(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
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

    result = invoke_tool(tool, EvalParams(code="print('ok')"), session=session)

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
    assert captured["timeout"] == podman_eval_module._EVAL_TIMEOUT_SECONDS


def test_evaluate_python_accepts_large_scripts(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
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

    result = invoke_tool(tool, EvalParams(code=code), session=session)

    assert result.success
    assert captured["script"] == code
    assert len(code) > 2_000


def test_evaluate_python_rejects_control_characters(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    with pytest.raises(ToolValidationError, match="unsupported control characters"):
        invoke_tool(
            tool,
            EvalParams(code="print('ok')\x01"),
            session=session,
        )


def test_evaluate_python_marks_failure_on_nonzero_exit(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
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

    result = invoke_tool(tool, EvalParams(code="fail"), session=session)

    assert not result.success
    assert result.message == "Evaluation failed (exit code 2)."
    payload = cast(EvalResult, result.value)
    assert payload.stderr == "boom"


def test_evaluate_python_reports_timeout(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
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

    result = invoke_tool(tool, EvalParams(code="while True: pass"), session=session)

    assert not result.success
    assert result.message == "Evaluation timed out."
    payload = cast(EvalResult, result.value)
    assert payload.stderr == "Execution timed out."


def test_evaluate_python_missing_cli_raises(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    def _fail(*_: object, **__: object) -> CompletedProcess[str]:
        raise FileNotFoundError("missing podman")

    monkeypatch.setattr(section, "run_python_script", _fail)

    with pytest.raises(ToolValidationError, match="Podman CLI is required"):
        invoke_tool(tool, EvalParams(code="0"), session=session)


def test_evaluate_python_truncates_streams(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
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

    result = invoke_tool(tool, EvalParams(code="0"), session=session)

    payload = cast(EvalResult, result.value)
    assert payload.stdout.endswith("...")
    assert payload.stderr.endswith("...")


def test_evaluate_python_rejects_reads_writes_and_globals(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    read = EvalFileRead(path=vfs_module.VfsPath(("docs", "notes.txt")))
    write = EvalFileWrite(
        path=vfs_module.VfsPath(("reports", "out.txt")),
        content="value",
        mode="create",
    )

    with pytest.raises(ToolValidationError, match="reads are not supported"):
        invoke_tool(
            tool,
            EvalParams(code="0", reads=(read,)),
            session=session,
        )

    with pytest.raises(ToolValidationError, match="writes are not supported"):
        invoke_tool(
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
        )

    with pytest.raises(ToolValidationError, match="globals are not supported"):
        invoke_tool(
            tool,
            EvalParams(code="0", globals={"value": "1"}),
            session=session,
        )
