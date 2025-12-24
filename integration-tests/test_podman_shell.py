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

"""End-to-end smoke tests for the Podman shell tool."""

# pyright: reportInvalidTypeForm=false, reportArgumentType=false

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import cast

import pytest

from tests.tools.helpers import build_tool_context, find_tool
from weakincentives.contrib.tools import (
    EvalParams,
    EvalResult,
    PodmanSandboxConfig,
    PodmanSandboxSection,
    PodmanShellParams,
    PodmanShellResult,
    ReadFileParams,
    ReadFileResult,
    WriteFileParams,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


@pytest.mark.integration
@pytest.mark.podman
def test_shell_execute_creates_files(tmp_path: Path) -> None:
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        pytest.skip("Podman integration requires a running podman machine.")
    assert connection is not None
    assert connection is not None
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    connection_name = connection.get("connection_name")
    section = PodmanSandboxSection(
        session=session,
        config=PodmanSandboxConfig(cache_dir=tmp_path),
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    container_name: str | None = None
    try:
        params = PodmanShellParams(
            command=("sh", "-c", "echo 'hello' > test.txt && cat test.txt"),
        )
        result = handler(params, context=build_tool_context(session))

        assert result.success
        assert result.value is not None
        value = cast(PodmanShellResult, result.value)
        assert "hello" in value.stdout
        handle = section._workspace_handle
        assert handle is not None
        container_name = handle.descriptor.container_name
    finally:
        section.close()
        if container_name:
            _wait_for_container_removal(container_name, connection_name)


@pytest.mark.integration
@pytest.mark.podman
def test_podman_vfs_round_trip(tmp_path: Path) -> None:
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        pytest.skip("Podman integration requires a running podman machine.")
    assert connection is not None
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    section = PodmanSandboxSection(
        session=session, config=PodmanSandboxConfig(cache_dir=tmp_path)
    )
    try:
        write_tool = find_tool(section, "write_file")
        read_tool = find_tool(section, "read_file")
        write_handler = write_tool.handler
        read_handler = read_tool.handler
        assert write_handler is not None
        assert read_handler is not None

        write_handler(
            WriteFileParams(file_path="notes.txt", content="hello world"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )
        result = read_handler(
            ReadFileParams(file_path="notes.txt"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )
        assert result.success
        assert result.value is not None
        read_value = cast(ReadFileResult, result.value)
        assert "hello world" in read_value.content
        # With the new architecture, filesystem operations work directly on
        # the overlay directory without starting a container
    finally:
        section.close()


@pytest.mark.integration
@pytest.mark.podman
def test_evaluate_python_writes_file(tmp_path: Path) -> None:
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        pytest.skip("Podman integration requires a running podman machine.")
    assert connection is not None
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    connection_name = connection.get("connection_name")
    section = PodmanSandboxSection(
        session=session, config=PodmanSandboxConfig(cache_dir=tmp_path)
    )
    container_name: str | None = None
    try:
        eval_tool = find_tool(section, "evaluate_python")
        eval_handler = eval_tool.handler
        assert eval_handler is not None

        params = EvalParams(code="print('integration-ok')")
        result = eval_handler(params, context=build_tool_context(session))

        assert result.success
        payload = cast(EvalResult, result.value)
        assert payload is not None
        assert payload.stdout.strip() == "integration-ok"
        assert payload.stderr == ""
        assert payload.value_repr is None
        handle = section._workspace_handle
        assert handle is not None
        container_name = handle.descriptor.container_name
    finally:
        section.close()
        if container_name:
            _wait_for_container_removal(container_name, connection_name)


@pytest.mark.integration
@pytest.mark.podman
def test_podman_container_has_no_network(tmp_path: Path) -> None:
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        pytest.skip("Podman integration requires a running podman machine.")
    assert connection is not None
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    connection_name = connection.get("connection_name")
    section = PodmanSandboxSection(
        session=session, config=PodmanSandboxConfig(cache_dir=tmp_path)
    )
    container_name: str | None = None
    try:
        tool = find_tool(section, "shell_execute")
        handler = tool.handler
        assert handler is not None

        script = (
            "import socket, sys\n"
            "addr = ('93.184.216.34', 80)\n"
            "try:\n"
            "    socket.create_connection(addr, 2)\n"
            "except OSError:\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        params = PodmanShellParams(command=("python3", "-c", script))
        result = handler(params, context=build_tool_context(session))

        assert result.success
        assert result.value is not None
        value = cast(PodmanShellResult, result.value)
        assert value.exit_code == 0
        handle = section._workspace_handle
        assert handle is not None
        container_name = handle.descriptor.container_name
        assert _container_network_mode(container_name, connection_name) == "none"
    finally:
        section.close()
        if container_name:
            _wait_for_container_removal(container_name, connection_name)


def _wait_for_container_removal(
    container_name: str, connection_name: str | None, timeout: float = 10.0
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _container_exists(container_name, connection_name):
            return
        time.sleep(0.5)
    raise AssertionError(f"Container {container_name} still exists after cleanup.")


def _container_exists(container_name: str, connection_name: str | None) -> bool:
    cmd = ["podman"]
    if connection_name:
        cmd.extend(["--connection", connection_name])
    cmd.extend(["container", "exists", container_name])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _container_network_mode(container_name: str, connection_name: str | None) -> str:
    cmd = ["podman"]
    if connection_name:
        cmd.extend(["--connection", connection_name])
    cmd.extend(
        [
            "container",
            "inspect",
            "--format",
            "{{.HostConfig.NetworkMode}}",
            container_name,
        ]
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()
