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

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest

from tests.tools.helpers import build_tool_context, find_tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import (
    PodmanShellParams,
    PodmanShellResult,
    PodmanToolsSection,
)


@pytest.mark.integration
@pytest.mark.podman
def test_shell_execute_creates_files(tmp_path: Path) -> None:
    connection = _resolve_podman_connection()
    if connection is None:
        pytest.skip("Podman integration requires a running podman machine.")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    assert connection is not None  # narrow for type-checkers
    base_url = connection["base_url"]
    identity = connection["identity"]
    assert base_url is not None and identity is not None
    section = PodmanToolsSection(
        session=session,
        cache_dir=tmp_path,
        base_url=base_url,
        identity=identity,
        connection_name=connection.get("connection_name"),
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    params = PodmanShellParams(
        command=("sh", "-c", "echo 'hello' > test.txt && cat test.txt"),
    )
    result = handler(params, context=build_tool_context(bus, session))

    assert result.success
    assert result.value is not None
    value = cast(PodmanShellResult, result.value)
    assert "hello" in value.stdout


def _resolve_podman_connection() -> dict[str, str | None] | None:
    env_base_url = os.environ.get("PODMAN_BASE_URL")
    env_identity = os.environ.get("PODMAN_IDENTITY")
    if env_base_url and env_identity:
        return {
            "base_url": env_base_url,
            "identity": env_identity,
            "connection_name": os.environ.get("PODMAN_CONNECTION"),
        }

    try:
        result = subprocess.run(
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

    for connection in connections:
        if connection.get("Default"):
            return {
                "base_url": connection.get("URI"),
                "identity": connection.get("Identity"),
                "connection_name": connection.get("Name"),
            }
    return None
