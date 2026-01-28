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

"""Workspace lifecycle tests for PodmanSandboxSection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import weakincentives.contrib.tools.podman as podman_module
import weakincentives.contrib.tools.podman_connection as podman_connection_module
from tests.tools.helpers import build_tool_context, find_tool
from tests.tools.podman_test_helpers import (
    ExecResponse,
    FakeCliRunner,
    FakeImageCollection,
    FakePodmanClient,
    make_section,
)
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    PodmanSandboxConfig,
    PodmanSandboxSection,
    PodmanShellParams,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


def test_section_exposes_new_client(
    session_and_dispatcher: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    assert section.new_client() is client


def test_close_stops_and_removes_container(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    handle = section._workspace_handle
    assert handle is not None
    container = client.containers.get(handle.descriptor.container_id)

    section.close()

    assert container.stop_calls == 1
    assert container.remove_calls == 1
    assert section._workspace_handle is None


def test_close_is_idempotent(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    handle = section._workspace_handle
    assert handle is not None
    container = client.containers.get(handle.descriptor.container_id)

    section.close()
    section.close()

    assert container.stop_calls == 1
    assert container.remove_calls == 1


def test_close_without_workspace(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    section.close()


def test_close_handles_client_factory_failure(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    def _raise() -> FakePodmanClient:
        raise RuntimeError("boom")

    section._client_factory = _raise

    section.close()


def test_close_handles_missing_container(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    class _BrokenContainers:
        def get(self, container_id: str) -> None:
            self.last_requested = container_id
            raise RuntimeError("missing container")

    class _BrokenClient:
        def __init__(self) -> None:
            self.containers = _BrokenContainers()
            self.images = FakeImageCollection()

        def close(self) -> None:
            self.closed = True

    section._client_factory = lambda: _BrokenClient()

    section.close()


def test_workspace_reuse_between_calls(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner(
        [ExecResponse(exit_code=0), ExecResponse(exit_code=0)]
    )
    section = make_section(
        session=session, client=client, cache_dir=tmp_path, runner=cli_runner
    )
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    assert len(client.containers._containers) == 1


def test_touch_workspace_handles_missing_handle(
    tmp_path: Path,
) -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    section = make_section(
        session=session, client=FakePodmanClient(), cache_dir=tmp_path
    )
    section._touch_workspace()


def test_readiness_failure_raises(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    client.containers.set_readiness_exit_code(1)
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "shell_execute")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            PodmanShellParams(command=("true",)),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_exec_runner_property_exposes_runner(
    session_and_dispatcher: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    runner = FakeCliRunner()
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )

    assert section.exec_runner is runner


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


def test_section_auto_resolves_connection(
    monkeypatch: pytest.MonkeyPatch,
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cli_runner = FakeCliRunner([ExecResponse(exit_code=0, stdout="auto")])
    resolved = podman_connection_module.PodmanConnectionInfo(
        base_url="ssh://detected",
        identity="/tmp/key",
        connection_name="auto-conn",
    )

    def _fake_resolve(
        *, preferred_name: str | None = None
    ) -> podman_connection_module.PodmanConnectionInfo | None:
        return resolved

    monkeypatch.setattr(
        podman_connection_module,
        "resolve_podman_connection",
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
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    assert section.connection_name == "auto-conn"


def test_connection_resolution_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODMAN_BASE_URL", "ssh://env")
    monkeypatch.setenv("PODMAN_IDENTITY", "/tmp/env")
    result = podman_connection_module.resolve_podman_connection(preferred_name="custom")
    assert result is not None
    assert result.base_url == "ssh://env"
    assert result.identity == "/tmp/env"
    assert result.connection_name == "custom"


def test_run_cli_cp_honors_connection_flag(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    runner = FakeCliRunner()
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
        connection_name="remote",
    )

    section.run_cli_cp(source="/tmp/src", destination="/tmp/dst")

    call = runner.calls[-1]
    assert call[:4] == ["podman", "--connection", "remote", "cp"]


def test_run_cli_cp_without_connection_flag(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    """Test branch 1241->1243: run_cli_cp without connection_name."""
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    runner = FakeCliRunner()
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
        connection_name=None,  # No connection name
    )

    section.run_cli_cp(source="/tmp/src", destination="/tmp/dst")

    call = runner.calls[-1]
    # Without connection_name, command should be "podman cp ..."
    assert call[:2] == ["podman", "cp"]
    assert "--connection" not in call


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

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fake_run)

    result = podman_connection_module.resolve_podman_connection(
        preferred_name="desired"
    )
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

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fake_run)

    result = podman_connection_module.resolve_podman_connection()
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

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fake_run)

    result = podman_connection_module.resolve_podman_connection()
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

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fail)

    assert podman_connection_module.resolve_podman_connection() is None


def test_connection_resolution_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODMAN_BASE_URL", raising=False)
    monkeypatch.delenv("PODMAN_IDENTITY", raising=False)
    monkeypatch.delenv("PODMAN_CONNECTION", raising=False)

    def _fake_run(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="not json")

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fake_run)

    assert podman_connection_module.resolve_podman_connection() is None


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

    monkeypatch.setattr(podman_connection_module.subprocess, "run", _fake_run)

    assert (
        podman_connection_module.resolve_podman_connection(preferred_name="missing")
        is None
    )


def test_resolve_connection_static_handles_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch where the function is used (podman module), not where it's defined
    monkeypatch.setattr(
        podman_module,
        "resolve_podman_connection",
        lambda *, preferred_name=None: None,
    )
    assert PodmanSandboxSection.resolve_connection() is None


def test_resolve_connection_static_returns_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = podman_connection_module.PodmanConnectionInfo(
        base_url="ssh://detected",
        identity="/tmp/key",
        connection_name="resolved",
    )
    # Patch where the function is used (podman module), not where it's defined
    monkeypatch.setattr(
        podman_module,
        "resolve_podman_connection",
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
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    monkeypatch.setattr(
        podman_connection_module,
        "resolve_podman_connection",
        lambda *, preferred_name=None: None,
    )
    with pytest.raises(ToolValidationError):
        PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(
                cache_dir=tmp_path,
                client_factory=lambda: client,
                base_environment={"PATH": "/usr/bin"},
                exec_runner=FakeCliRunner(),
            ),
        )


def test_section_init_with_overlay_path_and_filesystem(
    session_and_dispatcher: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    """Test that PodmanSandboxSection accepts _overlay_path and _filesystem for cloning."""
    session, _dispatcher = session_and_dispatcher
    client = FakePodmanClient()
    cache_dir = tmp_path / "cache"

    # Create initial section
    section1 = make_section(session=session, client=client, cache_dir=cache_dir)

    # Write a file through the filesystem
    fs = section1.filesystem
    _ = fs.write("test_file.txt", "test content")

    # Create new session for clone
    new_dispatcher = InProcessDispatcher()
    new_session = Session(dispatcher=new_dispatcher)

    # Create a new section with the preserved overlay path and filesystem
    # Must provide base_url to avoid Podman connection resolution
    section2 = PodmanSandboxSection(
        session=new_session,
        config=PodmanSandboxConfig(
            cache_dir=cache_dir,
            client_factory=lambda: client,
            exec_runner=FakeCliRunner(),
            base_url="unix:///tmp/fake.sock",
        ),
        _overlay_path=section1._overlay_path,
        _filesystem=section1._filesystem,
    )

    # Filesystem instance should be the same
    assert section2.filesystem is section1.filesystem

    # Content should be accessible
    result = section2.filesystem.read("test_file.txt")
    assert result.content == "test content"

    # Overlay path should be the same
    assert section2._overlay_path == section1._overlay_path
