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

"""Tests for the wink CLI entry point."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from weakincentives import cli
from weakincentives.cli import wink
from weakincentives.cli.config import MCPServerConfig
from weakincentives.cli.wink_overrides import (
    OverridesInspectionError,
    WinkOverridesError,
)
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.runtime.logging import StructuredLogger


def test_cli_namespace_lists_wink_module() -> None:
    assert "wink" in cli.__dir__()


def test_main_dispatches_to_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_configure_logging(
        *, level: object, json_mode: object, env: object
    ) -> None:
        calls["configure"] = {"level": level, "json_mode": json_mode, "env": env}

    class FakeLogger:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def info(
            self, message: str, *, event: str, context: dict[str, object] | None = None
        ) -> None:
            payload = {"event": event}
            if context is not None:
                payload["context"] = context
            self.calls.append((message, payload))

    fake_logger = FakeLogger()

    def fake_get_logger(name: str, *, context: dict[str, object]) -> FakeLogger:
        calls["logger_name"] = name
        calls["logger_context"] = context
        return fake_logger

    run_args: dict[str, Any] = {}

    def fake_run_mcp_server(
        *,
        config: object,
        overrides_dir: object,
        env: object,
        logger: object,
    ) -> None:
        run_args["config"] = config
        run_args["overrides_dir"] = overrides_dir
        run_args["env"] = env
        run_args["logger"] = logger

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink, "run_mcp_server", fake_run_mcp_server)

    exit_code = wink.main(
        [
            "--config",
            "config.toml",
            "--overrides-dir",
            "overrides",
            "--log-level",
            "DEBUG",
            "--no-json-logs",
        ]
    )

    assert exit_code == 0
    assert calls["configure"]["level"] == "DEBUG"
    assert calls["configure"]["json_mode"] is False
    assert calls["configure"]["env"] is os.environ
    assert calls["logger_name"] == "weakincentives.cli.wink"
    assert calls["logger_context"] == {"component": "wink.cli"}
    assert fake_logger.calls == [
        ("Starting wink MCP server.", {"event": "wink.mcp.start"})
    ]
    assert run_args["config"] == Path("config.toml")
    assert run_args["overrides_dir"] == Path("overrides")
    assert run_args["env"] is os.environ
    assert run_args["logger"] is fake_logger


def test_main_uses_environment_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_mcp_server(
        *, config: object, overrides_dir: object, env: object, logger: object
    ) -> None:
        captured["config"] = config

    monkeypatch.setattr(wink, "run_mcp_server", fake_run_mcp_server)
    monkeypatch.setattr(wink, "configure_logging", lambda **_: None)
    monkeypatch.setattr(
        wink,
        "get_logger",
        lambda name, *, context: type("L", (), {"info": lambda *_, **__: None})(),
    )

    monkeypatch.setenv("WINK_CONFIG", str(Path("/env/config.toml")))

    exit_code = wink.main(["--overrides-dir", "overrides"])

    assert exit_code == 0
    assert captured["config"] == Path("/env/config.toml")


def test_main_returns_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[dict[str, object]] = []

    class FakeLogger:
        def info(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            events.append(
                {"event": event, "context": context or {}, "message": message}
            )

        def error(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            events.append(
                {"event": event, "context": context or {}, "message": message}
            )

        def exception(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            events.append(
                {"event": event, "context": context or {}, "message": message}
            )

    def fake_get_logger(name: str, *, context: dict[str, object]) -> FakeLogger:
        return FakeLogger()

    def fake_run_mcp_server(**_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(wink, "configure_logging", lambda **_: None)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink, "run_mcp_server", fake_run_mcp_server)

    exit_code = wink.main([])

    assert exit_code == 1
    assert any(record["event"] == "wink.mcp.error" for record in events)


def test_run_mcp_server_bootstraps_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    overrides_dir = Path(".wink/overrides")
    config_obj = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=overrides_dir,
        prompt_registry_modules=("demo.module",),
        listen_host="127.0.0.1",
        listen_port=4321,
    )

    load_calls: dict[str, Any] = {}

    def fake_load_config(
        config: object,
        *,
        cli_overrides: dict[str, object],
        env: dict[str, str],
    ) -> MCPServerConfig:
        load_calls["config"] = config
        load_calls["cli_overrides"] = cli_overrides
        load_calls["env"] = env
        return config_obj

    stores: list[Any] = []

    class FakeStore:
        def __init__(
            self, *, root_path: object, overrides_relative_path: object
        ) -> None:
            stores.append(
                {
                    "root_path": root_path,
                    "overrides_relative_path": overrides_relative_path,
                }
            )

    servers: list[Any] = []

    class FakeServer:
        def __init__(self, *, name: str, auth: object) -> None:
            self.name = name
            self.auth = auth
            self.tools: list[str] = []
            self.tool_handlers: dict[
                str, Callable[..., Coroutine[Any, Any, object]]
            ] = {}
            self.run_called = False
            servers.append(self)

        def tool(
            self,
            func: Callable[..., object],
            /,
            *,
            name: str,
        ) -> Callable[..., object]:
            self.tools.append(name)
            self.tool_handlers[name] = cast(
                Callable[..., Coroutine[Any, Any, object]], func
            )
            return func

        async def run_http_async(
            self,
            *,
            show_banner: bool,
            host: str,
            port: int,
            log_level: object,
        ) -> None:
            self.run_called = True
            self.run_args = {
                "show_banner": show_banner,
                "host": host,
                "port": port,
                "log_level": log_level,
            }

    records: list[dict[str, object]] = []
    fake_logger_contexts: list[Mapping[str, object] | None] = []

    class FakeLogger:
        def info(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            records.append(
                {"event": event, "context": context or {}, "message": message}
            )

        def error(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Unexpected error log")

        def exception(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            records.append(
                {"event": event, "context": context or {}, "message": message}
            )

    fake_logger = FakeLogger()

    def fake_get_logger(
        name: str,
        *,
        logger_override: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FakeLogger:
        assert name == wink.__name__
        assert logger_override is fake_logger
        fake_logger_contexts.append(context)
        return fake_logger

    def fake_anyio_run(
        async_fn: Callable[..., Coroutine[Any, Any, object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return asyncio.run(async_fn(*args, **kwargs))

    tool_calls: dict[str, Any] = {}
    now = datetime(2024, 1, 1, tzinfo=UTC)
    override_entry = SimpleNamespace(
        ns="demo",
        prompt="example",
        tag="latest",
        section_count=2,
        tool_count=1,
        content_hash="abc123",
        backing_file_path=Path("/tmp/override.json"),
        updated_at=now,
    )
    section_snapshot = SimpleNamespace(
        ns="demo",
        prompt="example",
        tag="latest",
        section_path=("intro",),
        expected_hash="hash",
        override_body="override",
        default_body="default",
        backing_file_path=Path("/tmp/section.json"),
        descriptor_version=2,
    )
    section_mutation = SimpleNamespace(
        ns="demo",
        prompt="example",
        tag="latest",
        section_path=("intro",),
        expected_hash="hash",
        override_body="override",
        descriptor_version=2,
        backing_file_path=Path("/tmp/section.json"),
        updated_at=now,
        warnings=("section",),
    )
    tool_snapshot = SimpleNamespace(
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search",
        expected_contract_hash="contract",
        override_description="desc",
        override_param_descriptions={"q": "query"},
        default_description="default",
        default_param_descriptions={"q": "default"},
        description="desc",
        param_descriptions={"q": "query"},
        backing_file_path=Path("/tmp/tool.json"),
        descriptor_version=3,
    )
    tool_mutation = SimpleNamespace(
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search",
        expected_contract_hash="contract",
        override_description="desc",
        override_param_descriptions={"q": "query"},
        description="desc",
        param_descriptions={"q": "query"},
        descriptor_version=3,
        backing_file_path=Path("/tmp/tool.json"),
        updated_at=now,
        warnings=("tool",),
    )

    def fake_list_overrides(
        *, config: MCPServerConfig, namespace: str | None = None
    ) -> tuple[SimpleNamespace, ...]:
        tool_calls.setdefault("list_overrides", []).append(
            {"config": config, "namespace": namespace}
        )
        return (override_entry,)

    def fake_fetch_section_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        section_path: str,
    ) -> SimpleNamespace:
        tool_calls.setdefault("get_section", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "section_path": section_path,
            }
        )
        return section_snapshot

    def fake_apply_section_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        section_path: str,
        body: str,
        expected_hash: str | None,
        descriptor_version: int | None,
        confirm: bool,
    ) -> SimpleNamespace:
        tool_calls.setdefault("write_section", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "section_path": section_path,
                "body": body,
                "expected_hash": expected_hash,
                "descriptor_version": descriptor_version,
                "confirm": confirm,
            }
        )
        return section_mutation

    def fake_remove_section_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        section_path: str,
        descriptor_version: int | None,
    ) -> SimpleNamespace:
        tool_calls.setdefault("delete_section", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "section_path": section_path,
                "descriptor_version": descriptor_version,
            }
        )
        return section_mutation

    def fake_fetch_tool_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        tool_name: str,
    ) -> SimpleNamespace:
        tool_calls.setdefault("get_tool", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "tool_name": tool_name,
            }
        )
        return tool_snapshot

    def fake_apply_tool_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        tool_name: str,
        description: str | None,
        param_descriptions: Mapping[str, str] | None,
        expected_contract_hash: str | None,
        descriptor_version: int | None,
        confirm: bool,
    ) -> SimpleNamespace:
        tool_calls.setdefault("write_tool", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "tool_name": tool_name,
                "description": description,
                "param_descriptions": dict(param_descriptions or {}),
                "expected_contract_hash": expected_contract_hash,
                "descriptor_version": descriptor_version,
                "confirm": confirm,
            }
        )
        return tool_mutation

    def fake_remove_tool_override(
        *,
        config: MCPServerConfig,
        store: object,
        ns: str,
        prompt: str,
        tag: str,
        tool_name: str,
        descriptor_version: int | None,
    ) -> SimpleNamespace:
        tool_calls.setdefault("delete_tool", []).append(
            {
                "config": config,
                "store": store,
                "ns": ns,
                "prompt": prompt,
                "tag": tag,
                "tool_name": tool_name,
                "descriptor_version": descriptor_version,
            }
        )
        return tool_mutation

    monkeypatch.setattr(wink, "load_config", fake_load_config)
    monkeypatch.setattr(wink, "LocalPromptOverridesStore", FakeStore)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink, "FastMCP", FakeServer)
    monkeypatch.setattr(wink.anyio, "run", fake_anyio_run)
    monkeypatch.setattr(wink, "list_overrides", fake_list_overrides)
    monkeypatch.setattr(wink, "fetch_section_override", fake_fetch_section_override)
    monkeypatch.setattr(wink, "apply_section_override", fake_apply_section_override)
    monkeypatch.setattr(wink, "remove_section_override", fake_remove_section_override)
    monkeypatch.setattr(wink, "fetch_tool_override", fake_fetch_tool_override)
    monkeypatch.setattr(wink, "apply_tool_override", fake_apply_tool_override)
    monkeypatch.setattr(wink, "remove_tool_override", fake_remove_tool_override)

    wink.run_mcp_server(
        config=Path("config.toml"),
        overrides_dir=Path("cli-overrides"),
        env={"SAMPLE": "1"},
        logger=cast(StructuredLogger, fake_logger),
    )

    assert load_calls["config"] == Path("config.toml")
    assert load_calls["cli_overrides"] == {"overrides_dir": Path("cli-overrides")}
    assert load_calls["env"] == {"SAMPLE": "1"}

    assert stores == [
        {
            "root_path": workspace,
            "overrides_relative_path": Path(".wink") / "overrides",
        }
    ]

    assert servers
    server = servers[0]
    assert server.name == "wink"
    assert server.auth is None
    assert server.run_called is True
    assert set(server.tools) == {
        "wink.list_overrides",
        "wink.get_section_override",
        "wink.write_section_override",
        "wink.delete_section_override",
        "wink.get_tool_override",
        "wink.write_tool_override",
        "wink.delete_tool_override",
    }

    list_payload = asyncio.run(
        server.tool_handlers["wink.list_overrides"](namespace="demo")
    )
    assert list_payload == {
        "overrides": [
            {
                "ns": "demo",
                "prompt": "example",
                "tag": "latest",
                "section_count": 2,
                "tool_count": 1,
                "content_hash": "abc123",
                "backing_file_path": str(Path("/tmp/override.json")),
                "updated_at": now.isoformat(),
            }
        ]
    }
    assert tool_calls["list_overrides"][0]["namespace"] == "demo"

    section_payload = asyncio.run(
        server.tool_handlers["wink.get_section_override"](
            "demo", "example", section_path="intro"
        )
    )
    assert section_payload["section_path"] == ["intro"]
    assert section_payload["override_body"] == "override"

    write_section_payload = asyncio.run(
        server.tool_handlers["wink.write_section_override"](
            "demo",
            "example",
            body="override",
            expected_hash="hash",
            descriptor_version=2,
            confirm=True,
        )
    )
    assert write_section_payload["warnings"] == ["section"]
    assert tool_calls["write_section"][0]["confirm"] is True

    delete_section_payload = asyncio.run(
        server.tool_handlers["wink.delete_section_override"](
            "demo",
            "example",
            descriptor_version=2,
        )
    )
    assert delete_section_payload["descriptor_version"] == 2

    tool_payload = asyncio.run(
        server.tool_handlers["wink.get_tool_override"](
            "demo", "example", tool_name="search"
        )
    )
    assert tool_payload["tool_name"] == "search"
    assert tool_payload["override_param_descriptions"] == {"q": "query"}

    write_tool_payload = asyncio.run(
        server.tool_handlers["wink.write_tool_override"](
            "demo",
            "example",
            tool_name="search",
            description="desc",
            param_descriptions={"q": "query"},
            expected_contract_hash="contract",
            descriptor_version=3,
            confirm=True,
        )
    )
    assert write_tool_payload["warnings"] == ["tool"]
    assert tool_calls["write_tool"][0]["confirm"] is True

    delete_tool_payload = asyncio.run(
        server.tool_handlers["wink.delete_tool_override"](
            "demo",
            "example",
            tool_name="search",
            descriptor_version=3,
        )
    )
    assert delete_tool_payload["descriptor_version"] == 3
    assert tool_calls["delete_tool"][0]["descriptor_version"] == 3

    assert any(record["event"] == "wink.mcp.config_resolved" for record in records)
    assert any(record["event"] == "wink.mcp.runtime_start" for record in records)
    assert any(record["event"] == "wink.mcp.runtime_stop" for record in records)


def test_run_mcp_server_logs_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeLogger:
        def info(
            self, *args: object, **kwargs: object
        ) -> None:  # pragma: no cover - noop
            pass

        def exception(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            events.append(event)

    fake_logger = FakeLogger()

    def fake_get_logger(
        name: str,
        *,
        logger_override: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FakeLogger:
        assert logger_override is fake_logger
        return fake_logger

    def fake_load_config(*args: object, **kwargs: object) -> MCPServerConfig:
        raise wink.ConfigError("boom")

    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink, "load_config", fake_load_config)

    with pytest.raises(wink.ConfigError):
        wink.run_mcp_server(
            config=Path("config.toml"),
            overrides_dir=None,
            env={},
            logger=cast(StructuredLogger, fake_logger),
        )

    assert "wink.mcp.config_error" in events


def test_run_mcp_server_logs_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    overrides_dir = Path(".wink/overrides")
    config_obj = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=overrides_dir,
        prompt_registry_modules=(),
        listen_host="127.0.0.1",
        listen_port=4321,
    )

    def fake_load_config(*args: object, **kwargs: object) -> MCPServerConfig:
        return config_obj

    class FakeStore:
        def __init__(self, **_: object) -> None:
            pass

    class FailingServer:
        def __init__(self, *, name: str, auth: object) -> None:
            self.name = name
            self.auth = auth

        def tool(
            self, func: Callable[..., object], *, name: str
        ) -> Callable[..., object]:
            return func

        async def run_http_async(self, **_: object) -> None:
            raise RuntimeError("boom")

    class FakeLogger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def info(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            self.events.append(event)

        def exception(
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            self.events.append(event)

        def error(  # pragma: no cover - unused in success cases
            self,
            message: str,
            *,
            event: str,
            context: dict[str, object] | None = None,
        ) -> None:
            self.events.append(event)

    fake_logger = FakeLogger()

    def fake_get_logger(
        name: str,
        *,
        logger_override: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FakeLogger:
        assert logger_override is fake_logger
        return fake_logger

    def fake_anyio_run(
        async_fn: Callable[..., Coroutine[Any, Any, object]],
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        return asyncio.run(async_fn(*args, **kwargs))

    monkeypatch.setattr(wink, "load_config", fake_load_config)
    monkeypatch.setattr(wink, "LocalPromptOverridesStore", FakeStore)
    monkeypatch.setattr(wink, "FastMCP", FailingServer)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.anyio, "run", fake_anyio_run)

    with pytest.raises(RuntimeError):
        wink.run_mcp_server(
            config=Path("config.toml"),
            overrides_dir=Path("cli-overrides"),
            env={},
            logger=cast(StructuredLogger, fake_logger),
        )

    assert "wink.mcp.runtime_failure" in fake_logger.events


def test_build_store_with_absolute_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Path]] = []

    class FakeStore:
        def __init__(self, *, root_path: Path, overrides_relative_path: Path) -> None:
            calls.append(
                {
                    "root_path": root_path,
                    "overrides_relative_path": overrides_relative_path,
                }
            )

    monkeypatch.setattr(wink, "LocalPromptOverridesStore", FakeStore)

    config = MCPServerConfig(
        workspace_root=tmp_path / "workspace",
        overrides_dir=tmp_path / "abs_overrides",
    )

    store, root = wink._build_store(config)

    assert isinstance(store, FakeStore)
    assert root == (tmp_path / "abs_overrides").resolve()
    assert calls == [
        {
            "root_path": tmp_path / "abs_overrides",
            "overrides_relative_path": Path(),
        }
    ]


def test_build_store_with_relative_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Path]] = []

    class FakeStore:
        def __init__(self, *, root_path: Path, overrides_relative_path: Path) -> None:
            calls.append(
                {
                    "root_path": root_path,
                    "overrides_relative_path": overrides_relative_path,
                }
            )

    monkeypatch.setattr(wink, "LocalPromptOverridesStore", FakeStore)

    config = MCPServerConfig(
        workspace_root=tmp_path / "workspace",
        overrides_dir=Path("relative/overrides"),
    )

    store, root = wink._build_store(config)

    assert isinstance(store, FakeStore)
    assert root == (tmp_path / "workspace" / "relative" / "overrides").resolve()
    assert calls == [
        {
            "root_path": tmp_path / "workspace",
            "overrides_relative_path": Path("relative/overrides"),
        }
    ]


def test_build_auth_provider_returns_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, Any]] = {}

    class FakeVerifier:
        def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
            captured["tokens"] = tokens

    monkeypatch.setattr(wink, "StaticTokenVerifier", FakeVerifier)

    config = MCPServerConfig(
        workspace_root=Path("/workspace"),
        overrides_dir=Path("/overrides"),
        auth_tokens={"client": "secret"},
    )

    verifier = wink._build_auth_provider(config)

    assert isinstance(verifier, FakeVerifier)
    assert captured["tokens"] == {"secret": {"client_id": "client", "scopes": []}}


def test_build_auth_provider_without_tokens() -> None:
    config = MCPServerConfig(
        workspace_root=Path("/workspace"),
        overrides_dir=Path("/overrides"),
    )

    assert wink._build_auth_provider(config) is None


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: WinkOverridesError("wink error"),
        lambda: OverridesInspectionError("inspection"),
        lambda: PromptOverridesError("prompt"),
        lambda: ValueError("validation"),
    ],
)
def test_call_tool_wraps_known_errors(
    monkeypatch: pytest.MonkeyPatch, exc_factory: Callable[[], Exception]
) -> None:
    logs: list[str] = []

    class FakeLogger:
        def __init__(self) -> None:
            self.contexts: list[Mapping[str, object]] = []

        def error(
            self,
            message: str,
            *,
            event: str,
            context: Mapping[str, object],
        ) -> None:
            logs.append(event)
            self.contexts.append(context)

    async def fake_run_sync(
        func: Callable[[], object], *args: object, **kwargs: object
    ) -> object:
        await asyncio.sleep(0)
        return func()

    monkeypatch.setattr(wink.anyio.to_thread, "run_sync", fake_run_sync)

    def failing() -> object:
        raise exc_factory()

    fake_logger = FakeLogger()
    context_payload = {"path": Path("/tmp/context.json")}

    with pytest.raises(wink.ToolError):
        asyncio.run(
            wink._call_tool(
                failing,
                logger=cast(StructuredLogger, fake_logger),
                event="test.event",
                context=context_payload,
            )
        )

    assert logs == ["test.event"]
    assert fake_logger.contexts[0]["path"] == str(Path("/tmp/context.json"))


def test_call_tool_wraps_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[str] = []

    class FakeLogger:
        def error(
            self,
            message: str,
            *,
            event: str,
            context: Mapping[str, object],
        ) -> None:
            logs.append(message)

    async def fake_run_sync(
        func: Callable[[], object], *args: object, **kwargs: object
    ) -> object:
        await asyncio.sleep(0)
        return func()

    monkeypatch.setattr(wink.anyio.to_thread, "run_sync", fake_run_sync)

    def failing() -> object:
        raise RuntimeError("boom")

    with pytest.raises(wink.ToolError) as excinfo:
        asyncio.run(
            wink._call_tool(
                failing,
                logger=cast(StructuredLogger, FakeLogger()),
                event="test.event",
                context={},
            )
        )

    assert "Unexpected wink MCP failure." in logs[0]
    assert "Unexpected wink MCP failure." in str(excinfo.value)


def test_write_section_override_requires_body(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_handlers: dict[str, Callable[..., Coroutine[Any, Any, object]]] = {}

    class FakeServer:
        def tool(
            self,
            func: Callable[..., object],
            /,
            *,
            name: str,
        ) -> Callable[..., object]:
            tool_handlers[name] = cast(Callable[..., Coroutine[Any, Any, object]], func)
            return func

    class FakeLogger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def error(
            self,
            message: str,
            *,
            event: str,
            context: Mapping[str, object],
        ) -> None:
            self.events.append(event)

    config = MCPServerConfig(
        workspace_root=Path("/workspace"),
        overrides_dir=Path("overrides"),
    )

    fake_logger = FakeLogger()

    wink._register_tools(
        server=cast(wink.FastMCP, FakeServer()),
        config=config,
        store=cast(LocalPromptOverridesStore, object()),
        logger=cast(StructuredLogger, fake_logger),
    )

    write_section = tool_handlers["wink.write_section_override"]

    with pytest.raises(wink.ToolError):
        asyncio.run(write_section("demo", "prompt"))

    assert fake_logger.events == ["wink.mcp.write_section_override.error"]
