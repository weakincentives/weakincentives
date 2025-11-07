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

import pytest

from weakincentives import cli
from weakincentives.cli import wink


def test_cli_namespace_lists_wink_module() -> None:
    assert "wink" in cli.__dir__()


def test_main_dispatches_to_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        calls["configure"] = {"level": level, "json_mode": json_mode}

    class FakeLogger:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def info(self, message: str, *, event: str) -> None:
            self.calls.append((message, {"event": event}))

    fake_logger = FakeLogger()

    def fake_get_logger(name: str) -> FakeLogger:
        calls["logger_name"] = name
        return fake_logger

    run_args: dict[str, object] = {}

    def fake_run_mcp_server(*, config: object, overrides_dir: object) -> None:
        run_args["config"] = config
        run_args["overrides_dir"] = overrides_dir

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
            "mcp",
        ]
    )

    assert exit_code == 0
    assert calls["configure"] == {"level": "DEBUG", "json_mode": False}
    assert calls["logger_name"] == "weakincentives.cli.wink"
    assert fake_logger.calls == [
        ("Starting wink MCP server.", {"event": "wink.mcp.start"})
    ]
    assert str(run_args["config"]) == "config.toml"
    assert str(run_args["overrides_dir"]) == "overrides"
