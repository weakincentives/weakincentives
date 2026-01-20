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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from weakincentives import cli
from weakincentives.cli import wink
from weakincentives.debug.bundle import BundleConfig, BundleWriter
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _create_test_bundle(target_dir: Path) -> Path:
    """Create a test debug bundle with session data."""
    session = Session()
    session.dispatch(_ExampleSlice("a"))

    with BundleWriter(target_dir, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})

    assert writer.path is not None
    return writer.path


def test_cli_namespace_lists_wink_module() -> None:
    assert "wink" in cli.__dir__()


def test_main_runs_debug_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_path = _create_test_bundle(tmp_path)

    calls: dict[str, Any] = {}

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        calls["configure"] = {"level": level, "json_mode": json_mode}

    class FakeLogger:
        def __init__(self) -> None:
            self.logs: list[tuple[str, dict[str, object]]] = []

        def info(
            self, message: str, *, event: str, context: object | None = None
        ) -> None:
            self.logs.append((message, {"event": event, "context": context}))

        def error(
            self, message: str, *, event: str, context: object | None = None
        ) -> None:
            self.logs.append((message, {"event": event, "context": context}))

    fake_logger = FakeLogger()

    def fake_get_logger(name: str) -> FakeLogger:
        calls["logger_name"] = name
        return fake_logger

    def fake_build_app(*args: object, **kwargs: object) -> str:
        calls["app_args"] = {"store": args[0], **kwargs}
        return "app"

    def fake_run_server(
        app: object, *, host: str, port: int, open_browser: bool, logger: object
    ) -> int:
        calls["run_args"] = {
            "app": app,
            "host": host,
            "port": port,
            "open_browser": open_browser,
            "logger": logger,
        }
        return 0

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "build_debug_app", fake_build_app)
    monkeypatch.setattr(wink.debug_app, "run_debug_server", fake_run_server)

    exit_code = wink.main(
        [
            "--log-level",
            "DEBUG",
            "--no-json-logs",
            "debug",
            str(bundle_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--no-open-browser",
        ]
    )

    assert exit_code == 0
    assert calls["configure"] == {"level": "DEBUG", "json_mode": False}
    assert calls["logger_name"] == "weakincentives.cli.wink"
    bundle_store = calls["app_args"]["store"]
    assert isinstance(bundle_store, wink.debug_app.BundleStore)
    assert bundle_store.meta.path == str(bundle_path)
    assert calls["app_args"]["logger"] == fake_logger
    assert calls["run_args"] == {
        "app": "app",
        "host": "0.0.0.0",
        "port": 9001,
        "open_browser": False,
        "logger": fake_logger,
    }


def test_main_handles_invalid_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_path = tmp_path / "missing.zip"

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        return None

    class FakeLogger:
        @staticmethod
        def error(*_: object, **__: object) -> None:
            return None

        @staticmethod
        def exception(*_: object, **__: object) -> None:
            return None

    def fake_get_logger(name: str) -> FakeLogger:
        return FakeLogger()

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)

    exit_code = wink.main(["debug", str(bundle_path)])

    assert exit_code == 2


def test_main_returns_parser_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeParser:
        @staticmethod
        def parse_args(argv: list[str] | None = None) -> object:
            raise SystemExit(5)

    monkeypatch.setattr(wink, "_build_parser", lambda: FakeParser())

    exit_code = wink.main(["--unknown"])

    assert exit_code == 5


def test_main_returns_zero_for_unknown_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeArgs:
        command = "other"
        log_level = None
        json_logs = True

    class FakeParser:
        @staticmethod
        def parse_args(argv: list[str] | None = None) -> FakeArgs:
            return FakeArgs()

    calls: dict[str, object] = {}

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        calls["configured"] = (level, json_mode)

    def fake_get_logger(name: str) -> object:
        calls["logger_name"] = name
        return object()

    monkeypatch.setattr(wink, "_build_parser", lambda: FakeParser())
    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)

    exit_code = wink.main(["other"])

    assert exit_code == 0
    assert calls["configured"] == (None, True)
    assert calls["logger_name"] == "weakincentives.cli.wink"
