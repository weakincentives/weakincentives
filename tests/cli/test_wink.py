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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from weakincentives import cli
from weakincentives.cli import optimize_app, wink
from weakincentives.runtime.session.snapshots import Snapshot


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _write_snapshot(path: Path) -> None:
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: (_ExampleSlice("a"),)},
        tags={"suite": "wink", "session_id": path.stem},
    )
    path.write_text(snapshot.to_json() + "\n")


def test_cli_namespace_lists_wink_module() -> None:
    assert "wink" in cli.__dir__()


def test_main_runs_debug_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path)

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

    real_loader = wink.debug_app.load_snapshot

    def fake_load_snapshot(path: Path) -> object:
        calls["loaded_path"] = path
        return real_loader(path)

    def fake_build_app(*args: object, **kwargs: object) -> str:
        calls["app_args"] = {"snapshot": args[0], **kwargs}
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
    monkeypatch.setattr(wink.debug_app, "load_snapshot", fake_load_snapshot)
    monkeypatch.setattr(wink.debug_app, "build_debug_app", fake_build_app)
    monkeypatch.setattr(wink.debug_app, "run_debug_server", fake_run_server)

    exit_code = wink.main(
        [
            "--log-level",
            "DEBUG",
            "--no-json-logs",
            "debug",
            str(snapshot_path),
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
    assert calls["loaded_path"] == snapshot_path
    snapshot_store = calls["app_args"]["snapshot"]
    assert isinstance(snapshot_store, wink.debug_app.SnapshotStore)
    assert snapshot_store.meta.path == str(snapshot_path)
    assert calls["app_args"]["logger"] == fake_logger
    assert calls["run_args"] == {
        "app": "app",
        "host": "0.0.0.0",
        "port": 9001,
        "open_browser": False,
        "logger": fake_logger,
    }


def test_main_runs_optimize_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path)

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

        def exception(
            self, message: str, *, event: str, context: object | None = None
        ) -> None:
            self.logs.append((message, {"event": event, "context": context}))

    fake_logger = FakeLogger()

    def fake_get_logger(name: str) -> FakeLogger:
        calls["logger_name"] = name
        return fake_logger

    real_loader = optimize_app.load_snapshot

    def fake_load_snapshot(path: Path) -> object:
        calls["loaded_path"] = path
        return real_loader(path)

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
    monkeypatch.setattr(optimize_app, "load_snapshot", fake_load_snapshot)
    monkeypatch.setattr(optimize_app, "build_optimize_app", fake_build_app)
    monkeypatch.setattr(optimize_app, "run_optimize_server", fake_run_server)

    exit_code = wink.main(
        [
            "optimize",
            str(snapshot_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9002",
            "--no-open-browser",
        ]
    )

    assert exit_code == 0
    assert calls["configure"] == {"level": None, "json_mode": True}
    assert calls["logger_name"] == "weakincentives.cli.wink"
    assert calls["loaded_path"] == snapshot_path
    assert isinstance(calls["app_args"]["store"], optimize_app.OptimizeStore)
    assert calls["app_args"]["logger"] == fake_logger
    assert calls["run_args"] == {
        "app": "app",
        "host": "0.0.0.0",
        "port": 9002,
        "open_browser": False,
        "logger": fake_logger,
    }


def test_main_handles_invalid_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "missing.jsonl"

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        return None

    class FakeLogger:
        def error(self, *_: object, **__: object) -> None:
            return None

        def exception(self, *_: object, **__: object) -> None:
            return None

    def fake_get_logger(name: str) -> FakeLogger:
        return FakeLogger()

    def fake_load_snapshot(path: Path) -> object:
        msg = f"{path} missing"
        raise wink.debug_app.SnapshotLoadError(msg)

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "load_snapshot", fake_load_snapshot)

    exit_code = wink.main(["debug", str(snapshot_path)])

    assert exit_code == 2


def test_optimize_handles_invalid_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        return None

    class FakeLogger:
        def exception(self, *_: object, **__: object) -> None:
            return None

    def fake_get_logger(name: str) -> FakeLogger:
        return FakeLogger()

    def fake_load_snapshot(path: Path) -> object:
        msg = f"{path} missing"
        raise optimize_app.SnapshotLoadError(msg)

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(optimize_app, "load_snapshot", fake_load_snapshot)

    exit_code = wink.main(["optimize", "missing.jsonl"])

    assert exit_code == 2


def test_main_returns_parser_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeParser:
        def parse_args(self, argv: list[str] | None = None) -> object:
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
        def parse_args(self, argv: list[str] | None = None) -> FakeArgs:
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
