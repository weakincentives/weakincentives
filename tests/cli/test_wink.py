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
from weakincentives.cli import wink
from weakincentives.dbc import dbc_enabled
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
    with dbc_enabled(False):
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

    def fake_run_server(
        snapshot_path_arg: Path,
        *,
        host: str,
        port: int,
        open_browser: bool,
        logger: object,
    ) -> int:
        calls["run_args"] = {
            "snapshot_path": snapshot_path_arg,
            "host": host,
            "port": port,
            "open_browser": open_browser,
            "logger": logger,
        }
        return 0

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
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
    assert calls["run_args"] == {
        "snapshot_path": snapshot_path,
        "host": "0.0.0.0",
        "port": 9001,
        "open_browser": False,
        "logger": fake_logger,
    }


def test_main_runs_static_export_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path)
    output_dir = tmp_path / "output"

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

    def fake_generate_static_site(
        snapshot_path_arg: Path,
        output_dir_arg: Path,
        *,
        base_path: str,
        logger: object,
    ) -> None:
        calls["generate_args"] = {
            "snapshot_path": snapshot_path_arg,
            "output_dir": output_dir_arg,
            "base_path": base_path,
            "logger": logger,
        }

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "generate_static_site", fake_generate_static_site)

    exit_code = wink.main(
        [
            "debug",
            str(snapshot_path),
            "--output",
            str(output_dir),
            "--base-path",
            "/reports/",
        ]
    )

    assert exit_code == 0
    assert calls["generate_args"]["snapshot_path"] == snapshot_path
    assert calls["generate_args"]["output_dir"] == output_dir
    assert calls["generate_args"]["base_path"] == "/reports/"
    assert calls["generate_args"]["logger"] == fake_logger


def test_main_handles_static_export_snapshot_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path)
    output_dir = tmp_path / "output"

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

    def fake_generate_static_site(
        snapshot_path_arg: Path,
        output_dir_arg: Path,
        *,
        base_path: str,
        logger: object,
    ) -> None:
        msg = "Invalid snapshot"
        raise wink.debug_app.SnapshotLoadError(msg)

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "generate_static_site", fake_generate_static_site)

    exit_code = wink.main(
        ["debug", str(snapshot_path), "--output", str(output_dir)]
    )

    assert exit_code == 2


def test_main_handles_static_export_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path)
    output_dir = tmp_path / "output"

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

    def fake_generate_static_site(
        snapshot_path_arg: Path,
        output_dir_arg: Path,
        *,
        base_path: str,
        logger: object,
    ) -> None:
        msg = "Permission denied"
        raise OSError(msg)

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "generate_static_site", fake_generate_static_site)

    exit_code = wink.main(
        ["debug", str(snapshot_path), "--output", str(output_dir)]
    )

    assert exit_code == 4


def test_main_handles_invalid_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "missing.jsonl"

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

    def fake_run_server(
        path: Path,
        *,
        host: str,
        port: int,
        open_browser: bool,
        logger: object,
    ) -> int:
        msg = f"{path} missing"
        raise wink.debug_app.SnapshotLoadError(msg)

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.debug_app, "run_debug_server", fake_run_server)

    exit_code = wink.main(["debug", str(snapshot_path)])

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
