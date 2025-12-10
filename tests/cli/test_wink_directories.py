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

"""Tests for directory handling in the wink debug CLI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from weakincentives.cli import wink
from weakincentives.dbc import dbc_enabled
from weakincentives.runtime.session.snapshots import Snapshot


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _write_snapshot(path: Path, *, created_at: datetime) -> None:
    snapshot = Snapshot(
        created_at=created_at,
        slices={_ExampleSlice: (_ExampleSlice(path.name),)},
        tags={
            "suite": "wink-directories",
            "name": path.name,
            "session_id": path.stem,
        },
    )
    with dbc_enabled(False):
        path.write_text(snapshot.to_json() + "\n")


def test_directory_argument_loads_latest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that directory path is passed to run_debug_server.

    The actual directory handling (selecting newest file) is tested
    in test_wink_debug_app.py::test_generate_static_site_from_directory.
    """
    with dbc_enabled(False):
        older = tmp_path / "old.jsonl"
        newer = tmp_path / "new.jsonl"
        now = datetime.now(UTC)
        _write_snapshot(older, created_at=now - timedelta(days=1))
        _write_snapshot(newer, created_at=now)
        now_ts = time.time()
        os.utime(older, (now_ts, now_ts))
        os.utime(newer, (now_ts + 1, now_ts + 1))

        calls: dict[str, Any] = {}

        def fake_configure_logging(*, level: object, json_mode: object) -> None:
            calls["configure"] = {"level": level, "json_mode": json_mode}

        class FakeLogger:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            def info(
                self, message: str, *, event: str, context: object | None = None
            ) -> None:
                self.calls.append((message, {"event": event, "context": context}))

            def error(
                self, message: str, *, event: str, context: object | None = None
            ) -> None:
                self.calls.append((message, {"event": event, "context": context}))

        fake_logger = FakeLogger()

        def fake_get_logger(name: str) -> FakeLogger:
            calls["logger_name"] = name
            return fake_logger

        def fake_run_server(
            snapshot_path: Path,
            *,
            host: str,
            port: int,
            open_browser: bool,
            logger: object,
        ) -> int:
            calls["run_args"] = {
                "snapshot_path": snapshot_path,
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
                "debug",
                str(tmp_path),
                "--no-open-browser",
            ]
        )

    assert exit_code == 0
    # Directory path is passed directly to run_debug_server;
    # directory handling happens inside generate_static_site
    assert calls["run_args"]["snapshot_path"] == tmp_path
