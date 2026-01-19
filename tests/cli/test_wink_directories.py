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
from pathlib import Path
from typing import Any

import pytest

from weakincentives.cli import wink
from weakincentives.debug.bundle import BundleConfig, BundleWriter, CaptureMode
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _create_test_bundle(target_dir: Path, name: str) -> Path:
    """Create a test debug bundle with session data."""
    session = Session()
    session.dispatch(_ExampleSlice(name))

    with BundleWriter(
        target_dir, config=BundleConfig(mode=CaptureMode.STANDARD)
    ) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": name})
        writer.write_request_output({"status": "ok"})

    assert writer.path is not None
    return writer.path


def test_directory_argument_loads_latest_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = _create_test_bundle(tmp_path, "old")
    time.sleep(0.01)
    newer = _create_test_bundle(tmp_path, "new")

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
            "debug",
            str(tmp_path),
            "--no-open-browser",
        ]
    )

    assert exit_code == 0
    bundle_store = calls["app_args"]["store"]
    assert isinstance(bundle_store, wink.debug_app.BundleStore)
    # The store should have loaded the newer bundle (most recent by mtime)
    assert bundle_store.path == newer.resolve()
