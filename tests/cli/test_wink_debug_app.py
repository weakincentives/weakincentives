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

"""Tests for the wink debug FastAPI application."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weakincentives.cli import debug_app
from weakincentives.runtime.session.snapshots import Snapshot


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _write_snapshot(path: Path, values: list[str]) -> None:
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: tuple(_ExampleSlice(value) for value in values)},
    )
    path.write_text(snapshot.to_json())


def test_load_snapshot_validates_schema(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path, ["one"])

    loaded = debug_app.load_snapshot(snapshot_path)

    assert loaded.meta.version == "1"
    assert loaded.meta.slices[0].count == 1


def test_load_snapshot_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "missing.json"

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)

    snapshot_path.write_text("{")

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)


def test_load_snapshot_recovers_from_unknown_types(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    payload = {
        "version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "slices": [
            {
                "slice_type": "__main__:UnknownType",
                "item_type": "__main__:UnknownType",
                "items": [{"value": "one"}],
            }
        ],
    }
    snapshot_path.write_text(json.dumps(payload))

    loaded = debug_app.load_snapshot(snapshot_path)

    assert loaded.meta.validation_error
    assert "__main__:UnknownType" in loaded.slices
    unknown_slice = loaded.slices["__main__:UnknownType"]
    assert unknown_slice.items == ({"value": "one"},)


def test_api_routes_expose_snapshot_data(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path, ["a", "b", "c"])
    logger = debug_app.get_logger("test.api")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    meta_response = client.get("/api/meta")
    assert meta_response.status_code == 200
    meta = meta_response.json()
    assert len(meta["slices"]) == 1
    slice_type = meta["slices"][0]["slice_type"]

    detail_response = client.get(f"/api/slices/{quote(slice_type)}?offset=1&limit=1")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["items"] == [{"value": "b"}]

    raw_response = client.get("/api/raw")
    assert raw_response.status_code == 200
    raw = raw_response.json()
    assert raw["version"] == "1"
    assert raw["slices"][0]["items"][0]["value"] == "a"


def test_reload_endpoint_replaces_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path, ["one"])
    logger = debug_app.get_logger("test.reload")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    _write_snapshot(snapshot_path, ["two", "three"])

    reload_response = client.post("/api/reload")
    assert reload_response.status_code == 200
    meta = reload_response.json()
    assert meta["slices"][0]["count"] == 2
    slice_type = meta["slices"][0]["slice_type"]

    detail = client.get(f"/api/slices/{quote(slice_type)}").json()
    assert [item["value"] for item in detail["items"]] == ["two", "three"]

    _write_snapshot(snapshot_path, ["invalid"])
    snapshot_path.write_text("not-json")
    reload_failed = client.post("/api/reload")
    assert reload_failed.status_code == 400

    meta_after_failure = client.get("/api/meta").json()
    assert meta_after_failure["slices"][0]["count"] == 2


def test_snapshot_listing_and_switch(tmp_path: Path) -> None:
    snapshot_one = tmp_path / "one.json"
    snapshot_two = tmp_path / "two.json"
    _write_snapshot(snapshot_one, ["a"])
    _write_snapshot(snapshot_two, ["b", "c"])

    now = time.time()
    time.sleep(0.01)
    time_one = now
    time_two = now + 1
    os.utime(snapshot_one, (time_one, time_one))
    os.utime(snapshot_two, (time_two, time_two))

    logger = debug_app.get_logger("test.switch")
    store = debug_app.SnapshotStore(
        snapshot_one, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    listing = client.get("/api/snapshots").json()
    assert listing[0]["name"] == "two.json"
    assert listing[1]["name"] == "one.json"

    switch_response = client.post("/api/switch", json={"path": str(snapshot_two)})
    assert switch_response.status_code == 200
    switched_meta = switch_response.json()
    assert switched_meta["path"] == str(snapshot_two)

    detail = client.get("/api/meta").json()
    assert detail["path"] == str(snapshot_two)


def test_snapshot_store_handles_errors_and_properties(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path, ["one"])

    missing_target = tmp_path / "missing.json"
    broken_link = tmp_path / "broken.json"
    broken_link.symlink_to(missing_target)

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger(__name__),
    )

    assert store.raw_payload["version"] == "1"
    assert store.path == snapshot_path.resolve()

    listing = store.list_snapshots()
    names = {entry["name"] for entry in listing}
    assert "snapshot.json" in names
    assert "broken.json" not in names

    with pytest.raises(KeyError, match="Unknown slice type: missing"):
        store.slice_items("missing")


def test_snapshot_store_switch_rejects_outside_root(tmp_path: Path) -> None:
    base_snapshot = tmp_path / "base.json"
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_snapshot = other_dir / "other.json"

    _write_snapshot(base_snapshot, ["base"])
    _write_snapshot(other_snapshot, ["other"])

    store = debug_app.SnapshotStore(
        base_snapshot,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.switch_root"),
    )

    with pytest.raises(debug_app.SnapshotLoadError) as excinfo:
        store.switch(other_snapshot)

    assert (
        str(excinfo.value)
        == f"Snapshot must live under {base_snapshot.parent.resolve()}"
    )


def test_normalize_path_requires_snapshots(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(debug_app.SnapshotLoadError) as excinfo:
        debug_app.SnapshotStore(
            empty_dir,
            loader=debug_app.load_snapshot,
            logger=debug_app.get_logger("test.empty"),
        )

    assert str(excinfo.value) == f"No snapshots found under {empty_dir.resolve()}"


def test_index_and_error_endpoints(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path, ["a"])

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_snapshot = other_dir / "other.json"
    _write_snapshot(other_snapshot, ["b"])

    logger = debug_app.get_logger("test.routes")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "<!doctype html>" in index_response.text.lower()

    missing_slice = client.get("/api/slices/missing")
    assert missing_slice.status_code == 404
    assert "Unknown slice type: missing" in missing_slice.json()["detail"]

    missing_path = client.post("/api/switch", json={"path": 123})
    assert missing_path.status_code == 400
    assert missing_path.json()["detail"] == "path is required"

    wrong_root = client.post("/api/switch", json={"path": str(other_snapshot)})
    assert wrong_root.status_code == 400
    assert "Snapshot must live under" in wrong_root.json()["detail"]


def test_run_debug_server_opens_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTimer:
        def __init__(
            self,
            interval: float,
            function: Callable[..., None],
            args: tuple[object, ...] | None = None,
            kwargs: dict[str, object] | None = None,
        ) -> None:
            calls["timer_interval"] = interval
            self._function = function
            self._args: tuple[object, ...] = tuple(args or ())
            self._kwargs: dict[str, object] = dict(kwargs or {})

        def start(self) -> None:
            calls["timer_started"] = True
            self._function(*self._args, **self._kwargs)

    monkeypatch.setattr(debug_app.threading, "Timer", FakeTimer)

    def fake_webbrowser_open(url: str) -> bool:
        calls["browser_url"] = url
        return True

    monkeypatch.setattr(debug_app.webbrowser, "open", fake_webbrowser_open)

    config_calls: dict[str, object] = {}

    class FakeConfig:
        def __init__(
            self, app: object, host: str, port: int, log_config: object
        ) -> None:
            config_calls["config"] = {
                "app": app,
                "host": host,
                "port": port,
                "log_config": log_config,
            }

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            config_calls["server_config"] = config

        def run(self) -> None:
            config_calls["run_called"] = True

    monkeypatch.setattr(debug_app.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(debug_app.uvicorn, "Server", FakeServer)

    logger = debug_app.get_logger("test.run")
    infos: list[dict[str, object]] = []

    def capture_info(message: str, *, event: str, context: dict[str, object]) -> None:
        infos.append({"message": message, "event": event, "context": context})

    monkeypatch.setattr(logger, "info", capture_info)

    app = FastAPI()

    exit_code = debug_app.run_debug_server(
        app=app,
        host="0.0.0.0",
        port=8123,
        open_browser=True,
        logger=logger,
    )

    assert exit_code == 0
    assert calls["timer_started"] is True
    assert calls["browser_url"] == "http://0.0.0.0:8123/"
    assert config_calls["config"] == {
        "app": app,
        "host": "0.0.0.0",
        "port": 8123,
        "log_config": None,
    }
    assert config_calls["run_called"] is True
    assert infos[0]["event"] == "debug.server.start"
