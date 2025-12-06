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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weakincentives.cli import debug_app
from weakincentives.dbc import dbc_enabled
from weakincentives.runtime.session.snapshots import Snapshot


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


@dataclass(slots=True, frozen=True)
class _ListSlice:
    value: object


def _write_snapshot(path: Path, values: list[str]) -> list[str]:
    session_ids: list[str] = []
    entries: list[str] = []
    for index, value in enumerate(values):
        session_id = f"{path.stem}-{index}"
        snapshot = Snapshot(
            created_at=datetime.now(UTC),
            slices={_ExampleSlice: (_ExampleSlice(value),)},
            tags={"suite": "wink-debug", "session_id": session_id},
        )
        entries.append(snapshot.to_json())
        session_ids.append(session_id)
    with dbc_enabled(active=False):
        path.write_text("\n".join(entries))
    return session_ids


def test_load_snapshot_validates_schema(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    session_ids = _write_snapshot(snapshot_path, ["one"])

    loaded = debug_app.load_snapshot(snapshot_path)

    assert len(loaded) == 1
    meta = loaded[0].meta
    assert meta.version == "1"
    assert meta.tags["suite"] == "wink-debug"
    assert meta.session_id == session_ids[0]
    assert meta.line_number == 1
    assert meta.slices[0].count == 1


def test_load_snapshot_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "missing.jsonl"

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)

    snapshot_path.write_text("")

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)

    snapshot_path.write_text("{")

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)

    payload_missing_session = {
        "version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "slices": [],
        "tags": {},
    }
    snapshot_path.write_text(json.dumps(payload_missing_session))

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.load_snapshot(snapshot_path)


def test_load_snapshot_recovers_from_unknown_types(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
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
        "tags": {"session_id": "unknown"},
    }
    snapshot_path.write_text(json.dumps(payload))

    loaded = debug_app.load_snapshot(snapshot_path)

    assert len(loaded) == 1
    entry = loaded[0]
    assert entry.meta.validation_error
    assert "__main__:UnknownType" in entry.slices
    unknown_slice = entry.slices["__main__:UnknownType"]
    assert unknown_slice.items == ({"value": "one"},)


def test_api_routes_expose_snapshot_data(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    session_ids = _write_snapshot(snapshot_path, ["a", "b", "c"])
    logger = debug_app.get_logger("test.api")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    meta_response = client.get("/api/meta")
    assert meta_response.status_code == 200
    meta = meta_response.json()
    assert meta["session_id"] == session_ids[0]
    assert meta["line_number"] == 1
    assert len(meta["slices"]) == 1
    assert meta["tags"]["suite"] == "wink-debug"
    slice_type = meta["slices"][0]["slice_type"]

    entries_response = client.get("/api/entries")
    entries = entries_response.json()
    assert [entry["session_id"] for entry in entries] == session_ids
    assert entries[0]["selected"] is True

    select_response = client.post("/api/select", json={"session_id": session_ids[1]})
    assert select_response.status_code == 200
    selected_meta = select_response.json()
    assert selected_meta["session_id"] == session_ids[1]

    detail_response = client.get(f"/api/slices/{quote(slice_type)}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    first_item = detail["items"][0]
    assert first_item["value"] == "b"
    assert first_item["__type__"] == (
        f"{_ExampleSlice.__module__}:{_ExampleSlice.__qualname__}"
    )

    raw_response = client.get("/api/raw")
    assert raw_response.status_code == 200
    raw = raw_response.json()
    assert raw["version"] == "1"
    raw_item = raw["slices"][0]["items"][0]
    assert raw_item["value"] == "b"
    assert raw_item["__type__"] == (
        f"{_ExampleSlice.__module__}:{_ExampleSlice.__qualname__}"
    )


def test_slice_pagination_with_query_params(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            _ExampleSlice: tuple(
                _ExampleSlice(value) for value in ("zero", "one", "two", "three")
            )
        },
        tags={"session_id": "paginate"},
    )
    snapshot_path.write_text(snapshot.to_json())

    logger = debug_app.get_logger("test.pagination")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]

    default_items = client.get(f"/api/slices/{quote(slice_type)}").json()["items"]
    assert [item["value"] for item in default_items] == [
        "zero",
        "one",
        "two",
        "three",
    ]

    paginated_items = client.get(
        f"/api/slices/{quote(slice_type)}", params={"offset": 1, "limit": 2}
    ).json()["items"]

    assert [item["value"] for item in paginated_items] == ["one", "two"]


def test_reload_endpoint_replaces_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["one"])
    logger = debug_app.get_logger("test.reload")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    updated_session_ids = _write_snapshot(snapshot_path, ["two", "three"])

    reload_response = client.post("/api/reload")
    assert reload_response.status_code == 200
    meta = reload_response.json()
    assert meta["session_id"] == updated_session_ids[0]
    assert meta["slices"][0]["count"] == 1
    slice_type = meta["slices"][0]["slice_type"]

    detail = client.get(f"/api/slices/{quote(slice_type)}").json()
    assert [item["value"] for item in detail["items"]] == ["two"]

    _write_snapshot(snapshot_path, ["invalid"])
    snapshot_path.write_text("not-json")
    reload_failed = client.post("/api/reload")
    assert reload_failed.status_code == 400

    meta_after_failure = client.get("/api/meta").json()
    assert meta_after_failure["session_id"] == updated_session_ids[0]
    assert meta_after_failure["slices"][0]["count"] == 1


def test_snapshot_listing_and_switch(tmp_path: Path) -> None:
    snapshot_one = tmp_path / "one.jsonl"
    snapshot_two = tmp_path / "two.jsonl"
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
    assert listing[0]["name"] == "two.jsonl"
    assert listing[1]["name"] == "one.jsonl"

    switch_response = client.post("/api/switch", json={"path": str(snapshot_two)})
    assert switch_response.status_code == 200
    switched_meta = switch_response.json()
    assert switched_meta["path"] == str(snapshot_two)

    detail = client.get("/api/meta").json()
    assert detail["path"] == str(snapshot_two)
    assert detail["session_id"] == "two-0"


def test_snapshot_store_handles_errors_and_properties(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["one"])

    missing_target = tmp_path / "missing.jsonl"
    broken_link = tmp_path / "broken.jsonl"
    broken_link.symlink_to(missing_target)

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger(__name__),
    )

    raw_payload = store.raw_payload
    assert raw_payload["version"] == "1"
    tags_value = raw_payload.get("tags")
    assert isinstance(tags_value, Mapping)
    tags = cast("Mapping[str, object]", tags_value)
    assert tags.get("suite") == "wink-debug"
    assert "session_id" in tags
    assert store.path == snapshot_path.resolve()
    assert len(store.entries) == 1

    listing = store.list_snapshots()
    names = {entry["name"] for entry in listing}
    assert "snapshot.jsonl" in names
    assert "broken.jsonl" not in names

    entry_listing = store.list_entries()
    assert entry_listing[0]["selected"] is True

    with pytest.raises(KeyError, match="Unknown slice type: missing"):
        store.slice_items("missing")


def test_snapshot_loading_ignores_blank_lines(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    session_ids = _write_snapshot(snapshot_path, ["first", "second"])
    snapshot_path.write_text("\n" + snapshot_path.read_text() + "\n\n")

    loaded = debug_app.load_snapshot(snapshot_path)

    assert [entry.meta.session_id for entry in loaded] == session_ids


def test_api_slice_offset_and_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            _ExampleSlice: (
                _ExampleSlice("one"),
                _ExampleSlice("two"),
                _ExampleSlice("three"),
            )
        },
        tags={"session_id": "multi"},
    )
    snapshot_path.write_text(snapshot.to_json())
    logger = debug_app.get_logger("test.api.slices")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    limited = client.get(f"/api/slices/{quote(slice_type)}?offset=1&limit=1").json()
    assert [item["value"] for item in limited["items"]] == ["two"]

    missing = client.get("/api/slices/unknown")
    assert missing.status_code == 404


def test_api_slice_renders_markdown(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    markdown_text = "# Heading\n\nSome **bold** markdown content."
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: (_ExampleSlice(markdown_text),)},
        tags={"session_id": "markdown"},
    )
    snapshot_path.write_text(snapshot.to_json())
    logger = debug_app.get_logger("test.api.markdown")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    detail = client.get(f"/api/slices/{quote(slice_type)}").json()

    item = detail["items"][0]["value"]
    assert item["__markdown__"]["text"] == markdown_text
    assert "<h1" in item["__markdown__"]["html"]
    assert "<strong>bold</strong>" in item["__markdown__"]["html"]


def test_markdown_renderer_handles_nested_values(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    pre_rendered = {"__markdown__": {"text": "keep", "html": "<p>keep</p>\n"}}
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            _ListSlice: (
                _ListSlice(
                    {
                        "content": ["* bullet point with detail", pre_rendered, 7],
                    }
                ),
            )
        },
        tags={"session_id": "nested"},
    )
    snapshot_path.write_text(snapshot.to_json())
    logger = debug_app.get_logger("test.api.markdown.nested")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    detail = client.get(f"/api/slices/{quote(slice_type)}").json()

    content = detail["items"][0]["value"]["content"]
    assert content[0]["__markdown__"]["html"].startswith("<ul>")
    assert content[1] == pre_rendered
    assert content[2] == 7


def test_api_select_errors_and_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["alpha"])
    logger = debug_app.get_logger("test.api.select")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    bad_session = client.post("/api/select", json={"session_id": "missing"})
    assert bad_session.status_code == 400

    bad_line = client.post("/api/select", json={"line_number": 99})
    assert bad_line.status_code == 400

    by_line = client.post("/api/select", json={"line_number": 1})
    assert by_line.status_code == 200

    snapshot_path.write_text(
        "\n".join(
            Snapshot(
                created_at=datetime.now(UTC),
                slices={_ExampleSlice: (_ExampleSlice("beta"),)},
                tags={"session_id": "beta"},
            ).to_json()
            for _ in range(1)
        )
    )
    reload_response = client.post("/api/reload")
    assert reload_response.status_code == 200
    assert reload_response.json()["session_id"] == "beta"


def test_list_snapshots_skips_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.jsonl"
    _write_snapshot(base, ["value"])
    bad = tmp_path / "bad.jsonl"
    bad.write_text("invalid")

    original_stat = Path.stat

    def fake_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path.name == "bad.jsonl":
            raise OSError("fail")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(
        debug_app.SnapshotStore,
        "_iter_snapshot_files",
        staticmethod(lambda root: [base, bad]),
    )

    store = debug_app.SnapshotStore(
        base, loader=debug_app.load_snapshot, logger=debug_app.get_logger("test.list")
    )
    entries = store.list_snapshots()

    assert [entry["name"] for entry in entries] == ["base.jsonl"]


def test_snapshot_store_reload_fallbacks(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["original"])
    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.reload_fallback"),
    )

    _write_snapshot(snapshot_path, ["replacement"])
    meta = store.reload()

    assert meta.session_id.endswith("0")


def test_snapshot_store_select_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["only"])
    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.select_errors"),
    )

    with pytest.raises(debug_app.SnapshotLoadError):
        store.select(session_id="missing")

    with pytest.raises(debug_app.SnapshotLoadError):
        store.select(line_number=99)


def test_snapshot_store_rejects_empty_loader(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    def _empty_loader(path: Path) -> tuple[debug_app.LoadedSnapshot, ...]:
        assert path == snapshot_path
        return ()

    with pytest.raises(debug_app.SnapshotLoadError):
        debug_app.SnapshotStore(
            snapshot_path,
            loader=_empty_loader,
            logger=debug_app.get_logger("test.empty_loader"),
        )


def test_snapshot_store_switch_rejects_outside_root(tmp_path: Path) -> None:
    base_snapshot = tmp_path / "base.jsonl"
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_snapshot = other_dir / "other.jsonl"

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
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["a"])

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_snapshot = other_dir / "other.jsonl"
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

    bad_switch_session = client.post(
        "/api/switch", json={"path": str(snapshot_path), "session_id": 123}
    )
    assert bad_switch_session.status_code == 400

    bad_switch_line = client.post(
        "/api/switch", json={"path": str(snapshot_path), "line_number": "one"}
    )
    assert bad_switch_line.status_code == 400

    missing_selection = client.post("/api/select", json={})
    assert missing_selection.status_code == 400
    assert "session_id or line_number is required" in missing_selection.json()["detail"]

    bad_line_number = client.post("/api/select", json={"line_number": "one"})
    assert bad_line_number.status_code == 400
    assert "line_number must be an integer" in bad_line_number.json()["detail"]

    bad_session = client.post("/api/select", json={"session_id": 123})
    assert bad_session.status_code == 400


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

        @staticmethod
        def run() -> None:
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
