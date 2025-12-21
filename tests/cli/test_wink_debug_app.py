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
    with dbc_enabled(False):
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


def test_load_snapshot_recovers_from_unknown_policy_types(tmp_path: Path) -> None:
    """Verify snapshots with __main__ policy types can be loaded for display."""
    snapshot_path = tmp_path / "snapshot.jsonl"
    payload = {
        "version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "slices": [
            {
                "slice_type": "__main__:ReviewResponse",
                "item_type": "__main__:ReviewResponse",
                "items": [{"score": 10}],
            }
        ],
        "policies": {"__main__:ReviewResponse": "state"},
        "tags": {"session_id": "policy-test"},
    }
    snapshot_path.write_text(json.dumps(payload))

    loaded = debug_app.load_snapshot(snapshot_path)

    assert len(loaded) == 1
    entry = loaded[0]
    assert entry.meta.validation_error
    assert "__main__:ReviewResponse" in entry.slices
    review_slice = entry.slices["__main__:ReviewResponse"]
    assert review_slice.items == ({"score": 10},)


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
    tags = cast(Mapping[str, object], tags_value)
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


def test_run_debug_server_without_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that run_debug_server works without opening browser."""
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

    logger = debug_app.get_logger("test.run_no_browser")
    infos: list[dict[str, object]] = []

    def capture_info(message: str, *, event: str, context: dict[str, object]) -> None:
        infos.append({"message": message, "event": event, "context": context})

    monkeypatch.setattr(logger, "info", capture_info)

    # Ensure webbrowser.open is not called
    browser_opened = False

    def should_not_open(url: str) -> bool:
        nonlocal browser_opened
        browser_opened = True
        return True

    monkeypatch.setattr(debug_app.webbrowser, "open", should_not_open)

    app = FastAPI()

    exit_code = debug_app.run_debug_server(
        app=app,
        host="127.0.0.1",
        port=9000,
        open_browser=False,
        logger=logger,
    )

    assert exit_code == 0
    assert browser_opened is False
    assert config_calls["run_called"] is True
    assert infos[0]["event"] == "debug.server.start"


# --- Filesystem Explorer Tests ---


def _create_test_archive(path: Path, files: dict[str, str]) -> Path:
    """Create a test ZIP archive with the given files."""
    import json
    import zipfile

    archive_path = path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        total_bytes = 0
        for file_path, content in files.items():
            content_bytes = content.encode("utf-8")
            total_bytes += len(content_bytes)
            zf.writestr(file_path, content_bytes)

        # Add metadata
        metadata = {
            "version": "1",
            "created_at": "2024-01-15T10:30:00+00:00",
            "session_id": path.stem,
            "root_path": "/",
            "file_count": len(files),
            "total_bytes": total_bytes,
        }
        zf.writestr("_wink_metadata.json", json.dumps(metadata))

    return archive_path


def test_filesystem_archive_store_loads_archive(tmp_path: Path) -> None:
    """Test that FilesystemArchiveStore loads a companion archive."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    archive_path = _create_test_archive(
        snapshot_path,
        {"src/main.py": "print('hello')", "README.md": "# Test"},
    )

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    result = store.load_for_snapshot(snapshot_path)

    assert result is True
    assert store.has_archive is True
    assert store.archive_path == archive_path
    assert store.metadata is not None
    assert store.metadata.file_count == 2
    assert store.tree is not None


def test_filesystem_archive_store_no_archive(tmp_path: Path) -> None:
    """Test that FilesystemArchiveStore handles missing archives."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    result = store.load_for_snapshot(snapshot_path)

    assert result is False
    assert store.has_archive is False
    assert store.archive_path is None
    assert store.metadata is None
    assert store.tree is None


def test_filesystem_archive_store_read_file(tmp_path: Path) -> None:
    """Test reading files from an archive."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"src/main.py": "print('hello')"})

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    content = store.read_file("src/main.py")
    assert content.content == "print('hello')"
    assert content.size_bytes == len("print('hello')")
    assert content.binary is False
    assert content.error is None


def test_filesystem_archive_store_read_file_with_pagination(tmp_path: Path) -> None:
    """Test reading files with pagination."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"file.txt": "line1\nline2\nline3\nline4\n"})

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    content = store.read_file("file.txt", offset=1, limit=2)
    assert content.content == "line2\nline3\n"
    assert content.total_lines == 4
    assert content.truncated is True
    assert content.offset == 1
    assert content.limit == 2


def test_filesystem_archive_store_read_missing_file(tmp_path: Path) -> None:
    """Test reading a missing file from an archive."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"exists.txt": "content"})

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    content = store.read_file("missing.txt")
    assert content.content is None
    assert content.error is not None
    assert "not found" in content.error.lower()


def test_filesystem_tree_api_route(tmp_path: Path) -> None:
    """Test the /api/filesystem/tree route."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(
        snapshot_path,
        {
            "src/main.py": "print('hello')",
            "src/utils.py": "# utils",
            "README.md": "# Test",
        },
    )

    logger = debug_app.get_logger("test.fs.tree")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/tree")
    assert response.status_code == 200

    data = response.json()
    assert data["has_archive"] is True
    assert data["metadata"]["file_count"] == 3
    assert data["tree"]["name"] == "/"
    assert data["tree"]["type"] == "directory"


def test_filesystem_file_api_route(tmp_path: Path) -> None:
    """Test the /api/filesystem/file route."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"src/main.py": "print('hello')"})

    logger = debug_app.get_logger("test.fs.file")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/file/src/main.py")
    assert response.status_code == 200

    data = response.json()
    assert data["content"] == "print('hello')"
    assert data["binary"] is False


def test_filesystem_file_api_route_with_pagination(tmp_path: Path) -> None:
    """Test the /api/filesystem/file route with pagination."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"file.txt": "line1\nline2\nline3\n"})

    logger = debug_app.get_logger("test.fs.file.pagination")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/file/file.txt?offset=1&limit=1")
    assert response.status_code == 200

    data = response.json()
    assert data["content"] == "line2\n"
    assert data["truncated"] is True


def test_filesystem_download_api_route(tmp_path: Path) -> None:
    """Test the /api/filesystem/download route."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"src/main.py": "print('hello')"})

    logger = debug_app.get_logger("test.fs.download")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/download/src/main.py")
    assert response.status_code == 200
    assert response.content == b"print('hello')"
    assert "attachment" in response.headers["content-disposition"]


def test_filesystem_download_missing_file(tmp_path: Path) -> None:
    """Test downloading a missing file returns 404."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"exists.txt": "content"})

    logger = debug_app.get_logger("test.fs.download.missing")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/download/missing.txt")
    assert response.status_code == 404


def test_filesystem_tree_no_archive(tmp_path: Path) -> None:
    """Test /api/filesystem/tree when no archive exists."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    logger = debug_app.get_logger("test.fs.tree.noarchive")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/tree")
    assert response.status_code == 200

    data = response.json()
    assert data["has_archive"] is False
    assert data["archive_path"] is None
    assert data["metadata"] is None
    assert data["tree"] is None


def test_filesystem_switch_reloads_archive(tmp_path: Path) -> None:
    """Test that switching snapshots reloads the filesystem archive."""
    snapshot_one = tmp_path / "one.jsonl"
    snapshot_two = tmp_path / "two.jsonl"
    _write_snapshot(snapshot_one, ["a"])
    _write_snapshot(snapshot_two, ["b"])

    _create_test_archive(snapshot_one, {"one.txt": "one content"})
    _create_test_archive(snapshot_two, {"two.txt": "two content"})

    logger = debug_app.get_logger("test.fs.switch")
    store = debug_app.SnapshotStore(
        snapshot_one, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    # Initial check - should have one.txt
    tree_response = client.get("/api/filesystem/tree")
    assert tree_response.json()["has_archive"] is True

    file_response = client.get("/api/filesystem/file/one.txt")
    assert file_response.json()["content"] == "one content"

    # Switch to snapshot two
    switch_response = client.post("/api/switch", json={"path": str(snapshot_two)})
    assert switch_response.status_code == 200

    # Should now have two.txt
    file_response = client.get("/api/filesystem/file/two.txt")
    assert file_response.json()["content"] == "two content"

    # one.txt should not be found
    old_file_response = client.get("/api/filesystem/file/one.txt")
    assert old_file_response.json()["error"] is not None


def test_filesystem_archive_store_read_no_archive(tmp_path: Path) -> None:
    """Test reading file when no archive is loaded."""
    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    # Don't load any archive
    content = store.read_file("any.txt")
    assert content.error == "No archive loaded"


def test_filesystem_archive_store_read_raw_no_archive(tmp_path: Path) -> None:
    """Test reading raw file when no archive is loaded."""
    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    raw = store.read_file_raw("any.txt")
    assert raw is None


def test_filesystem_archive_store_binary_file(tmp_path: Path) -> None:
    """Test reading a binary file from an archive."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create archive with binary content
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write binary content that isn't valid UTF-8
        zf.writestr("binary.dat", bytes([0xFF, 0xFE, 0x00, 0x01]))
        # Add metadata
        metadata = {
            "version": "1",
            "created_at": "2024-01-15T10:30:00+00:00",
            "session_id": "test",
            "root_path": "/",
            "file_count": 1,
            "total_bytes": 4,
        }
        zf.writestr("_wink_metadata.json", json.dumps(metadata))

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    content = store.read_file("binary.dat")
    assert content.binary is True
    assert content.content is None
    assert "Binary" in (content.error or "")


def test_filesystem_archive_store_missing_metadata(tmp_path: Path) -> None:
    """Test loading archive without metadata file."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create archive without metadata
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file.txt", "content")

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    result = store.load_for_snapshot(snapshot_path)

    assert result is True
    assert store.has_archive is True
    # Metadata should be None when not present
    assert store.metadata is None


def test_filesystem_tree_api_with_missing_metadata(tmp_path: Path) -> None:
    """Test /api/filesystem/tree when archive has no metadata."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create archive without metadata
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file.txt", "content")

    logger = debug_app.get_logger("test.fs.tree.nometadata")
    store = debug_app.SnapshotStore(
        snapshot_path, loader=debug_app.load_snapshot, logger=logger
    )
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/filesystem/tree")
    assert response.status_code == 200

    data = response.json()
    assert data["has_archive"] is True
    # metadata should be None since archive has no metadata file
    assert data["metadata"] is None
    # tree should still exist
    assert data["tree"] is not None


def test_filesystem_archive_store_corrupted_archive(tmp_path: Path) -> None:
    """Test loading a corrupted archive file."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create corrupted archive
    archive_path = snapshot_path.with_suffix(".fs.zip")
    archive_path.write_bytes(b"not a zip file")

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    result = store.load_for_snapshot(snapshot_path)

    assert result is False
    assert store.has_archive is False


def test_filesystem_archive_store_corrupted_metadata(tmp_path: Path) -> None:
    """Test loading archive with corrupted metadata."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create archive with invalid JSON metadata
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file.txt", "content")
        zf.writestr("_wink_metadata.json", "not valid json")

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    result = store.load_for_snapshot(snapshot_path)

    # Should still load successfully with fallback metadata
    assert result is True
    assert store.has_archive is True
    assert store.metadata is not None
    assert store.metadata.file_count == 1


def test_filesystem_archive_store_read_file_archive_error(tmp_path: Path) -> None:
    """Test reading file when archive read fails."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create valid archive first
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file.txt", "content")
        metadata = {"version": "1", "session_id": "test", "file_count": 1}
        zf.writestr("_wink_metadata.json", json.dumps(metadata))

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    # Now corrupt the archive
    archive_path.write_bytes(b"corrupted")

    content = store.read_file("file.txt")
    assert content.error is not None
    assert "Failed" in content.error


def test_filesystem_archive_store_read_raw_archive_error(tmp_path: Path) -> None:
    """Test reading raw file when archive read fails."""
    import zipfile

    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])

    # Create valid archive first
    archive_path = snapshot_path.with_suffix(".fs.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file.txt", "content")
        metadata = {"version": "1", "session_id": "test", "file_count": 1}
        zf.writestr("_wink_metadata.json", json.dumps(metadata))

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    # Now corrupt the archive
    archive_path.write_bytes(b"corrupted")

    raw = store.read_file_raw("file.txt")
    assert raw is None


def test_filesystem_archive_store_read_raw_missing_file(tmp_path: Path) -> None:
    """Test reading raw file that doesn't exist."""
    snapshot_path = tmp_path / "session.jsonl"
    _write_snapshot(snapshot_path, ["value"])
    _create_test_archive(snapshot_path, {"exists.txt": "content"})

    store = debug_app.FilesystemArchiveStore(logger=debug_app.get_logger("test.fs"))
    store.load_for_snapshot(snapshot_path)

    raw = store.read_file_raw("missing.txt")
    assert raw is None
