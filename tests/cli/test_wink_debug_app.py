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

"""Tests for the wink debug static site generator."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from weakincentives.cli import debug_app
from weakincentives.dbc import dbc_enabled
from weakincentives.runtime.session.snapshots import Snapshot


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


@dataclass(slots=True, frozen=True)
class _ListSlice:
    value: object


@dataclass(slots=True, frozen=True)
class _MixedSlice:
    """Slice with various value types for testing markdown rendering edge cases."""

    string_value: str
    int_value: int
    bool_value: bool
    none_value: None
    dict_value: dict[str, object]


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


def test_snapshot_loading_ignores_blank_lines(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    session_ids = _write_snapshot(snapshot_path, ["first", "second"])
    snapshot_path.write_text("\n" + snapshot_path.read_text() + "\n\n")

    loaded = debug_app.load_snapshot(snapshot_path)

    assert [entry.meta.session_id for entry in loaded] == session_ids


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


def test_snapshot_store_reload_fallbacks(tmp_path: Path) -> None:
    """Test that reload falls back to index 0 when session_id no longer exists."""
    snapshot_path = tmp_path / "snapshot.jsonl"
    # Create a snapshot with a specific session_id
    original_session_id = "unique-session-12345"
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: (_ExampleSlice("original"),)},
        tags={"session_id": original_session_id},
    )
    snapshot_path.write_text(snapshot.to_json())

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.reload_fallback"),
    )

    # Now write a completely different snapshot with a different session_id
    new_snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: (_ExampleSlice("replacement"),)},
        tags={"session_id": "different-session-67890"},
    )
    snapshot_path.write_text(new_snapshot.to_json())

    # Reload - the old session_id won't exist, so it should fall back to index 0
    meta = store.reload()

    assert meta.session_id == "different-session-67890"


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


# ---------------------------------------------------------------------------
# SnapshotStore Method Coverage Tests
# ---------------------------------------------------------------------------


def test_snapshot_store_raw_text_property(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.raw_text"),
    )

    raw_text = store.raw_text
    assert "test" in raw_text
    assert "version" in raw_text


def test_snapshot_store_select_returns_meta(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    session_ids = _write_snapshot(snapshot_path, ["one", "two"])

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.select_meta"),
    )

    # Select by session_id
    meta = store.select(session_id=session_ids[1])
    assert meta.session_id == session_ids[1]

    # Select by line_number
    meta = store.select(line_number=1)
    assert meta.line_number == 1


def test_snapshot_store_switch_to_valid_path(tmp_path: Path) -> None:
    snapshot_one = tmp_path / "one.jsonl"
    snapshot_two = tmp_path / "two.jsonl"
    _write_snapshot(snapshot_one, ["first"])
    _write_snapshot(snapshot_two, ["second"])

    store = debug_app.SnapshotStore(
        snapshot_one,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.switch"),
    )

    _ = store.switch(snapshot_two, session_id=snapshot_two.stem + "-0")
    assert "second" in store.raw_text


def test_snapshot_store_switch_to_directory(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    snapshot_one = snapshots_dir / "one.jsonl"
    snapshot_two = snapshots_dir / "two.jsonl"
    _write_snapshot(snapshot_one, ["first"])
    _write_snapshot(snapshot_two, ["second"])

    # Set mtime to control ordering
    now = time.time()
    os.utime(snapshot_one, (now, now))
    os.utime(snapshot_two, (now + 1, now + 1))

    store = debug_app.SnapshotStore(
        snapshot_one,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.switch_dir"),
    )

    # Switch to the directory - should pick newest (two.jsonl)
    _ = store.switch(snapshots_dir)
    assert store.path == snapshot_two.resolve()


def test_snapshot_store_select_no_filters(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["only"])

    store = debug_app.SnapshotStore(
        snapshot_path,
        loader=debug_app.load_snapshot,
        logger=debug_app.get_logger("test.select_none"),
    )

    # Select with no filters returns index 0
    meta = store.select()
    assert meta.line_number == 1  # First entry


# ---------------------------------------------------------------------------
# Markdown Rendering Edge Cases
# ---------------------------------------------------------------------------


def test_render_markdown_values_with_existing_wrapper(tmp_path: Path) -> None:
    """Test that already-wrapped markdown values are passed through."""
    snapshot_path = tmp_path / "snapshot.jsonl"
    # Create a snapshot with a list and nested structure
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ListSlice: (_ListSlice(["# Markdown", "plain text"]),)},
        tags={"session_id": "list_test"},
    )
    snapshot_path.write_text(snapshot.to_json())
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.markdown_list"),
    )

    # Verify the list items are processed
    slices_dir = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "slices"
    )
    slice_files = list(slices_dir.glob("*.json"))
    slice_data = json.loads(slice_files[0].read_text())
    items = slice_data["items"][0]["value"]
    assert isinstance(items, list)


def test_render_markdown_values_with_primitives(tmp_path: Path) -> None:
    """Test that primitive values (int, bool, None) are passed through unchanged."""
    snapshot_path = tmp_path / "snapshot.jsonl"
    # Create a snapshot with various primitive types
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            _MixedSlice: (
                _MixedSlice(
                    string_value="hello",
                    int_value=42,
                    bool_value=True,
                    none_value=None,
                    dict_value={"nested": "value"},
                ),
            )
        },
        tags={"session_id": "primitives_test"},
    )
    snapshot_path.write_text(snapshot.to_json())
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.markdown_primitives"),
    )

    # Verify primitive values are preserved
    slices_dir = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "slices"
    )
    slice_files = list(slices_dir.glob("*.json"))
    slice_data = json.loads(slice_files[0].read_text())
    item = slice_data["items"][0]

    # Primitive values should be unchanged
    assert item["int_value"] == 42
    assert item["bool_value"] is True
    assert item["none_value"] is None
    assert item["dict_value"] == {"nested": "value"}


def test_render_markdown_values_with_existing_markdown_wrapper(tmp_path: Path) -> None:
    """Test that already-wrapped markdown mappings are passed through unchanged."""
    snapshot_path = tmp_path / "snapshot.jsonl"
    # Create a snapshot with a dict that has __markdown__ key
    # (simulates pre-rendered markdown)
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            _MixedSlice: (
                _MixedSlice(
                    string_value="normal",
                    int_value=1,
                    bool_value=False,
                    none_value=None,
                    dict_value={
                        "__markdown__": {"text": "# Already wrapped", "html": "<h1>"},
                    },
                ),
            )
        },
        tags={"session_id": "markdown_wrapper_test"},
    )
    snapshot_path.write_text(snapshot.to_json())
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.markdown_wrapper"),
    )

    # Verify the __markdown__ dict is preserved as-is
    slices_dir = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "slices"
    )
    slice_files = list(slices_dir.glob("*.json"))
    slice_data = json.loads(slice_files[0].read_text())
    item = slice_data["items"][0]

    # The __markdown__ dict should be passed through unchanged
    assert item["dict_value"]["__markdown__"]["text"] == "# Already wrapped"


# ---------------------------------------------------------------------------
# Static Site Generation Tests
# ---------------------------------------------------------------------------


def test_generate_static_site_creates_output_structure(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["one", "two"])
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.static"),
    )

    # Check directory structure
    assert (output_dir / "index.html").exists()
    assert (output_dir / "static" / "style.css").exists()
    assert (output_dir / "static" / "app.js").exists()
    assert (output_dir / "data" / "manifest.json").exists()

    # Check manifest
    manifest = json.loads((output_dir / "data" / "manifest.json").read_text())
    assert manifest["version"] == "1"
    assert len(manifest["snapshots"]) == 1
    assert manifest["default_snapshot"] == "snapshot.jsonl"

    # Check entries
    entries_path = output_dir / "data" / "snapshots" / "snapshot.jsonl" / "entries.json"
    assert entries_path.exists()
    entries = json.loads(entries_path.read_text())
    assert len(entries) == 2


def test_generate_static_site_from_directory(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    snapshot_one = snapshots_dir / "one.jsonl"
    snapshot_two = snapshots_dir / "two.jsonl"
    _write_snapshot(snapshot_one, ["a"])
    _write_snapshot(snapshot_two, ["b"])

    # Set modification times to control ordering
    now = time.time()
    os.utime(snapshot_one, (now, now))
    os.utime(snapshot_two, (now + 1, now + 1))

    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshots_dir,
        output_dir,
        logger=debug_app.get_logger("test.static.dir"),
    )

    manifest = json.loads((output_dir / "data" / "manifest.json").read_text())
    assert len(manifest["snapshots"]) == 2
    # Most recent first (two.jsonl)
    assert manifest["snapshots"][0]["file"] == "two.jsonl"
    assert manifest["default_snapshot"] == "two.jsonl"


def test_generate_static_site_writes_slice_data(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test_value"])
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.static.slices"),
    )

    # Find slice file
    slices_dir = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "slices"
    )
    slice_files = list(slices_dir.glob("*.json"))
    assert len(slice_files) == 1

    slice_data = json.loads(slice_files[0].read_text())
    assert slice_data["items"][0]["value"] == "test_value"


def test_generate_static_site_renders_markdown(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    markdown_text = "# Heading\n\nSome **bold** markdown content."
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={_ExampleSlice: (_ExampleSlice(markdown_text),)},
        tags={"session_id": "markdown"},
    )
    snapshot_path.write_text(snapshot.to_json())
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.static.markdown"),
    )

    slices_dir = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "slices"
    )
    slice_files = list(slices_dir.glob("*.json"))
    slice_data = json.loads(slice_files[0].read_text())

    item = slice_data["items"][0]["value"]
    assert item["__markdown__"]["text"] == markdown_text
    assert "<h1" in item["__markdown__"]["html"]


def test_generate_static_site_writes_raw_json(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["raw_test"])
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.static.raw"),
    )

    raw_path = (
        output_dir
        / "data"
        / "snapshots"
        / "snapshot.jsonl"
        / "entries"
        / "1"
        / "raw.json"
    )
    raw_data = json.loads(raw_path.read_text())
    assert raw_data["version"] == "1"


def test_generate_static_site_skips_invalid_files(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    valid = snapshots_dir / "valid.jsonl"
    invalid = snapshots_dir / "invalid.jsonl"
    _write_snapshot(valid, ["good"])
    invalid.write_text("not json")

    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshots_dir,
        output_dir,
        logger=debug_app.get_logger("test.static.skip"),
    )

    manifest = json.loads((output_dir / "data" / "manifest.json").read_text())
    assert len(manifest["snapshots"]) == 1
    assert manifest["snapshots"][0]["file"] == "valid.jsonl"


def test_generate_static_site_error_on_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(debug_app.SnapshotLoadError, match="No snapshot files found"):
        debug_app.generate_static_site(
            empty_dir,
            output_dir,
            logger=debug_app.get_logger("test.static.empty"),
        )


# ---------------------------------------------------------------------------
# Debug Server Tests
# ---------------------------------------------------------------------------


def test_run_debug_server_opens_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])

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

    # Mock the TCP server to not actually start
    class FakeTCPServer:
        allow_reuse_address = True

        def __init__(self, address: tuple[str, int], handler: object) -> None:
            calls["server_address"] = address
            calls["server_started"] = True

        def __enter__(self) -> FakeTCPServer:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def serve_forever(self) -> None:
            # Simulate immediate shutdown
            raise KeyboardInterrupt

    monkeypatch.setattr(debug_app.socketserver, "TCPServer", FakeTCPServer)

    logger = debug_app.get_logger("test.run")
    infos: list[dict[str, object]] = []

    def capture_info(message: str, *, event: str, context: dict[str, object]) -> None:
        infos.append({"message": message, "event": event, "context": context})

    monkeypatch.setattr(logger, "info", capture_info)

    exit_code = debug_app.run_debug_server(
        snapshot_path,
        host="0.0.0.0",
        port=8123,
        open_browser=True,
        logger=logger,
    )

    assert exit_code == 0
    assert calls["timer_started"] is True
    assert calls["browser_url"] == "http://0.0.0.0:8123/"
    assert calls["server_address"] == ("0.0.0.0", 8123)
    # Find the server start event
    start_events = [i for i in infos if i["event"] == "debug.server.start"]
    assert len(start_events) >= 1


def test_run_debug_server_without_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])

    timer_calls: list[object] = []

    class FakeTimer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            timer_calls.append((args, kwargs))

        def start(self) -> None:
            pass

    monkeypatch.setattr(debug_app.threading, "Timer", FakeTimer)

    class FakeTCPServer:
        allow_reuse_address = True

        def __init__(self, address: tuple[str, int], handler: object) -> None:
            pass

        def __enter__(self) -> FakeTCPServer:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(debug_app.socketserver, "TCPServer", FakeTCPServer)

    exit_code = debug_app.run_debug_server(
        snapshot_path,
        open_browser=False,
        logger=debug_app.get_logger("test.run.nobrowser"),
    )

    assert exit_code == 0
    assert len(timer_calls) == 0  # No timer started without browser


def test_reload_handler_regenerates_static_files(tmp_path: Path) -> None:
    """Test that the HTTP reload handler regenerates static files."""
    import io

    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])
    output_dir = tmp_path / "output"

    # Generate initial static site
    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.reload_handler"),
    )

    # Create a mock request for the reload handler
    class MockRequestHandler(debug_app._ReloadHandler):
        def __init__(self) -> None:
            self.wfile = io.BytesIO()
            self._headers_buffer: list[bytes] = []
            self.path = "/api/reload"
            self.requestline = "POST /api/reload HTTP/1.1"

        def send_response(self, code: int) -> None:
            self._response_code = code

        def send_header(self, keyword: str, value: str) -> None:
            self._headers_buffer.append(f"{keyword}: {value}".encode())

        def end_headers(self) -> None:
            pass

    # Assign class attributes
    MockRequestHandler.snapshot_path = snapshot_path
    MockRequestHandler.output_dir = output_dir
    MockRequestHandler.logger = debug_app.get_logger("test.reload_handler.mock")

    handler = MockRequestHandler()
    handler.do_POST()

    # Check response
    handler.wfile.seek(0)
    response_body = handler.wfile.read().decode()
    response = json.loads(response_body)
    assert response["success"] is True


def test_reload_handler_returns_404_for_unknown_paths(tmp_path: Path) -> None:
    """Test that the HTTP handler returns 404 for non-reload paths."""
    import io

    snapshot_path = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshot_path,
        output_dir,
        logger=debug_app.get_logger("test.404"),
    )

    class MockRequestHandler(debug_app._ReloadHandler):
        def __init__(self) -> None:
            self.wfile = io.BytesIO()
            self.path = "/api/unknown"
            self.requestline = "POST /api/unknown HTTP/1.1"
            self._error_code: int | None = None

        def send_error(self, code: int, message: str) -> None:
            self._error_code = code

    MockRequestHandler.snapshot_path = snapshot_path
    MockRequestHandler.output_dir = output_dir
    MockRequestHandler.logger = debug_app.get_logger("test.404.mock")

    handler = MockRequestHandler()
    handler.do_POST()

    assert handler._error_code == 404


def test_reload_handler_error_returns_400(tmp_path: Path) -> None:
    """Test that reload errors return 400 response."""
    import io

    # Create a valid snapshot initially in a subdirectory
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    snapshot_path = snapshots_dir / "snapshot.jsonl"
    _write_snapshot(snapshot_path, ["test"])
    output_dir = tmp_path / "output"

    debug_app.generate_static_site(
        snapshots_dir,
        output_dir,
        logger=debug_app.get_logger("test.reload_error"),
    )

    # Remove all snapshot files from the directory to cause error on reload
    for f in snapshots_dir.glob("*.jsonl"):
        f.unlink()

    class MockRequestHandler(debug_app._ReloadHandler):
        def __init__(self) -> None:
            self.wfile = io.BytesIO()
            self._headers_buffer: list[bytes] = []
            self.path = "/api/reload"
            self.requestline = "POST /api/reload HTTP/1.1"
            self._response_code: int | None = None

        def send_response(self, code: int) -> None:
            self._response_code = code

        def send_header(self, keyword: str, value: str) -> None:
            pass

        def end_headers(self) -> None:
            pass

    MockRequestHandler.snapshot_path = snapshots_dir
    MockRequestHandler.output_dir = output_dir
    MockRequestHandler.logger = debug_app.get_logger("test.reload_error.mock")

    handler = MockRequestHandler()
    handler.do_POST()

    assert handler._response_code == 400
    handler.wfile.seek(0)
    response = json.loads(handler.wfile.read().decode())
    assert response["success"] is False


def test_reload_handler_log_message_suppressed() -> None:
    """Test that the log_message method suppresses output."""

    class MockHandler(debug_app._ReloadHandler):
        def __init__(self) -> None:
            pass

    handler = MockHandler()
    # This should not raise
    handler.log_message("test %s", "arg")
