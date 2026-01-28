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
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weakincentives.cli import debug_app
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_ROOT_DIR,
    BundleConfig,
    BundleManifest,
    BundleWriter,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


@dataclass(slots=True, frozen=True)
class _ListSlice:
    value: object


def _create_test_bundle(
    target_dir: Path,
    values: list[str],
) -> Path:
    """Create a test debug bundle with session data."""
    session = Session()

    for value in values:
        session.dispatch(_ExampleSlice(value))

    with BundleWriter(
        target_dir,
        config=BundleConfig(),
    ) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})
        writer.write_config({"adapter": "test"})
        writer.write_metrics({"tokens": 100})

    assert writer.path is not None
    return writer.path


def _create_minimal_bundle(
    target_dir: Path,
    session_content: str | None = None,
    manifest_override: dict[str, object] | None = None,
) -> Path:
    """Create a minimal bundle directly for edge case testing."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    zip_path = target_dir / zip_name

    manifest = BundleManifest(
        format_version=BUNDLE_FORMAT_VERSION,
        bundle_id=bundle_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    manifest_dict = json.loads(manifest.to_json())
    if manifest_override:
        manifest_dict.update(manifest_override)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/manifest.json",
            json.dumps(manifest_dict, indent=2),
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json",
            json.dumps({"task": "test"}),
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json",
            json.dumps({"status": "ok"}),
        )
        if session_content:
            zf.writestr(
                f"{BUNDLE_ROOT_DIR}/session/after.jsonl",
                session_content,
            )

    return zip_path


def test_bundle_store_loads_bundle(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["one"])

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))
    meta = store.get_meta()

    assert meta["bundle_id"]
    assert meta["status"] == "success"
    assert len(meta["slices"]) == 1


def test_bundle_store_errors_on_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.zip"

    with pytest.raises(debug_app.BundleLoadError):
        debug_app.BundleStore(missing_path, logger=debug_app.get_logger("test"))


def test_bundle_store_errors_on_invalid(tmp_path: Path) -> None:
    # Invalid zip file
    invalid_path = tmp_path / "invalid.zip"
    invalid_path.write_text("not a zip")

    with pytest.raises(debug_app.BundleLoadError):
        debug_app.BundleStore(invalid_path, logger=debug_app.get_logger("test"))


def test_api_routes_expose_bundle_data(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["a", "b", "c"])
    logger = debug_app.get_logger("test.api")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    meta_response = client.get("/api/meta")
    assert meta_response.status_code == 200
    meta = meta_response.json()
    assert meta["bundle_id"]
    assert meta["status"] == "success"
    assert len(meta["slices"]) == 1
    slice_type = meta["slices"][0]["slice_type"]

    manifest_response = client.get("/api/manifest")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["format_version"] == BUNDLE_FORMAT_VERSION

    detail_response = client.get(f"/api/slices/{quote(slice_type)}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    # Items are in reverse order (last dispatched first in snapshot)
    values = [item["value"] for item in detail["items"]]
    assert set(values) == {"a", "b", "c"}

    request_input = client.get("/api/request/input")
    assert request_input.status_code == 200
    assert request_input.json() == {"task": "test"}

    request_output = client.get("/api/request/output")
    assert request_output.status_code == 200
    assert request_output.json() == {"status": "ok"}


def test_slice_pagination_with_query_params(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["zero", "one", "two", "three"])

    logger = debug_app.get_logger("test.pagination")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]

    default_items = client.get(f"/api/slices/{quote(slice_type)}").json()["items"]
    values = [item["value"] for item in default_items]
    assert len(values) == 4

    paginated_items = client.get(
        f"/api/slices/{quote(slice_type)}", params={"offset": 1, "limit": 2}
    ).json()["items"]

    assert len(paginated_items) == 2


def test_reload_endpoint_replaces_bundle(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["one"])
    logger = debug_app.get_logger("test.reload")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    original_bundle_id = client.get("/api/meta").json()["bundle_id"]

    reload_response = client.post("/api/reload")
    assert reload_response.status_code == 200
    meta = reload_response.json()
    assert meta["bundle_id"] == original_bundle_id

    # Corrupt the bundle
    bundle_path.write_text("not-a-zip")
    reload_failed = client.post("/api/reload")
    assert reload_failed.status_code == 400


def test_bundle_listing_and_switch(tmp_path: Path) -> None:
    bundle_one = _create_test_bundle(tmp_path, ["a"])
    time.sleep(0.01)
    bundle_two = _create_test_bundle(tmp_path, ["b", "c"])

    now = time.time()
    time_one = now - 1
    time_two = now
    os.utime(bundle_one, (time_one, time_one))
    os.utime(bundle_two, (time_two, time_two))

    logger = debug_app.get_logger("test.switch")
    store = debug_app.BundleStore(bundle_one, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    listing = client.get("/api/bundles").json()
    assert len(listing) == 2
    # Most recent first
    assert listing[0]["path"] == str(bundle_two)

    switch_response = client.post("/api/switch", json={"path": str(bundle_two)})
    assert switch_response.status_code == 200
    switched_meta = switch_response.json()
    assert switched_meta["path"] == str(bundle_two)


def test_bundle_store_handles_errors(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["one"])

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger(__name__))

    assert store.path == bundle_path.resolve()

    listing = store.list_bundles()
    assert len(listing) == 1
    assert listing[0]["selected"] is True

    with pytest.raises(KeyError, match="Unknown slice type: missing"):
        store.get_slice_items("missing")


def test_api_slice_offset_and_errors(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["one", "two", "three"])
    logger = debug_app.get_logger("test.api.slices")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    limited = client.get(f"/api/slices/{quote(slice_type)}?offset=1&limit=1").json()
    assert len(limited["items"]) == 1

    missing = client.get("/api/slices/unknown")
    assert missing.status_code == 404


def test_api_slice_renders_markdown(tmp_path: Path) -> None:
    markdown_text = "# Heading\n\nSome **bold** markdown content."
    session = Session()
    session.dispatch(_ExampleSlice(markdown_text))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})

    assert writer.path is not None
    logger = debug_app.get_logger("test.api.markdown")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    detail = client.get(f"/api/slices/{quote(slice_type)}").json()

    item = detail["items"][0]["value"]
    assert item["__markdown__"]["text"] == markdown_text
    assert "<h1" in item["__markdown__"]["html"]
    assert "<strong>bold</strong>" in item["__markdown__"]["html"]


def test_markdown_renderer_handles_nested_values(tmp_path: Path) -> None:
    pre_rendered = {"__markdown__": {"text": "keep", "html": "<p>keep</p>\n"}}
    session = Session()
    session.dispatch(
        _ListSlice(
            {
                "content": ["* bullet point with detail", pre_rendered, 7],
            }
        )
    )

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})

    assert writer.path is not None
    logger = debug_app.get_logger("test.api.markdown.nested")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    detail = client.get(f"/api/slices/{quote(slice_type)}").json()

    content = detail["items"][0]["value"]["content"]
    assert content[0]["__markdown__"]["html"].startswith("<ul>")
    assert content[1] == pre_rendered
    assert content[2] == 7


def test_bundle_store_switch_rejects_outside_root(tmp_path: Path) -> None:
    base_bundle = _create_test_bundle(tmp_path, ["base"])
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_bundle = _create_test_bundle(other_dir, ["other"])

    store = debug_app.BundleStore(
        base_bundle, logger=debug_app.get_logger("test.switch_root")
    )

    with pytest.raises(debug_app.BundleLoadError) as excinfo:
        store.switch(other_bundle)

    assert "Bundle must live under" in str(excinfo.value)


def test_normalize_path_requires_bundles(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(debug_app.BundleLoadError) as excinfo:
        debug_app.BundleStore(empty_dir, logger=debug_app.get_logger("test.empty"))

    assert "No bundles found" in str(excinfo.value)


def test_index_and_error_endpoints(tmp_path: Path) -> None:
    bundle_path = _create_test_bundle(tmp_path, ["a"])

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_bundle = _create_test_bundle(other_dir, ["b"])

    logger = debug_app.get_logger("test.routes")
    store = debug_app.BundleStore(bundle_path, logger=logger)
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

    wrong_root = client.post("/api/switch", json={"path": str(other_bundle)})
    assert wrong_root.status_code == 400
    assert "Bundle must live under" in wrong_root.json()["detail"]


def test_api_logs_endpoint(tmp_path: Path) -> None:
    """Test the logs API endpoint."""
    # Create a bundle with logs
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            logging.getLogger("test.logs").info("Test log message")

    assert writer.path is not None
    logger = debug_app.get_logger("test.logs.api")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    logs_response = client.get("/api/logs")
    assert logs_response.status_code == 200
    logs = logs_response.json()
    assert "entries" in logs
    assert "total" in logs


def test_api_transcript_endpoint(tmp_path: Path) -> None:
    """Test the transcript API endpoint."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.collector.entry",
                    "context": {
                        "prompt_name": "test",
                        "transcript_source": "main",
                        "entry_type": "user",
                        "sequence_number": 1,
                        "raw_json": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hi"},
                            }
                        ),
                        "parsed": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    transcript_response = client.get("/api/transcript")
    assert transcript_response.status_code == 200
    transcript = transcript_response.json()
    assert transcript["total"] >= 1
    assert transcript["entries"][0]["entry_type"] == "user"

    facets_response = client.get("/api/transcript/facets")
    assert facets_response.status_code == 200
    facets = facets_response.json()
    assert "sources" in facets


def test_api_config_and_metrics_endpoints(tmp_path: Path) -> None:
    """Test config and metrics endpoints."""
    bundle_path = _create_test_bundle(tmp_path, ["test"])
    logger = debug_app.get_logger("test.config.metrics")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    config_response = client.get("/api/config")
    assert config_response.status_code == 200
    assert config_response.json() == {"adapter": "test"}

    metrics_response = client.get("/api/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json() == {"tokens": 100}


def test_api_error_endpoint_not_found(tmp_path: Path) -> None:
    """Test error endpoint returns 404 when no error in bundle."""
    bundle_path = _create_test_bundle(tmp_path, ["test"])
    logger = debug_app.get_logger("test.error")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    error_response = client.get("/api/error")
    assert error_response.status_code == 404


def test_api_files_endpoints(tmp_path: Path) -> None:
    """Test file listing and content endpoints."""
    bundle_path = _create_test_bundle(tmp_path, ["test"])
    logger = debug_app.get_logger("test.files")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    files_response = client.get("/api/files")
    assert files_response.status_code == 200
    files = files_response.json()
    assert "manifest.json" in files
    assert "request/input.json" in files

    file_content = client.get("/api/files/manifest.json")
    assert file_content.status_code == 200
    content = file_content.json()
    assert content["type"] == "json"

    missing_file = client.get("/api/files/nonexistent.json")
    assert missing_file.status_code == 404


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


def test_bundle_without_session(tmp_path: Path) -> None:
    """Test loading a bundle without session data."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))
    meta = store.get_meta()

    assert meta["bundle_id"]
    assert len(meta["slices"]) == 0


def test_logs_endpoint_with_level_filter(tmp_path: Path) -> None:
    """Test filtering logs by level."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            test_logger = logging.getLogger("test.level_filter")
            test_logger.setLevel(logging.DEBUG)
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")

    assert writer.path is not None
    logger = debug_app.get_logger("test.logs.filter")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    # Filter by WARNING level
    warning_logs = client.get("/api/logs", params={"level": "WARNING"}).json()
    assert warning_logs["total"] >= 0


def test_list_bundles_with_broken_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that listing bundles skips files that fail stat."""
    good_bundle = _create_test_bundle(tmp_path, ["value"])
    bad = tmp_path / "bad.zip"
    bad.write_text("invalid")

    # Create store first, before monkeypatching stat
    store = debug_app.BundleStore(good_bundle, logger=debug_app.get_logger("test.list"))

    original_stat = Path.stat

    def fake_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path.name == "bad.zip":
            raise OSError("fail")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)

    # Patch iter_bundle_files where it's imported and used (debug_app module)
    monkeypatch.setattr(
        debug_app,
        "iter_bundle_files",
        lambda root: [good_bundle, bad],
    )

    entries = store.list_bundles()

    assert len(entries) == 1
    assert entries[0]["name"] == good_bundle.name


def test_logs_endpoint_no_logs(tmp_path: Path) -> None:
    """Test logs endpoint when bundle has no logs."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)
    logger = debug_app.get_logger("test.logs.none")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    logs_response = client.get("/api/logs")
    assert logs_response.status_code == 200
    logs = logs_response.json()
    assert logs["entries"] == []
    assert logs["total"] == 0


def test_logs_facets_endpoint(tmp_path: Path) -> None:
    """Test log facets endpoint returns unique loggers and events."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            logger1 = logging.getLogger("app.service")
            logger1.setLevel(logging.INFO)
            logger1.info("Service started", extra={"event": "service.start"})

            logger2 = logging.getLogger("app.database")
            logger2.setLevel(logging.INFO)
            logger2.info("DB connected", extra={"event": "db.connect"})
            logger2.info("Query executed", extra={"event": "db.query"})

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.facets")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    facets = client.get("/api/logs/facets").json()

    assert "loggers" in facets
    assert "events" in facets
    assert "levels" in facets
    assert len(facets["loggers"]) >= 2
    assert len(facets["events"]) >= 2


def test_transcript_endpoints_and_filters(tmp_path: Path) -> None:
    """Test transcript endpoints return entries and support filtering."""
    import logging

    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)

            transcript_logger.debug(
                "Transcript entry: user",
                extra={
                    "event": "transcript.collector.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "transcript_source": "main",
                        "entry_type": "user",
                        "sequence_number": 1,
                        "raw_json": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hello"},
                            }
                        ),
                        "parsed": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hello"},
                        },
                    },
                },
            )

            transcript_logger.debug(
                "Transcript entry: assistant",
                extra={
                    "event": "transcript.collector.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "transcript_source": "main",
                        "entry_type": "assistant",
                        "sequence_number": 2,
                        "raw_json": None,
                        "parsed": {
                            "type": "assistant",
                            "message": {"role": "assistant", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    result = client.get("/api/transcript").json()
    assert result["total"] == 2
    assert len(result["entries"]) == 2
    assert result["entries"][0]["transcript_source"] == "main"

    facets = client.get("/api/transcript/facets").json()
    assert any(item["name"] == "main" for item in facets["sources"])
    assert any(item["name"] == "user" for item in facets["entry_types"])
    assert any(item["name"] == "assistant" for item in facets["entry_types"])

    # Search (server-side)
    searched = client.get("/api/transcript", params={"search": "Hello"}).json()
    assert searched["total"] == 1
    assert searched["entries"][0]["entry_type"] == "user"

    # Include filters
    filtered = client.get(
        "/api/transcript", params={"source": "main", "entry_type": "assistant"}
    ).json()
    assert filtered["total"] == 1
    assert filtered["entries"][0]["entry_type"] == "assistant"

    # Exclude filters
    excluded = client.get(
        "/api/transcript", params={"exclude_entry_type": "assistant"}
    ).json()
    assert excluded["total"] == 1
    assert excluded["entries"][0]["entry_type"] == "user"

    # Pagination branches
    limited = client.get("/api/transcript", params={"limit": 1}).json()
    assert limited["total"] == 2
    assert len(limited["entries"]) == 1

    offset_only = client.get("/api/transcript", params={"offset": 1}).json()
    assert offset_only["total"] == 2
    assert len(offset_only["entries"]) == 1


def test_logs_filter_by_logger(tmp_path: Path) -> None:
    """Test filtering logs by logger name."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            logger1 = logging.getLogger("app.service")
            logger1.setLevel(logging.INFO)
            logger1.info("Service message")

            logger2 = logging.getLogger("app.database")
            logger2.setLevel(logging.INFO)
            logger2.info("Database message")

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.filter.logger")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    # Filter by specific logger
    logs = client.get("/api/logs", params={"logger": "app.service"}).json()
    assert all(
        "service" in entry.get("logger", "").lower() for entry in logs["entries"]
    )


def test_logs_filter_by_event(tmp_path: Path) -> None:
    """Test filtering logs by event name."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            test_log = logging.getLogger("test.events")
            test_log.setLevel(logging.INFO)
            test_log.info("Event A", extra={"event": "event.a"})
            test_log.info("Event B", extra={"event": "event.b"})

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.filter.event")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    # Filter by specific event
    logs = client.get("/api/logs", params={"event": "event.a"}).json()
    assert all(entry.get("event") == "event.a" for entry in logs["entries"])


def test_logs_exclude_logger(tmp_path: Path) -> None:
    """Test excluding logs by logger name."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            logger1 = logging.getLogger("keep.this")
            logger1.setLevel(logging.INFO)
            logger1.info("Keep message")

            logger2 = logging.getLogger("exclude.this")
            logger2.setLevel(logging.INFO)
            logger2.info("Exclude message")

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.exclude.logger")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    # Exclude specific logger
    logs = client.get("/api/logs", params={"exclude_logger": "exclude.this"}).json()
    assert all(entry.get("logger") != "exclude.this" for entry in logs["entries"])


def test_logs_search_filter(tmp_path: Path) -> None:
    """Test full-text search in logs."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            test_log = logging.getLogger("test.search")
            test_log.setLevel(logging.INFO)
            test_log.info("Find the needle in the haystack")
            test_log.info("Another message without it")

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.search")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    # Search for "needle"
    logs = client.get("/api/logs", params={"search": "needle"}).json()
    assert logs["total"] >= 1
    assert any(
        "needle" in entry.get("message", "").lower() for entry in logs["entries"]
    )


def test_config_endpoint_missing(tmp_path: Path) -> None:
    """Test config endpoint returns 404 when no config in bundle."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)
    logger = debug_app.get_logger("test.config.missing")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    config_response = client.get("/api/config")
    assert config_response.status_code == 404


def test_metrics_endpoint_missing(tmp_path: Path) -> None:
    """Test metrics endpoint returns 404 when no metrics in bundle."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)
    logger = debug_app.get_logger("test.metrics.missing")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    metrics_response = client.get("/api/metrics")
    assert metrics_response.status_code == 404


def test_error_endpoint_with_error(tmp_path: Path) -> None:
    """Test error endpoint returns error when present in bundle."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    # Manually add error.json
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr(
            "debug_bundle/error.json",
            '{"type": "ValueError", "message": "test error"}',
        )

    logger = debug_app.get_logger("test.error.present")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    error_response = client.get("/api/error")
    assert error_response.status_code == 200
    error = error_response.json()
    assert error["type"] == "ValueError"


def test_file_endpoint_text_file(tmp_path: Path) -> None:
    """Test reading a non-JSON text file."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    # Manually add a text file
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/README.txt", "This is plain text content")

    logger = debug_app.get_logger("test.file.text")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    file_response = client.get("/api/files/README.txt")
    assert file_response.status_code == 200
    content = file_response.json()
    assert content["type"] == "text"
    assert content["content"] == "This is plain text content"


def test_file_endpoint_binary_file(tmp_path: Path) -> None:
    """Test reading a binary file."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    # Manually add a binary file
    with zipfile.ZipFile(bundle_path, "a") as zf:
        # Add binary data that can't be decoded as UTF-8
        zf.writestr("debug_bundle/binary.dat", b"\x80\x81\x82\x83\xff\xfe")

    logger = debug_app.get_logger("test.file.binary")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    file_response = client.get("/api/files/binary.dat")
    assert file_response.status_code == 200
    content = file_response.json()
    assert content["type"] == "binary"
    assert content["content"] is None


def test_bundle_store_close(tmp_path: Path) -> None:
    """Test that BundleStore can be closed."""
    bundle_path = _create_test_bundle(tmp_path, ["one"])
    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Should be able to close without error
    store.close()

    # Closing again should also work
    store.close()


def test_bundle_store_from_directory(tmp_path: Path) -> None:
    """Test creating BundleStore from a directory with multiple bundles."""
    # Create two bundles
    bundle_one = _create_test_bundle(tmp_path, ["a"])
    time.sleep(0.01)
    bundle_two = _create_test_bundle(tmp_path, ["b"])

    # Set mtimes to make bundle_two newest
    now = time.time()
    os.utime(bundle_one, (now - 1, now - 1))
    os.utime(bundle_two, (now, now))

    # Create store from directory - should pick newest bundle
    store = debug_app.BundleStore(tmp_path, logger=debug_app.get_logger("test.dir"))

    assert store.path == bundle_two.resolve()


def test_slice_offset_only(tmp_path: Path) -> None:
    """Test getting slices with offset but no limit."""
    bundle_path = _create_test_bundle(tmp_path, ["a", "b", "c", "d"])
    logger = debug_app.get_logger("test.slice.offset")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    result = client.get(f"/api/slices/{quote(slice_type)}", params={"offset": 2}).json()

    # Should return items starting from offset 2
    assert len(result["items"]) == 2


def test_logs_offset_only(tmp_path: Path) -> None:
    """Test getting logs with offset but no limit."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            test_logger = logging.getLogger("test.offset")
            test_logger.setLevel(logging.INFO)
            for i in range(5):
                test_logger.info(f"Message {i}")

    assert writer.path is not None
    logger = debug_app.get_logger("test.logs.offset")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    result = client.get("/api/logs", params={"offset": 2}).json()
    # Should return logs starting from offset 2
    assert result["total"] >= 5


def test_slice_offset_beyond_count(tmp_path: Path) -> None:
    """Test getting slices with offset beyond item count returns empty."""
    bundle_path = _create_test_bundle(tmp_path, ["a", "b"])
    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Get meta to find slice type
    meta = store.get_meta()
    slice_type = meta["slices"][0]["slice_type"]

    # Query with offset beyond the number of items (2)
    result = store.get_slice_items(str(slice_type), offset=100, limit=10)

    # Should return empty items without raising KeyError
    assert result["items"] == []


def test_reload_without_existing_cache(tmp_path: Path) -> None:
    """Test reload when cache file doesn't exist yet."""
    bundle_path = _create_test_bundle(tmp_path, ["test"])
    cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Cache should exist now
    assert cache_path.exists()

    # Remove cache
    cache_path.unlink()

    # Reload should still work (it should handle missing cache)
    result = store.reload()
    assert result["bundle_id"]


def test_logs_with_pagination(tmp_path: Path) -> None:
    """Test logs with both offset and limit."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            test_logger = logging.getLogger("test.pagination")
            test_logger.setLevel(logging.INFO)
            for i in range(10):
                test_logger.info(f"Message {i}")

    assert writer.path is not None
    logger = debug_app.get_logger("test.logs.pagination")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    result = client.get("/api/logs", params={"offset": 2, "limit": 3}).json()
    assert len(result["entries"]) == 3
