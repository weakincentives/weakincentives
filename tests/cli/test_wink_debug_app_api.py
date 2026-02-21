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

"""Tests for core API routes: meta, manifest, slices, reload, switch, static."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.cli.conftest import ExampleSlice, ListSlice, create_test_bundle
from weakincentives.cli import debug_app
from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleConfig,
)
from weakincentives.runtime.session import Session


def test_api_routes_expose_bundle_data(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["a", "b", "c"])
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


def test_slice_pagination_with_query_params(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["zero", "one", "two", "three"])

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
    bundle_path = create_test_bundle(tmp_path, ["one"])
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
    bundle_one = create_test_bundle(tmp_path, ["a"])
    time.sleep(0.01)
    bundle_two = create_test_bundle(tmp_path, ["b", "c"])

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


def test_api_slice_offset_and_errors(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["one", "two", "three"])
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
    session.dispatch(ExampleSlice(markdown_text))

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
        ListSlice(
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


def test_index_and_error_endpoints(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["a"])

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_bundle = create_test_bundle(other_dir, ["b"])

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


def test_static_files_have_no_cache_header(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["a"])
    logger = debug_app.get_logger("test.routes")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"


def test_api_config_and_metrics_endpoints(tmp_path: Path) -> None:
    """Test config and metrics endpoints."""
    bundle_path = create_test_bundle(tmp_path, ["test"])
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
    bundle_path = create_test_bundle(tmp_path, ["test"])
    logger = debug_app.get_logger("test.error")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    error_response = client.get("/api/error")
    assert error_response.status_code == 404


def test_slice_offset_only(tmp_path: Path) -> None:
    """Test getting slices with offset but no limit."""
    bundle_path = create_test_bundle(tmp_path, ["a", "b", "c", "d"])
    logger = debug_app.get_logger("test.slice.offset")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    slice_type = client.get("/api/meta").json()["slices"][0]["slice_type"]
    result = client.get(f"/api/slices/{quote(slice_type)}", params={"offset": 2}).json()

    # Should return items starting from offset 2
    assert len(result["items"]) == 2


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
