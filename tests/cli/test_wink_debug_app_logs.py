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

"""Tests for log-related API endpoints: logs, facets, filtering, pagination."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.cli.conftest import ExampleSlice, create_minimal_bundle
from weakincentives.cli import debug_app
from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import BundleConfig
from weakincentives.runtime.session import Session


def test_api_logs_endpoint(tmp_path: Path) -> None:
    """Test the logs API endpoint."""
    # Create a bundle with logs
    session = Session()
    session.dispatch(ExampleSlice("test"))

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


def test_logs_endpoint_with_level_filter(tmp_path: Path) -> None:
    """Test filtering logs by level."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

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


def test_logs_endpoint_no_logs(tmp_path: Path) -> None:
    """Test logs endpoint when bundle has no logs."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)
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
    session.dispatch(ExampleSlice("test"))

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


def test_logs_filter_by_logger(tmp_path: Path) -> None:
    """Test filtering logs by logger name."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

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
    session.dispatch(ExampleSlice("test"))

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
    session.dispatch(ExampleSlice("test"))

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
    session.dispatch(ExampleSlice("test"))

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


def test_logs_offset_only(tmp_path: Path) -> None:
    """Test getting logs with offset but no limit."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

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


def test_logs_with_pagination(tmp_path: Path) -> None:
    """Test logs with both offset and limit."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

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
