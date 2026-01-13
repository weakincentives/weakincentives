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

"""Tests for the log collector context manager."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from weakincentives.debug import collect_all_logs
from weakincentives.runtime import get_logger


@pytest.fixture(autouse=True)
def enable_debug_logging() -> Iterator[None]:
    """Enable debug-level logging for all test loggers."""
    test_logger = logging.getLogger("weakincentives.test")
    original_level = test_logger.level
    test_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        test_logger.setLevel(original_level)


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """Return a temporary log file path."""
    return tmp_path / "test_logs.jsonl"


def test_collect_all_logs_writes_structured_logs(log_file: Path) -> None:
    """Verify that structured logs are written to file."""
    structured = get_logger("weakincentives.test.debug")

    with collect_all_logs(log_file):
        structured.info(
            "Test message",
            event="test.event",
            context={"key": "value"},
        )

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["event"] == "test.event"
    assert entry["message"] == "Test message"
    assert entry["context"]["key"] == "value"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "weakincentives.test.debug"
    assert "T" in entry["timestamp"]  # ISO format


def test_collect_all_logs_writes_multiple_entries(log_file: Path) -> None:
    """Verify that multiple logs are written to the JSONL file."""
    structured = get_logger("weakincentives.test.debug")

    with collect_all_logs(log_file):
        structured.debug("Debug message", event="debug.event", context={})
        structured.info("Info message", event="info.event", context={"a": 1})

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["event"] == "debug.event"
    assert first["level"] == "DEBUG"

    second = json.loads(lines[1])
    assert second["event"] == "info.event"
    assert second["context"]["a"] == 1


def test_collect_all_logs_creates_parent_directories(tmp_path: Path) -> None:
    """Verify that parent directories are created if they don't exist."""
    log_file = tmp_path / "nested" / "dirs" / "logs.jsonl"
    structured = get_logger("weakincentives.test.debug")

    with collect_all_logs(log_file):
        structured.info("Test", event="test.event", context={})

    assert log_file.exists()
    assert log_file.parent.exists()


def test_collect_all_logs_respects_level_filter(log_file: Path) -> None:
    """Verify that level filtering works."""
    structured = get_logger("weakincentives.test.debug")

    with collect_all_logs(log_file, level=logging.WARNING):
        structured.debug("Debug", event="debug.skip", context={})
        structured.info("Info", event="info.skip", context={})
        structured.warning("Warning", event="warning.capture", context={})
        structured.error("Error", event="error.capture", context={})

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    assert json.loads(lines[0])["event"] == "warning.capture"
    assert json.loads(lines[1])["event"] == "error.capture"


def test_collect_all_logs_captures_all_loggers(log_file: Path) -> None:
    """Verify that logs from all logger hierarchies are captured."""
    other_logger = get_logger("other.namespace")
    target_logger = get_logger("myapp.target")

    logging.getLogger("myapp").setLevel(logging.DEBUG)
    logging.getLogger("other").setLevel(logging.DEBUG)

    with collect_all_logs(log_file):
        other_logger.info("From other", event="other.event", context={})
        target_logger.info("From target", event="target.event", context={})

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "other.event"
    assert json.loads(lines[1])["event"] == "target.event"


def test_collect_all_logs_handles_non_structured_logs(log_file: Path) -> None:
    """Verify that plain logs without event/context are handled."""
    plain_logger = logging.getLogger("weakincentives.test.plain")
    plain_logger.setLevel(logging.DEBUG)

    with collect_all_logs(log_file):
        plain_logger.info("Plain message without structure")

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["event"] == ""
    assert entry["message"] == "Plain message without structure"
    assert entry["context"] == {}


def test_collect_all_logs_removes_handler_on_exit(log_file: Path) -> None:
    """Verify that handler is properly removed on context exit."""
    root_logger = logging.getLogger()
    initial_handlers = len(root_logger.handlers)

    with collect_all_logs(log_file):
        assert len(root_logger.handlers) == initial_handlers + 1

    assert len(root_logger.handlers) == initial_handlers


def test_collect_all_logs_removes_handler_on_exception(log_file: Path) -> None:
    """Verify that handler is removed even when exception occurs."""
    root_logger = logging.getLogger()
    initial_handlers = len(root_logger.handlers)

    with pytest.raises(RuntimeError):
        with collect_all_logs(log_file):
            raise RuntimeError("Test exception")

    assert len(root_logger.handlers) == initial_handlers


def test_collect_all_logs_multiple_files(tmp_path: Path) -> None:
    """Verify that multiple collection contexts write to separate files."""
    file1 = tmp_path / "log1.jsonl"
    file2 = tmp_path / "log2.jsonl"
    structured = get_logger("weakincentives.test.multi")

    with collect_all_logs(file1):
        structured.info("First context", event="first.event", context={})

        with collect_all_logs(file2):
            structured.info("Second context", event="second.event", context={})

        structured.info("Back to first", event="first.again", context={})

    lines1 = file1.read_text().strip().split("\n")
    assert len(lines1) == 3

    lines2 = file2.read_text().strip().split("\n")
    assert len(lines2) == 1
    assert json.loads(lines2[0])["event"] == "second.event"


def test_collect_all_logs_path_property(log_file: Path) -> None:
    """Verify that collector.path returns the correct path."""
    with collect_all_logs(log_file) as collector:
        pass

    assert collector.path == log_file


def test_collect_all_logs_expands_user_path(tmp_path: Path) -> None:
    """Verify that paths are handled correctly."""
    log_file = tmp_path / "logs.jsonl"
    structured = get_logger("weakincentives.test.expand")

    with collect_all_logs(str(log_file)) as collector:
        structured.info("Test", event="test.event", context={})

    assert collector.path == log_file
    assert log_file.exists()


def test_collect_all_logs_with_warning_level(log_file: Path) -> None:
    """Verify warning-level log capture."""
    structured = get_logger("weakincentives.test.warning")

    with collect_all_logs(log_file):
        structured.warning(
            "Something might be wrong",
            event="test.warning",
            context={"severity": "medium"},
        )

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["level"] == "WARNING"
    assert entry["context"]["severity"] == "medium"


def test_collect_all_logs_with_error_level(log_file: Path) -> None:
    """Verify error-level log capture."""
    structured = get_logger("weakincentives.test.error")

    with collect_all_logs(log_file):
        structured.error(
            "Something went wrong",
            event="test.error",
            context={"error_code": 500},
        )

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["level"] == "ERROR"
    assert entry["context"]["error_code"] == 500


def test_collect_all_logs_handles_non_mapping_context(log_file: Path) -> None:
    """Verify that logs with non-Mapping context are handled gracefully."""
    plain_logger = logging.getLogger("weakincentives.test.nonmapping")
    plain_logger.setLevel(logging.DEBUG)

    with collect_all_logs(log_file):
        record = plain_logger.makeRecord(
            name="weakincentives.test.nonmapping",
            level=logging.INFO,
            fn="test_log_collector.py",
            lno=0,
            msg="Log with bad context",
            args=(),
            exc_info=None,
            func=None,
            extra={"event": "test.bad.context", "context": "not-a-mapping"},
        )
        plain_logger.handle(record)

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["event"] == "test.bad.context"
    assert entry["context"] == {}


def test_collect_all_logs_handles_emit_exception(
    log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that exceptions in emit are handled gracefully."""
    structured = get_logger("weakincentives.test.emit_error")

    # Track handleError calls
    errors_handled: list[logging.LogRecord] = []
    original_handle_error = logging.Handler.handleError

    def track_handle_error(self: logging.Handler, record: logging.LogRecord) -> None:
        errors_handled.append(record)
        original_handle_error(self, record)

    monkeypatch.setattr(logging.Handler, "handleError", track_handle_error)

    with collect_all_logs(log_file):
        # Close the file to force an I/O error on write
        # We need to get access to the handler's file on the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, "_file"):
                handler._file.close()
                break

        # This should trigger handleError instead of raising
        structured.info("Test", event="test.event", context={})

    # handleError should have been called
    assert len(errors_handled) >= 1


def test_collect_all_logs_handles_non_serializable_context(
    log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that non-JSON-serializable context is handled gracefully."""
    plain_logger = logging.getLogger("weakincentives.test.nonserializable")
    plain_logger.setLevel(logging.DEBUG)

    # Track handleError calls
    errors_handled: list[logging.LogRecord] = []
    original_handle_error = logging.Handler.handleError

    def track_handle_error(self: logging.Handler, record: logging.LogRecord) -> None:
        errors_handled.append(record)
        original_handle_error(self, record)

    monkeypatch.setattr(logging.Handler, "handleError", track_handle_error)

    with collect_all_logs(log_file):
        # Create a log record with non-serializable context
        record = plain_logger.makeRecord(
            name="weakincentives.test.nonserializable",
            level=logging.INFO,
            fn="test_log_collector.py",
            lno=0,
            msg="Log with non-serializable context",
            args=(),
            exc_info=None,
            func=None,
            extra={"event": "test.nonserializable", "context": {"obj": object()}},
        )
        plain_logger.handle(record)

    # handleError should have been called due to serialization failure
    assert len(errors_handled) >= 1
