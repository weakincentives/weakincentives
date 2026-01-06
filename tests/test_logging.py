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

"""Tests for structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from types import TracebackType
from typing import cast

import pytest

from weakincentives.runtime.logging import (
    JSONValue,
    StructuredLogger,
    StructuredLogPayload,
    _coerce_level,
    _unwrap_logger,
    configure_logging,
    get_logger,
)


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture(logger: logging.Logger) -> Iterator[list[logging.LogRecord]]:
    handler = _CaptureHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def reset_logging_state() -> Iterator[None]:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_structured_logger_emits_structured_records() -> None:
    logger = get_logger("tests.logging").bind(component="unit-test")
    base_logger = logger.logger
    base_logger.setLevel(logging.INFO)

    with _capture(base_logger) as records:
        logger.info("structured", event="tests.event", context={"attempt": 1})

    assert len(records) == 1
    record = records[0]
    assert record.event == "tests.event"
    assert record.context == {"component": "unit-test", "attempt": 1}
    assert record.getMessage() == "structured"


def test_structured_logger_handles_none_extra_and_merges_mapping() -> None:
    logger = get_logger("tests.logging.extra")
    base_logger = logger.logger
    base_logger.setLevel(logging.INFO)

    with _capture(base_logger) as records:
        logger.info("none-extra", event="tests.none", extra=None)
        logger.info("with-extra", extra={"event": "tests.extra", "count": 2})

    assert [record.event for record in records] == ["tests.none", "tests.extra"]
    assert records[0].context == {}
    assert records[1].context == {"count": 2}


def test_get_logger_preserves_override_context() -> None:
    override = logging.getLogger("override")
    adapter = StructuredLogger(override, context={"existing": True})

    logger = get_logger("ignored", logger_override=adapter, context={"bound": "value"})

    assert logger.logger is override
    assert logger.extra == {"existing": True, "bound": "value"}


def test_get_logger_accepts_plain_logger_override() -> None:
    override = logging.getLogger("plain")

    logger = get_logger("ignored", logger_override=override, context={"plain": True})

    assert logger.logger is override
    assert logger.extra == {"plain": True}


def test_get_logger_merges_standard_logger_adapter_context() -> None:
    base = logging.getLogger("adapter")
    adapter = logging.LoggerAdapter(base, {"source": "adapter"})

    logger = get_logger("ignored", logger_override=adapter)

    assert logger.logger is base
    assert logger.extra == {"source": "adapter"}


def test_get_logger_handles_nested_logger_adapter_override() -> None:
    base = logging.getLogger("nested.base")
    inner = logging.LoggerAdapter(base, {"inner": True})

    class _NestedAdapter(logging.LoggerAdapter):
        def __init__(self) -> None:
            super().__init__(logging.getLogger("outer"), {})
            self.logger = inner

    nested = _NestedAdapter()
    logger = get_logger("ignored", logger_override=nested)

    assert logger.logger is base
    assert logger.extra == {}


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_structured_logger_exception_logging_includes_context() -> None:
    logger = get_logger("tests.exception").bind(component="runner")
    base_logger = logger.logger
    base_logger.setLevel(logging.INFO)

    with _capture(base_logger) as records:
        try:
            _raise_runtime_error()
        except RuntimeError:
            logger.exception("failed", event="tests.error", context={"prompt": "demo"})

    assert len(records) == 1
    record = records[0]
    assert record.event == "tests.error"
    assert record.context == {"component": "runner", "prompt": "demo"}
    assert record.exc_info is not None


def test_structured_logger_requires_event_metadata() -> None:
    logger = get_logger("tests.missing")
    logger.logger.setLevel(logging.INFO)

    with pytest.raises(TypeError):
        logger.info("missing-event", extra={"detail": True})


def test_configure_logging_respects_existing_handlers() -> None:
    root = logging.getLogger()
    root_handler = logging.NullHandler()
    root.handlers = [root_handler]

    configure_logging(level="DEBUG", json_mode=True)

    assert root.handlers == [root_handler]
    assert root.level == logging.DEBUG


def test_configure_logging_honors_env_toggle() -> None:
    configure_logging(force=True, env={"WEAKINCENTIVES_LOG_FORMAT": "json"})

    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert handler.formatter.__class__.__name__ == "_JsonFormatter"


def test_configure_logging_explicit_level_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAKINCENTIVES_LOG_LEVEL", "WARNING")

    configure_logging(level=logging.NOTSET, force=True)

    root = logging.getLogger()
    assert root.level == logging.NOTSET


def test_configure_logging_force_installs_json_formatter() -> None:
    configure_logging(level="INFO", json_mode=True, force=True)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert handler.formatter.__class__.__name__ == "_JsonFormatter"

    exc_info: tuple[type[BaseException], BaseException, TracebackType] | None = None
    try:
        _raise_runtime_error()
    except RuntimeError as error:
        traceback = error.__traceback__
        assert traceback is not None
        exc_info = (error.__class__, error, traceback)
    assert exc_info is not None

    non_serializable = cast(JSONValue, object())
    context_payload = cast(
        StructuredLogPayload, {"key": "value", "object": non_serializable}
    )

    record = root.makeRecord(
        name="tests.force",
        level=logging.INFO,
        fn="tests/test_logging.py",
        lno=0,
        msg="payload",
        args=(),
        exc_info=exc_info,
        func=None,
        extra={
            "event": "tests.force",
            "context": context_payload,
        },
    )

    rendered = handler.format(record)
    payload = json.loads(rendered)
    assert payload["event"] == "tests.force"
    assert payload["context"]["key"] == "value"
    assert "object" in payload["context"]
    assert payload["message"] == "payload"


def test_configure_logging_json_mode_emits_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    configure_logging(json_mode=True, force=True)

    logger = get_logger("tests.logging.json").bind(component="json-test")
    logger.logger.setLevel(logging.INFO)

    logger.info("payload", event="tests.json", context={"key": "value"})

    root = logging.getLogger()
    handler = root.handlers[0]
    handler.flush()

    output = stream.getvalue().strip()
    assert output

    payload = json.loads(output)
    assert payload["event"] == "tests.json"
    assert payload["context"] == {"component": "json-test", "key": "value"}
    assert payload["message"] == "payload"
    assert payload["logger"] == "tests.logging.json"


def test_configure_logging_defaults_to_text_formatter() -> None:
    configure_logging(force=True)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert handler.formatter.__class__.__name__ != "_JsonFormatter"


def test_configure_logging_accepts_integer_level() -> None:
    configure_logging(level=logging.WARNING, force=True)

    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_coerce_level_defaults_to_info() -> None:
    assert _coerce_level(None) == logging.INFO


def test_unwrap_logger_raises_for_invalid_adapter() -> None:
    class _BrokenAdapter(logging.LoggerAdapter):
        def __init__(self) -> None:
            super().__init__(logging.getLogger("broken"), {})
            self.logger = object()

    adapter = _BrokenAdapter()

    with pytest.raises(TypeError):
        _unwrap_logger(adapter)


def test_get_logger_handles_adapter_without_extra_mapping() -> None:
    """Cover branch where LoggerAdapter.extra is not a Mapping."""
    base = logging.getLogger("adapter.no.extra")

    class _NoExtraAdapter(logging.LoggerAdapter):
        def __init__(self) -> None:
            super().__init__(base, {})
            self.extra = None  # Not a Mapping

    adapter = _NoExtraAdapter()
    logger = get_logger("ignored", logger_override=adapter)

    assert logger.logger is base
    assert logger.extra == {}


def test_json_formatter_handles_missing_event_and_context() -> None:
    """Cover branches where event is None and context is falsy."""
    configure_logging(json_mode=True, force=True)

    root = logging.getLogger()
    handler = root.handlers[0]

    # Create record without event or context attributes
    record = root.makeRecord(
        name="tests.bare",
        level=logging.INFO,
        fn="tests/test_logging.py",
        lno=0,
        msg="bare message",
        args=(),
        exc_info=None,
        func=None,
        extra={},
    )

    rendered = handler.format(record)
    payload = json.loads(rendered)
    assert payload["message"] == "bare message"
    assert "event" not in payload
    assert "context" not in payload


def test_text_formatter_handles_missing_event_and_context() -> None:
    """Cover branches where third-party loggers lack event/context fields."""
    configure_logging(json_mode=False, force=True)

    root = logging.getLogger()
    handler = root.handlers[0]

    # Create record without event or context attributes (simulating third-party library)
    record = root.makeRecord(
        name="claude_agent_sdk._internal.transport",
        level=logging.INFO,
        fn="subprocess_cli.py",
        lno=113,
        msg="Using bundled Claude Code CLI: /opt/venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude",
        args=(),
        exc_info=None,
        func=None,
        extra={},
    )

    # This should NOT raise KeyError: 'event'
    rendered = handler.format(record)

    # Verify the output contains the expected fields
    assert "INFO" in rendered
    assert "claude_agent_sdk._internal.transport" in rendered
    assert "Using bundled Claude Code CLI" in rendered


def test_text_formatter_includes_event_and_context_when_present() -> None:
    """Verify text formatter includes structured fields when provided."""
    configure_logging(json_mode=False, force=True)

    root = logging.getLogger()
    handler = root.handlers[0]

    record = root.makeRecord(
        name="weakincentives.adapters.claude_agent_sdk.adapter",
        level=logging.INFO,
        fn="adapter.py",
        lno=320,
        msg="claude_agent_sdk.evaluate.start",
        args=(),
        exc_info=None,
        func=None,
        extra={
            "event": "sdk.evaluate.start",
            "context": {"component": "claude_agent_sdk", "model": "claude-sonnet-4-5"},
        },
    )

    rendered = handler.format(record)

    assert "INFO" in rendered
    assert "sdk.evaluate.start" in rendered
    assert "claude_agent_sdk.evaluate.start" in rendered
    assert "component" in rendered
    assert "claude-sonnet-4-5" in rendered


def test_text_formatter_handles_exception_info() -> None:
    """Verify text formatter includes exception traceback."""
    configure_logging(json_mode=False, force=True)

    root = logging.getLogger()
    handler = root.handlers[0]

    exc_info: tuple[type[BaseException], BaseException, TracebackType] | None = None
    try:
        _raise_runtime_error()
    except RuntimeError as error:
        traceback = error.__traceback__
        assert traceback is not None
        exc_info = (error.__class__, error, traceback)
    assert exc_info is not None

    record = root.makeRecord(
        name="tests.exception",
        level=logging.ERROR,
        fn="tests/test_logging.py",
        lno=0,
        msg="something failed",
        args=(),
        exc_info=exc_info,
        func=None,
        extra={"event": "tests.error", "context": {}},
    )

    rendered = handler.format(record)

    assert "ERROR" in rendered
    assert "something failed" in rendered
    assert "RuntimeError" in rendered
    assert "boom" in rendered


def test_text_formatter_handles_stack_info() -> None:
    """Verify text formatter includes stack info when present."""
    configure_logging(json_mode=False, force=True)

    root = logging.getLogger()
    handler = root.handlers[0]

    record = root.makeRecord(
        name="tests.stack",
        level=logging.WARNING,
        fn="tests/test_logging.py",
        lno=0,
        msg="stack trace follows",
        args=(),
        exc_info=None,
        func=None,
        extra={},
        sinfo="Stack (most recent call last):\n  File test.py, line 1",
    )

    rendered = handler.format(record)

    assert "WARNING" in rendered
    assert "stack trace follows" in rendered
    assert "Stack (most recent call last)" in rendered


def test_configure_logging_defaults_to_text_formatter_class() -> None:
    """Verify text formatter class is used by default."""
    configure_logging(force=True)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert handler.formatter.__class__.__name__ == "_TextFormatter"
