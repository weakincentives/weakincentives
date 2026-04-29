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

"""Tests for ``weakincentives.logging``."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from weakincentives.logging import (
    JsonFormatter,
    StructuredLogger,
    configure_logging,
    get_logger,
)


@pytest.fixture
def reset_root_handlers() -> Iterator[None]:
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_configure_logging_writes_json_lines(
    reset_root_handlers: None,
) -> None:
    del reset_root_handlers
    sink = io.StringIO()
    configure_logging(level=logging.INFO, stream=sink)
    logger = get_logger("test.alpha")
    logger.info("started", run_id="abc", iteration=2)
    payload = json.loads(sink.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.alpha"
    assert payload["message"] == "started"
    assert payload["fields"] == {"run_id": "abc", "iteration": 2}


def test_configure_logging_is_idempotent(
    reset_root_handlers: None,
) -> None:
    del reset_root_handlers
    sink = io.StringIO()
    configure_logging(stream=sink)
    configure_logging(stream=sink)
    handlers = [
        h for h in logging.getLogger().handlers if h.get_name() == "weakincentives"
    ]
    assert len(handlers) == 1


def test_logger_supports_all_levels(reset_root_handlers: None) -> None:
    del reset_root_handlers
    sink = io.StringIO()
    configure_logging(level=logging.DEBUG, stream=sink)
    logger = get_logger("test.beta")
    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    levels = [
        json.loads(line)["level"]
        for line in sink.getvalue().splitlines()
        if line.strip()
    ]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR"]


def _raise_boom() -> None:
    raise RuntimeError("boom")


def test_logger_exception_captures_traceback(
    reset_root_handlers: None,
) -> None:
    del reset_root_handlers
    sink = io.StringIO()
    configure_logging(stream=sink)
    logger = get_logger("test.gamma")
    try:
        _raise_boom()
    except RuntimeError:
        logger.exception("crashed")
    payload = json.loads(sink.getvalue().strip())
    assert "RuntimeError" in payload["exception"]


def test_json_formatter_omits_fields_when_no_extra() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert "fields" not in payload


def test_get_logger_returns_structured_logger() -> None:
    assert isinstance(get_logger("x"), StructuredLogger)
    assert get_logger("x").logger.name == "x"


def test_logger_emits_via_existing_handler(
    reset_root_handlers: None,
) -> None:
    del reset_root_handlers
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.setFormatter(JsonFormatter())
    handler.set_name("weakincentives")
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    get_logger("test.delta").info("noted")
    assert "noted" in sink.getvalue()
