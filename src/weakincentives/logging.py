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

"""Structured logging built on top of the standard ``logging`` module.

:class:`StructuredLogger` wraps a :class:`logging.Logger` and serializes
the message + structured fields as a single JSON object so production
sinks (Datadog, GCP, journald, …) can index by field. The built-in
:func:`configure_logging` wires up a sensible default handler if the
caller hasn't done it themselves.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TextIO, cast, final, override

__all__ = [
    "JsonFormatter",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
]


@final
class JsonFormatter(logging.Formatter):
    """Format records as a JSON object, one per line."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "wink_extra", None)
        if isinstance(extra, dict):
            payload["fields"] = cast("dict[str, object]", extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    formatter: logging.Formatter | None = None,
) -> None:
    """Install a :class:`JsonFormatter` on the root logger.

    Idempotent: calling it twice replaces the previously installed
    weakincentives handler so tests and demos can re-configure without
    leaking handlers.
    """
    sink = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(sink)
    handler.setFormatter(formatter if formatter is not None else JsonFormatter())
    handler.set_name("weakincentives")
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        if existing.get_name() == "weakincentives":
            root.removeHandler(existing)
    root.addHandler(handler)


@dataclass
class StructuredLogger:
    """Wrap a :class:`logging.Logger` with a fluent ``log(event, **fields)``.

    Each call emits a single ``logging`` record whose ``wink_extra``
    attribute carries the structured fields. The built-in
    :class:`JsonFormatter` reads them; foreign formatters ignore them.
    """

    name: str
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def debug(self, event: str, /, **fields: object) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, /, **fields: object) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, /, **fields: object) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, /, **fields: object) -> None:
        self._emit(logging.ERROR, event, fields)

    def exception(self, event: str, /, **fields: object) -> None:
        self._emit(logging.ERROR, event, fields, exc_info=True)

    def _emit(
        self,
        level: int,
        event: str,
        fields: Mapping[str, object],
        *,
        exc_info: bool = False,
    ) -> None:
        self._logger.log(
            level,
            event,
            extra={"wink_extra": dict(fields)},
            exc_info=exc_info,
        )


def get_logger(name: str) -> StructuredLogger:
    """Return a :class:`StructuredLogger` bound to ``name``."""
    return StructuredLogger(name=name)
