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

"""Log collector context manager for capturing logs during prompt evaluation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast, override

from ..types import JSONValue


class _LogCollectorHandler(logging.Handler):
    """Handler that captures log records and writes them to a file as JSONL."""

    def __init__(
        self,
        file_handle: IO[str],
        level: int = logging.DEBUG,
    ) -> None:
        super().__init__(level=level)
        self._file = file_handle

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Convert log record to JSON and write to file."""
        try:
            json_line = self._record_to_json(record)
            _ = self._file.write(json_line + "\n")
            self._file.flush()
        except (OSError, TypeError, ValueError):
            # OSError: file write errors
            # TypeError/ValueError: JSON serialization errors (non-serializable context)
            self.handleError(record)

    @staticmethod
    def _record_to_json(record: logging.LogRecord) -> str:
        """Convert a logging.LogRecord to a JSON string."""
        # Extract structured fields from weakincentives logging framework
        event = getattr(record, "event", "")
        raw_context = getattr(record, "context", {})

        # Ensure context is a proper dict
        context_dict: dict[str, JSONValue] = {}
        if isinstance(raw_context, Mapping):
            source = cast("Mapping[str, JSONValue]", raw_context)
            context_dict = dict(source)

        data: dict[str, JSONValue] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event if isinstance(event, str) else "",
            "message": record.getMessage(),
            "context": context_dict,
        }
        return json.dumps(data, ensure_ascii=False)


class _LogCollectorContext:
    """Context object returned by collect_all_logs."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @property
    def path(self) -> Path:
        """Return the path to the log file."""
        return self._path


@contextmanager
def collect_all_logs(
    target: str | Path,
    *,
    level: int = logging.DEBUG,
) -> Iterator[_LogCollectorContext]:
    """Capture all log records and write them to a file.

    This context manager attaches a logging handler to the root logger
    and captures all log records emitted during the context. The captured
    entries are written to a JSONL file (one JSON object per line) using
    the standard structured logging format.

    Args:
        target: Path to the output file. Will be created if it doesn't exist,
            or appended to if it already exists. Parent directories are
            created automatically.
        level: Minimum log level to capture. Defaults to ``logging.DEBUG``
            to capture all logs.

    Yields:
        A context object with a ``path`` property for the resolved log file path.

    Example::

        from weakincentives.debug import collect_all_logs

        with collect_all_logs("./logs/session.log") as collector:
            response = adapter.evaluate(prompt, session=session)

        print(f"Logs written to: {collector.path}")

    File format (JSONL)::

        {"timestamp": "2024-01-15T10:30:00+00:00", "level": "INFO", ...}
        {"timestamp": "2024-01-15T10:30:01+00:00", "level": "DEBUG", ...}

    """
    path = Path(target).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    with path.open("a", encoding="utf-8") as file_handle:
        handler = _LogCollectorHandler(file_handle, level=level)
        root_logger.addHandler(handler)
        try:
            yield _LogCollectorContext(path)
        finally:
            root_logger.removeHandler(handler)
            handler.close()


__all__ = [
    "collect_all_logs",
]
