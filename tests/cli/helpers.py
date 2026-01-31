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

"""Test helpers for CLI module tests.

This module provides reusable fake implementations for CLI testing,
avoiding duplication of ad-hoc mocks across test files.

Example::

    from tests.cli.helpers import FakeLogger, FakeContextManager

    def test_logging_behavior() -> None:
        logger = FakeLogger()
        # ... use logger in test
        assert len(logger.logs) == 1

"""

from __future__ import annotations

from types import TracebackType


class FakeLogger:
    """Fake logger that records all log calls for assertion.

    Captures info, error, and exception calls with their arguments.

    Example::

        logger = FakeLogger()
        logger.info("message", event="test.event", context={"key": "value"})
        assert logger.logs[0] == ("message", {"event": "test.event", "context": {"key": "value"}})
    """

    def __init__(self) -> None:
        self.logs: list[tuple[str, dict[str, object]]] = []

    def info(self, message: str, *, event: str, context: object | None = None) -> None:
        """Record an info-level log entry."""
        self.logs.append((message, {"event": event, "context": context}))

    def error(self, message: str, *, event: str, context: object | None = None) -> None:
        """Record an error-level log entry."""
        self.logs.append((message, {"event": event, "context": context}))

    def exception(
        self, message: str, *, event: str, context: object | None = None
    ) -> None:
        """Record an exception-level log entry."""
        self.logs.append((message, {"event": event, "context": context}))


class FakeContextManager:
    """Fake context manager for testing context protocol compliance.

    Tracks enter/exit calls and can be configured to raise on exit.

    Example::

        cm = FakeContextManager()
        with cm:
            pass
        assert cm.entered
        assert cm.exited
    """

    def __init__(self, *, raise_on_exit: Exception | None = None) -> None:
        self.entered = False
        self.exited = False
        self._raise_on_exit = raise_on_exit

    def __enter__(self) -> FakeContextManager:
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.exited = True
        if self._raise_on_exit is not None:
            raise self._raise_on_exit
        return False


__all__ = [
    "FakeContextManager",
    "FakeLogger",
]
