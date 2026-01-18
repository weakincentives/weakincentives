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

"""Clock control helpers for tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weakincentives import deadlines


class FrozenUtcNow:
    """Controller for :func:`weakincentives.deadlines._utcnow` during tests."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, anchor: datetime | None = None
    ) -> None:
        self._current = anchor if anchor is not None else datetime.now(UTC)
        monkeypatch.setattr(deadlines, "_utcnow", self.now)

    def now(self) -> datetime:
        """Return the frozen current time."""

        return self._current

    def set(self, current: datetime) -> datetime:
        """Reset the frozen clock to the provided datetime."""

        self._current = current
        return self._current

    def advance(self, delta: timedelta) -> datetime:
        """Move the frozen clock forward by ``delta``."""

        self._current += delta
        return self._current


@pytest.fixture
def frozen_utcnow(monkeypatch: pytest.MonkeyPatch) -> FrozenUtcNow:
    """Provide a controllable :func:`_utcnow` override for deadline tests."""

    return FrozenUtcNow(monkeypatch)


class ControllableClock:
    """A controllable clock for testing time-based behavior without sleeping.

    Use as a replacement for time.monotonic() in tests. Advance the clock
    manually to simulate time passage without actual delays.

    Example::

        clock = ControllableClock()
        mailbox = InMemoryMailbox(name="test", clock=clock)

        mailbox.send("hello")
        messages = mailbox.receive(visibility_timeout=10)

        # Advance past visibility timeout
        clock.advance(11)

        # Message should now be requeued
        messages = mailbox.receive()
    """

    def __init__(self, start: float = 0.0) -> None:
        self._current = start

    def __call__(self) -> float:
        """Return the current clock time."""
        return self._current

    def advance(self, seconds: float) -> float:
        """Advance the clock by the given number of seconds.

        Args:
            seconds: Time to advance in seconds.

        Returns:
            The new current time.
        """
        self._current += seconds
        return self._current

    def set(self, value: float) -> float:
        """Set the clock to an absolute value.

        Args:
            value: The new clock value.

        Returns:
            The new current time.
        """
        self._current = value
        return self._current


__all__ = ["ControllableClock", "FrozenUtcNow", "frozen_utcnow"]
