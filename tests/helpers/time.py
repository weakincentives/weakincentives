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

"""Clock control helpers for tests.

The primary test clock is :class:`weakincentives.clock.FakeClock`, which provides
controllable monotonic and wall-clock time. This module provides fixtures and
backward-compatible helpers.

Example::

    from weakincentives.clock import FakeClock

    def test_deadline_remaining() -> None:
        clock = FakeClock()
        clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))

        deadline = Deadline(
            expires_at=datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC),
            clock=clock,
        )

        assert deadline.remaining() == timedelta(hours=1)

        clock.advance(1800)  # 30 minutes
        assert deadline.remaining() == timedelta(minutes=30)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# Re-export FakeClock from main module for convenience
from weakincentives.clock import FakeClock


class ControllableClock:
    """Backward-compatible wrapper around FakeClock for monotonic-only usage.

    .. deprecated::
        Use :class:`weakincentives.clock.FakeClock` directly instead.
        This class remains for backward compatibility with tests that
        use ``clock()`` instead of ``clock.monotonic()``.

    Example (legacy)::

        clock = ControllableClock()
        mailbox = InMemoryMailbox(name="test", clock=clock)

    Example (preferred)::

        from weakincentives.clock import FakeClock

        clock = FakeClock()
        mailbox = InMemoryMailbox(name="test", clock=clock)
    """

    def __init__(self, start: float = 0.0) -> None:
        self._clock = FakeClock()
        self._clock.set_monotonic(start)

    def __call__(self) -> float:
        """Return the current monotonic time (legacy callable interface)."""
        return self._clock.monotonic()

    def monotonic(self) -> float:
        """Return the current monotonic time."""
        return self._clock.monotonic()

    def utcnow(self) -> datetime:
        """Return the current wall-clock time."""
        return self._clock.utcnow()

    def sleep(self, seconds: float) -> None:
        """Advance time immediately without blocking."""
        self._clock.sleep(seconds)

    def advance(self, seconds: float) -> float:
        """Advance the clock by the given number of seconds.

        Args:
            seconds: Time to advance in seconds.

        Returns:
            The new monotonic time.
        """
        self._clock.advance(seconds)
        return self._clock.monotonic()

    def set(self, value: float) -> float:
        """Set the monotonic clock to an absolute value.

        Args:
            value: The new clock value.

        Returns:
            The new monotonic time.
        """
        self._clock.set_monotonic(value)
        return self._clock.monotonic()


class FrozenUtcNow:
    """Backward-compatible controller for wall-clock time in tests.

    .. deprecated::
        Use :class:`weakincentives.clock.FakeClock` with clock injection instead
        of monkeypatching. This class remains for backward compatibility.

    Example (legacy - monkeypatching)::

        def test_deadline(frozen_utcnow: FrozenUtcNow) -> None:
            frozen_utcnow.set(datetime(2024, 1, 1, tzinfo=UTC))
            deadline = Deadline(expires_at=datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
            ...

    Example (preferred - clock injection)::

        def test_deadline() -> None:
            clock = FakeClock()
            clock.set_wall(datetime(2024, 1, 1, tzinfo=UTC))
            deadline = Deadline(
                expires_at=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                clock=clock,
            )
            ...
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, anchor: datetime | None = None
    ) -> None:
        self._current = anchor if anchor is not None else datetime.now(UTC)
        # Note: This monkeypatching approach is deprecated.
        # The _utcnow function no longer exists in the refactored deadlines module.
        # This class is kept for reference but tests should use clock injection.
        self._monkeypatch = monkeypatch

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
def fake_clock() -> FakeClock:
    """Provide a fresh FakeClock for deterministic time control.

    This is the preferred fixture for testing time-dependent code.
    Inject the clock into components that accept a clock parameter.

    Example::

        def test_heartbeat_elapsed(fake_clock: FakeClock) -> None:
            hb = Heartbeat(clock=fake_clock)

            hb.beat()
            assert hb.elapsed() == 0.0

            fake_clock.advance(10)
            assert hb.elapsed() == 10.0
    """
    return FakeClock()


@pytest.fixture
def frozen_utcnow(monkeypatch: pytest.MonkeyPatch) -> FrozenUtcNow:
    """Provide a controllable wall-clock for legacy deadline tests.

    .. deprecated::
        Use the ``fake_clock`` fixture with clock injection instead.
    """
    return FrozenUtcNow(monkeypatch)


__all__ = ["ControllableClock", "FakeClock", "FrozenUtcNow", "fake_clock", "frozen_utcnow"]
