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

"""Controllable time abstractions for testable time-dependent code.

This module provides protocols and implementations for time operations,
enabling deterministic testing without real delays.

There are two distinct time domains:

- **Monotonic time** (float seconds): For measuring elapsed time, timeouts,
  and rate limiting. Guaranteed to never go backwards.

- **Wall-clock time** (UTC datetime): For timestamps, deadlines, and recording
  when events occurred. Can jump due to NTP adjustments.

Example (production)::

    from weakincentives.clock import SYSTEM_CLOCK

    start = SYSTEM_CLOCK.monotonic()
    SYSTEM_CLOCK.sleep(1.0)
    elapsed = SYSTEM_CLOCK.monotonic() - start  # ~1.0

Example (testing)::

    from weakincentives.clock import FakeClock

    clock = FakeClock()
    start = clock.monotonic()
    clock.sleep(10)  # Advances instantly, no real delay
    assert clock.monotonic() - start == 10
"""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, runtime_checkable


@runtime_checkable
class MonotonicClock(Protocol):
    """Protocol for monotonic time measurement.

    Monotonic clocks measure elapsed time and are guaranteed to never
    go backwards. They are suitable for timeouts, rate limiting, and
    measuring durations.

    The zero point is arbitrary and not related to wall-clock time.

    Use this narrow protocol when your component only needs to measure
    elapsed time and does not need wall-clock timestamps or sleep.

    Example::

        def with_timeout(clock: MonotonicClock, timeout: float) -> bool:
            deadline = clock.monotonic() + timeout
            while clock.monotonic() < deadline:
                if try_operation():
                    return True
            return False
    """

    def monotonic(self) -> float:
        """Return monotonic time in seconds since an arbitrary reference point.

        Returns:
            Current monotonic time as a float. The value increases
            monotonically and never decreases, even across system clock
            adjustments.

        Note:
            Compare values from the same clock instance only. Values from
            different MonotonicClock instances are not comparable.
        """
        ...


@runtime_checkable
class WallClock(Protocol):
    """Protocol for wall-clock time measurement.

    Wall clocks provide the current UTC datetime. They are suitable for
    timestamps, deadlines, and recording when events occurred.

    Unlike monotonic clocks, wall clocks can jump (NTP adjustments,
    daylight saving, etc.) and should not be used for measuring durations.

    Use this narrow protocol when your component only needs timestamps
    and does not need elapsed time measurement or sleep.

    Example::

        def log_event(clock: WallClock, message: str) -> dict:
            return {
                "timestamp": clock.utcnow().isoformat(),
                "message": message,
            }
    """

    def utcnow(self) -> datetime:
        """Return current UTC datetime as a timezone-aware object.

        Returns:
            A timezone-aware datetime in UTC. Always has ``tzinfo=UTC``.

        Warning:
            Do not use wall-clock time to measure durations. Wall-clock
            time can jump forwards or backwards due to NTP synchronization.
            Use :meth:`MonotonicClock.monotonic` for duration measurement.
        """
        ...


@runtime_checkable
class Sleeper(Protocol):
    """Protocol for sleep/delay operations.

    Separating sleep from clock allows tests to advance time without
    actually sleeping, while production code uses real delays.

    Use this narrow protocol when your component only needs to pause
    execution and does not need time measurement.

    Example::

        def poll_with_backoff(sleeper: Sleeper, max_attempts: int) -> bool:
            for attempt in range(max_attempts):
                if check_condition():
                    return True
                sleeper.sleep(2 ** attempt)  # Exponential backoff
            return False
    """

    def sleep(self, seconds: float) -> None:
        """Pause execution for the specified duration.

        Args:
            seconds: Duration to sleep in seconds. Should be non-negative.
                Behavior for negative values is implementation-defined.

        Note:
            In production (SystemClock), this blocks the current thread.
            In tests (FakeClock), this advances simulated time immediately
            without blocking.
        """
        ...


@runtime_checkable
class Clock(MonotonicClock, WallClock, Sleeper, Protocol):
    """Unified clock combining monotonic, wall-clock, and sleep capabilities.

    This is the most common clock protocol to depend on. Use it when your
    component needs multiple time capabilities. For single-purpose use,
    prefer the narrower protocols (:class:`MonotonicClock`, :class:`WallClock`,
    or :class:`Sleeper`) to reduce coupling.

    Implementations:
        - :class:`SystemClock`: Production implementation using system time.
        - :class:`FakeClock`: Test implementation with controllable time.

    Example::

        def rate_limited_fetch(clock: Clock, url: str, min_interval: float) -> bytes:
            last_fetch = clock.monotonic()
            # ... fetch logic ...
            elapsed = clock.monotonic() - last_fetch
            if elapsed < min_interval:
                clock.sleep(min_interval - elapsed)
            return data

    Typical usage pattern::

        from weakincentives.clock import SYSTEM_CLOCK, Clock

        def my_function(clock: Clock = SYSTEM_CLOCK) -> None:
            # Use clock.monotonic(), clock.utcnow(), clock.sleep()
            pass
    """

    pass


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Production clock using system time functions.

    This is the default clock implementation used throughout WINK. It
    delegates directly to Python's standard library time functions:

    - :meth:`monotonic` uses ``time.monotonic()``
    - :meth:`utcnow` uses ``datetime.now(UTC)``
    - :meth:`sleep` uses ``time.sleep()``

    SystemClock is stateless and immutable. Multiple instances behave
    identically, but prefer using the :data:`SYSTEM_CLOCK` singleton.

    Example::

        from weakincentives.clock import SYSTEM_CLOCK

        start = SYSTEM_CLOCK.monotonic()
        SYSTEM_CLOCK.sleep(1.0)
        elapsed = SYSTEM_CLOCK.monotonic() - start  # ~1.0

    For testing, inject :class:`FakeClock` instead to control time
    deterministically without real delays.
    """

    def monotonic(self) -> float:
        """Return monotonic time from the system clock.

        Returns:
            Current value of ``time.monotonic()``, representing seconds
            since an arbitrary reference point that does not change during
            process lifetime.
        """
        return _time.monotonic()

    def utcnow(self) -> datetime:
        """Return current UTC datetime from the system clock.

        Returns:
            Timezone-aware datetime representing the current wall-clock
            time in UTC.
        """
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        """Block the current thread for the specified duration.

        Args:
            seconds: Duration to sleep in seconds. Negative values
                are treated as zero by the underlying ``time.sleep()``.

        Note:
            This blocks the calling thread. For non-blocking delays in
            async code, use ``asyncio.sleep()`` instead.
        """
        _time.sleep(seconds)


# Module-level singleton for production use
SYSTEM_CLOCK: Final[Clock] = SystemClock()
"""Default system clock singleton for production use.

Use this as the default value for clock parameters in function signatures.
This pattern enables dependency injection for testing while keeping
production code simple.

Example::

    from weakincentives.clock import SYSTEM_CLOCK, Clock, FakeClock

    def process_with_timeout(data: bytes, clock: Clock = SYSTEM_CLOCK) -> Result:
        deadline = clock.monotonic() + 30.0
        # ... processing logic using clock ...

    # Production: uses real time
    result = process_with_timeout(data)

    # Testing: uses controllable fake time
    fake = FakeClock()
    result = process_with_timeout(data, clock=fake)
    fake.advance(30.0)  # Simulate timeout instantly
"""


@dataclass
class FakeClock:
    """Controllable clock for deterministic testing.

    FakeClock simulates time without real delays, enabling fast and
    reproducible tests for time-dependent code. Both monotonic and
    wall-clock time advance together when :meth:`advance` or :meth:`sleep`
    is called.

    Initial state:
        - Monotonic time starts at ``0.0``
        - Wall-clock time starts at ``2024-01-01T00:00:00Z``

    Example::

        from weakincentives.clock import FakeClock

        clock = FakeClock()

        # Sleep advances time instantly (no real delay)
        start = clock.monotonic()
        clock.sleep(10)
        assert clock.monotonic() - start == 10

        # Manual advance for testing timeouts
        clock.advance(60)
        assert clock.monotonic() - start == 70

        # Wall clock advances in sync
        assert clock.utcnow().minute == 1  # 70 seconds from midnight

    Testing pattern::

        def test_timeout_handling():
            clock = FakeClock()
            handler = TimeoutHandler(clock=clock)

            handler.start_operation()
            clock.advance(30.0)  # Simulate 30 seconds passing

            assert handler.is_timed_out()

    Thread-safety:
        All operations are thread-safe. Multiple threads can safely read
        and advance the clock concurrently.
    """

    _monotonic: float = 0.0
    _wall: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def monotonic(self) -> float:
        """Return current simulated monotonic time.

        Returns:
            Simulated monotonic time in seconds, starting from 0.0.
        """
        with self._lock:
            return self._monotonic

    def utcnow(self) -> datetime:
        """Return current simulated wall-clock time.

        Returns:
            Simulated UTC datetime, starting from 2024-01-01T00:00:00Z.
        """
        with self._lock:
            return self._wall

    def sleep(self, seconds: float) -> None:
        """Advance simulated time instantly without blocking.

        This method advances both monotonic and wall-clock time by the
        specified duration, then returns immediately. No actual delay
        occurs, making tests run fast.

        Args:
            seconds: Duration to advance in seconds. Must be non-negative.

        Raises:
            ValueError: If seconds is negative.
        """
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Advance both monotonic and wall-clock time by the given duration.

        Use this to simulate time passing in tests, such as checking
        timeout behavior or scheduled operations.

        Args:
            seconds: Duration to advance in seconds. Must be non-negative.

        Raises:
            ValueError: If seconds is negative.

        Example::

            clock = FakeClock()
            clock.advance(3600)  # Advance by 1 hour
            assert clock.utcnow().hour == 1
        """
        if seconds < 0:
            msg = "Cannot advance time by negative seconds"
            raise ValueError(msg)
        with self._lock:
            self._monotonic += seconds
            self._wall += timedelta(seconds=seconds)

    def set_monotonic(self, value: float) -> None:
        """Set monotonic time to an absolute value.

        Use this to set up specific test conditions. Note that this
        only affects monotonic time; wall-clock time is unchanged.

        Args:
            value: New monotonic time value in seconds.

        Example::

            clock = FakeClock()
            clock.set_monotonic(1000.0)
            assert clock.monotonic() == 1000.0
        """
        with self._lock:
            self._monotonic = value

    def set_wall(self, value: datetime) -> None:
        """Set wall-clock time to an absolute value.

        Use this to set up specific test conditions, such as testing
        behavior at specific dates or times. Note that this only affects
        wall-clock time; monotonic time is unchanged.

        Args:
            value: New wall-clock time. Must be a timezone-aware datetime
                (typically with ``tzinfo=UTC``).

        Raises:
            ValueError: If value is naive (has no timezone info).

        Example::

            from datetime import UTC, datetime
            clock = FakeClock()
            clock.set_wall(datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC))
            assert clock.utcnow().year == 2025
        """
        if value.tzinfo is None:
            msg = "Wall clock time must be timezone-aware"
            raise ValueError(msg)
        with self._lock:
            self._wall = value


__all__ = [
    "SYSTEM_CLOCK",
    "Clock",
    "FakeClock",
    "MonotonicClock",
    "Sleeper",
    "SystemClock",
    "WallClock",
]
