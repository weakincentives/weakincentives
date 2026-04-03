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

There are two sleep domains:

- **Synchronous** (:class:`Sleeper`): Blocks the calling thread.
- **Asynchronous** (:class:`AsyncSleeper`): Yields control to the event loop.

Example (production)::

    from weakincentives.clock import SYSTEM_CLOCK

    start = SYSTEM_CLOCK.monotonic()
    SYSTEM_CLOCK.sleep(1.0)
    elapsed = SYSTEM_CLOCK.monotonic() - start  # ~1.0

Example (async production)::

    from weakincentives.clock import SYSTEM_CLOCK

    await SYSTEM_CLOCK.async_sleep(1.0)  # Yields to event loop

Example (testing — sync)::

    from weakincentives.clock import FakeClock

    clock = FakeClock()
    start = clock.monotonic()
    clock.sleep(10)  # Advances instantly, no real delay
    assert clock.monotonic() - start == 10

Example (testing — async background loop)::

    clock = FakeClock()
    component = MyComponent(async_sleeper=clock)
    async with component.run():       # starts background poll loop
        clock.advance(0.01)           # wake loop's async_sleep
        await asyncio.sleep(0)        # yield so loop coroutine runs
"""

from __future__ import annotations

import asyncio as _asyncio
import threading
import time as _time
from asyncio import Future
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
    """

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...


@runtime_checkable
class WallClock(Protocol):
    """Protocol for wall-clock time measurement.

    Wall clocks provide the current UTC datetime. They are suitable for
    timestamps, deadlines, and recording when events occurred.

    Unlike monotonic clocks, wall clocks can jump (NTP adjustments,
    daylight saving, etc.) and should not be used for measuring durations.
    """

    def utcnow(self) -> datetime:
        """Return current UTC datetime (timezone-aware)."""
        ...


@runtime_checkable
class Sleeper(Protocol):
    """Protocol for synchronous sleep/delay operations.

    Separating sleep from clock allows tests to advance time without
    actually sleeping, while production code uses real delays.
    """

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration in seconds."""
        ...


@runtime_checkable
class AsyncSleeper(Protocol):
    """Protocol for asynchronous sleep/delay operations.

    The async counterpart of :class:`Sleeper` for code running in an
    ``asyncio`` event loop.  Production code delegates to
    :func:`asyncio.sleep`; :class:`FakeClock` advances time instantly
    without yielding, enabling deterministic async tests.
    """

    async def async_sleep(self, seconds: float) -> None:
        """Asynchronously sleep for the specified duration in seconds."""
        ...


@runtime_checkable
class Clock(MonotonicClock, WallClock, Sleeper, AsyncSleeper, Protocol):
    """Unified clock combining monotonic, wall-clock, and sleep.

    Components that need multiple time capabilities should depend on
    this combined protocol. Components that only need monotonic time
    or wall-clock time should depend on the narrower protocol.
    """


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Production clock using system time functions.

    This is the default clock used throughout WINK. It delegates to:

    - ``time.monotonic()`` for monotonic time
    - ``datetime.now(UTC)`` for wall-clock time
    - ``time.sleep()`` for delays

    Example::

        clock = SystemClock()
        start = clock.monotonic()
        clock.sleep(1.0)
        elapsed = clock.monotonic() - start  # ~1.0
    """

    def monotonic(self) -> float:
        """Return monotonic time from time.monotonic()."""
        return _time.monotonic()

    def utcnow(self) -> datetime:
        """Return current UTC datetime."""
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        """Sleep using time.sleep()."""
        _time.sleep(seconds)

    async def async_sleep(self, seconds: float) -> None:
        """Asynchronously sleep using asyncio.sleep()."""
        await _asyncio.sleep(seconds)


# Module-level singleton for production use
SYSTEM_CLOCK: Final[Clock] = SystemClock()
"""Default system clock instance.

Use this as the default value for clock parameters throughout WINK.
Tests can inject FakeClock instead for deterministic behavior.
"""


def _resolve_future(future: Future[None]) -> None:
    """Resolve an async future, handling cross-thread scenarios.

    Guards against cancellation races: the task may be cancelled between
    the ``future.done()`` check in ``advance()`` and actual resolution.
    """
    loop = future.get_loop()
    try:
        running_loop = _asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        # Same event loop thread — safe to resolve directly.
        future.set_result(None)
    elif loop.is_running():
        # Different thread — re-check done() inside callback to guard
        # against cancellation between enqueue and execution.
        def _safe_resolve() -> None:
            if not future.done():  # pragma: no branch
                future.set_result(None)

        _ = loop.call_soon_threadsafe(_safe_resolve)
    elif not future.done():  # pragma: no cover
        future.set_result(None)


@dataclass
class FakeClock:
    """Controllable clock for deterministic testing.

    Both monotonic and wall-clock time advance together when ``advance()``
    is called.

    **Sync sleep** (``sleep()``) advances time instantly without blocking.

    **Async sleep** (``async_sleep()``) *suspends* the calling coroutine
    until ``advance()`` moves time past the requested deadline. This enables
    deterministic control of async background loops — the test decides
    exactly when each sleep completes by calling ``advance()``.

    Example (sync)::

        clock = FakeClock()
        start = clock.monotonic()
        clock.sleep(10)  # Advances immediately, no real delay
        assert clock.monotonic() - start == 10

    Example (async background loop control)::

        clock = FakeClock()
        # Background loop calls: await clock.async_sleep(interval)
        # It suspends until test advances time:
        clock.advance(interval)
        await asyncio.sleep(0)  # yield to let the loop coroutine run

    Thread-safety:
        All operations are thread-safe. Multiple threads can read and
        advance the clock concurrently. ``advance()`` from a non-event-loop
        thread uses ``call_soon_threadsafe`` to resolve futures.
    """

    _monotonic: float = 0.0
    _wall: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _waiters: list[tuple[float, Future[None]]] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list, repr=False
    )

    def monotonic(self) -> float:
        """Return current monotonic time."""
        with self._lock:
            return self._monotonic

    def utcnow(self) -> datetime:
        """Return current wall-clock time."""
        with self._lock:
            return self._wall

    def sleep(self, seconds: float) -> None:
        """Advance time immediately without blocking (wakes async waiters)."""
        self.advance(seconds)

    async def async_sleep(self, seconds: float) -> None:
        """Suspend until ``advance()`` moves time past the deadline.

        For ``seconds <= 0``, advances time and returns immediately
        (preserving yield-point semantics for ``await async_sleep(0)``).

        For ``seconds > 0``, registers a waiter and suspends the coroutine.
        The coroutine resumes when ``advance()`` or ``sleep()`` moves
        monotonic time to or past the deadline.
        """
        if seconds < 0:
            msg = "Cannot advance time by negative seconds"
            raise ValueError(msg)
        if seconds == 0:
            return
        with self._lock:
            deadline = self._monotonic + seconds
            loop = _asyncio.get_running_loop()
            future: Future[None] = loop.create_future()
            self._waiters.append((deadline, future))
        await future

    def advance(self, seconds: float) -> None:
        """Advance both clocks and wake async waiters past their deadline.

        Args:
            seconds: Duration to advance in seconds (must be non-negative).

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            msg = "Cannot advance time by negative seconds"
            raise ValueError(msg)
        with self._lock:
            self._monotonic += seconds
            self._wall += timedelta(seconds=seconds)
            ready: list[Future[None]] = []
            remaining: list[tuple[float, Future[None]]] = []
            for deadline, future in self._waiters:
                if deadline <= self._monotonic:
                    ready.append(future)
                else:
                    remaining.append((deadline, future))
            self._waiters = remaining
        for future in ready:
            if not future.done():
                _resolve_future(future)

    def set_monotonic(self, value: float) -> None:
        """Set monotonic time to an absolute value."""
        with self._lock:
            self._monotonic = value

    def set_wall(self, value: datetime) -> None:
        """Set wall-clock time to an absolute value.

        Args:
            value: Must be timezone-aware UTC datetime.

        Raises:
            ValueError: If value is not timezone-aware.
        """
        if value.tzinfo is None:
            msg = "Wall clock time must be timezone-aware"
            raise ValueError(msg)
        with self._lock:
            self._wall = value


__all__ = [
    "SYSTEM_CLOCK",
    "AsyncSleeper",
    "Clock",
    "FakeClock",
    "MonotonicClock",
    "Sleeper",
    "SystemClock",
    "WallClock",
]
