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

This module provides deterministic time control for testing time-dependent code
without actual waiting. The primary class is :class:`DeterministicClock` which
replaces all time operations with controlled versions.

Example::

    def test_timeout(deterministic_clock):
        clock = deterministic_clock

        # Code that normally waits 30 seconds runs instantly
        clock.advance(30.0)

        # Sleeps are tracked but don't actually wait
        assert clock.total_slept >= 0

For threaded tests, use synchronization instead of sleeps::

    def test_worker(deterministic_clock):
        started = threading.Event()

        def worker():
            started.set()
            # do work

        thread = threading.Thread(target=worker)
        thread.start()
        started.wait()  # Replaces time.sleep(0.1)
        thread.join()
"""

from __future__ import annotations

import threading
import time as _real_time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from weakincentives import deadlines

if TYPE_CHECKING:
    from types import ModuleType


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


class DeterministicClock:
    """Deterministic clock for testing time-dependent code.

    Replaces ``time.monotonic()``, ``time.time()``, and ``time.sleep()``
    with controlled versions. Time only advances when explicitly requested
    via :meth:`advance` or when :meth:`sleep` is called.

    The clock patches both the ``weakincentives.runtime._clock`` module
    (for production code) and can optionally patch the ``time`` module
    directly in specified modules.

    Thread Safety:
        The clock is thread-safe. Multiple threads can call :meth:`monotonic`,
        :meth:`time`, and :meth:`sleep` concurrently. The :meth:`advance`
        method will wake up all sleeping threads whose sleep duration has
        been satisfied by the advancement.

    Example::

        def test_wait_until(monkeypatch):
            clock = DeterministicClock(monkeypatch, initial_monotonic=1000.0)

            def predicate():
                return clock.monotonic() > 1010.0

            # Start waiting in background
            result = [None]
            def waiter():
                result[0] = wait_until(predicate, timeout=20.0)

            thread = threading.Thread(target=waiter)
            thread.start()

            # Advance time to satisfy predicate
            clock.advance(15.0)
            thread.join()

            assert result[0] is True

    Example with auto-advance on sleep::

        def test_sleeper(monkeypatch):
            clock = DeterministicClock(monkeypatch, auto_advance_on_sleep=True)

            # Sleep automatically advances clock
            clock.sleep(5.0)
            assert clock.monotonic() >= 5.0

    Attributes:
        total_slept: Total seconds passed to sleep() calls.
        sleep_count: Number of times sleep() was called.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        initial_monotonic: float = 1000.0,
        initial_time: float | None = None,
        auto_advance_on_sleep: bool = True,
        modules_to_patch: tuple[ModuleType, ...] = (),
    ) -> None:
        """Initialize the deterministic clock.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            initial_monotonic: Starting value for monotonic time.
            initial_time: Starting value for wall clock time.
                Defaults to current time if not specified.
            auto_advance_on_sleep: If True, sleep() advances the clock.
                If False, sleep() returns immediately without advancing
                (useful for testing sleep call counts).
            modules_to_patch: Additional modules to patch ``time.monotonic``,
                ``time.time``, and ``time.sleep`` in. The clock module
                is always patched.
        """
        self._lock = threading.Lock()
        self._monotonic = initial_monotonic
        self._time = initial_time if initial_time is not None else _real_time.time()
        self._auto_advance = auto_advance_on_sleep
        self._monkeypatch = monkeypatch
        self._condition = threading.Condition(self._lock)
        self._sleepers: list[tuple[float, threading.Event]] = []

        # Track sleep statistics
        self.total_slept: float = 0.0
        self.sleep_count: int = 0

        # Import modules that use time functions directly
        from weakincentives.runtime import lease_extender, lifecycle, watchdog
        from weakincentives.runtime.mailbox import _in_memory

        # Patch time module in the source modules
        # These modules import time at module level, so we patch the time module
        # that was imported into each module's namespace
        monkeypatch.setattr(watchdog.time, "monotonic", self.monotonic)
        monkeypatch.setattr(lifecycle.time, "monotonic", self.monotonic)
        monkeypatch.setattr(lifecycle.time, "sleep", self.sleep)
        monkeypatch.setattr(_in_memory.time, "monotonic", self.monotonic)
        monkeypatch.setattr(lease_extender.time, "monotonic", self.monotonic)

        # Patch additional modules
        for mod in modules_to_patch:
            monkeypatch.setattr(mod, "monotonic", self.monotonic, raising=False)
            monkeypatch.setattr(mod, "time", self.time, raising=False)
            monkeypatch.setattr(mod, "sleep", self.sleep, raising=False)

    def monotonic(self) -> float:
        """Return the current monotonic time."""
        with self._lock:
            return self._monotonic

    def time(self) -> float:
        """Return the current wall clock time."""
        with self._lock:
            return self._time

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration.

        If ``auto_advance_on_sleep`` is True, this advances the clock
        and returns immediately. Otherwise, it waits until enough time
        has been advanced via :meth:`advance`.
        """
        if seconds <= 0:
            return

        self.sleep_count += 1
        self.total_slept += seconds

        if self._auto_advance:
            # Immediately advance clock and return
            self.advance(seconds)
            return

        # Register as a sleeper and wait for advance()
        wake_event = threading.Event()
        with self._lock:
            wake_time = self._monotonic + seconds
            self._sleepers.append((wake_time, wake_event))

        # Wait for wake event with a real timeout as safety net
        # This ensures tests don't hang if advance() is never called
        wake_event.wait(timeout=5.0)

    def advance(self, seconds: float) -> float:
        """Advance both monotonic and wall clock time.

        Args:
            seconds: Seconds to advance. Must be non-negative.

        Returns:
            New monotonic time value.

        Wakes up any threads waiting in :meth:`sleep` whose sleep
        duration has been satisfied.
        """
        if seconds < 0:
            msg = "Cannot advance clock backwards"
            raise ValueError(msg)

        with self._lock:
            self._monotonic += seconds
            self._time += seconds
            new_monotonic = self._monotonic

            # Wake sleepers whose time has come
            remaining: list[tuple[float, threading.Event]] = []
            for wake_time, event in self._sleepers:
                if wake_time <= new_monotonic:
                    event.set()
                else:
                    remaining.append((wake_time, event))
            self._sleepers = remaining
            self._condition.notify_all()

        return new_monotonic

    def set_monotonic(self, value: float) -> None:
        """Set the monotonic clock to a specific value.

        Use :meth:`advance` for normal time progression. This method
        is for setting up initial test conditions.
        """
        with self._lock:
            self._monotonic = value

    def set_time(self, value: float) -> None:
        """Set the wall clock to a specific value."""
        with self._lock:
            self._time = value

    def wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.01,
    ) -> bool:
        """Wait until predicate returns True or timeout expires.

        This is a test utility that polls in real time (not simulated).
        Use for waiting on threading.Event or other real synchronization.

        Args:
            predicate: Callable returning True when done waiting.
            timeout: Maximum real seconds to wait.
            poll_interval: Real seconds between predicate checks.

        Returns:
            True if predicate returned True, False if timeout expired.
        """
        deadline = _real_time.monotonic() + timeout
        while _real_time.monotonic() < deadline:
            if predicate():
                return True
            _real_time.sleep(poll_interval)
        return predicate()


@pytest.fixture
def deterministic_clock(monkeypatch: pytest.MonkeyPatch) -> DeterministicClock:
    """Provide a deterministic clock for testing time-dependent code.

    The clock patches ``weakincentives.runtime._clock`` so that
    production code using the clock module gets deterministic time.

    Example::

        def test_timeout(deterministic_clock):
            clock = deterministic_clock
            clock.advance(30.0)  # Instant 30-second jump
    """
    return DeterministicClock(monkeypatch)


__all__ = [
    "DeterministicClock",
    "FrozenUtcNow",
    "deterministic_clock",
    "frozen_utcnow",
]
