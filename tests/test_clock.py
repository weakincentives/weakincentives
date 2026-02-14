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

"""Tests for :mod:`weakincentives.clock`, including the AsyncSleeper protocol."""

from __future__ import annotations

import asyncio

import pytest

from weakincentives.clock import (
    SYSTEM_CLOCK,
    AsyncSleeper,
    Clock,
    FakeClock,
    SystemClock,
)


def _run(coro: object) -> object:
    """Run a coroutine synchronously."""
    return asyncio.run(coro)  # type: ignore[arg-type]


class TestAsyncSleeperProtocol:
    """Tests for the AsyncSleeper protocol."""

    def test_system_clock_satisfies_async_sleeper(self) -> None:
        assert isinstance(SYSTEM_CLOCK, AsyncSleeper)

    def test_fake_clock_satisfies_async_sleeper(self) -> None:
        clock = FakeClock()
        assert isinstance(clock, AsyncSleeper)

    def test_system_clock_satisfies_clock(self) -> None:
        assert isinstance(SYSTEM_CLOCK, Clock)

    def test_fake_clock_satisfies_clock(self) -> None:
        clock = FakeClock()
        assert isinstance(clock, Clock)


class TestFakeClockAsyncSleep:
    """Tests for FakeClock.async_sleep."""

    def test_async_sleep_advances_monotonic(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()

        _run(clock.async_sleep(5.0))

        assert clock.monotonic() - start == 5.0

    def test_async_sleep_advances_wall_clock(self) -> None:
        clock = FakeClock()
        start = clock.utcnow()

        _run(clock.async_sleep(10.0))

        elapsed = (clock.utcnow() - start).total_seconds()
        assert elapsed == 10.0

    def test_async_sleep_zero_is_noop(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()

        _run(clock.async_sleep(0.0))

        assert clock.monotonic() == start

    def test_async_sleep_negative_raises(self) -> None:
        clock = FakeClock()

        with pytest.raises(ValueError, match="negative"):
            _run(clock.async_sleep(-1.0))


class TestSystemClockAsyncSleep:
    """Tests for SystemClock.async_sleep."""

    def test_async_sleep_completes(self) -> None:
        clock = SystemClock()
        _run(clock.async_sleep(0.0))

    def test_async_sleep_is_awaitable(self) -> None:
        clock = SystemClock()
        coro = clock.async_sleep(0.0)
        assert asyncio.iscoroutine(coro)
        _run(coro)
