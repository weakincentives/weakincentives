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
    """Tests for FakeClock.async_sleep waiter-based behavior."""

    def test_async_sleep_suspends_until_advance(self) -> None:
        """async_sleep(N) suspends; advance(N) wakes it."""
        clock = FakeClock()

        async def run() -> None:
            start = clock.monotonic()
            # Schedule advance after yielding control.
            asyncio.get_running_loop().call_soon(clock.advance, 5.0)
            await clock.async_sleep(5.0)
            assert clock.monotonic() - start == 5.0

        _run(run())

    def test_async_sleep_advance_wakes_wall_clock(self) -> None:
        """Wall clock also advances when waiters are woken."""
        clock = FakeClock()

        async def run() -> None:
            start = clock.utcnow()
            asyncio.get_running_loop().call_soon(clock.advance, 10.0)
            await clock.async_sleep(10.0)
            elapsed = (clock.utcnow() - start).total_seconds()
            assert elapsed == 10.0

        _run(run())

    def test_async_sleep_zero_returns_immediately(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()

        _run(clock.async_sleep(0.0))

        assert clock.monotonic() == start

    def test_async_sleep_negative_raises(self) -> None:
        clock = FakeClock()

        with pytest.raises(ValueError, match="negative"):
            _run(clock.async_sleep(-1.0))

    def test_async_sleep_partial_advance_does_not_wake(self) -> None:
        """Advancing less than the deadline keeps the waiter suspended."""
        clock = FakeClock()
        woke = False

        async def sleeper() -> None:
            nonlocal woke
            await clock.async_sleep(1.0)
            woke = True

        async def run() -> None:
            task = asyncio.create_task(sleeper())
            await asyncio.sleep(0)  # let sleeper register waiter
            clock.advance(0.5)
            await asyncio.sleep(0)  # yield — sleeper should NOT wake
            assert not woke
            clock.advance(0.5)
            await asyncio.sleep(0)  # yield — sleeper should wake
            assert woke
            await task

        _run(run())

    def test_async_sleep_multiple_waiters(self) -> None:
        """Multiple concurrent async_sleeps wake at their respective deadlines."""
        clock = FakeClock()
        order: list[str] = []

        async def sleeper(name: str, seconds: float) -> None:
            await clock.async_sleep(seconds)
            order.append(name)

        async def run() -> None:
            t1 = asyncio.create_task(sleeper("short", 1.0))
            t2 = asyncio.create_task(sleeper("long", 3.0))
            await asyncio.sleep(0)  # let both register
            clock.advance(1.0)
            await asyncio.sleep(0)
            assert order == ["short"]
            clock.advance(2.0)
            await asyncio.sleep(0)
            assert order == ["short", "long"]
            await t1
            await t2

        _run(run())

    def test_sync_sleep_wakes_async_waiters(self) -> None:
        """sync sleep() advances time and wakes async waiters."""
        clock = FakeClock()

        async def run() -> None:
            woke = False

            async def waiter() -> None:
                nonlocal woke
                await clock.async_sleep(1.0)
                woke = True

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)  # let waiter register
            clock.sleep(1.0)
            await asyncio.sleep(0)  # yield to let waiter run
            assert woke
            await task

        _run(run())


class TestFakeClockEdgeCases:
    """Tests for FakeClock edge cases and _resolve_future branches."""

    def test_advance_skips_already_done_future(self) -> None:
        """advance() skips futures that are already done (cancelled)."""
        clock = FakeClock()

        async def run() -> None:
            task = asyncio.create_task(clock.async_sleep(1.0))
            await asyncio.sleep(0)  # let it register
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # advance should skip the cancelled future without error
            clock.advance(1.0)

        _run(run())

    def test_advance_from_non_loop_thread(self) -> None:
        """advance() from a thread with no running loop uses call_soon_threadsafe."""
        import threading

        clock = FakeClock()
        woke = False

        async def run() -> None:
            nonlocal woke

            async def sleeper() -> None:
                nonlocal woke
                await clock.async_sleep(1.0)
                woke = True

            task = asyncio.create_task(sleeper())
            await asyncio.sleep(0)  # let sleeper register

            # Advance from a different thread
            done = threading.Event()

            def bg() -> None:
                clock.advance(1.0)
                done.set()

            t = threading.Thread(target=bg)
            t.start()
            done.wait(timeout=2.0)
            t.join()

            await asyncio.sleep(0)  # let the resolved future propagate
            await asyncio.sleep(0)
            assert woke
            await task

        _run(run())

    def test_resolve_future_no_running_loop(self) -> None:
        """_resolve_future works when called with no running event loop."""
        from weakincentives.clock import _resolve_future

        loop = asyncio.new_event_loop()
        future: asyncio.Future[None] = loop.create_future()
        # Call from a context with no running loop
        _resolve_future(future)
        assert future.done()
        loop.close()


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
