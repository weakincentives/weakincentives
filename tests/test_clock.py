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

"""Tests for ``weakincentives.clock``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.clock import (
    SYSTEM_CLOCK,
    AsyncSleeper,
    Budget,
    BudgetTracker,
    Clock,
    Deadline,
    FakeClock,
    MonotonicClock,
    Sleeper,
    SystemClock,
    WallClock,
)
from weakincentives.core import (
    BudgetExceededError,
    DeadlineExceededError,
    Usage,
)

# ---------------------------------------------------------------------------
# Protocol membership
# ---------------------------------------------------------------------------


def test_system_clock_satisfies_all_protocols() -> None:
    clock = SystemClock()
    assert isinstance(clock, WallClock)
    assert isinstance(clock, MonotonicClock)
    assert isinstance(clock, Sleeper)
    assert isinstance(clock, AsyncSleeper)
    assert isinstance(clock, Clock)


def test_fake_clock_satisfies_all_protocols() -> None:
    clock = FakeClock()
    assert isinstance(clock, Clock)


def test_module_singleton_is_a_clock() -> None:
    assert isinstance(SYSTEM_CLOCK, Clock)


# ---------------------------------------------------------------------------
# SystemClock behavior
# ---------------------------------------------------------------------------


def test_system_clock_returns_aware_utc() -> None:
    now = SystemClock().utcnow()
    assert now.tzinfo is not None


def test_system_clock_monotonic_increases() -> None:
    clock = SystemClock()
    a = clock.monotonic()
    b = clock.monotonic()
    assert b >= a


def test_system_clock_sleep_zero_returns() -> None:
    SystemClock().sleep(0)


def test_system_clock_async_sleep_zero_returns() -> None:
    asyncio.run(SystemClock().async_sleep(0))


# ---------------------------------------------------------------------------
# FakeClock behavior
# ---------------------------------------------------------------------------


def test_fake_clock_advances_wall_and_monotonic() -> None:
    clock = FakeClock()
    start_wall = clock.utcnow()
    start_mono = clock.monotonic()
    clock.advance(5.0)
    assert clock.utcnow() == start_wall + timedelta(seconds=5)
    assert clock.monotonic() == start_mono + 5.0


def test_fake_clock_sleep_advances_time() -> None:
    clock = FakeClock()
    start = clock.monotonic()
    clock.sleep(2.0)
    assert clock.monotonic() == start + 2.0


def test_fake_clock_async_sleep_advances_time() -> None:
    clock = FakeClock()
    start = clock.monotonic()
    asyncio.run(clock.async_sleep(3.0))
    assert clock.monotonic() == start + 3.0


def test_fake_clock_rejects_negative_advance() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="cannot advance"):
        clock.advance(-1.0)


def test_fake_clock_set_wall_requires_aware_datetime() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.set_wall(datetime(2025, 1, 1))


def test_fake_clock_set_wall_overrides_time() -> None:
    clock = FakeClock()
    target = datetime(2030, 6, 1, 12, 0, 0, tzinfo=UTC)
    clock.set_wall(target)
    assert clock.utcnow() == target


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------


def test_deadline_create_requires_aware_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Deadline.create(expires_at=datetime(2030, 1, 1))


def test_deadline_create_requires_future_time() -> None:
    clock = FakeClock()
    near = clock.utcnow() + timedelta(milliseconds=500)
    with pytest.raises(ValueError, match="at least 1s"):
        Deadline.create(expires_at=near, clock=clock)


def test_deadline_remaining_and_expired() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=10),
        clock=clock,
    )
    assert deadline.remaining() == timedelta(seconds=10)
    assert not deadline.expired()
    clock.advance(10.5)
    assert deadline.remaining() == timedelta()
    assert deadline.expired()


def test_deadline_check_raises_when_expired() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=2),
        clock=clock,
    )
    deadline.check()
    clock.advance(3)
    with pytest.raises(DeadlineExceededError):
        deadline.check()


# ---------------------------------------------------------------------------
# Budget / BudgetTracker
# ---------------------------------------------------------------------------


def test_budget_tracker_records_and_checks() -> None:
    tracker = BudgetTracker(budget=Budget(max_total_tokens=100))
    tracker.record(Usage(input_tokens=10, output_tokens=20, total_tokens=30))
    tracker.check()
    assert tracker.consumed.total_tokens == 30


def test_budget_tracker_exceeds_total() -> None:
    tracker = BudgetTracker(budget=Budget(max_total_tokens=10))
    tracker.record(Usage(total_tokens=11))
    with pytest.raises(BudgetExceededError, match="total tokens"):
        tracker.check()


def test_budget_tracker_exceeds_input() -> None:
    tracker = BudgetTracker(budget=Budget(max_input_tokens=5))
    tracker.record(Usage(input_tokens=6))
    with pytest.raises(BudgetExceededError, match="input tokens"):
        tracker.check()


def test_budget_tracker_exceeds_output() -> None:
    tracker = BudgetTracker(budget=Budget(max_output_tokens=5))
    tracker.record(Usage(output_tokens=6))
    with pytest.raises(BudgetExceededError, match="output tokens"):
        tracker.check()


def test_budget_tracker_record_cumulative() -> None:
    tracker = BudgetTracker(budget=Budget())
    tracker.record_cumulative("a", Usage(total_tokens=5))
    tracker.record_cumulative("a", Usage(total_tokens=12))
    tracker.record_cumulative("b", Usage(total_tokens=3))
    assert tracker.consumed.total_tokens == 15


def test_budget_tracker_with_deadline_propagates() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=5),
        clock=clock,
    )
    tracker = BudgetTracker(budget=Budget(deadline=deadline))
    tracker.check()
    clock.advance(6)
    with pytest.raises(DeadlineExceededError):
        tracker.check()


def test_budget_tracker_unlimited_budget() -> None:
    tracker = BudgetTracker(budget=Budget())
    tracker.record(Usage(total_tokens=10**9))
    tracker.check()
