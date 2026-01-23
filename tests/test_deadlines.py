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

"""Tests for the :mod:`weakincentives.deadlines` module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline


def test_deadline_rejects_naive_datetime() -> None:
    naive = datetime.now()
    with pytest.raises(ValueError):
        Deadline(naive)


def test_deadline_rejects_past_datetime() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    with pytest.raises(ValueError):
        Deadline(anchor - timedelta(seconds=10), clock=clock)


def test_deadline_requires_future_second() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)
    clock.set_wall(anchor)

    with pytest.raises(ValueError):
        Deadline(anchor + timedelta(milliseconds=500), clock=clock)


def test_deadline_remaining_uses_override() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=30), clock=clock)

    remaining = deadline.remaining(now=anchor + timedelta(seconds=5))

    assert remaining == timedelta(seconds=25)


def test_deadline_remaining_uses_clock() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=30), clock=clock)

    assert deadline.remaining() == timedelta(seconds=30)

    clock.advance(10)  # 10 seconds pass
    assert deadline.remaining() == timedelta(seconds=20)


def test_deadline_remaining_rejects_naive_datetime() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=30), clock=clock)

    with pytest.raises(ValueError):
        deadline.remaining(now=datetime(2024, 1, 1, 12, 0))


def test_fake_clock_set_wall_rejects_naive_datetime() -> None:
    clock = FakeClock()

    with pytest.raises(ValueError, match="timezone-aware"):
        clock.set_wall(datetime(2024, 1, 1, 12, 0))  # naive datetime


def test_fake_clock_advance_rejects_negative_seconds() -> None:
    clock = FakeClock()

    with pytest.raises(ValueError, match="negative"):
        clock.advance(-5)


def test_deadline_started_at_defaults_to_now() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=30), clock=clock)

    assert deadline.started_at == anchor


def test_deadline_started_at_can_be_explicit() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    explicit_start = anchor - timedelta(minutes=5)
    deadline = Deadline(
        anchor + timedelta(seconds=30),
        started_at=explicit_start,
        clock=clock,
    )

    assert deadline.started_at == explicit_start


def test_deadline_started_at_rejects_naive_datetime() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    with pytest.raises(ValueError, match="started_at must be timezone-aware"):
        Deadline(
            anchor + timedelta(seconds=30),
            started_at=datetime(2024, 1, 1, 11, 55),  # naive
            clock=clock,
        )


def test_deadline_elapsed_uses_clock() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=60), clock=clock)

    assert deadline.elapsed() == timedelta(seconds=0)

    clock.advance(10)  # 10 seconds pass
    assert deadline.elapsed() == timedelta(seconds=10)


def test_deadline_elapsed_uses_override() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=60), clock=clock)

    elapsed = deadline.elapsed(now=anchor + timedelta(seconds=25))

    assert elapsed == timedelta(seconds=25)


def test_deadline_elapsed_rejects_naive_datetime() -> None:
    clock = FakeClock()
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    clock.set_wall(anchor)

    deadline = Deadline(anchor + timedelta(seconds=30), clock=clock)

    with pytest.raises(ValueError, match="elapsed now must be timezone-aware"):
        deadline.elapsed(now=datetime(2024, 1, 1, 12, 0, 10))
