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

"""Tests for Clock abstraction."""

from datetime import UTC, datetime

from weakincentives.runtime.clock import FakeClock


def test_fake_clock_advance(clock: FakeClock) -> None:
    """Test FakeClock.advance() advances monotonic time."""
    clock = FakeClock()
    initial = clock.monotonic()

    clock.advance(10.5)

    assert clock.monotonic() == initial + 10.5


def test_fake_clock_set_now(clock: FakeClock) -> None:
    """Test FakeClock.set_now() sets current datetime and resets monotonic."""
    clock = FakeClock()
    clock.advance(5.0)  # Advance first

    new_time = datetime(2025, 6, 15, 12, 30, 45, tzinfo=UTC)
    clock.set_now(new_time)

    # Monotonic should be reset to 0
    assert clock.monotonic() == 0.0
    # now() should return the set time
    assert clock.now() == new_time


def test_fake_clock_now_advances_with_monotonic(clock: FakeClock) -> None:
    """Test that FakeClock.now() advances as monotonic time advances."""
    clock = FakeClock()
    start_time = clock.now()

    clock.advance(3600.0)  # 1 hour

    end_time = clock.now()
    assert (end_time - start_time).total_seconds() == 3600.0


def test_fake_clock_asleep(clock: FakeClock) -> None:
    """Test FakeClock.asleep() advances time without real delay."""
    import asyncio

    clock = FakeClock()
    initial = clock.monotonic()

    asyncio.run(clock.asleep(2.5))

    # Time should have advanced without real delay
    assert clock.monotonic() == initial + 2.5
