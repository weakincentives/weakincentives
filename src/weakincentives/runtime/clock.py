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

"""
Controllable time source for deterministic testing.

See specs/CLOCK.md for complete specification and rationale.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """
    Controllable time source for deterministic testing.

    All time-dependent operations must go through a Clock instance.
    Production code uses SystemClock (real time), tests use FakeClock
    (simulated time).
    """

    def now(self) -> datetime:
        """
        Current UTC datetime.

        Production: Returns actual wall clock time
        Tests: Returns simulated time controlled by test harness
        """
        ...

    def monotonic(self) -> float:
        """
        Monotonic timestamp in seconds.

        Used for measuring intervals and timeouts. Guaranteed to never
        go backwards, even if system clock is adjusted.

        Production: Returns time.monotonic()
        Tests: Returns simulated monotonic counter
        """
        ...

    def sleep(self, seconds: float) -> None:
        """
        Block for specified duration.

        Production: time.sleep(seconds)
        Tests: Advance simulated time without real delay
        """
        ...

    async def asleep(self, seconds: float) -> None:
        """
        Async sleep for specified duration.

        Production: await asyncio.sleep(seconds)
        Tests: Advance simulated time without real delay
        """
        ...


class SystemClock:
    """Real clock for production use."""

    def now(self) -> datetime:  # noqa: PLR6301 - implements Clock protocol
        return datetime.now(UTC)

    def monotonic(self) -> float:  # noqa: PLR6301 - implements Clock protocol
        return time.monotonic()

    def sleep(self, seconds: float) -> None:  # noqa: PLR6301 - implements Clock protocol
        time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:  # noqa: PLR6301 - implements Clock protocol
        await asyncio.sleep(seconds)


@dataclass
class FakeClock:
    """
    Controllable clock for deterministic testing.

    Example:
        clock = FakeClock()
        clock.advance(10.0)  # Jump forward 10 seconds
        assert clock.monotonic() == 10.0
    """

    _utc_epoch: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC)
    )
    _monotonic: float = 0.0

    def now(self) -> datetime:
        """Current simulated UTC time."""
        return self._utc_epoch + timedelta(seconds=self._monotonic)

    def monotonic(self) -> float:
        """Current simulated monotonic timestamp."""
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        """Advance simulated time without blocking."""
        self._monotonic += seconds

    async def asleep(self, seconds: float) -> None:
        """Async advance simulated time without blocking."""
        self._monotonic += seconds

    def advance(self, seconds: float) -> None:
        """Manually advance simulated time."""
        self._monotonic += seconds

    def set_now(self, dt: datetime) -> None:
        """Set current simulated datetime."""
        self._utc_epoch = dt
        self._monotonic = 0.0
