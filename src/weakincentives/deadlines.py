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

"""Deadline utilities for orchestrating prompt evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .clock import SYSTEM_CLOCK, WallClock
from .dataclasses import FrozenDataclassMixin

__all__ = ["Deadline"]


@dataclass(slots=True, frozen=True)
class Deadline(FrozenDataclassMixin):
    """Immutable value object describing a wall-clock expiration.

    Example::

        from datetime import UTC, datetime, timedelta
        from weakincentives import Deadline

        # Create a deadline 1 hour from now
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        # Check remaining time
        remaining = deadline.remaining()

        # Check elapsed time since deadline was created
        elapsed = deadline.elapsed()

    For testing, inject a controllable clock::

        from weakincentives.clock import TestClock

        clock = TestClock()
        clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))

        deadline = Deadline(
            expires_at=datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC),
            clock=clock,
        )
        assert deadline.remaining() == timedelta(hours=1)
        assert deadline.elapsed() == timedelta(0)

        clock.advance(1800)  # 30 minutes
        assert deadline.remaining() == timedelta(minutes=30)
        assert deadline.elapsed() == timedelta(minutes=30)
    """

    expires_at: datetime
    started_at: datetime | None = None
    """When the deadline tracking started. Defaults to creation time."""
    clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)
    """Clock for time operations. Defaults to system clock."""

    def __post_init__(self) -> None:
        expires_at = self.expires_at
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            msg = "Deadline expires_at must be timezone-aware."
            raise ValueError(msg)

        now = self.clock.utcnow()

        # Default started_at to now if not provided
        if self.started_at is None:
            object.__setattr__(self, "started_at", now)
        else:
            started_at = self.started_at
            if started_at.tzinfo is None or started_at.utcoffset() is None:
                msg = "Deadline started_at must be timezone-aware."
                raise ValueError(msg)

        if expires_at <= now:
            msg = "Deadline expires_at must be in the future."
            raise ValueError(msg)

        if expires_at - now < timedelta(seconds=1):
            msg = "Deadline must be at least one second in the future."
            raise ValueError(msg)

    def remaining(self, *, now: datetime | None = None) -> timedelta:
        """Return the remaining duration before expiration.

        Args:
            now: Override current time. If None, uses the clock's utcnow().
                Must be timezone-aware.

        Returns:
            Duration until expiration. May be negative if deadline has passed.

        Raises:
            ValueError: If now is provided but not timezone-aware.
        """
        current = now if now is not None else self.clock.utcnow()
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "Deadline remaining now must be timezone-aware."
            raise ValueError(msg)

        return self.expires_at - current

    def elapsed(self, *, now: datetime | None = None) -> timedelta:
        """Return the duration since tracking started.

        Args:
            now: Override current time. If None, uses the clock's utcnow().
                Must be timezone-aware.

        Returns:
            Duration since started_at. Always non-negative under normal use.

        Raises:
            ValueError: If now is provided but not timezone-aware.
        """
        current = now if now is not None else self.clock.utcnow()
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "Deadline elapsed now must be timezone-aware."
            raise ValueError(msg)

        # started_at is guaranteed to be set by __post_init__
        assert self.started_at is not None  # nosec B101
        return current - self.started_at
