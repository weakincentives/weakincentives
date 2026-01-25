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

"""Deadline utilities for time-bound operations.

This module provides the ``Deadline`` class, an immutable value object for
tracking wall-clock expirations. Deadlines are useful for setting time limits
on operations, tracking elapsed time, and ensuring timezone-aware time handling.

All datetime values must be timezone-aware (typically UTC). The module integrates
with the clock abstraction layer to support both production use with real time
and testing with controllable fake clocks.
"""

from __future__ import annotations

from dataclasses import field
from datetime import datetime, timedelta

from .clock import SYSTEM_CLOCK, WallClock
from .dataclasses import FrozenDataclass

__all__ = ["Deadline"]


@FrozenDataclass()
class Deadline:
    """Immutable value object representing a wall-clock expiration time.

    A Deadline tracks when an operation should expire and how much time has
    elapsed since tracking began. All datetime values must be timezone-aware
    to ensure correct behavior across timezones.

    Args:
        expires_at: When the deadline expires. Must be timezone-aware and
            at least 1 second in the future at creation time.
        started_at: When deadline tracking started. Defaults to the current
            time at creation. Must be timezone-aware if provided.
        clock: Clock implementation for time operations. Defaults to
            ``SYSTEM_CLOCK``. Inject a ``TestClock`` for deterministic testing.

    Raises:
        ValueError: If ``expires_at`` is not timezone-aware.
        ValueError: If ``expires_at`` is not in the future.
        ValueError: If ``expires_at`` is less than 1 second in the future.
        ValueError: If ``started_at`` is provided but not timezone-aware.

    Example::

        from datetime import UTC, datetime, timedelta
        from weakincentives import Deadline

        # Create a deadline 1 hour from now
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        # Check remaining time until expiration
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
    """The timezone-aware datetime when this deadline expires."""
    started_at: datetime | None = None
    """When deadline tracking started. Defaults to creation time if not provided."""
    clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)
    """Clock used for ``remaining()`` and ``elapsed()`` calculations. Inject a
    ``TestClock`` for deterministic testing. Excluded from repr and equality."""

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
        """Return the time remaining until this deadline expires.

        Use this to check how much time is left for an operation, implement
        timeouts, or decide whether to continue or abort.

        Args:
            now: Override current time for calculation. If ``None``, uses the
                clock's ``utcnow()``. Must be timezone-aware if provided.

        Returns:
            A timedelta representing time until expiration. Positive if the
            deadline is still in the future, negative if it has passed.

        Raises:
            ValueError: If ``now`` is provided but not timezone-aware.

        Example::

            if deadline.remaining() < timedelta(seconds=30):
                # Not enough time, abort operation
                raise TimeoutError("Insufficient time remaining")
        """
        current = now if now is not None else self.clock.utcnow()
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "Deadline remaining now must be timezone-aware."
            raise ValueError(msg)

        return self.expires_at - current

    def elapsed(self, *, now: datetime | None = None) -> timedelta:
        """Return the time elapsed since this deadline started tracking.

        Use this to measure how long an operation has been running, for
        logging, metrics, or progress reporting.

        Args:
            now: Override current time for calculation. If ``None``, uses the
                clock's ``utcnow()``. Must be timezone-aware if provided.

        Returns:
            A timedelta representing time since ``started_at``. Always
            non-negative under normal operation (when ``now >= started_at``).

        Raises:
            ValueError: If ``now`` is provided but not timezone-aware.

        Example::

            elapsed = deadline.elapsed()
            logger.info(f"Operation running for {elapsed.total_seconds():.1f}s")
        """
        current = now if now is not None else self.clock.utcnow()
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "Deadline elapsed now must be timezone-aware."
            raise ValueError(msg)

        # started_at is guaranteed to be set by __post_init__
        assert self.started_at is not None  # nosec B101
        return current - self.started_at
