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

"""Concrete clocks plus :class:`Deadline` and :class:`Budget` implementations.

The spine declares :class:`weakincentives.core.Deadline` and
:class:`weakincentives.core.Budget` as protocols only; concrete classes
live here so they can be injected without coupling the core to a clock
implementation.
"""

# `_BudgetTracker` deliberately reaches into peer attributes within this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, final, runtime_checkable

from weakincentives.core import (
    BudgetExceededError,
    DeadlineExceededError,
    Usage,
)

__all__ = [
    "SYSTEM_CLOCK",
    "AsyncSleeper",
    "Budget",
    "BudgetTracker",
    "Clock",
    "Deadline",
    "FakeClock",
    "MonotonicClock",
    "Sleeper",
    "SystemClock",
    "WallClock",
]


@runtime_checkable
class WallClock(Protocol):
    """Wall-clock UTC source."""

    def utcnow(self) -> datetime: ...


@runtime_checkable
class MonotonicClock(Protocol):
    """Monotonic clock for elapsed-time measurements."""

    def monotonic(self) -> float: ...


@runtime_checkable
class Sleeper(Protocol):
    """Synchronous delay primitive."""

    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class AsyncSleeper(Protocol):
    """Asynchronous delay primitive."""

    async def async_sleep(self, seconds: float) -> None: ...


@runtime_checkable
class Clock(WallClock, MonotonicClock, Sleeper, AsyncSleeper, Protocol):
    """Composite clock satisfying every time primitive."""


@final
class SystemClock:
    """Clock backed by :mod:`time` and :mod:`datetime`."""

    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return _time.monotonic()

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)

    async def async_sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)


SYSTEM_CLOCK: Clock = SystemClock()


@final
class FakeClock:
    """Controllable clock for deterministic tests.

    Wall and monotonic time advance only when :meth:`advance` is called or
    when an explicit setter is used. ``sleep`` and ``async_sleep`` advance
    monotonic time without blocking.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._wall = datetime(2025, 1, 1, tzinfo=UTC)
        self._mono = 0.0

    def utcnow(self) -> datetime:
        with self._lock:
            return self._wall

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    async def async_sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance backwards")
        with self._lock:
            self._mono += seconds
            self._wall = self._wall + timedelta(seconds=seconds)

    def set_wall(self, when: datetime) -> None:
        if when.tzinfo is None:
            raise ValueError("set_wall requires a timezone-aware datetime")
        with self._lock:
            self._wall = when


# =============================================================================
# Deadline
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Deadline:
    """Wall-clock cutoff with an injected clock.

    Use :meth:`create` to build one from a future :class:`datetime`. The
    created deadline must be at least one second in the future and use a
    timezone-aware UTC time.
    """

    expires_at: datetime
    _clock: WallClock

    @classmethod
    def create(
        cls, *, expires_at: datetime, clock: WallClock = SYSTEM_CLOCK
    ) -> Deadline:
        if expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        now = clock.utcnow()
        if expires_at <= now + timedelta(seconds=1):
            raise ValueError("deadline must be at least 1s in the future")
        return cls(expires_at=expires_at, _clock=clock)

    def remaining(self) -> timedelta:
        delta = self.expires_at - self._clock.utcnow()
        return delta if delta > timedelta() else timedelta()

    def expired(self) -> bool:
        return self._clock.utcnow() >= self.expires_at

    def check(self) -> None:
        """Raise :class:`DeadlineExceededError` if the deadline has passed."""
        if self.expired():
            raise DeadlineExceededError(
                f"deadline expired at {self.expires_at.isoformat()}"
            )


# =============================================================================
# Budget
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Budget:
    """Resource envelope for a session: token caps + an optional deadline."""

    deadline: Deadline | None = None
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


def _zero_usage() -> Usage:
    return Usage()


@dataclass(slots=True)
class BudgetTracker:
    """Thread-safe accumulator that enforces a :class:`Budget`."""

    budget: Budget
    _consumed: Usage = field(default_factory=_zero_usage)
    _by_id: dict[str, Usage] = field(default_factory=lambda: {})
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def consumed(self) -> Usage:
        with self._lock:
            return self._consumed

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._consumed = _add_usage(self._consumed, usage)

    def record_cumulative(self, evaluation_id: str, usage: Usage) -> None:
        """Record absolute usage for ``evaluation_id``.

        Subsequent calls with the same id replace the previous value, so
        adapters can update partial usage without double-counting.
        """
        with self._lock:
            previous = self._by_id.get(evaluation_id, Usage())
            self._by_id[evaluation_id] = usage
            delta = _subtract_usage(usage, previous)
            self._consumed = _add_usage(self._consumed, delta)

    def check(self) -> None:
        with self._lock:
            consumed = self._consumed
            budget = self.budget
        if budget.deadline is not None:
            budget.deadline.check()
        if (
            budget.max_total_tokens is not None
            and consumed.total_tokens > budget.max_total_tokens
        ):
            raise BudgetExceededError(
                f"total tokens {consumed.total_tokens} > {budget.max_total_tokens}"
            )
        if (
            budget.max_input_tokens is not None
            and consumed.input_tokens > budget.max_input_tokens
        ):
            raise BudgetExceededError(
                f"input tokens {consumed.input_tokens} > {budget.max_input_tokens}"
            )
        if (
            budget.max_output_tokens is not None
            and consumed.output_tokens > budget.max_output_tokens
        ):
            raise BudgetExceededError(
                f"output tokens {consumed.output_tokens} > {budget.max_output_tokens}"
            )


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _subtract_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens - right.input_tokens,
        output_tokens=left.output_tokens - right.output_tokens,
        total_tokens=left.total_tokens - right.total_tokens,
    )
