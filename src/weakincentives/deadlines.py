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

from dataclasses import field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from .runtime.clock import Clock

__all__ = ["Deadline"]


def _default_clock() -> Clock:
    """Create default clock instance."""
    from .runtime.clock import SystemClock

    return SystemClock()


@FrozenDataclass()
class Deadline:
    """Immutable value object describing a wall-clock expiration."""

    expires_at: datetime
    clock: Clock = field(default_factory=_default_clock)

    def __post_init__(self) -> None:
        expires_at = self.expires_at
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            msg = "Deadline expires_at must be timezone-aware."
            raise ValueError(msg)

        now = self.clock.now()
        if expires_at <= now:
            msg = "Deadline expires_at must be in the future."
            raise ValueError(msg)

        if expires_at - now < timedelta(seconds=1):
            msg = "Deadline must be at least one second in the future."
            raise ValueError(msg)

    def remaining(self, *, now: datetime | None = None) -> timedelta:
        """Return the remaining duration before expiration."""

        current = now or self.clock.now()
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "Deadline remaining now must be timezone-aware."
            raise ValueError(msg)

        return self.expires_at - current
