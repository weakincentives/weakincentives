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

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

__all__ = ["Deadline"]


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


@dataclass(slots=True, frozen=True)
class Deadline:
    """Immutable value object describing a wall-clock expiration."""

    expires_at: datetime

    def __post_init__(self) -> None:
        expires_at = self.expires_at
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            msg = "Deadline expires_at must be timezone-aware."
            raise ValueError(msg)

        now = _utcnow()
        if expires_at <= now:
            msg = "Deadline expires_at must be in the future."
            raise ValueError(msg)

        if expires_at - now < timedelta(seconds=1):
            msg = "Deadline must be at least one second in the future."
            raise ValueError(msg)

    def remaining(self, *, now: datetime | None = None) -> timedelta:
        """Return the remaining duration before expiration."""

        current = now or _utcnow()
        return self.expires_at - current
