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

"""Clock control helpers for tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weakincentives import deadlines


class FrozenUtcNow:
    """Controller for :func:`weakincentives.deadlines._utcnow` during tests."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, anchor: datetime | None = None):
        self._current = anchor if anchor is not None else datetime.now(UTC)
        monkeypatch.setattr(deadlines, "_utcnow", self.now)

    def now(self) -> datetime:
        """Return the frozen current time."""

        return self._current

    def set(self, current: datetime) -> datetime:
        """Reset the frozen clock to the provided datetime."""

        self._current = current
        return self._current

    def advance(self, delta: timedelta) -> datetime:
        """Move the frozen clock forward by ``delta``."""

        self._current += delta
        return self._current


@pytest.fixture
def frozen_utcnow(monkeypatch: pytest.MonkeyPatch) -> FrozenUtcNow:
    """Provide a controllable :func:`_utcnow` override for deadline tests."""

    return FrozenUtcNow(monkeypatch)


__all__ = ["FrozenUtcNow", "frozen_utcnow"]
