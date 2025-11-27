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

from weakincentives.deadlines import Deadline
from tests.helpers import FrozenUtcNow


def test_deadline_rejects_naive_datetime() -> None:
    naive = datetime.now()
    with pytest.raises(ValueError):
        Deadline(naive)


def test_deadline_rejects_past_datetime(frozen_utcnow: FrozenUtcNow) -> None:
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    with pytest.raises(ValueError):
        Deadline(anchor - timedelta(seconds=10))


def test_deadline_requires_future_second(frozen_utcnow: FrozenUtcNow) -> None:
    anchor = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    with pytest.raises(ValueError):
        Deadline(anchor + timedelta(milliseconds=500))


def test_deadline_remaining_uses_override(frozen_utcnow: FrozenUtcNow) -> None:
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=30))

    remaining = deadline.remaining(now=anchor + timedelta(seconds=5))

    assert remaining == timedelta(seconds=25)


def test_deadline_remaining_rejects_naive_datetime(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=30))

    with pytest.raises(ValueError):
        deadline.remaining(now=datetime(2024, 1, 1, 12, 0))
