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

"""Tests for wait_until utility."""

from __future__ import annotations

import time

from weakincentives.runtime import wait_until

# =============================================================================
# wait_until Tests
# =============================================================================


def test_wait_until_returns_true_when_predicate_succeeds() -> None:
    """wait_until returns True when predicate becomes True."""
    counter = {"value": 0}

    def predicate() -> bool:
        counter["value"] += 1
        return counter["value"] >= 3

    result = wait_until(predicate, timeout=1.0, poll_interval=0.01)
    assert result is True
    assert counter["value"] >= 3


def test_wait_until_returns_false_on_timeout() -> None:
    """wait_until returns False when timeout expires."""
    result = wait_until(lambda: False, timeout=0.1, poll_interval=0.01)
    assert result is False


def test_wait_until_returns_immediately_if_predicate_true() -> None:
    """wait_until returns immediately if predicate is already True."""
    start = time.monotonic()
    result = wait_until(lambda: True, timeout=10.0, poll_interval=1.0)
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.5  # Should return well before 1 second poll interval
