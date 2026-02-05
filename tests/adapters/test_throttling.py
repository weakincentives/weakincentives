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

"""Tests for throttle policy and backoff utilities."""

from __future__ import annotations

from datetime import timedelta

import pytest

from weakincentives.adapters.core import PROMPT_EVALUATION_PHASE_REQUEST
from weakincentives.adapters.throttle import (
    ThrottleError,
    ThrottlePolicy,
    details_from_error,
    jittered_backoff,
    new_throttle_policy,
    sleep_for,
    throttle_details,
)


def test_jittered_backoff_returns_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, _b: 0.0
    )

    policy = new_throttle_policy(base_delay=timedelta(milliseconds=10))
    retry_after = timedelta(milliseconds=30)

    delay = jittered_backoff(policy=policy, attempt=1, retry_after=retry_after)

    assert delay == retry_after


def test_throttle_policy_validation() -> None:
    with pytest.raises(ValueError):
        new_throttle_policy(max_attempts=0)
    with pytest.raises(ValueError):
        new_throttle_policy(base_delay=timedelta(0))
    with pytest.raises(ValueError):
        new_throttle_policy(max_delay=timedelta(0))
    with pytest.raises(ValueError):
        new_throttle_policy(max_total_delay=timedelta(0))


def test_jittered_backoff_respects_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = new_throttle_policy(
        base_delay=timedelta(seconds=1),
        max_delay=timedelta(seconds=4),
        max_total_delay=timedelta(seconds=10),
    )
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, b: b
    )

    delay = jittered_backoff(policy=policy, attempt=2, retry_after=timedelta(seconds=2))

    assert delay == timedelta(seconds=2)


def test_jittered_backoff_clamps_to_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = new_throttle_policy(
        base_delay=timedelta(seconds=1),
        max_delay=timedelta(seconds=4),
        max_total_delay=timedelta(seconds=10),
    )
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, _b: 0
    )

    delay = jittered_backoff(policy=policy, attempt=2, retry_after=timedelta(seconds=1))

    assert delay == timedelta(seconds=1)


def test_jittered_backoff_with_non_positive_base() -> None:
    policy = ThrottlePolicy(
        max_attempts=1,
        base_delay=timedelta(0),
        max_delay=timedelta(seconds=1),
        max_total_delay=timedelta(seconds=1),
    )

    assert jittered_backoff(policy=policy, attempt=1, retry_after=None) == timedelta(0)


def test_sleep_for_uses_sleeper() -> None:
    from weakincentives.clock import FakeClock

    clock = FakeClock()
    initial = clock.monotonic()

    sleep_for(timedelta(milliseconds=150), sleeper=clock)

    # FakeClock advances time on sleep instead of blocking
    assert clock.monotonic() == initial + 0.15


def test_throttle_error_properties() -> None:
    """Test ThrottleError exposes details through properties."""
    details = throttle_details(
        kind="rate_limit",
        retry_after=timedelta(seconds=5),
        attempts=3,
        retry_safe=False,
    )
    error = ThrottleError(
        "Rate limited",
        prompt_name="test",
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        details=details,
    )

    assert error.kind == "rate_limit"
    assert error.retry_after == timedelta(seconds=5)
    assert error.attempts == 3
    assert error.retry_safe is False


def test_throttle_details_defaults() -> None:
    """Test throttle_details with default values."""
    details = throttle_details(kind="timeout")

    assert details.kind == "timeout"
    assert details.retry_after is None
    assert details.attempts == 1
    assert details.retry_safe is True
    assert details.provider_payload is None


def test_details_from_error_extracts_and_updates() -> None:
    """Test details_from_error extracts details and updates attempts."""
    original_details = throttle_details(
        kind="quota_exhausted",
        retry_after=timedelta(seconds=30),
        provider_payload={"error": "quota exceeded"},
    )
    error = ThrottleError(
        "Quota exhausted",
        prompt_name="test",
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        details=original_details,
    )

    new_details = details_from_error(error, attempts=5, retry_safe=False)

    assert new_details.kind == "quota_exhausted"
    assert new_details.retry_after == timedelta(seconds=30)
    assert new_details.attempts == 5
    assert new_details.retry_safe is False
    assert new_details.provider_payload == {"error": "quota exceeded"}
