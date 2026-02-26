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

"""Throttling policy, errors, and retry logic for provider adapters."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

from ..clock import SYSTEM_CLOCK, Sleeper
from ..dataclasses import FrozenDataclassMixin
from .core import PromptEvaluationError, PromptEvaluationPhase

ThrottleKind = Literal["rate_limit", "quota_exhausted", "timeout", "unknown"]
"""Classification for throttling scenarios."""

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BASE_DELAY = timedelta(milliseconds=500)
_DEFAULT_MAX_DELAY = timedelta(seconds=8)
_DEFAULT_MAX_TOTAL_DELAY = timedelta(seconds=30)


@dataclass(slots=True, frozen=True)
class ThrottlePolicy(FrozenDataclassMixin):
    """Configuration for throttle retry handling."""

    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    base_delay: timedelta = _DEFAULT_BASE_DELAY
    max_delay: timedelta = _DEFAULT_MAX_DELAY
    max_total_delay: timedelta = _DEFAULT_MAX_TOTAL_DELAY


def new_throttle_policy(
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay: timedelta = _DEFAULT_BASE_DELAY,
    max_delay: timedelta = _DEFAULT_MAX_DELAY,
    max_total_delay: timedelta = _DEFAULT_MAX_TOTAL_DELAY,
) -> ThrottlePolicy:
    """Return a throttle policy instance with validation."""

    if max_attempts < 1:
        msg = "Throttle max_attempts must be at least 1."
        raise ValueError(msg)
    if base_delay <= timedelta(0):
        msg = "Throttle base_delay must be positive."
        raise ValueError(msg)
    if max_delay <= timedelta(0):
        msg = "Throttle max_delay must be positive."
        raise ValueError(msg)
    if max_total_delay <= timedelta(0):
        msg = "Throttle max_total_delay must be positive."
        raise ValueError(msg)
    return ThrottlePolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        max_total_delay=max_total_delay,
    )


@dataclass(slots=True, frozen=True)
class ThrottleDetails(FrozenDataclassMixin):
    """Provider throttle metadata tracked alongside PromptEvaluationError details."""

    kind: ThrottleKind
    retry_after: timedelta | None = None
    attempts: int = 1
    retry_safe: bool = True
    provider_payload: dict[str, Any] | None = None


class ThrottleError(PromptEvaluationError):
    """Raised when a provider throttles a request."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        details: ThrottleDetails,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=details.provider_payload,
        )
        self.details = details

    @property
    def kind(self) -> ThrottleKind:
        return self.details.kind

    @property
    def retry_after(self) -> timedelta | None:
        return self.details.retry_after

    @property
    def attempts(self) -> int:
        return self.details.attempts

    @property
    def retry_safe(self) -> bool:
        return self.details.retry_safe


def throttle_details(
    *,
    kind: ThrottleKind,
    retry_after: timedelta | None = None,
    attempts: int = 1,
    retry_safe: bool = True,
    provider_payload: dict[str, Any] | None = None,
) -> ThrottleDetails:
    """Convenience wrapper for constructing throttle detail payloads."""

    return ThrottleDetails(
        kind=kind,
        retry_after=retry_after,
        attempts=attempts,
        retry_safe=retry_safe,
        provider_payload=provider_payload,
    )


def details_from_error(
    error: ThrottleError, *, attempts: int, retry_safe: bool
) -> ThrottleDetails:
    """Extract throttle details from an error with updated attempt info."""
    return throttle_details(
        kind=error.kind,
        retry_after=error.retry_after,
        attempts=attempts,
        retry_safe=retry_safe,
        provider_payload=error.provider_payload,
    )


def sleep_for(delay: timedelta, *, sleeper: Sleeper = SYSTEM_CLOCK) -> None:
    """Sleep for the specified duration.

    Args:
        delay: Duration to sleep.
        sleeper: Sleep implementation. Defaults to system clock.
            Inject FakeClock for instant advancement in tests.
    """
    sleeper.sleep(delay.total_seconds())


def jittered_backoff(
    *,
    policy: ThrottlePolicy,
    attempt: int,
    retry_after: timedelta | None,
    _uniform: Callable[[float, float], float] = random.uniform,
) -> timedelta:
    """Calculate a jittered backoff delay based on policy and attempt number."""
    capped = min(policy.max_delay, policy.base_delay * 2 ** max(attempt - 1, 0))
    base = max(capped, retry_after or timedelta(0))
    if base <= timedelta(0):
        return policy.base_delay

    jitter_seconds = _uniform(0, base.total_seconds())  # nosec B311
    delay = timedelta(seconds=jitter_seconds)
    delay = max(delay, policy.base_delay)
    if retry_after is not None and delay < retry_after:
        return retry_after
    return delay


__all__ = [
    "ThrottleDetails",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "details_from_error",
    "jittered_backoff",
    "new_throttle_policy",
    "sleep_for",
    "throttle_details",
]
