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
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from ..dataclasses import FrozenDataclass
from .core import PromptEvaluationError, PromptEvaluationPhase

ThrottleKind = Literal["rate_limit", "quota_exhausted", "timeout", "unknown"]
"""Classification for throttling scenarios."""


class ClockProvider(Protocol):
    """Protocol for obtaining the current time."""

    def __call__(self) -> datetime:
        """Return the current UTC timestamp."""
        ...


class SleeperProvider(Protocol):
    """Protocol for sleeping for a duration."""

    def __call__(self, delay: timedelta) -> None:
        """Sleep for the specified duration."""
        ...


class JitterProvider(Protocol):
    """Protocol for generating jitter values."""

    def __call__(self, low: float, high: float) -> float:
        """Return a random float in [low, high]."""
        ...


def _default_clock() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _default_sleeper(delay: timedelta) -> None:
    """Sleep for the specified duration using time.sleep."""
    time.sleep(delay.total_seconds())


def _default_jitter(low: float, high: float) -> float:
    """Return a random float in [low, high] using random.uniform."""
    return random.uniform(low, high)  # nosec B311


_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BASE_DELAY = timedelta(milliseconds=500)
_DEFAULT_MAX_DELAY = timedelta(seconds=8)
_DEFAULT_MAX_TOTAL_DELAY = timedelta(seconds=30)


@FrozenDataclass()
class ThrottleProviders:
    """Injectable providers for throttle operations.

    These providers enable deterministic testing by allowing clock, sleep,
    and jitter functions to be replaced with controlled implementations.
    """

    clock: ClockProvider = _default_clock
    sleeper: SleeperProvider = _default_sleeper
    jitter: JitterProvider = _default_jitter


def new_throttle_providers(
    *,
    clock: ClockProvider | None = None,
    sleeper: SleeperProvider | None = None,
    jitter: JitterProvider | None = None,
) -> ThrottleProviders:
    """Return a throttle providers instance with optional overrides.

    Arguments default to the standard implementations when None.
    """
    return ThrottleProviders(
        clock=clock or _default_clock,
        sleeper=sleeper or _default_sleeper,
        jitter=jitter or _default_jitter,
    )


# Singleton for default providers
DEFAULT_THROTTLE_PROVIDERS = ThrottleProviders()


@FrozenDataclass()
class ThrottlePolicy:
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


@FrozenDataclass()
class ThrottleDetails:
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


def sleep_for(
    delay: timedelta,
    *,
    providers: ThrottleProviders | None = None,
) -> None:
    """Sleep for the specified duration.

    Args:
        delay: The duration to sleep.
        providers: Optional providers for testing. Uses defaults if None.
    """
    sleeper = (providers or DEFAULT_THROTTLE_PROVIDERS).sleeper
    sleeper(delay)


def jittered_backoff(
    *,
    policy: ThrottlePolicy,
    attempt: int,
    retry_after: timedelta | None,
    providers: ThrottleProviders | None = None,
) -> timedelta:
    """Calculate a jittered backoff delay based on policy and attempt number.

    Args:
        policy: The throttle policy configuration.
        attempt: The current attempt number (1-indexed).
        retry_after: Optional server-provided retry-after duration.
        providers: Optional providers for testing. Uses defaults if None.

    Returns:
        The computed delay duration.
    """
    jitter_fn = (providers or DEFAULT_THROTTLE_PROVIDERS).jitter

    capped = min(policy.max_delay, policy.base_delay * 2 ** max(attempt - 1, 0))
    base = max(capped, retry_after or timedelta(0))
    if base <= timedelta(0):
        return policy.base_delay

    jitter_seconds = jitter_fn(0, base.total_seconds())
    delay = timedelta(seconds=jitter_seconds)
    delay = max(delay, policy.base_delay)
    if retry_after is not None and delay < retry_after:
        return retry_after
    return delay


__all__ = [
    "DEFAULT_THROTTLE_PROVIDERS",
    "ClockProvider",
    "JitterProvider",
    "SleeperProvider",
    "ThrottleDetails",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "ThrottleProviders",
    "details_from_error",
    "jittered_backoff",
    "new_throttle_policy",
    "new_throttle_providers",
    "sleep_for",
    "throttle_details",
]
