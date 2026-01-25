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

"""Throttling policy, errors, and retry logic for provider adapters.

This module provides infrastructure for handling rate limits and quota
exhaustion from LLM providers. Key components:

- ThrottlePolicy: Configuration for retry behavior (delays, attempts, caps)
- ThrottleDetails: Metadata about a throttling event
- ThrottleError: Exception raised when requests are throttled
- jittered_backoff: Compute retry delays with exponential backoff and jitter

Typical usage in an adapter::

    from weakincentives.adapters.throttle import (
        ThrottleError,
        ThrottlePolicy,
        jittered_backoff,
        new_throttle_policy,
        sleep_for,
    )

    policy = new_throttle_policy(max_attempts=3)
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return call_provider(request)
        except ThrottleError as e:
            if attempt == policy.max_attempts or not e.retry_safe:
                raise
            delay = jittered_backoff(policy=policy, attempt=attempt, retry_after=e.retry_after)
            sleep_for(delay)
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Literal

from ..clock import SYSTEM_CLOCK, Sleeper
from ..dataclasses import FrozenDataclass
from .core import PromptEvaluationError, PromptEvaluationPhase

ThrottleKind = Literal["rate_limit", "quota_exhausted", "timeout", "unknown"]
"""Classification of throttling scenarios from LLM providers.

- "rate_limit": Too many requests in a time window. Usually temporary.
- "quota_exhausted": Usage limit reached (daily/monthly caps). May require waiting longer.
- "timeout": Request took too long to complete. May indicate overload.
- "unknown": Throttle type could not be determined from provider response.
"""

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BASE_DELAY = timedelta(milliseconds=500)
_DEFAULT_MAX_DELAY = timedelta(seconds=8)
_DEFAULT_MAX_TOTAL_DELAY = timedelta(seconds=30)


@FrozenDataclass()
class ThrottlePolicy:
    """Configuration for throttle retry handling with exponential backoff.

    Controls how retries are performed when a provider throttles requests.
    Use `new_throttle_policy()` to construct instances with validation.

    Attributes:
        max_attempts: Maximum number of retry attempts before giving up.
            Defaults to 5.
        base_delay: Initial delay between retries. Also serves as the minimum
            delay floor. Defaults to 500ms.
        max_delay: Upper bound on individual retry delays (before jitter).
            Exponential backoff is capped at this value. Defaults to 8s.
        max_total_delay: Maximum cumulative time spent waiting across all
            retries. Prevents unbounded retry loops. Defaults to 30s.

    Example:
        >>> policy = new_throttle_policy(max_attempts=3, base_delay=timedelta(seconds=1))
        >>> policy.max_attempts
        3
    """

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
    """Create a validated throttle policy for retry handling.

    Factory function that constructs a ThrottlePolicy with input validation.
    All duration parameters must be positive; max_attempts must be at least 1.

    Args:
        max_attempts: Maximum retry attempts. Must be >= 1.
        base_delay: Initial/minimum delay between retries. Must be positive.
        max_delay: Cap on individual retry delays. Must be positive.
        max_total_delay: Cap on cumulative retry wait time. Must be positive.

    Returns:
        A validated ThrottlePolicy instance.

    Raises:
        ValueError: If any parameter violates its constraints.

    Example:
        >>> policy = new_throttle_policy(
        ...     max_attempts=3,
        ...     base_delay=timedelta(seconds=1),
        ...     max_delay=timedelta(seconds=10),
        ... )
    """
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
    """Metadata describing a throttling event from a provider.

    Captures information about why a request was throttled and whether
    it can be safely retried. Attached to ThrottleError instances and
    used by retry logic to determine backoff behavior.

    Attributes:
        kind: Classification of the throttle scenario. One of "rate_limit"
            (too many requests), "quota_exhausted" (usage limit reached),
            "timeout" (request took too long), or "unknown".
        retry_after: Provider-suggested wait time before retrying, if provided.
            When set, backoff logic respects this as a minimum delay.
        attempts: Number of attempts made so far, including the current one.
            Starts at 1 for the initial request.
        retry_safe: Whether the request is safe to retry. False for requests
            that may have had side effects or are not idempotent.
        provider_payload: Raw error payload from the provider for debugging.
            May contain provider-specific error codes or messages.
    """

    kind: ThrottleKind
    retry_after: timedelta | None = None
    attempts: int = 1
    retry_safe: bool = True
    provider_payload: dict[str, Any] | None = None


class ThrottleError(PromptEvaluationError):
    """Raised when a provider throttles a request due to rate limits or quotas.

    This exception signals that a request was rejected by the provider and may
    be retryable. The `details` attribute contains throttle metadata including
    the kind of throttle, suggested retry delay, and whether retry is safe.

    Attributes:
        details: ThrottleDetails with full throttle metadata.
        kind: Shortcut to details.kind - the throttle classification.
        retry_after: Shortcut to details.retry_after - suggested wait time.
        attempts: Shortcut to details.attempts - number of attempts made.
        retry_safe: Shortcut to details.retry_safe - whether retry is safe.

    Example:
        >>> try:
        ...     result = adapter.evaluate(prompt)
        ... except ThrottleError as e:
        ...     if e.retry_safe and e.attempts < 3:
        ...         delay = e.retry_after or timedelta(seconds=1)
        ...         sleep_for(delay)
        ...         # retry...
    """

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
        """The throttle classification (rate_limit, quota_exhausted, etc.)."""
        return self.details.kind

    @property
    def retry_after(self) -> timedelta | None:
        """Provider-suggested wait time before retrying, or None if not specified."""
        return self.details.retry_after

    @property
    def attempts(self) -> int:
        """Number of attempts made so far, including the one that raised this error."""
        return self.details.attempts

    @property
    def retry_safe(self) -> bool:
        """Whether this request can be safely retried without side effects."""
        return self.details.retry_safe


def throttle_details(
    *,
    kind: ThrottleKind,
    retry_after: timedelta | None = None,
    attempts: int = 1,
    retry_safe: bool = True,
    provider_payload: dict[str, Any] | None = None,
) -> ThrottleDetails:
    """Create a ThrottleDetails instance with the given parameters.

    Convenience factory for constructing throttle metadata. Typically used
    by adapter implementations when translating provider errors.

    Args:
        kind: The throttle classification (rate_limit, quota_exhausted, etc.).
        retry_after: Provider-suggested delay before retrying.
        attempts: Number of attempts made, including current. Defaults to 1.
        retry_safe: Whether the request can be safely retried. Defaults to True.
        provider_payload: Raw provider error data for debugging.

    Returns:
        A ThrottleDetails instance with the specified values.

    Example:
        >>> details = throttle_details(
        ...     kind="rate_limit",
        ...     retry_after=timedelta(seconds=5),
        ... )
    """
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
    """Create updated ThrottleDetails from an existing ThrottleError.

    Extracts throttle metadata from an error and returns a new ThrottleDetails
    with updated attempt count and retry safety. Used by retry loops to track
    cumulative retry state across multiple attempts.

    Args:
        error: The ThrottleError to extract details from.
        attempts: Updated attempt count to record.
        retry_safe: Updated retry safety flag.

    Returns:
        A new ThrottleDetails preserving kind, retry_after, and provider_payload
        from the error, but with the new attempts and retry_safe values.

    Example:
        >>> try:
        ...     result = adapter.evaluate(prompt)
        ... except ThrottleError as e:
        ...     updated = details_from_error(e, attempts=2, retry_safe=True)
    """
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
) -> timedelta:
    """Calculate a jittered exponential backoff delay for retry timing.

    Implements "full jitter" backoff: computes an exponential delay based on
    attempt number, caps it at max_delay, then applies uniform random jitter
    between 0 and the capped value. The result is floored at base_delay and
    respects any provider-specified retry_after minimum.

    The algorithm:
    1. Compute exponential delay: base_delay * 2^(attempt-1)
    2. Cap at max_delay
    3. Take the larger of capped delay or retry_after
    4. Apply uniform jitter in [0, base]
    5. Floor at base_delay
    6. Respect retry_after as absolute minimum

    Args:
        policy: ThrottlePolicy defining delay bounds.
        attempt: Current attempt number (1-indexed).
        retry_after: Provider-suggested minimum delay, or None.

    Returns:
        The computed delay to wait before the next retry attempt.

    Example:
        >>> policy = new_throttle_policy(base_delay=timedelta(seconds=1))
        >>> delay = jittered_backoff(policy=policy, attempt=2, retry_after=None)
        >>> timedelta(seconds=1) <= delay <= timedelta(seconds=2)
        True
    """
    capped = min(policy.max_delay, policy.base_delay * 2 ** max(attempt - 1, 0))
    base = max(capped, retry_after or timedelta(0))
    if base <= timedelta(0):
        return policy.base_delay

    jitter_seconds = random.uniform(0, base.total_seconds())  # nosec B311
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
