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

"""Built-in feedback providers.

This module contains ready-to-use feedback providers for common scenarios:

- **DeadlineFeedback**: Reports remaining time until deadline, warning as
  time runs low.

Example:
    >>> from weakincentives.prompt import (
    ...     DeadlineFeedback,
    ...     FeedbackProviderConfig,
    ...     FeedbackTrigger,
    ... )
    >>>
    >>> config = FeedbackProviderConfig(
    ...     provider=DeadlineFeedback(warning_threshold_seconds=120),
    ...     trigger=FeedbackTrigger(every_n_seconds=30),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .feedback import Feedback, FeedbackContext

# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _format_duration(seconds: float) -> str:
    """Format a duration as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration (e.g., "45 seconds", "5 minutes", "1.5 hours").
    """
    if seconds < _SECONDS_PER_MINUTE:
        return f"{int(seconds)} seconds"
    if seconds < _SECONDS_PER_HOUR:
        minutes = int(seconds / _SECONDS_PER_MINUTE)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds / _SECONDS_PER_HOUR:.1f} hours"


# ---------------------------------------------------------------------------
# DeadlineFeedback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeadlineFeedback:
    """Feedback provider that reports remaining time until deadline.

    This provider helps agents manage their time during unattended execution
    by periodically reporting how much time remains. When time runs low
    (below warning_threshold_seconds), the feedback severity increases and
    includes suggestions to wrap up.

    Attributes:
        warning_threshold_seconds: Remaining seconds at which severity becomes
            "warning" and suggestions are included. Defaults to 120 (2 minutes).

    Example:
        >>> config = FeedbackProviderConfig(
        ...     provider=DeadlineFeedback(warning_threshold_seconds=60),
        ...     trigger=FeedbackTrigger(every_n_seconds=30),
        ... )

    Output examples:

        When plenty of time remains::

            [Feedback - Deadline]

            The work so far took 12 minutes. You have 8 minutes remaining
            to complete the task.

        When time is running low::

            [Feedback - Deadline]

            The work so far took 18 minutes. You have 90 seconds remaining
            to complete the task.

            → Prioritize completing critical remaining work.
            → Consider summarizing progress and remaining tasks.

        When deadline has passed::

            [Feedback - Deadline]

            The work took 20 minutes. You have reached the time deadline.

            → Wrap up immediately.
    """

    warning_threshold_seconds: float = 120.0  # 2 minutes

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Deadline"

    # PLR6301: Method doesn't use self, but must be instance method per Protocol
    def should_run(self, *, context: FeedbackContext) -> bool:
        """Return True only if a deadline is configured."""
        return context.deadline is not None

    def provide(self, *, context: FeedbackContext) -> Feedback:
        """Produce feedback about elapsed and remaining time.

        Args:
            context: Feedback context with deadline access.

        Returns:
            Feedback with time elapsed, time remaining, and appropriate severity.

        Raises:
            ValueError: If called without a deadline (should_run prevents this).
        """
        if context.deadline is None:
            raise ValueError("DeadlineFeedback.provide() requires a deadline")

        remaining = context.deadline.remaining().total_seconds()
        elapsed = context.deadline.elapsed().total_seconds()
        elapsed_str = _format_duration(elapsed)

        # Deadline has passed
        if remaining <= 0:
            return Feedback(
                provider_name=self.name,
                summary=(
                    f"The work took {elapsed_str}. You have reached the time deadline."
                ),
                suggestions=("Wrap up immediately.",),
                severity="warning",
            )

        # Build feedback based on remaining time
        remaining_str = _format_duration(remaining)
        summary = (
            f"The work so far took {elapsed_str}. "
            f"You have {remaining_str} remaining to complete the task."
        )
        suggestions: tuple[str, ...] = ()
        severity: Literal["info", "caution", "warning"] = "info"

        if remaining <= self.warning_threshold_seconds:
            severity = "warning"
            suggestions = (
                "Prioritize completing critical remaining work.",
                "Consider summarizing progress and remaining tasks.",
            )

        return Feedback(
            provider_name=self.name,
            summary=summary,
            suggestions=suggestions,
            severity=severity,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DeadlineFeedback",
]
