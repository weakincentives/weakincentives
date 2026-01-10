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

"""Built-in feedback providers for common progress feedback patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .observer import Feedback, FeedbackContext

if TYPE_CHECKING:
    from ..runtime.session.protocols import SessionProtocol

# Duration constants for _format_duration
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < _SECONDS_PER_MINUTE:
        return f"{int(seconds)} seconds"
    if seconds < _SECONDS_PER_HOUR:
        minutes = int(seconds / _SECONDS_PER_MINUTE)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds / _SECONDS_PER_HOUR:.1f} hours"


@dataclass(frozen=True)
class DeadlineFeedback:
    """Report remaining time until deadline.

    This feedback provider produces feedback about time remaining until the
    deadline, helping agents manage their time during unattended execution.

    Default trigger: every 30 seconds (configured separately via FeedbackTrigger).

    Example:
        >>> from weakincentives.prompt import (
        ...     DeadlineFeedback,
        ...     FeedbackProviderConfig,
        ...     FeedbackTrigger,
        ... )
        >>>
        >>> config = FeedbackProviderConfig(
        ...     provider=DeadlineFeedback(),
        ...     trigger=FeedbackTrigger(every_n_seconds=30),
        ... )
    """

    warning_threshold_seconds: float = 120  # 2 minutes

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Deadline"

    def should_run(  # noqa: PLR6301 - required by FeedbackProvider protocol
        self,
        session: SessionProtocol,
        *,
        context: FeedbackContext,
    ) -> bool:
        """Only run if a deadline is set."""
        return context.deadline is not None

    def provide(
        self,
        session: SessionProtocol,
        *,
        context: FeedbackContext,
    ) -> Feedback:
        """Analyze time remaining and produce appropriate feedback."""
        if context.deadline is None:
            # Shouldn't happen if should_run is called first, but defensive
            return Feedback(
                provider_name=self.name,
                summary="No deadline configured.",
                severity="info",
            )

        remaining = context.deadline.remaining().total_seconds()

        if remaining <= 0:
            return Feedback(
                provider_name=self.name,
                summary="You have reached the time deadline.",
                suggestions=("Wrap up immediately.",),
                severity="warning",
            )

        summary = f"You have {_format_duration(remaining)} remaining."
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


__all__ = [
    "DeadlineFeedback",
]
