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

"""Built-in trajectory observers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .types import Assessment, ObserverContext

if TYPE_CHECKING:
    from ...runtime.session.protocols import SessionProtocol

__all__ = ["DeadlineObserver"]

# Time constants for duration formatting
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


@dataclass(frozen=True, slots=True)
class DeadlineObserver:
    """Report remaining time until deadline.

    This observer produces assessments about the time remaining before the
    deadline expires. It can be used to help agents prioritize work when
    time is limited.

    Attributes:
        warning_threshold_seconds: When remaining time falls below this threshold,
            the assessment severity is elevated to "warning" and suggestions are
            provided to help the agent wrap up.
    """

    warning_threshold_seconds: float = 120  # 2 minutes

    @property
    def name(self) -> str:
        """Return the observer name."""
        return "Deadline"

    def should_run(  # noqa: PLR6301
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> bool:
        """Return True if a deadline is configured."""
        return context.deadline is not None

    def observe(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> Assessment:
        """Produce an assessment about the remaining time."""
        if context.deadline is None:
            # Defensive - should_run prevents this path
            return Assessment(
                observer_name=self.name,
                summary="No deadline configured.",
                severity="info",
            )

        remaining = context.deadline.remaining().total_seconds()

        if remaining <= 0:
            return Assessment(
                observer_name=self.name,
                summary="You have reached the time deadline.",
                suggestions=("Wrap up immediately.",),
                severity="warning",
            )

        summary = f"You have {self._format_duration(remaining)} remaining."
        suggestions: tuple[str, ...] = ()
        severity: Literal["info", "caution", "warning"] = "info"

        if remaining <= self.warning_threshold_seconds:
            severity = "warning"
            suggestions = (
                "Prioritize completing critical remaining work.",
                "Consider summarizing progress and remaining tasks.",
            )

        return Assessment(
            observer_name=self.name,
            summary=summary,
            suggestions=suggestions,
            severity=severity,
        )

    def _format_duration(self, seconds: float) -> str:  # noqa: PLR6301
        """Format a duration in seconds as a human-readable string."""
        if seconds < _SECONDS_PER_MINUTE:
            return f"{int(seconds)} seconds"
        if seconds < _SECONDS_PER_HOUR:
            minutes = int(seconds / _SECONDS_PER_MINUTE)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        return f"{seconds / _SECONDS_PER_HOUR:.1f} hours"
