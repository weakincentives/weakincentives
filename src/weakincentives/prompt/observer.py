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

"""Trajectory observer types for ongoing assessment of agent progress.

Trajectory observers provide ongoing assessment of agent progress during
unattended execution. Unlike tool policies that gate individual calls,
observers analyze patterns over time and inject feedback into the agent's
context. This enables soft course-correction without hard intervention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from ..deadlines import Deadline
    from ..filesystem import Filesystem
    from ..runtime.events import ToolInvoked
    from ..runtime.session.protocols import SessionProtocol
    from ._prompt_resources import PromptResources
    from .protocols import PromptProtocol


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


@FrozenDataclass()
class Observation:
    """Single observation about the trajectory."""

    category: str
    description: str
    evidence: str | None = None


@FrozenDataclass()
class Assessment:
    """Structured output from trajectory observation.

    Assessments are produced by trajectory observers and delivered immediately
    to the agent via hook response. They are also stored in the session's
    Assessment slice for trigger calculations and debugging.
    """

    observer_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=_utcnow)
    call_index: int = 0

    def render(self) -> str:
        """Render as concise text for context injection."""
        lines = [
            f"[Trajectory Assessment - {self.observer_name}]",
            "",
            self.summary,
        ]

        if self.observations:
            lines.append("")
            lines.extend(
                f"• {obs.category}: {obs.description}" for obs in self.observations
            )

        if self.suggestions:
            lines.append("")
            lines.extend(f"→ {suggestion}" for suggestion in self.suggestions)

        return "\n".join(lines)


@FrozenDataclass()
class RecordAssessment:
    """Event dispatched when an observer produces an assessment."""

    assessment: Assessment


@dataclass(slots=True, frozen=True)
class ObserverContext:
    """Context provided to observers during assessment.

    Provides access to session state and prompt resources, mirroring the
    ToolContext interface for consistency.
    """

    session: SessionProtocol
    prompt: PromptProtocol[Any]
    deadline: Deadline | None = None

    @property
    def resources(self) -> PromptResources:
        """Access resources from the prompt's resource context."""
        return self.prompt.resources

    @property
    def filesystem(self) -> Filesystem | None:
        """Return the filesystem resource, if available."""
        # Import here to avoid circular import at module load time
        from ..filesystem import Filesystem

        return self.resources.get_optional(Filesystem)

    @property
    def last_assessment(self) -> Assessment | None:
        """Most recent assessment, if any."""
        return self.session[Assessment].latest()

    @property
    def tool_call_count(self) -> int:
        """Total tool calls in session."""
        from ..runtime.events import ToolInvoked

        return len(self.session[ToolInvoked].all())

    def tool_calls_since_last_assessment(self) -> int:
        """Number of tool calls since last assessment."""
        last = self.last_assessment
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Retrieve the N most recent tool invocations."""
        from ..runtime.events import ToolInvoked

        records = self.session[ToolInvoked].all()
        return records[-n:] if len(records) >= n else records


class TrajectoryObserver(Protocol):
    """Protocol for programmatic assessment of agent trajectory.

    Observers analyze patterns over time and produce feedback for course
    correction. They run after tool execution and their assessments are
    injected into the agent's context.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this observer."""
        ...

    def should_run(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> bool:
        """Determine if observer should produce an assessment.

        Called after trigger conditions are met to check if the observer
        should actually run. Use this for additional filtering beyond
        trigger conditions.
        """
        ...

    def observe(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> Assessment:
        """Analyze trajectory and produce feedback.

        Called when trigger conditions are met and should_run returns True.
        Returns an Assessment that will be rendered and injected into the
        agent's context.
        """
        ...


@FrozenDataclass()
class ObserverTrigger:
    """Conditions that trigger observer execution.

    Triggers are OR'd together: if either condition is met, the observer runs.
    """

    every_n_calls: int | None = None
    every_n_seconds: float | None = None


@FrozenDataclass()
class ObserverConfig:
    """Configuration for a trajectory observer."""

    observer: TrajectoryObserver
    trigger: ObserverTrigger


def _should_trigger(trigger: ObserverTrigger, context: ObserverContext) -> bool:
    """Check if any trigger condition is met."""
    if (
        trigger.every_n_calls is not None
        and context.tool_calls_since_last_assessment() >= trigger.every_n_calls
    ):
        return True

    if trigger.every_n_seconds is not None:
        last = context.last_assessment
        if last is not None:
            elapsed = (_utcnow() - last.timestamp).total_seconds()
            if elapsed >= trigger.every_n_seconds:
                return True
        else:
            # No previous assessment - trigger
            return True

    return False


def run_observers(
    *,
    observers: Sequence[ObserverConfig],
    context: ObserverContext,
    session: SessionProtocol,
) -> str | None:
    """Run observers and return rendered assessment if triggered.

    Iterates through configured observers, checks trigger conditions, and
    runs the first matching observer. The assessment is recorded in the
    session and returned as rendered text for context injection.

    Returns None if no observer triggered or produced an assessment.
    """
    for config in observers:
        if _should_trigger(config.trigger, context) and config.observer.should_run(
            session, context=context
        ):
            assessment = config.observer.observe(session, context=context)
            assessment = replace(assessment, call_index=context.tool_call_count)
            # Store assessment in session slice for history and trigger calculations
            session[Assessment].append(assessment)
            return assessment.render()

    return None


__all__ = [
    "Assessment",
    "Observation",
    "ObserverConfig",
    "ObserverContext",
    "ObserverTrigger",
    "RecordAssessment",
    "TrajectoryObserver",
    "run_observers",
]
