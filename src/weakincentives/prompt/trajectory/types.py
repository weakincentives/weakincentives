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

# pyright: reportImportCycles=false

"""Core types for trajectory observation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ...dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from ...deadlines import Deadline
    from ...filesystem import Filesystem
    from ...runtime.events import ToolInvoked
    from ...runtime.session.protocols import SessionProtocol
    from .._prompt_resources import PromptResources
    from ..protocols import PromptProtocol


__all__ = [
    "Assessment",
    "Observation",
    "ObserverConfig",
    "ObserverContext",
    "ObserverTrigger",
    "TrajectoryObserver",
]


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

    Assessments are produced by trajectory observers and injected as additional
    context into the agent's conversation. They provide feedback on agent progress
    without blocking execution.
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
        from ...filesystem import Filesystem

        return self.resources.get_optional(Filesystem)

    @property
    def last_assessment(self) -> Assessment | None:
        """Most recent assessment, if any."""
        return self.session[Assessment].latest()

    @property
    def tool_call_count(self) -> int:
        """Total tool calls in session."""
        from ...runtime.events import ToolInvoked

        return len(self.session[ToolInvoked].all())

    def tool_calls_since_last_assessment(self) -> int:
        """Number of tool calls since last assessment."""
        last = self.last_assessment
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Retrieve the N most recent tool invocations."""
        from ...runtime.events import ToolInvoked

        records = self.session[ToolInvoked].all()
        return records[-n:] if len(records) >= n else records


@FrozenDataclass()
class ObserverTrigger:
    """Conditions that trigger observer execution.

    Triggers are OR'd together: if either condition is met, the observer runs.
    """

    every_n_calls: int | None = None
    every_n_seconds: float | None = None


class TrajectoryObserver(Protocol):
    """Programmatic assessment of agent trajectory.

    Trajectory observers analyze patterns over time and produce feedback
    that is injected into the agent's context. They provide soft course-correction
    without hard intervention.
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
        """Determine if observer should produce an assessment."""
        ...

    def observe(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> Assessment:
        """Analyze trajectory and produce feedback."""
        ...


@FrozenDataclass()
class ObserverConfig:
    """Configuration for a trajectory observer."""

    observer: TrajectoryObserver
    trigger: ObserverTrigger
