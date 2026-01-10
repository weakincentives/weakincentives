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

"""Feedback provider types for ongoing agent progress feedback.

Feedback providers deliver ongoing feedback about agent progress during
unattended execution. Unlike tool policies that gate individual calls,
providers analyze patterns over time and inject feedback into the agent's
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
    """Single observation about the agent's trajectory."""

    category: str
    description: str
    evidence: str | None = None


@FrozenDataclass()
class Feedback:
    """Structured feedback from a feedback provider.

    Feedback is produced by feedback providers and delivered immediately
    to the agent via hook response. It is also stored in the session's
    Feedback slice for trigger calculations and debugging.
    """

    provider_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=_utcnow)
    call_index: int = 0

    def render(self) -> str:
        """Render as concise text for context injection."""
        lines = [
            f"[Feedback - {self.provider_name}]",
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
class RecordFeedback:
    """Event dispatched when a provider produces feedback."""

    feedback: Feedback


@dataclass(slots=True, frozen=True)
class FeedbackContext:
    """Context provided to feedback providers.

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
    def last_feedback(self) -> Feedback | None:
        """Most recent feedback, if any."""
        return self.session[Feedback].latest()

    @property
    def tool_call_count(self) -> int:
        """Total tool calls in session."""
        from ..runtime.events import ToolInvoked

        return len(self.session[ToolInvoked].all())

    def tool_calls_since_last_feedback(self) -> int:
        """Number of tool calls since last feedback."""
        last = self.last_feedback
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Retrieve the N most recent tool invocations."""
        from ..runtime.events import ToolInvoked

        records = self.session[ToolInvoked].all()
        return records[-n:] if len(records) >= n else records


class FeedbackProvider(Protocol):
    """Protocol for programmatic feedback about agent progress.

    Providers analyze patterns over time and produce feedback for course
    correction. They run after tool execution and their feedback is
    injected into the agent's context.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this provider."""
        ...

    def should_run(
        self,
        session: SessionProtocol,
        *,
        context: FeedbackContext,
    ) -> bool:
        """Determine if provider should produce feedback.

        Called after trigger conditions are met to check if the provider
        should actually run. Use this for additional filtering beyond
        trigger conditions.
        """
        ...

    def provide(
        self,
        session: SessionProtocol,
        *,
        context: FeedbackContext,
    ) -> Feedback:
        """Analyze trajectory and produce feedback.

        Called when trigger conditions are met and should_run returns True.
        Returns Feedback that will be rendered and injected into the
        agent's context.
        """
        ...


@FrozenDataclass()
class FeedbackTrigger:
    """Conditions that trigger feedback provider execution.

    Triggers are OR'd together: if either condition is met, the provider runs.
    """

    every_n_calls: int | None = None
    every_n_seconds: float | None = None


@FrozenDataclass()
class FeedbackProviderConfig:
    """Configuration for a feedback provider."""

    provider: FeedbackProvider
    trigger: FeedbackTrigger


def _should_trigger(trigger: FeedbackTrigger, context: FeedbackContext) -> bool:
    """Check if any trigger condition is met."""
    if (
        trigger.every_n_calls is not None
        and context.tool_calls_since_last_feedback() >= trigger.every_n_calls
    ):
        return True

    if trigger.every_n_seconds is not None:
        last = context.last_feedback
        if last is not None:
            elapsed = (_utcnow() - last.timestamp).total_seconds()
            if elapsed >= trigger.every_n_seconds:
                return True
        else:
            # No previous feedback - trigger
            return True

    return False


def run_feedback_providers(
    *,
    providers: Sequence[FeedbackProviderConfig],
    context: FeedbackContext,
    session: SessionProtocol,
) -> str | None:
    """Run feedback providers and return rendered feedback if triggered.

    Iterates through configured providers, checks trigger conditions, and
    runs the first matching provider. The feedback is recorded in the
    session and returned as rendered text for context injection.

    Returns None if no provider triggered or produced feedback.
    """
    for config in providers:
        if _should_trigger(config.trigger, context) and config.provider.should_run(
            session, context=context
        ):
            feedback = config.provider.provide(session, context=context)
            feedback = replace(feedback, call_index=context.tool_call_count)
            # Store feedback in session slice for history and trigger calculations
            session[Feedback].append(feedback)
            return feedback.render()

    return None


__all__ = [
    "Feedback",
    "FeedbackContext",
    "FeedbackProvider",
    "FeedbackProviderConfig",
    "FeedbackTrigger",
    "Observation",
    "RecordFeedback",
    "run_feedback_providers",
]
