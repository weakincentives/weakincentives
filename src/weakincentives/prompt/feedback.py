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

"""Feedback providers for ongoing agent progress assessment.

This module implements feedback providers that deliver contextual guidance to
agents during unattended execution. Unlike tool policies that gate individual
calls, feedback providers analyze patterns over time and inject feedback into
the agent's context for soft course-correction.

Key concepts:

- **FeedbackProvider**: Protocol for producing feedback based on session state.
- **FeedbackTrigger**: Conditions that determine when a provider runs.
- **Feedback**: Structured feedback with summary, observations, and suggestions.
- **FeedbackContext**: Context provided to providers for state access.

Example:
    >>> from weakincentives.prompt import (
    ...     DeadlineFeedback,
    ...     FeedbackProviderConfig,
    ...     FeedbackTrigger,
    ...     PromptTemplate,
    ... )
    >>>
    >>> template = PromptTemplate[OutputType](
    ...     ns="my-agent",
    ...     key="main",
    ...     feedback_providers=(
    ...         FeedbackProviderConfig(
    ...             provider=DeadlineFeedback(),
    ...             trigger=FeedbackTrigger(every_n_seconds=30),
    ...         ),
    ...     ),
    ... )
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


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@FrozenDataclass()
class Observation:
    """A single observation about the agent's behavior or trajectory.

    Observations provide structured evidence for feedback. Each observation
    has a category (e.g., "Pattern", "Resource") and a description of what
    was observed.

    Attributes:
        category: Classification of the observation (e.g., "Loop", "Drift").
        description: Human-readable description of what was observed.
        evidence: Optional supporting evidence (e.g., file path, tool name).
    """

    category: str
    description: str
    evidence: str | None = None


@FrozenDataclass()
class Feedback:
    """Structured feedback produced by a feedback provider.

    Feedback is delivered immediately to the agent via the adapter's hook
    mechanism and stored in the session's Feedback slice for history tracking.

    Attributes:
        provider_name: Name of the provider that produced this feedback.
        summary: Brief description of the feedback (shown prominently).
        observations: Supporting observations with categorized evidence.
        suggestions: Actionable suggestions for the agent.
        severity: Urgency level - "info", "caution", or "warning".
        timestamp: When the feedback was produced.
        call_index: Tool call count when feedback was produced (set by runner).
    """

    provider_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=_utcnow)
    call_index: int = 0

    def render(self) -> str:
        """Render feedback as text for context injection.

        Returns:
            Formatted text suitable for injection into agent context.
        """
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


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class FeedbackContext:
    """Context provided to feedback providers.

    Provides access to session state, prompt resources, and helper methods
    for analyzing agent trajectory. Mirrors the ToolContext interface for
    consistency.

    Attributes:
        session: The current session for state access.
        prompt: The prompt being executed.
        deadline: Optional deadline for time-aware feedback.
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
        """Return the filesystem resource if available, otherwise None."""
        from ..filesystem import Filesystem

        return self.resources.get_optional(Filesystem)

    @property
    def last_feedback(self) -> Feedback | None:
        """Return the most recent feedback, or None if no feedback exists."""
        return self.session[Feedback].latest()

    @property
    def tool_call_count(self) -> int:
        """Return the total number of tool calls in the session."""
        from ..runtime.events import ToolInvoked

        return len(self.session[ToolInvoked].all())

    def tool_calls_since_last_feedback(self) -> int:
        """Return the number of tool calls since the last feedback.

        If no feedback has been produced yet, returns the total tool call count.
        """
        last = self.last_feedback
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Return the N most recent tool invocations.

        Args:
            n: Maximum number of tool calls to return.

        Returns:
            Sequence of ToolInvoked events, oldest first. Empty if n=0.
        """
        from ..runtime.events import ToolInvoked

        if n <= 0:
            return ()
        records = self.session[ToolInvoked].all()
        return records[-n:] if len(records) >= n else records


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class FeedbackProvider(Protocol):
    """Protocol for feedback providers.

    Feedback providers analyze session state and produce contextual feedback
    for agents. They run after tool execution when trigger conditions are met.

    Implement this protocol to create custom feedback providers:

        @dataclass(frozen=True)
        class MyProvider:
            @property
            def name(self) -> str:
                return "MyProvider"

            def should_run(
                self,
                session: SessionProtocol,
                *,
                context: FeedbackContext,
            ) -> bool:
                return True  # Additional filtering beyond trigger

            def provide(
                self,
                session: SessionProtocol,
                *,
                context: FeedbackContext,
            ) -> Feedback:
                return Feedback(
                    provider_name=self.name,
                    summary="Status update",
                )
    """

    @property
    def name(self) -> str:
        """Return a unique identifier for this provider."""
        ...

    def should_run(
        self,
        session: SessionProtocol,
        *,
        context: FeedbackContext,
    ) -> bool:
        """Determine if the provider should produce feedback.

        Called after trigger conditions are met. Use this for additional
        filtering (e.g., only run if a deadline is configured).

        Args:
            session: The current session.
            context: Feedback context with state access.

        Returns:
            True if the provider should run, False to skip.
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

        Args:
            session: The current session.
            context: Feedback context with state access.

        Returns:
            Feedback to be rendered and injected into agent context.
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@FrozenDataclass()
class FeedbackTrigger:
    """Conditions that determine when a feedback provider runs.

    Trigger conditions are OR'd together: if any condition is met, the
    provider is evaluated. If no conditions are specified, the trigger
    never fires.

    Attributes:
        every_n_calls: Run after this many tool calls since last feedback.
        every_n_seconds: Run after this many seconds since last feedback.

    Example:
        >>> # Run every 10 tool calls OR every 60 seconds
        >>> trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=60)
    """

    every_n_calls: int | None = None
    every_n_seconds: float | None = None


@FrozenDataclass()
class FeedbackProviderConfig:
    """Configuration pairing a feedback provider with its trigger.

    Attributes:
        provider: The feedback provider instance.
        trigger: Conditions that determine when the provider runs.

    Example:
        >>> config = FeedbackProviderConfig(
        ...     provider=DeadlineFeedback(),
        ...     trigger=FeedbackTrigger(every_n_seconds=30),
        ... )
    """

    provider: FeedbackProvider
    trigger: FeedbackTrigger


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _should_trigger(trigger: FeedbackTrigger, context: FeedbackContext) -> bool:
    """Check if trigger conditions are met.

    Args:
        trigger: The trigger configuration to check.
        context: Feedback context with state for evaluation.

    Returns:
        True if any trigger condition is met, False otherwise.
    """
    # Check call count condition
    if (
        trigger.every_n_calls is not None
        and context.tool_calls_since_last_feedback() >= trigger.every_n_calls
    ):
        return True

    # Check time-based condition
    if trigger.every_n_seconds is not None:
        last = context.last_feedback
        if last is not None:
            elapsed = (_utcnow() - last.timestamp).total_seconds()
            if elapsed >= trigger.every_n_seconds:
                return True
        else:
            # No previous feedback - trigger immediately
            return True

    return False


def run_feedback_providers(
    *,
    providers: Sequence[FeedbackProviderConfig],
    context: FeedbackContext,
    session: SessionProtocol,
) -> str | None:
    """Run feedback providers and return rendered feedback if triggered.

    Iterates through configured providers in order. For each provider:
    1. Check if trigger conditions are met
    2. Check if provider.should_run() returns True
    3. Call provider.provide() to get feedback
    4. Store feedback in session and return rendered text

    First matching provider wins; subsequent providers are not evaluated.

    Args:
        providers: Sequence of provider configurations to evaluate.
        context: Feedback context with session state.
        session: The current session for storing feedback.

    Returns:
        Rendered feedback text if a provider triggered, None otherwise.
    """
    for config in providers:
        if _should_trigger(config.trigger, context) and config.provider.should_run(
            session, context=context
        ):
            feedback = config.provider.provide(session, context=context)
            # Update call_index to current tool call count
            feedback = replace(feedback, call_index=context.tool_call_count)
            # Store in session for history and trigger calculations
            session[Feedback].append(feedback)
            return feedback.render()

    return None


def collect_feedback(
    *,
    prompt: PromptProtocol[Any],
    session: SessionProtocol,
    deadline: Deadline | None = None,
) -> str | None:
    """Collect feedback from providers configured on the prompt.

    This is the primary entry point for running feedback providers. It creates
    the FeedbackContext and invokes run_feedback_providers.

    Args:
        prompt: The prompt with feedback_providers configured.
        session: The current session for state access and feedback storage.
        deadline: Optional deadline for time-aware feedback providers.

    Returns:
        Rendered feedback text if a provider triggered, None otherwise.

    Example:
        >>> feedback_text = collect_feedback(
        ...     prompt=prompt,
        ...     session=session,
        ...     deadline=deadline,
        ... )
        >>> if feedback_text:
        ...     # Inject feedback into agent context
        ...     pass
    """
    context = FeedbackContext(
        session=session,
        prompt=prompt,
        deadline=deadline,
    )
    return run_feedback_providers(
        providers=prompt.feedback_providers,
        context=context,
        session=session,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Feedback",
    "FeedbackContext",
    "FeedbackProvider",
    "FeedbackProviderConfig",
    "FeedbackTrigger",
    "Observation",
    "collect_feedback",
    "run_feedback_providers",
]
