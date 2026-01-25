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
    was observed. Observations are typically created within a FeedbackProvider
    and included in the Feedback.observations tuple.

    Attributes:
        category: Classification of the observation (e.g., "Loop", "Drift",
            "Resource", "Pattern"). Used as a prefix when rendering feedback.
        description: Human-readable description of what was observed.
        evidence: Optional supporting evidence (e.g., file path, tool name,
            or other context). Not rendered directly but useful for debugging.

    Example:
        >>> obs = Observation(
        ...     category="Pattern",
        ...     description="Agent has called read_file 5 times on the same path",
        ...     evidence="/path/to/file.py",
        ... )
    """

    category: str
    description: str
    evidence: str | None = None


@FrozenDataclass()
class Feedback:
    """Structured feedback produced by a feedback provider.

    Feedback is delivered immediately to the agent via the adapter's hook
    mechanism and stored in the session's Feedback slice for history tracking.
    Use the ``render()`` method to convert feedback to text for injection
    into the agent's context.

    Attributes:
        provider_name: Name of the provider that produced this feedback.
            Used as a header when rendering (e.g., "[Feedback - DeadlineFeedback]").
        summary: Brief description of the feedback (shown prominently).
            Should be 1-2 sentences explaining the main point.
        observations: Supporting observations with categorized evidence.
            Each observation is rendered as a bullet point.
        suggestions: Actionable suggestions for the agent. Each suggestion
            is rendered with an arrow prefix (e.g., "-> Focus on tests").
        severity: Urgency level - "info" for status updates, "caution" for
            potential issues, "warning" for urgent course corrections.
        timestamp: When the feedback was produced (defaults to current UTC time).
        call_index: Tool call count when feedback was produced. Set automatically
            by ``run_feedback_providers()``; do not set manually.
        prompt_name: Name of the prompt that produced this feedback. Set
            automatically by ``run_feedback_providers()``; do not set manually.

    Example:
        >>> feedback = Feedback(
        ...     provider_name="DeadlineFeedback",
        ...     summary="50% of allocated time has elapsed.",
        ...     observations=(
        ...         Observation("Progress", "3 of 5 tasks completed"),
        ...     ),
        ...     suggestions=("Focus on remaining critical tasks",),
        ...     severity="caution",
        ... )
        >>> print(feedback.render())
        [Feedback - DeadlineFeedback]
        <BLANKLINE>
        50% of allocated time has elapsed.
        <BLANKLINE>
        * Progress: 3 of 5 tasks completed
        <BLANKLINE>
        -> Focus on remaining critical tasks
    """

    provider_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=_utcnow)
    call_index: int = 0
    prompt_name: str = ""

    def render(self) -> str:
        """Render feedback as text for context injection.

        Produces a human-readable text block with the following structure:

        1. Header line: ``[Feedback - {provider_name}]``
        2. Blank line
        3. Summary text
        4. Observations (if any): bullet points with category prefix
        5. Suggestions (if any): arrow-prefixed actionable items

        Returns:
            Formatted text suitable for injection into agent context.
            The output is designed to be clear and actionable for LLM agents.
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
    """Context provided to feedback providers for state access and analysis.

    Provides access to session state, prompt resources, and helper methods
    for analyzing agent trajectory. Mirrors the ToolContext interface for
    consistency across the library.

    Tool call counts and recent tool calls are scoped to the current prompt
    to ensure triggers behave consistently when sessions are reused across
    multiple prompt evaluations.

    Attributes:
        session: The current session for state access. Use this to query
            slices (e.g., ``context.session[ToolInvoked].all()``).
        prompt: The prompt being executed. Provides access to prompt
            configuration and resources.
        deadline: Optional deadline for time-aware feedback providers.
            Providers can use this to calculate remaining time.

    Example:
        >>> # Inside a FeedbackProvider.provide() method:
        >>> def provide(self, *, context: FeedbackContext) -> Feedback:
        ...     call_count = context.tool_call_count
        ...     recent = context.recent_tool_calls(5)
        ...     # Analyze recent tool usage patterns...
        ...     return Feedback(provider_name=self.name, summary="...")
    """

    session: SessionProtocol
    prompt: PromptProtocol[Any]
    deadline: Deadline | None = None

    @property
    def prompt_name(self) -> str:
        """Return the prompt name for filtering events.

        Used internally by feedback providers and runners to scope trigger
        calculations to the current prompt.
        """
        return self.prompt.name or f"{self.prompt.ns}:{self.prompt.key}"

    @property
    def resources(self) -> PromptResources:
        """Access resources from the prompt's resource context.

        Returns the PromptResources instance configured on the prompt,
        allowing feedback providers to access injected dependencies like
        filesystems, clocks, or custom services.

        Returns:
            The prompt's resource container for dependency resolution.

        Example:
            >>> fs = context.resources.get_optional(Filesystem)
            >>> if fs:
            ...     # Use filesystem for analysis
            ...     pass
        """
        return self.prompt.resources

    @property
    def filesystem(self) -> Filesystem | None:
        """Return the filesystem resource if available, otherwise None.

        Convenience accessor for the Filesystem resource. Useful for feedback
        providers that need to inspect file state or working directory.

        Returns:
            The Filesystem instance if configured, None otherwise.
        """
        from ..filesystem import Filesystem

        return self.resources.get_optional(Filesystem)

    def _feedback_for_prompt(self) -> Sequence[Feedback]:
        """Return all feedback for the current prompt."""
        prompt_name = self.prompt_name
        return tuple(
            fb for fb in self.session[Feedback].all() if fb.prompt_name == prompt_name
        )

    @property
    def last_feedback(self) -> Feedback | None:
        """Return the most recent feedback for the current prompt.

        Only considers feedback matching this prompt's name to ensure triggers
        behave consistently when sessions are reused across prompts.
        """
        feedback_list = self._feedback_for_prompt()
        return feedback_list[-1] if feedback_list else None

    def _tool_calls_for_prompt(self) -> Sequence[ToolInvoked]:
        """Return all tool calls for the current prompt."""
        from ..runtime.events import ToolInvoked

        prompt_name = self.prompt_name
        return tuple(
            event
            for event in self.session[ToolInvoked].all()
            if event.prompt_name == prompt_name
        )

    @property
    def tool_call_count(self) -> int:
        """Return the number of tool calls for the current prompt.

        Only counts ToolInvoked events matching this prompt's name to ensure
        triggers behave consistently when sessions are reused across prompts.
        """
        return len(self._tool_calls_for_prompt())

    def tool_calls_since_last_feedback(self) -> int:
        """Return the number of tool calls since the last feedback.

        If no feedback has been produced yet, returns the total tool call count
        for the current prompt.
        """
        last = self.last_feedback
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Return the N most recent tool invocations for the current prompt.

        Args:
            n: Maximum number of tool calls to return.

        Returns:
            Sequence of ToolInvoked events, oldest first. Empty if n=0.
        """
        if n <= 0:
            return ()
        records = self._tool_calls_for_prompt()
        return records[-n:] if len(records) >= n else records


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class FeedbackProvider(Protocol):
    """Protocol for feedback providers that analyze agent trajectory.

    Feedback providers analyze session state and produce contextual feedback
    for agents. They run after tool execution when trigger conditions are met,
    allowing for soft course-correction without blocking tool execution.

    Implement this protocol to create custom feedback providers. Providers
    should be immutable (frozen dataclass) and stateless - all state should
    be accessed via the FeedbackContext.

    Example:
        >>> @dataclass(frozen=True)
        ... class LoopDetector:
        ...     '''Detects repetitive tool call patterns.'''
        ...
        ...     threshold: int = 3
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "LoopDetector"
        ...
        ...     def should_run(self, *, context: FeedbackContext) -> bool:
        ...         return context.tool_call_count >= self.threshold
        ...
        ...     def provide(self, *, context: FeedbackContext) -> Feedback:
        ...         recent = context.recent_tool_calls(self.threshold)
        ...         # Analyze for loops...
        ...         return Feedback(
        ...             provider_name=self.name,
        ...             summary="Potential loop detected",
        ...             severity="caution",
        ...         )

    Access session state via ``context.session`` for consistency with the
    ToolContext pattern used elsewhere in the library.
    """

    @property
    def name(self) -> str:
        """Return a unique identifier for this provider.

        The name is used in the feedback header (e.g., "[Feedback - MyProvider]")
        and for debugging/logging. Should be descriptive and consistent.

        Returns:
            A short, descriptive name for this provider (e.g., "DeadlineFeedback").
        """
        ...

    def should_run(self, *, context: FeedbackContext) -> bool:
        """Determine if the provider should produce feedback.

        Called after trigger conditions are met. Use this for additional
        filtering (e.g., only run if a deadline is configured).

        Args:
            context: Feedback context with state access (includes session).

        Returns:
            True if the provider should run, False to skip.
        """
        ...

    def provide(self, *, context: FeedbackContext) -> Feedback:
        """Analyze trajectory and produce feedback.

        Called when trigger conditions are met and should_run returns True.

        Args:
            context: Feedback context with state access (includes session).

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

    Time-based triggers fire on the first opportunity (before the interval
    elapses) to ensure providers run at least once early in execution.
    Call-based triggers wait until the specified number of calls occur.

    Attributes:
        every_n_calls: Run after this many tool calls since last feedback.
            Set to 10 to run after every 10 tool calls. None to disable.
        every_n_seconds: Run after this many seconds since last feedback.
            Set to 60.0 to run every minute. None to disable.

    Example:
        >>> # Run every 10 tool calls OR every 60 seconds (whichever comes first)
        >>> trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=60)
        >>>
        >>> # Run only on time interval (e.g., for deadline tracking)
        >>> time_trigger = FeedbackTrigger(every_n_seconds=30)
        >>>
        >>> # Run only on call count (e.g., for loop detection)
        >>> call_trigger = FeedbackTrigger(every_n_calls=5)
    """

    every_n_calls: int | None = None
    every_n_seconds: float | None = None


@FrozenDataclass()
class FeedbackProviderConfig:
    """Configuration pairing a feedback provider with its trigger.

    Used to configure feedback providers on a PromptTemplate via the
    ``feedback_providers`` parameter. Multiple providers can be configured,
    and they are evaluated in order (first match wins).

    Attributes:
        provider: The feedback provider instance implementing FeedbackProvider.
        trigger: Conditions that determine when the provider runs. The provider's
            ``should_run()`` method is called only after trigger conditions are met.

    Example:
        >>> from weakincentives.prompt import PromptTemplate
        >>>
        >>> config = FeedbackProviderConfig(
        ...     provider=DeadlineFeedback(),
        ...     trigger=FeedbackTrigger(every_n_seconds=30),
        ... )
        >>>
        >>> template = PromptTemplate[str](
        ...     ns="my-agent",
        ...     key="main",
        ...     feedback_providers=(config,),
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
            # No previous feedback exists - trigger on first opportunity.
            # This ensures time-based providers fire at least once early in execution
            # rather than waiting the full interval from an undefined start time.
            return True

    return False


def run_feedback_providers(
    *,
    providers: Sequence[FeedbackProviderConfig],
    context: FeedbackContext,
) -> str | None:
    """Run feedback providers and return rendered feedback if triggered.

    Iterates through configured providers in order. For each provider:

    1. Check if trigger conditions are met (call count or time elapsed)
    2. Check if ``provider.should_run()`` returns True (additional filtering)
    3. Call ``provider.provide()`` to generate feedback
    4. Store feedback in session and return rendered text

    First matching provider wins; subsequent providers are not evaluated.
    This allows providers to be ordered by priority.

    Note:
        This function automatically sets ``call_index`` and ``prompt_name``
        on the returned Feedback before storing it in the session. Providers
        do not need to set these fields.

    Args:
        providers: Sequence of provider configurations to evaluate, in priority
            order. Earlier providers take precedence.
        context: Feedback context with session state and helper methods.

    Returns:
        Rendered feedback text (from ``Feedback.render()``) if a provider
        triggered, None if no providers matched.
    """
    for config in providers:
        if _should_trigger(config.trigger, context) and config.provider.should_run(
            context=context
        ):
            feedback = config.provider.provide(context=context)
            # Update call_index and prompt_name for trigger state tracking
            feedback = replace(
                feedback,
                call_index=context.tool_call_count,
                prompt_name=context.prompt_name,
            )
            # Store in session for history and trigger calculations
            _ = context.session.dispatch(feedback)
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
    a FeedbackContext from the provided arguments and delegates to
    ``run_feedback_providers()``.

    Typically called by adapter hooks after each tool execution to check if
    any feedback should be injected into the agent's context.

    Args:
        prompt: The prompt with ``feedback_providers`` configured. Providers
            are evaluated in the order they appear in the tuple.
        session: The current session for state access and feedback storage.
            Feedback is automatically dispatched to the session when produced.
        deadline: Optional deadline for time-aware feedback providers (e.g.,
            DeadlineFeedback). Providers can access this via ``context.deadline``.

    Returns:
        Rendered feedback text if a provider triggered, None otherwise.
        The returned text is suitable for injection into the agent's context.

    Example:
        >>> # Called from an adapter's post-tool hook:
        >>> feedback_text = collect_feedback(
        ...     prompt=prompt,
        ...     session=session,
        ...     deadline=deadline,
        ... )
        >>> if feedback_text:
        ...     # Inject feedback into agent context
        ...     adapter.inject_context(feedback_text)
    """
    context = FeedbackContext(
        session=session,
        prompt=prompt,
        deadline=deadline,
    )
    return run_feedback_providers(
        providers=prompt.feedback_providers,
        context=context,
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
