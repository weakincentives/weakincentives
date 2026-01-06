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

"""Task completion checking abstractions for Claude Agent SDK.

This module provides a generic mechanism for verifying task completion status
before allowing an agent to stop or produce final output. Checkers can inspect
session state, filesystem contents, and use LLM-based verification.

Example:
    >>> from weakincentives.adapters.claude_agent_sdk import (
    ...     PlanBasedChecker,
    ...     TaskCompletionChecker,
    ... )
    >>> from weakincentives.runtime import Session
    >>>
    >>> checker = PlanBasedChecker()
    >>> result = checker.check(session, tentative_output={"summary": "Done"})
    >>> if not result.complete:
    ...     print(result.feedback)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ...dataclasses import FrozenDataclass
from ...runtime.logging import get_logger

if TYPE_CHECKING:
    from ...filesystem import Filesystem
    from ...runtime.session import Session
    from ..core import ProviderAdapter

__all__ = [
    "CompositeChecker",
    "LLMJudgeChecker",
    "PlanBasedChecker",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
]

logger = get_logger(__name__)


@FrozenDataclass()
class TaskCompletionResult:
    """Result of a task completion check.

    Attributes:
        complete: True if all tasks are complete and the agent can stop.
        feedback: Natural language feedback explaining the completion status.
            When complete is False, this contains a reminder of what remains.
            When complete is True, this may contain a summary of completed work.
    """

    complete: bool
    feedback: str | None = None

    @classmethod
    def ok(cls, feedback: str | None = None) -> TaskCompletionResult:
        """Create a result indicating tasks are complete."""
        return cls(complete=True, feedback=feedback)

    @classmethod
    def incomplete(cls, feedback: str) -> TaskCompletionResult:
        """Create a result indicating tasks are not complete."""
        return cls(complete=False, feedback=feedback)


@dataclass(slots=True)
class TaskCompletionContext:
    """Context provided to task completion checkers.

    This context provides access to all resources a checker might need to
    evaluate task completion status.

    Attributes:
        session: The session containing state slices (Plan, etc.).
        tentative_output: The output the agent is attempting to produce.
            For StructuredOutput tool calls, this is the output payload.
            For end_turn stops, this may be None.
        filesystem: Optional filesystem for checking file-based completion
            criteria (e.g., verifying test files exist).
        adapter: Optional adapter for LLM-based verification. When provided,
            checkers can use LLM-as-judge to evaluate completion.
        stop_reason: The reason for the stop attempt (e.g., "end_turn",
            "tool_use", "max_turns_reached").
    """

    session: Session
    tentative_output: Any = None
    filesystem: Filesystem | None = None
    adapter: ProviderAdapter | None = None
    stop_reason: str | None = None


@runtime_checkable
class TaskCompletionChecker(Protocol):
    """Protocol for task completion verification.

    Implementations can check various aspects of completion:
    - Plan state (all steps marked done)
    - File existence (required outputs created)
    - LLM verification (using an LLM to judge completion)
    - Custom business logic

    Checkers are designed to be composable via CompositeChecker.
    """

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check if tasks are complete.

        Args:
            context: Context with session, output, filesystem, and adapter.

        Returns:
            Result indicating completion status with natural language feedback.
        """
        ...


class PlanBasedChecker(TaskCompletionChecker):
    """Plan-based task completion checker.

    This checker examines the session's Plan state to determine if all
    tasks are complete. It checks for steps with status != "done" and
    returns incomplete feedback if any are found.
    """

    def __init__(self, plan_type: type | None = None) -> None:
        """Initialize the checker.

        Args:
            plan_type: The Plan dataclass type to check. If None, attempts to
                import from weakincentives.contrib.tools.planning.
        """
        self._plan_type = plan_type

    def _resolve_plan_type(self) -> type | None:
        """Resolve the Plan type, importing if necessary."""
        if self._plan_type is not None:
            return self._plan_type

        try:
            from ...contrib.tools.planning import Plan
        except ImportError:
            return None
        else:
            return Plan

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check if all plan tasks are complete.

        Args:
            context: Context with session containing Plan state.

        Returns:
            Complete if all steps are done or no plan exists.
            Incomplete with feedback listing remaining tasks otherwise.
        """
        plan_type = self._resolve_plan_type()
        if plan_type is None:
            logger.debug(
                "task_completion.plan_checker.no_plan_type",
                event="plan_checker.no_plan_type",
            )
            return TaskCompletionResult.ok("No planning tools available.")

        try:
            plan = context.session[plan_type].latest()
        except (KeyError, AttributeError):
            logger.debug(
                "task_completion.plan_checker.no_plan_slice",
                event="plan_checker.no_plan_slice",
            )
            return TaskCompletionResult.ok("No plan configured in session.")

        if plan is None:
            logger.debug(
                "task_completion.plan_checker.no_plan",
                event="plan_checker.no_plan",
            )
            return TaskCompletionResult.ok("No plan has been created.")

        steps = getattr(plan, "steps", ())
        incomplete_steps = [
            step for step in steps if getattr(step, "status", "done") != "done"
        ]

        if not incomplete_steps:
            step_count = len(steps)
            logger.debug(
                "task_completion.plan_checker.all_complete",
                event="plan_checker.all_complete",
                context={"step_count": step_count},
            )
            return TaskCompletionResult.ok(
                f"All {step_count} task(s) in the plan are complete."
            )

        incomplete_count = len(incomplete_steps)
        total_count = len(steps)
        incomplete_titles = [
            getattr(step, "title", f"Step {i}")
            for i, step in enumerate(incomplete_steps)
        ]

        max_titles_in_message = 3
        task_list = ", ".join(incomplete_titles[:max_titles_in_message])
        if len(incomplete_titles) > max_titles_in_message:
            task_list += "..."
        feedback = (
            f"You have {incomplete_count} incomplete task(s) out of {total_count}. "
            f"Please either complete all remaining tasks or update the plan to "
            f"remove tasks that are no longer needed before producing output: "
            f"{task_list}"
        )

        logger.info(
            "task_completion.plan_checker.incomplete",
            event="plan_checker.incomplete",
            context={
                "incomplete_count": incomplete_count,
                "total_count": total_count,
                "incomplete_titles": incomplete_titles[:5],
            },
        )

        return TaskCompletionResult.incomplete(feedback)


class LLMJudgeChecker(TaskCompletionChecker):
    """Placeholder for LLM-as-judge task completion verification.

    .. warning::
        This is a placeholder implementation that does NOT perform actual
        LLM verification. It always returns ``ok()`` when an adapter is
        available. A full implementation would use the adapter to evaluate
        completion criteria via an LLM call.

    This checker is intended to use an LLM to evaluate whether the agent
    has completed its tasks based on the session state, tentative output,
    and custom criteria. The full implementation would:

    1. Build a verification prompt from the context and criteria
    2. Call the adapter to evaluate completion
    3. Parse the LLM response to determine pass/fail

    Requires an adapter to be provided in the context for LLM calls.
    """

    def __init__(
        self,
        *,
        criteria: str | None = None,
        require_adapter: bool = True,
    ) -> None:
        """Initialize the LLM judge checker.

        .. warning::
            This is a placeholder. See class docstring for details.

        Args:
            criteria: Custom criteria for the LLM to evaluate. If None,
                uses a default prompt asking if tasks appear complete.
            require_adapter: If True, returns incomplete when no adapter
                is available. If False, returns ok when no adapter.
        """
        warnings.warn(
            "LLMJudgeChecker is a placeholder implementation that does not "
            "perform actual LLM verification. It will always return ok() "
            "when an adapter is available.",
            stacklevel=2,
        )
        self._criteria = criteria or (
            "Based on the session state and output, determine if the agent "
            "has successfully completed all requested tasks. Consider whether "
            "the output addresses the original request and if any work remains."
        )
        self._require_adapter = require_adapter

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check completion using LLM-as-judge.

        Args:
            context: Context with adapter for LLM calls.

        Returns:
            Result based on LLM evaluation of completion status.
        """
        if context.adapter is None:
            if self._require_adapter:
                return TaskCompletionResult.incomplete(
                    "LLM verification required but no adapter available."
                )
            return TaskCompletionResult.ok("No adapter for LLM verification.")

        # Build the verification prompt (for future LLM call)
        _ = self._build_verification_prompt(context)

        try:
            # Use adapter for verification
            # Note: This is a simplified implementation. A full implementation
            # would use a proper Prompt with structured output for the verdict.
            logger.debug(
                "task_completion.llm_judge.checking",
                event="llm_judge.checking",
                context={"criteria": self._criteria[:100]},
            )

            # For now, we return a placeholder indicating LLM verification
            # would be performed. A full implementation would make the LLM call.
            return TaskCompletionResult.ok(
                "LLM verification passed (implementation pending)."
            )

        except Exception as e:
            logger.warning(
                "task_completion.llm_judge.error",
                event="llm_judge.error",
                context={"error": str(e)},
            )
            # On error, allow completion to avoid blocking
            return TaskCompletionResult.ok(
                f"LLM verification skipped due to error: {e}"
            )

    def _build_verification_prompt(self, context: TaskCompletionContext) -> str:
        """Build the verification prompt for the LLM."""
        parts = [
            "# Task Completion Verification",
            "",
            "## Criteria",
            self._criteria,
            "",
        ]

        if context.tentative_output is not None:
            parts.extend([
                "## Tentative Output",
                str(context.tentative_output),
                "",
            ])

        if context.stop_reason:
            parts.extend([
                "## Stop Reason",
                context.stop_reason,
                "",
            ])

        parts.extend([
            "## Instructions",
            "Evaluate whether the task is complete based on the criteria above.",
            "Respond with COMPLETE if done, or INCOMPLETE with explanation.",
        ])

        return "\n".join(parts)


class CompositeChecker(TaskCompletionChecker):
    """Combines multiple checkers with configurable logic.

    Checkers are evaluated in order. By default (all_must_pass=True),
    all checkers must return complete for the composite to be complete.
    With all_must_pass=False, any checker returning complete is sufficient.
    """

    def __init__(
        self,
        checkers: tuple[TaskCompletionChecker, ...],
        *,
        all_must_pass: bool = True,
    ) -> None:
        """Initialize the composite checker.

        Args:
            checkers: Tuple of checkers to evaluate.
            all_must_pass: If True (default), all checkers must pass.
                If False, any checker passing is sufficient.
        """
        self._checkers = checkers
        self._all_must_pass = all_must_pass

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check completion using all configured checkers.

        Args:
            context: Context passed to each checker.

        Returns:
            Combined result based on all_must_pass setting.
        """
        if not self._checkers:
            return TaskCompletionResult.ok("No checkers configured.")

        results: list[TaskCompletionResult] = []
        for checker in self._checkers:
            result = checker.check(context)
            results.append(result)

            # Short-circuit based on mode
            if self._all_must_pass and not result.complete:
                # First failure stops evaluation
                return result
            if not self._all_must_pass and result.complete:
                # First success stops evaluation
                return result

        # All checkers evaluated
        if self._all_must_pass:
            # All passed
            feedbacks = [r.feedback for r in results if r.feedback]
            combined = " ".join(feedbacks) if feedbacks else "All checks passed."
            return TaskCompletionResult.ok(combined)

        # None passed (only reached if all_must_pass=False)
        feedbacks = [r.feedback for r in results if r.feedback and not r.complete]
        combined = " ".join(feedbacks) if feedbacks else "No checks passed."
        return TaskCompletionResult.incomplete(combined)
