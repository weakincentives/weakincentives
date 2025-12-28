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

"""Completion handler types for determining task completion status.

Completion handlers allow prompts to define custom logic for determining
whether a task has been completed. This is used by the Claude Agent SDK
adapter's stop hook to decide whether to continue execution when there
is remaining budget.

Example::

    from weakincentives.prompt import (
        CompletionContext,
        CompletionHandler,
        CompletionResult,
        Prompt,
    )

    def check_review_complete(
        output: ReviewResult,
        *,
        context: CompletionContext,
    ) -> CompletionResult:
        # Check if review found all required sections
        if not output.has_security_section:
            return CompletionResult(
                complete=False,
                reason="Missing security analysis section",
            )
        return CompletionResult(complete=True)

    prompt = Prompt(template).with_completion_handler(check_review_complete)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ..budget import BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..resources import ResourceRegistry
from ..runtime.events._types import TokenUsage

if TYPE_CHECKING:
    from ..runtime.session.protocols import SessionProtocol
    from .protocols import PromptProtocol, RenderedPromptProtocol

OutputT_contra = TypeVar("OutputT_contra", contravariant=True)


@FrozenDataclass()
class CompletionResult:
    """Result from a completion handler indicating task completion status.

    Attributes:
        complete: Whether the task has been completed successfully.
        reason: Optional explanation for why the task is incomplete.
            Useful for logging and debugging.
    """

    complete: bool
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class CompletionContext:
    """Context passed to completion handlers for evaluating task completion.

    Provides access to execution state, session, budgets, and response
    information needed to determine if a task is complete.

    Attributes:
        prompt: The prompt that was evaluated.
        rendered_prompt: The rendered prompt that was sent to the model.
        session: Session for state access.
        stop_reason: The reason execution stopped (e.g., "end_turn", "tool_use").
        deadline: Optional deadline for the execution.
        budget_tracker: Optional budget tracker for token consumption.
        resources: Resource registry for accessing injected resources.
        usage: Token usage from the current evaluation.
        response_text: The text response from the model, if any.
    """

    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    session: SessionProtocol
    stop_reason: str | None
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    usage: TokenUsage | None = None
    response_text: str | None = None

    def has_remaining_budget(self) -> bool:
        """Check if there is remaining budget for continued execution.

        Returns True if:
        - No budget tracker is configured, or
        - Budget limits have not been exceeded

        Also checks deadline if configured.
        """
        # Check deadline
        if self.deadline is not None and self.deadline.remaining().total_seconds() <= 0:
            return False

        # Check token budget
        if self.budget_tracker is None:
            return True

        budget = self.budget_tracker.budget
        consumed = self.budget_tracker.consumed
        consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)

        return not (
            budget.max_total_tokens is not None
            and consumed_total >= budget.max_total_tokens
        )


class CompletionHandler(Protocol[OutputT_contra]):
    """Protocol for completion handler callables.

    A completion handler receives the structured output from prompt evaluation
    and determines whether the task has been completed.

    The handler is called by the stop hook after the model returns a response.
    If the handler returns ``complete=False`` and there is remaining budget,
    execution will continue.

    Example::

        def my_handler(
            output: MyOutput,
            *,
            context: CompletionContext,
        ) -> CompletionResult:
            if output.needs_revision:
                return CompletionResult(
                    complete=False,
                    reason="Output requires revision",
                )
            return CompletionResult(complete=True)
    """

    def __call__(
        self,
        output: OutputT_contra,
        *,
        context: CompletionContext,
    ) -> CompletionResult:
        """Evaluate whether the task is complete.

        Args:
            output: The structured output from prompt evaluation.
            context: Execution context with session, budget, and response info.

        Returns:
            CompletionResult indicating whether the task is complete.
        """
        ...


__all__ = [
    "CompletionContext",
    "CompletionHandler",
    "CompletionResult",
]
