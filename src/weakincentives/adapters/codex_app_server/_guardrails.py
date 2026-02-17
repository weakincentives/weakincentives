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

"""Guardrails support for the Codex App Server adapter.

Provides feedback collection after tool calls and task completion checking.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from ...budget import BudgetTracker, TokenUsage
from ...deadlines import Deadline
from ...prompt.feedback import collect_feedback
from ...prompt.task_completion import TaskCompletionContext
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from ...filesystem import Filesystem
    from ...prompt.protocols import PromptProtocol

logger: StructuredLogger = get_logger(
    __name__, context={"component": "codex_app_server"}
)


def append_feedback(
    content_items: list[dict[str, str]],
    *,
    is_error: bool,
    prompt: PromptProtocol[Any] | None,
    session: SessionProtocol | None,
    deadline: Deadline | None,
) -> None:
    """Collect and append feedback after a successful tool call."""
    if is_error or prompt is None or session is None:
        return
    feedback_text = collect_feedback(prompt=prompt, session=session, deadline=deadline)
    if feedback_text:
        content_items.append({"type": "inputText", "text": feedback_text})


def accumulate_usage(current: TokenUsage | None, new: TokenUsage) -> TokenUsage:
    """Sum token usage across continuation rounds."""
    if current is None:
        return new
    return TokenUsage(
        input_tokens=(current.input_tokens or 0) + (new.input_tokens or 0),
        output_tokens=(current.output_tokens or 0) + (new.output_tokens or 0),
        cached_tokens=(current.cached_tokens or 0) + (new.cached_tokens or 0),
    )


def resolve_filesystem(prompt: PromptProtocol[Any] | None) -> Filesystem | None:
    """Extract filesystem from prompt resources if available."""
    if prompt is None:
        return None
    from ...filesystem import Filesystem as FsType

    with contextlib.suppress(Exception):
        return prompt.resources.get_optional(FsType)
    return None


def check_task_completion(  # noqa: PLR0911
    *,
    prompt: PromptProtocol[Any] | None,
    session: SessionProtocol,
    accumulated_text: str | None,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
) -> tuple[bool, str | None]:
    """Check if the task is complete according to the prompt's checker.

    Returns (should_continue, feedback). When should_continue is True,
    feedback is guaranteed to be a non-empty string that the caller should
    use as the prompt text for the next turn.
    """
    if prompt is None:
        return False, None

    checker = prompt.task_completion_checker
    if checker is None:
        return False, None

    # Don't continue if deadline is exhausted.
    if deadline is not None and deadline.remaining().total_seconds() <= 0:
        logger.debug(
            "codex_app_server.task_completion.deadline_exhausted",
            event="task_completion.deadline_exhausted",
        )
        return False, None

    # Don't continue if budget is exhausted.
    if budget_tracker is not None:
        from ...budget import BudgetExceededError

        try:
            budget_tracker.check()
        except BudgetExceededError:
            logger.debug(
                "codex_app_server.task_completion.budget_exhausted",
                event="task_completion.budget_exhausted",
            )
            return False, None

    filesystem = resolve_filesystem(prompt)
    context = TaskCompletionContext(
        session=session,
        tentative_output=accumulated_text,
        filesystem=filesystem,
    )
    result = checker.check(context)

    if result.complete:
        logger.debug(
            "codex_app_server.task_completion.complete",
            event="task_completion.complete",
            context={"feedback": result.feedback},
        )
        return False, None

    # Incomplete without feedback — nothing actionable to send as a new turn.
    if not result.feedback:
        logger.debug(
            "codex_app_server.task_completion.incomplete_no_feedback",
            event="task_completion.incomplete_no_feedback",
        )
        return False, None

    logger.info(
        "codex_app_server.task_completion.incomplete",
        event="task_completion.incomplete",
        context={"feedback": result.feedback},
    )
    return True, result.feedback
