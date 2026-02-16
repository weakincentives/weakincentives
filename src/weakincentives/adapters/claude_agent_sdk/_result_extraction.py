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

"""Result extraction and validation for Claude Agent SDK adapter.

Functions for extracting structured output, text, and token usage from SDK
response messages, and for validating completion requirements.
"""

from __future__ import annotations

import contextlib
from typing import Any

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...filesystem import Filesystem
from ...prompt import RenderedPrompt
from ...prompt.protocols import PromptProtocol
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol
from ...serde import parse, schema
from ..core import PromptEvaluationError
from ._schema_normalization import _normalize_claude_output_schema
from ._task_completion import TaskCompletionContext

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk"}
)


def build_output_format[OutputT](
    rendered: RenderedPrompt[OutputT],
) -> dict[str, Any] | None:
    """Generate SDK output format from prompt output type."""
    output_type = rendered.output_type

    if output_type is None or output_type is type(None):
        return None

    return {
        "type": "json_schema",
        "schema": _normalize_claude_output_schema(schema(output_type)),
    }


def try_parse_structured_output[OutputT](
    message: Any,  # noqa: ANN401
    rendered: RenderedPrompt[OutputT],
) -> OutputT | None:
    """Attempt to parse structured output from a message."""
    if not (hasattr(message, "structured_output") and message.structured_output):
        return None
    output_type = rendered.output_type
    if not output_type or output_type is type(None):
        return None  # pragma: no cover - defensive for prompts without output type
    try:
        return parse(output_type, message.structured_output, extra="ignore")
    except (TypeError, ValueError) as error:
        logger.warning(
            "claude_agent_sdk.parse.structured_output_error",
            event="parse.structured_output_error",
            context={"error": str(error)},
        )
        return None


def extract_result[OutputT](
    messages: list[Any],
    rendered: RenderedPrompt[OutputT],
    budget_tracker: BudgetTracker | None,
    prompt_name: str,
) -> tuple[str | None, OutputT | None, TokenUsage | None]:
    """Extract text, structured output, and usage from SDK messages."""
    from claude_agent_sdk.types import ResultMessage

    result_text: str | None = None
    structured_output: OutputT | None = None
    total_input_tokens = 0
    total_output_tokens = 0

    for message in reversed(messages):
        if isinstance(message, ResultMessage):
            if hasattr(message, "result") and message.result:
                result_text = message.result
            structured_output = try_parse_structured_output(message, rendered)

        if hasattr(message, "usage") and message.usage:
            usage_dict = message.usage
            if isinstance(usage_dict, dict):
                total_input_tokens += usage_dict.get("input_tokens", 0)
                total_output_tokens += usage_dict.get("output_tokens", 0)

    usage = TokenUsage(
        input_tokens=total_input_tokens or None,
        output_tokens=total_output_tokens or None,
        cached_tokens=None,
    )

    if budget_tracker and (total_input_tokens or total_output_tokens):
        budget_tracker.record_cumulative(prompt_name, usage)

    return result_text, structured_output, usage


def raise_if_missing_required_structured_output[OutputT](  # noqa: PLR0913
    *,
    rendered: RenderedPrompt[OutputT],
    prompt_name: str,
    messages: list[Any],
    result_text: str | None,
    output: OutputT | None,
    stop_reason: str | None,
    stderr_buffer: list[str],
) -> None:
    """Raise when structured output is required but no response was produced."""
    output_type = rendered.output_type
    if output_type is None or output_type is type(None):
        return
    if output is not None or result_text is not None:
        return

    message_type_counts: dict[str, int] = {}
    for message in messages:
        message_type = type(message).__name__
        message_type_counts[message_type] = message_type_counts.get(message_type, 0) + 1
    stderr_tail = [line.rstrip() for line in stderr_buffer[-20:]]

    logger.warning(
        "claude_agent_sdk.evaluate.missing_structured_output",
        event="sdk.evaluate.missing_structured_output",
        context={
            "prompt_name": prompt_name,
            "message_count": len(messages),
            "message_type_counts": message_type_counts,
            "stop_reason": stop_reason,
            "stderr_tail": stderr_tail or None,
        },
    )

    raise PromptEvaluationError(
        message="Structured output prompt returned no text and no structured output.",
        prompt_name=prompt_name,
        phase="response",
        provider_payload={
            "output_type": (
                output_type.__name__
                if hasattr(output_type, "__name__")
                else str(output_type)
            ),
            "message_count": len(messages),
            "message_type_counts": message_type_counts,
            "stop_reason": stop_reason,
            "stderr_tail": stderr_tail or None,
        },
    )


def verify_task_completion(  # noqa: PLR0913
    output: Any,  # noqa: ANN401
    session: SessionProtocol,
    stop_reason: str | None,
    prompt_name: str,
    *,
    deadline: Deadline | None = None,
    budget_tracker: BudgetTracker | None = None,
    prompt: PromptProtocol[Any] | None = None,
    adapter: Any = None,  # noqa: ANN401
) -> None:
    """Verify task completion if checker is configured.

    Resolves the checker from the prompt. Skips verification if deadline or
    budget is exhausted (partial output is acceptable when resources run out).

    Args:
        output: The structured output to verify.
        session: The session containing state.
        stop_reason: Why the agent stopped.
        prompt_name: Name of the prompt for error reporting.
        deadline: Optional deadline to check for exhaustion.
        budget_tracker: Optional budget tracker to check for exhaustion.
        prompt: Optional prompt for filesystem access and checker resolution.
        adapter: The adapter instance (passed through to context).
    """
    from ._task_completion import resolve_checker

    checker = resolve_checker(prompt=prompt)
    if output is None or checker is None:
        return

    # Skip verification if deadline exceeded - can't do more work
    if deadline is not None and deadline.remaining().total_seconds() <= 0:
        logger.debug(
            "claude_agent_sdk.verify.deadline_exceeded",
            event="sdk.verify.deadline_exceeded",
            context={"prompt_name": prompt_name, "stop_reason": stop_reason},
        )
        return

    # Skip verification if budget exhausted - can't do more work
    if budget_tracker is not None:
        budget = budget_tracker.budget
        consumed = budget_tracker.consumed
        consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)
        if (
            budget.max_total_tokens is not None
            and consumed_total >= budget.max_total_tokens
        ):
            logger.debug(
                "claude_agent_sdk.verify.budget_exhausted",
                event="sdk.verify.budget_exhausted",
                context={"prompt_name": prompt_name, "stop_reason": stop_reason},
            )
            return

    # Get filesystem from prompt resources if available.
    # checker is resolved from prompt so prompt is always set here,
    # but guard defensively for type safety.
    filesystem: Filesystem | None = None
    if prompt is not None:  # pragma: no branch - invariant of resolve_checker
        with contextlib.suppress(LookupError, AttributeError):
            filesystem = prompt.resources.get(Filesystem)

    context = TaskCompletionContext(
        session=session,
        tentative_output=output,
        stop_reason=stop_reason or "structured_output",
        filesystem=filesystem,
        adapter=adapter,
    )
    completion = checker.check(context)
    if not completion.complete:
        # Log warning but don't fail - the task completion checker provides
        # feedback during execution via hooks. At the end, we should return
        # whatever output the agent produced, even if tasks are incomplete.
        # This allows the agent to make progress even if it doesn't complete
        # all planned tasks within the available turns/budget.
        logger.warning(
            "claude_agent_sdk.evaluate.incomplete_tasks",
            event="sdk.evaluate.incomplete_tasks",
            context={
                "prompt_name": prompt_name,
                "feedback": completion.feedback,
                "stop_reason": stop_reason,
                "has_output": output is not None,
            },
        )
        # Don't raise an error - let the response be returned with whatever
        # output the agent managed to produce
