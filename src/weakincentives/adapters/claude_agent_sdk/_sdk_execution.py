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

"""SDK query execution loop for Claude Agent SDK adapter.

Functions for running the SDK client query, collecting response messages,
managing continuation rounds, and enforcing deadline/budget constraints.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ...filesystem import Filesystem
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ..core import PromptEvaluationError
from ._hooks import HookContext
from ._message_extraction import _extract_message_content
from ._task_completion import TaskCompletionContext
from ._visibility_signal import VisibilityExpansionSignal

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk"}
)


def check_continuation_constraints(
    hook_context: HookContext,
    continuation_round: int,
) -> bool:
    """Check deadline and budget constraints before a round.

    Returns True if constraints are satisfied and we should continue.
    Returns False if constraints are exceeded and we should stop.
    """
    if hook_context.deadline and hook_context.deadline.remaining().total_seconds() <= 0:
        logger.info(
            "claude_agent_sdk.sdk_query.deadline_exceeded",
            event="sdk_query.deadline_exceeded",
            context={
                "continuation_round": continuation_round,
                "prompt_name": hook_context.prompt_name,
            },
        )
        return False

    if hook_context.budget_tracker:
        try:
            hook_context.budget_tracker.check()
        except Exception as budget_error:
            logger.info(
                "claude_agent_sdk.sdk_query.token_budget_exceeded",
                event="sdk_query.token_budget_exceeded",
                context={
                    "continuation_round": continuation_round,
                    "total_input_tokens": hook_context.stats.total_input_tokens,
                    "total_output_tokens": hook_context.stats.total_output_tokens,
                    "error": str(budget_error),
                },
            )
            return False

    return True


def update_token_stats(
    hook_context: HookContext,
    content: dict[str, Any],
) -> None:
    """Update token statistics from message content."""
    if content.get("input_tokens"):
        hook_context.stats.total_input_tokens += content["input_tokens"]
    if content.get("output_tokens"):
        hook_context.stats.total_output_tokens += content["output_tokens"]

    if hook_context.budget_tracker:
        hook_context.budget_tracker.record_cumulative(
            hook_context.prompt_name,
            TokenUsage(
                input_tokens=hook_context.stats.total_input_tokens,
                output_tokens=hook_context.stats.total_output_tokens,
            ),
        )


def check_task_completion(
    checker: Any,  # noqa: ANN401
    round_messages: list[Any],
    hook_context: HookContext,
) -> tuple[bool, str | None]:
    """Check if task is complete and return (should_continue, feedback)."""
    if not round_messages:
        return (False, None)

    last_message = round_messages[-1]
    tentative_output = getattr(last_message, "structured_output", None)
    if tentative_output is None:
        tentative_output = getattr(last_message, "result", None)

    # Resolve filesystem from prompt resources so file-based checkers work
    filesystem: Filesystem | None = None
    with contextlib.suppress(LookupError, AttributeError):
        filesystem = hook_context.prompt.resources.get(Filesystem)

    completion_context = TaskCompletionContext(
        session=hook_context.session,
        tentative_output=tentative_output,
        stop_reason="message_stream_complete",
        filesystem=filesystem,
    )

    result = checker.check(completion_context)

    if not result.complete and result.feedback:
        return (True, result.feedback)

    if result.complete:
        logger.debug(
            "claude_agent_sdk.sdk_query.task_complete",
            event="sdk_query.task_complete",
            context={"feedback": result.feedback},
        )

    return (False, None)


def log_message_received(
    message: Any,  # noqa: ANN401
    messages: list[Any],
    continuation_round: int,
    hook_context: HookContext,
    content: dict[str, Any],
) -> None:
    """Log message receipt at DEBUG level."""
    logger.debug(
        "claude_agent_sdk.sdk_query.message_received",
        event="sdk_query.message_received",
        context={
            "message_type": type(message).__name__,
            "message_index": len(messages) - 1,
            "continuation_round": continuation_round,
            "cumulative_input_tokens": hook_context.stats.total_input_tokens,
            "cumulative_output_tokens": hook_context.stats.total_output_tokens,
            **content,
        },
    )


def should_continue_loop(
    continuation_round: int,
    max_continuation_rounds: int | None,
) -> bool:
    """Check if continuation loop should continue."""
    return (
        max_continuation_rounds is None or continuation_round < max_continuation_rounds
    )


def check_and_raise_visibility_signal(
    visibility_signal: VisibilityExpansionSignal,
) -> None:
    """Check for visibility expansion signal and raise if present."""
    stored_exc = visibility_signal.get_and_clear()
    if stored_exc is not None:
        logger.debug(
            "claude_agent_sdk.sdk_query.visibility_expansion_detected",
            event="sdk_query.visibility_expansion_detected",
            context={
                "section_keys": stored_exc.section_keys,
                "reason": stored_exc.reason,
            },
        )
        raise stored_exc


def resolve_response_wait_timeout(
    *,
    hook_context: HookContext,
    continuation_round: int,
    message_count: int,
) -> tuple[float | None, bool]:
    """Resolve wait timeout for the next response message.

    Returns:
        Tuple of (timeout_seconds, should_stop_stream_reading).
    """
    if hook_context.deadline is None:
        return (None, False)

    wait_timeout = hook_context.deadline.remaining().total_seconds()
    if wait_timeout <= 0:
        logger.info(
            "claude_agent_sdk.sdk_query.deadline_exceeded_during_stream_wait",
            event="sdk_query.deadline_exceeded_during_stream_wait",
            context={
                "prompt_name": hook_context.prompt_name,
                "continuation_round": continuation_round,
                "message_count": message_count,
                "deadline_remaining_seconds": wait_timeout,
            },
        )
        return (None, True)

    return (wait_timeout, False)


async def next_response_message(
    *,
    response_stream: Any,  # noqa: ANN401
    hook_context: HookContext,
    continuation_round: int,
    message_count: int,
) -> Any | None:  # noqa: ANN401
    """Read the next message from the SDK response stream."""
    wait_timeout, should_stop = resolve_response_wait_timeout(
        hook_context=hook_context,
        continuation_round=continuation_round,
        message_count=message_count,
    )
    if should_stop:
        return None

    try:
        if wait_timeout is None:
            return await anext(response_stream)
        return await asyncio.wait_for(
            anext(response_stream),
            timeout=wait_timeout,
        )
    except StopAsyncIteration:
        return None
    except TimeoutError as error:
        raise PromptEvaluationError(
            message="Deadline exceeded while waiting for Claude SDK response stream.",
            prompt_name=hook_context.prompt_name,
            phase="response",
            provider_payload={
                "continuation_round": continuation_round,
                "message_count": message_count,
                "deadline_remaining_seconds": (
                    hook_context.deadline.remaining().total_seconds()
                    if hook_context.deadline is not None
                    else None
                ),
            },
        ) from error


async def collect_round_messages(
    *,
    client: Any,  # noqa: ANN401
    messages: list[Any],
    continuation_round: int,
    hook_context: HookContext,
) -> list[Any]:
    """Collect a single response round from the SDK."""
    round_messages: list[Any] = []
    response_stream = client.receive_response()

    while True:
        message = await next_response_message(
            response_stream=response_stream,
            hook_context=hook_context,
            continuation_round=continuation_round,
            message_count=len(messages),
        )
        if message is None:
            break

        messages.append(message)
        round_messages.append(message)

        content = _extract_message_content(message)
        update_token_stats(hook_context, content)
        log_message_received(
            message, messages, continuation_round, hook_context, content
        )

    return round_messages


def _log_sdk_options(options_kwargs: dict[str, Any]) -> None:
    """Log SDK options at DEBUG level (excluding sensitive data)."""
    logger.debug(
        "claude_agent_sdk.sdk_query.options",
        event="sdk_query.options",
        context={
            "model": options_kwargs.get("model"),
            "cwd": options_kwargs.get("cwd"),
            "permission_mode": options_kwargs.get("permission_mode"),
            "max_turns": options_kwargs.get("max_turns"),
            "max_budget_usd": options_kwargs.get("max_budget_usd"),
            "reasoning": options_kwargs.get("reasoning"),
            "has_output_format": "output_format" in options_kwargs,
            "allowed_tools": options_kwargs.get("allowed_tools"),
            "disallowed_tools": options_kwargs.get("disallowed_tools"),
            "has_mcp_servers": "mcp_servers" in options_kwargs,
            "betas": options_kwargs.get("betas"),
        },
    )


async def _run_continuation_loop(
    *,
    client: Any,  # noqa: ANN401
    messages: list[Any],
    hook_context: HookContext,
    checker: Any | None,  # noqa: ANN401
) -> None:
    """Run the SDK message collection and continuation loop."""
    has_constraints = bool(hook_context.deadline or hook_context.budget_tracker)
    max_continuation_rounds = None if has_constraints else 100
    continuation_round = 0

    logger.debug(
        "claude_agent_sdk.sdk_query.loop_config",
        event="sdk_query.loop_config",
        context={
            "has_constraints": has_constraints,
            "has_deadline": hook_context.deadline is not None,
            "has_budget": hook_context.budget_tracker is not None,
            "max_rounds": max_continuation_rounds,
            "prompt_name": hook_context.prompt_name,
        },
    )

    while should_continue_loop(  # pragma: no branch
        continuation_round, max_continuation_rounds
    ):
        if not check_continuation_constraints(hook_context, continuation_round):
            break

        round_messages = await collect_round_messages(
            client=client,
            messages=messages,
            continuation_round=continuation_round,
            hook_context=hook_context,
        )

        if not round_messages:
            logger.warning(
                "claude_agent_sdk.sdk_query.empty_message_stream",
                event="sdk_query.empty_message_stream",
                context={
                    "continuation_round": continuation_round,
                    "prompt_name": hook_context.prompt_name,
                },
            )
            break

        if checker is not None:
            should_cont, feedback = check_task_completion(
                checker, round_messages, hook_context
            )
            if should_cont and feedback:
                logger.info(
                    "claude_agent_sdk.sdk_query.continuation_required",
                    event="sdk_query.continuation_required",
                    context={
                        "feedback": feedback[:200],
                        "continuation_round": continuation_round + 1,
                    },
                )
                continuation_round += 1
                await client.query(
                    prompt=feedback,
                    session_id=hook_context.prompt_name,
                )
                continue

        break


async def run_sdk_query(  # noqa: PLR0913
    *,
    options_kwargs: dict[str, Any],
    prompt_text: str,
    hook_context: HookContext,
    visibility_signal: VisibilityExpansionSignal,
    stderr_buffer: list[str],
    task_completion_checker: Any | None,  # noqa: ANN401
) -> list[Any]:
    """Execute the SDK query and return message list."""
    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import ClaudeAgentOptions

    logger.debug(
        "claude_agent_sdk.sdk_query.entry",
        event="sdk_query.entry",
        context={
            "prompt_text_preview": (prompt_text or "")[:500],
            "has_output_format": options_kwargs.get("output_format") is not None,
        },
    )
    _log_sdk_options(options_kwargs)

    options = ClaudeAgentOptions(**options_kwargs)
    client = ClaudeSDKClient(options=options)

    logger.debug(
        "claude_agent_sdk.sdk_query.connecting",
        event="sdk_query.connecting",
        context={"prompt_name": hook_context.prompt_name},
    )

    await client.connect(prompt=None)
    await client.query(prompt=prompt_text, session_id=hook_context.prompt_name)

    logger.debug(
        "claude_agent_sdk.sdk_query.executing",
        event="sdk_query.executing",
        context={"prompt_name": hook_context.prompt_name},
    )

    messages: list[Any] = []
    try:
        await _run_continuation_loop(
            client=client,
            messages=messages,
            hook_context=hook_context,
            checker=task_completion_checker,
        )
    finally:
        if client._transport is not None:
            await client._transport.end_input()
        logger.debug(
            "claude_agent_sdk.sdk_query.disconnecting",
            event="sdk_query.disconnecting",
            context={
                "prompt_name": hook_context.prompt_name,
            },
        )
        await client.disconnect()

    logger.debug(
        "claude_agent_sdk.sdk_query.complete",
        event="sdk_query.complete",
        context={
            "message_count": len(messages),
            "stderr_line_count": len(stderr_buffer),
            "stats_tool_count": hook_context.stats.tool_count,
            "stats_turn_count": hook_context.stats.turn_count,
            "stats_subagent_count": hook_context.stats.subagent_count,
            "stats_compact_count": hook_context.stats.compact_count,
            "stats_input_tokens": hook_context.stats.total_input_tokens,
            "stats_output_tokens": hook_context.stats.total_output_tokens,
            "stats_hook_errors": hook_context.stats.hook_errors,
        },
    )

    check_and_raise_visibility_signal(visibility_signal)
    return messages
