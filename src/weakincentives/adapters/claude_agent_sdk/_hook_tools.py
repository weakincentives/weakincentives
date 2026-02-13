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

"""Tool-processing helpers for Claude Agent SDK hooks.

Private helper functions used by :mod:`._hooks` for constraint checking,
tool transaction management, event dispatch, and feedback collection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from claude_agent_sdk.types import (
    PostToolUseHookInput,
    PostToolUseHookSpecificOutput,
    PreToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...prompt.feedback import collect_feedback
from ...runtime.events.types import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger
from ._hook_context import HookContext
from ._task_completion import TaskCompletionChecker, TaskCompletionResult

__all__ = [
    "_ParsedToolData",
    "_check_budget_constraint",
    "_check_deadline_constraint",
    "_compute_budget_info",
    "_dispatch_tool_invoked_event",
    "_handle_mcp_tool_post",
    "_handle_structured_output_completion",
    "_handle_tool_transaction",
    "_is_budget_exhausted",
    "_is_deadline_exceeded",
    "_is_tool_error_response",
    "_parse_tool_data",
    "_run_feedback_providers",
    "_setup_tool_execution_state",
    "_utcnow",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "sdk_hooks"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _compute_budget_info(budget_tracker: BudgetTracker | None) -> dict[str, Any]:
    """Compute budget info for logging."""
    if budget_tracker is None or not isinstance(budget_tracker, BudgetTracker):
        return {}
    budget = budget_tracker.budget
    consumed = budget_tracker.consumed
    consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)
    return {
        "budget_consumed_input": consumed.input_tokens,
        "budget_consumed_output": consumed.output_tokens,
        "budget_consumed_total": consumed_total,
        "budget_max_total": budget.max_total_tokens,
        "budget_remaining": (
            budget.max_total_tokens - consumed_total
            if budget.max_total_tokens
            else None
        ),
    }


def _is_deadline_exceeded(deadline: Deadline | None) -> bool:
    """Check if deadline has been exceeded."""
    return deadline is not None and deadline.remaining().total_seconds() <= 0


def _is_budget_exhausted(budget_tracker: BudgetTracker | None) -> bool:
    """Check if token budget has been exhausted."""
    if budget_tracker is None or not isinstance(budget_tracker, BudgetTracker):
        return False
    budget = budget_tracker.budget
    consumed = budget_tracker.consumed
    consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)
    return (
        budget.max_total_tokens is not None
        and consumed_total >= budget.max_total_tokens
    )


def _check_deadline_constraint(
    hook_context: HookContext, tool_name: str
) -> SyncHookJSONOutput | None:
    """Check deadline constraint and return deny response if exceeded."""
    if not _is_deadline_exceeded(hook_context.deadline):
        return None
    logger.warning(
        "claude_agent_sdk.hook.deadline_exceeded",
        event="hook.deadline_exceeded",
        context={
            "tool_name": tool_name,
            "elapsed_ms": hook_context.elapsed_ms,
            "tool_count": hook_context.stats.tool_count,
        },
    )
    output: PreToolUseHookSpecificOutput = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Deadline exceeded",
    }
    return {"hookSpecificOutput": output}


def _check_budget_constraint(
    hook_context: HookContext, tool_name: str
) -> SyncHookJSONOutput | None:
    """Check budget constraint and return deny response if exhausted."""
    budget_tracker = hook_context.budget_tracker
    if budget_tracker is None or not _is_budget_exhausted(budget_tracker):
        return None
    budget = budget_tracker.budget
    consumed = budget_tracker.consumed
    consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)
    logger.warning(
        "claude_agent_sdk.hook.budget_exhausted",
        event="hook.budget_exhausted",
        context={
            "tool_name": tool_name,
            "consumed_total": consumed_total,
            "max_total": budget.max_total_tokens,
            "elapsed_ms": hook_context.elapsed_ms,
        },
    )
    output: PreToolUseHookSpecificOutput = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Token budget exhausted",
    }
    return {"hookSpecificOutput": output}


def _setup_tool_execution_state(
    hook_context: HookContext,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str | None,
    hook_start: float,
) -> None:
    """Set up tool execution state for MCP or native tools.

    For MCP tools: enqueues tool_use_id in mcp_tool_state for BridgedTool.
    For native tools: begins transactional execution with snapshot.
    """
    is_mcp_tool = tool_name.startswith("mcp__wink__")
    if is_mcp_tool and hook_context.mcp_tool_state and tool_use_id is not None:
        hook_context.mcp_tool_state.enqueue(tool_name, tool_input, tool_use_id)
    elif tool_use_id is not None and not is_mcp_tool:
        hook_context.begin_tool_execution(tool_use_id=tool_use_id, tool_name=tool_name)
        hook_duration_ms = int((time.monotonic() - hook_start) * 1000)
        logger.debug(
            "claude_agent_sdk.hook.snapshot_taken",
            event="hook.snapshot_taken",
            context={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "hook_duration_ms": hook_duration_ms,
            },
        )


@dataclass(slots=True, frozen=True)
class _ParsedToolData:
    """Parsed tool data from PostToolUse hook input."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_error: str | None
    output_text: str
    result_raw: Any


def _parse_tool_data(input_data: PostToolUseHookInput) -> _ParsedToolData:
    """Parse tool data from PostToolUse hook input.

    Uses the SDK's PostToolUseHookInput TypedDict for type-safe access.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    tool_response = input_data.get("tool_response")

    # Extract error and output text from response
    tool_error: str | None = None
    output_text: str = ""

    if isinstance(tool_response, dict):
        stderr = tool_response.get("stderr")
        tool_error = stderr if stderr else None
        stdout = tool_response.get("stdout", "")
        output_text = stdout or str(tool_response)
    elif tool_response is not None:
        output_text = str(tool_response)

    return _ParsedToolData(
        tool_name=tool_name,
        tool_input=tool_input if isinstance(tool_input, dict) else {},
        tool_error=tool_error,
        output_text=output_text,
        result_raw=tool_response,
    )


def _handle_structured_output_completion(
    hook_context: HookContext,
    data: _ParsedToolData,
    task_completion_checker: TaskCompletionChecker | None,
    check_task_completion: Callable[[dict[str, Any]], TaskCompletionResult],
    stop_on_structured_output: bool,
) -> SyncHookJSONOutput | None:
    """Handle StructuredOutput task completion logic.

    Returns a hook response if stop/continue decision was made, None otherwise.
    """
    if data.tool_name != "StructuredOutput":
        return None

    # Only apply checker to main agent, not sub-agents
    if task_completion_checker is not None and not hook_context.stats.in_subagent:
        result = check_task_completion(data.tool_input)
        if not result.complete:
            logger.info(
                "claude_agent_sdk.hook.structured_output_incomplete",
                event="hook.structured_output_incomplete",
                context={
                    "feedback": result.feedback,
                    "elapsed_ms": hook_context.elapsed_ms,
                },
            )
            feedback_message = (
                result.feedback or "<blocker>Tasks are incomplete.</blocker>"
            )
            output: PostToolUseHookSpecificOutput = {
                "hookEventName": "PostToolUse",
                "additionalContext": feedback_message,
            }
            return {"continue_": True, "hookSpecificOutput": output}
        # Tasks complete - allow stop
        logger.debug(
            "claude_agent_sdk.hook.structured_output_complete",
            event="hook.structured_output_complete",
            context={
                "feedback": result.feedback,
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
            },
        )
        return {"continue_": False}

    if stop_on_structured_output:
        logger.debug(
            "claude_agent_sdk.hook.structured_output_stop",
            event="hook.structured_output_stop",
            context={
                "tool_name": data.tool_name,
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
            },
        )
        return {"continue_": False}

    return None


def _dispatch_tool_invoked_event(
    hook_context: HookContext, data: _ParsedToolData, tool_use_id: str | None
) -> None:
    """Dispatch ToolInvoked event for native tools."""
    event = ToolInvoked(
        prompt_name=hook_context.prompt_name,
        adapter=hook_context.adapter_name,
        name=data.tool_name,
        params=data.tool_input,
        result=data.result_raw,
        session_id=getattr(hook_context.session, "session_id", None),
        created_at=_utcnow(),
        usage=None,
        rendered_output=data.output_text[:1000] if data.output_text else "",
        call_id=tool_use_id,
        run_context=hook_context.run_context,
    )
    hook_context.session.dispatcher.dispatch(event)


def _handle_mcp_tool_post(
    hook_context: HookContext, data: _ParsedToolData
) -> SyncHookJSONOutput:
    """Handle post-processing for MCP tools.

    Runs feedback providers. MCP tools dispatch their own ToolInvoked events
    via the bridge, which also dequeues the tool_use_id.
    """
    result = _run_feedback_providers(hook_context, data)
    if result is not None:
        return result
    empty: SyncHookJSONOutput = {}
    return empty


def _run_feedback_providers(  # pragma: no cover - integration tested
    hook_context: HookContext, data: _ParsedToolData
) -> SyncHookJSONOutput | None:
    """Run feedback providers and return hook response if triggered."""
    feedback_text = collect_feedback(
        prompt=hook_context._prompt,
        session=hook_context.session,
        deadline=hook_context.deadline,
    )
    if feedback_text is not None:
        logger.debug(
            "claude_agent_sdk.hook.feedback_provided",
            event="hook.feedback_provided",
            context={
                "tool_name": data.tool_name,
                "feedback_length": len(feedback_text),
                "elapsed_ms": hook_context.elapsed_ms,
            },
        )
        output: PostToolUseHookSpecificOutput = {
            "hookEventName": "PostToolUse",
            "additionalContext": feedback_text,
        }
        return {"hookSpecificOutput": output}
    return None


def _handle_tool_transaction(
    hook_context: HookContext,
    data: _ParsedToolData,
    tool_use_id: str | None,
    hook_start: float,
) -> None:
    """Handle tool transaction - log completion and restore state on failure."""
    success = data.tool_error is None and not _is_tool_error_response(data.result_raw)
    logger.debug(
        "claude_agent_sdk.hook.tool_call.complete",
        event="hook.tool_call.complete",
        context={
            "tool_name": data.tool_name,
            "success": success,
            "call_id": tool_use_id,
            "tool_input": data.tool_input,
            "tool_response": data.result_raw,
            "output_text": data.output_text,
            "elapsed_ms": hook_context.elapsed_ms,
            "tool_count": hook_context.stats.tool_count,
            "turn_count": hook_context.stats.turn_count,
        },
    )
    if tool_use_id is not None:
        restored = hook_context.end_tool_execution(
            tool_use_id=tool_use_id, success=success
        )
        if restored:
            hook_duration_ms = int((time.monotonic() - hook_start) * 1000)
            logger.info(
                "claude_agent_sdk.hook.state_restored",
                event="hook.state_restored",
                context={
                    "tool_name": data.tool_name,
                    "tool_use_id": tool_use_id,
                    "reason": "tool_failure",
                    "hook_duration_ms": hook_duration_ms,
                    "elapsed_ms": hook_context.elapsed_ms,
                },
            )


def _is_tool_error_response(response: Any) -> bool:  # noqa: ANN401
    """Check if tool response indicates an error.

    Examines the response structure for error indicators like 'is_error' flags
    or error-related content patterns.
    """
    if not isinstance(response, dict):
        return False

    # Check explicit error flag
    if response.get("is_error") or response.get("isError"):
        return True

    # Check for error in content (Claude Agent SDK format)
    content = response.get("content")
    if isinstance(content, list) and content:
        first_item = content[0]
        if isinstance(first_item, dict):
            text = first_item.get("text", "")
            if isinstance(text, str):
                text_lower = text.lower()
                # Common error indicators in tool output
                if text_lower.startswith("error:") or text_lower.startswith("error -"):
                    return True

    return False
