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

"""Hook implementations for Claude Agent SDK state synchronization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...runtime.events._types import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from ...runtime.session.protocols import SessionProtocol

__all__ = [
    "AsyncHookCallback",
    "HookCallback",
    "HookContext",
    "create_post_tool_use_hook",
    "create_pre_tool_use_hook",
    "create_stop_hook",
    "create_user_prompt_submit_hook",
    "safe_hook_wrapper",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "sdk_hooks"})

HookCallback = Callable[
    [dict[str, Any], str | None, "HookContext"],
    dict[str, Any],
]
"""Type alias for synchronous hook callbacks."""

AsyncHookCallback = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]
"""Type alias for async hook callbacks matching SDK signature."""


class HookContext:
    """Context passed to hook callbacks for state access."""

    def __init__(
        self,
        *,
        session: SessionProtocol,
        adapter_name: str,
        prompt_name: str,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        self.session = session
        self.adapter_name = adapter_name
        self.prompt_name = prompt_name
        self.deadline = deadline
        self.budget_tracker = budget_tracker
        self.stop_reason: str | None = None
        self._tool_count = 0


def _utcnow() -> datetime:
    return datetime.now(UTC)


def create_pre_tool_use_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PreToolUse hook for constraint enforcement.

    The hook checks deadlines and budgets before tool execution, blocking
    tools that would violate constraints.

    Args:
        hook_context: Context with session, deadline, and budget references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_tool_use_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        tool_name = (
            input_data.get("tool_name", "") if isinstance(input_data, dict) else ""
        )

        if (
            hook_context.deadline
            and hook_context.deadline.remaining().total_seconds() <= 0
        ):
            logger.warning(
                "claude_agent_sdk.hook.deadline_exceeded",
                event="hook.deadline_exceeded",
                context={"tool_name": tool_name},
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Deadline exceeded",
                }
            }

        budget_tracker = hook_context.budget_tracker
        if budget_tracker is not None and isinstance(budget_tracker, BudgetTracker):
            budget = budget_tracker.budget
            consumed = budget_tracker.consumed
            consumed_total = (consumed.input_tokens or 0) + (
                consumed.output_tokens or 0
            )
            if (
                budget.max_total_tokens is not None
                and consumed_total >= budget.max_total_tokens
            ):
                logger.warning(
                    "claude_agent_sdk.hook.budget_exhausted",
                    event="hook.budget_exhausted",
                    context={"tool_name": tool_name},
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "Token budget exhausted",
                    }
                }

        return {}

    return pre_tool_use_hook


def create_post_tool_use_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PostToolUse hook for tool result recording.

    The hook publishes ToolInvoked events to the session bus.

    Args:
        hook_context: Context with session and adapter references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def post_tool_use_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = sdk_context

        tool_name = (
            input_data.get("tool_name", "") if isinstance(input_data, dict) else ""
        )
        tool_input = (
            input_data.get("tool_input", {}) if isinstance(input_data, dict) else {}
        )
        tool_output = (
            input_data.get("tool_output", {}) if isinstance(input_data, dict) else {}
        )
        tool_error = (
            input_data.get("tool_error") if isinstance(input_data, dict) else None
        )

        hook_context._tool_count += 1

        output_text = ""
        if isinstance(tool_output, dict):
            output_text = tool_output.get("content", str(tool_output))
        elif tool_output is not None:
            output_text = str(tool_output)

        event = ToolInvoked(
            prompt_name=hook_context.prompt_name,
            adapter=hook_context.adapter_name,
            name=tool_name,
            params=tool_input,
            result=tool_output,
            session_id=None,
            created_at=_utcnow(),
            usage=None,
            value=None,
            rendered_output=output_text[:1000] if output_text else "",
            call_id=tool_use_id,
        )
        hook_context.session.event_bus.publish(event)

        logger.debug(
            "claude_agent_sdk.hook.tool_invoked",
            event="hook.tool_invoked",
            context={
                "tool_name": tool_name,
                "success": tool_error is None,
                "call_id": tool_use_id,
            },
        )

        return {}

    return post_tool_use_hook


def create_user_prompt_submit_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a UserPromptSubmit hook for context injection.

    Currently a no-op placeholder for future session context injection.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def user_prompt_submit_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = input_data
        _ = tool_use_id
        _ = sdk_context
        _ = hook_context
        return {}

    return user_prompt_submit_hook


def create_stop_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a Stop hook for execution finalization.

    Records the stop reason for later use in result construction.

    Args:
        hook_context: Context to record stop reason.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def stop_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        stop_reason = (
            input_data.get("stopReason", "end_turn")
            if isinstance(input_data, dict)
            else "end_turn"
        )
        hook_context.stop_reason = stop_reason

        logger.debug(
            "claude_agent_sdk.hook.stop",
            event="hook.stop",
            context={"stop_reason": stop_reason},
        )

        return {}

    return stop_hook


def safe_hook_wrapper(
    hook_fn: HookCallback,
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Wrap a hook to catch exceptions and convert to responses.

    Prevents hook errors from crashing the SDK execution by catching
    exceptions and returning appropriate denial responses for constraint
    violations.

    Args:
        hook_fn: The hook function to wrap.
        input_data: Data passed to the hook.
        tool_use_id: Optional tool use identifier.
        context: Hook context.

    Returns:
        Hook response dict, potentially with denial for errors.
    """
    try:
        return hook_fn(input_data, tool_use_id, context)
    except Exception as error:
        error_name = type(error).__name__

        if error_name in {"DeadlineExceededError", "DeadlineExpired"}:
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data.get("hookEventName", "PreToolUse"),
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Deadline exceeded",
                }
            }

        if error_name in {"BudgetExhaustedError", "BudgetExceeded"}:
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data.get("hookEventName", "PreToolUse"),
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Budget exhausted",
                }
            }

        logger.exception(
            "claude_agent_sdk.hook.error",
            event="hook.error",
            context={"error": str(error), "error_type": error_name},
        )

        return {}
