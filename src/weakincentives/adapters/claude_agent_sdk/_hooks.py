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
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from ...budget import BudgetExceededError, BudgetTracker
from ...deadlines import Deadline
from ...runtime.events import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol
from ...tools.errors import DeadlineExceededError
from .._names import AdapterName

__all__ = [
    "HookCallback",
    "HookOutput",
    "build_post_tool_use_hook",
    "build_pre_tool_use_hook",
    "build_stop_hook",
    "build_user_prompt_submit_hook",
    "safe_hook_wrapper",
]


logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk.hooks"}
)

# Type aliases for hooks
HookOutput = dict[str, Any]
HookCallback = Callable[
    [dict[str, Any], str | None, dict[str, Any]],
    Awaitable[HookOutput],
]


def build_pre_tool_use_hook(
    *,
    session: SessionProtocol,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    adapter_name: AdapterName,
    prompt_name: str,
) -> HookCallback:
    """Build a PreToolUse hook that enforces constraints and injects state.

    Args:
        session: The session for state queries.
        deadline: Optional execution deadline.
        budget_tracker: Optional budget tracker.
        adapter_name: Name of the adapter for logging.
        prompt_name: Name of the prompt being evaluated.

    Returns:
        An async hook callback function.
    """

    async def pre_tool_use_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> HookOutput:
        """Intercept tool calls to enforce constraints and inject state."""
        del context  # Unused

        tool_name = input_data.get("tool_name", "")

        logger.debug(
            "pre_tool_use_hook.start",
            event="sdk.hook.pre_tool_use.start",
            context={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "prompt_name": prompt_name,
            },
        )

        # 1. Check deadline
        if deadline is not None and deadline.remaining() <= timedelta(0):
            logger.info(
                "pre_tool_use_hook.deadline_exceeded",
                event="sdk.hook.pre_tool_use.deadline_exceeded",
                context={"tool_name": tool_name, "prompt_name": prompt_name},
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Deadline exceeded",
                }
            }

        # 2. Check budget
        if budget_tracker is not None:
            try:
                budget_tracker.check()
            except BudgetExceededError as exc:
                logger.info(
                    "pre_tool_use_hook.budget_exceeded",
                    event="sdk.hook.pre_tool_use.budget_exceeded",
                    context={
                        "tool_name": tool_name,
                        "prompt_name": prompt_name,
                        "exceeded_dimension": exc.exceeded_dimension,
                    },
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Budget exhausted: {exc.exceeded_dimension}"
                        ),
                    }
                }

        # Allow tool execution
        return {}

    return pre_tool_use_hook


def build_post_tool_use_hook(
    *,
    session: SessionProtocol,
    adapter_name: AdapterName,
    prompt_name: str,
) -> HookCallback:
    """Build a PostToolUse hook that captures tool results and publishes events.

    Args:
        session: The session to publish events to.
        adapter_name: Name of the adapter for events.
        prompt_name: Name of the prompt being evaluated.

    Returns:
        An async hook callback function.
    """

    async def post_tool_use_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> HookOutput:
        """Capture tool results and publish to session."""
        del context  # Unused

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_output = input_data.get("tool_output", {})
        tool_error = input_data.get("tool_error")

        logger.debug(
            "post_tool_use_hook.record",
            event="sdk.hook.post_tool_use.record",
            context={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "success": tool_error is None,
                "prompt_name": prompt_name,
            },
        )

        # Extract message from tool output
        message = _extract_message(tool_output)
        if tool_error is not None:
            message = f"Error: {tool_error}"

        # Build and publish ToolInvoked event
        event = ToolInvoked(
            prompt_name=prompt_name,
            adapter=adapter_name,
            name=tool_name,
            params=tool_input,
            result=message,
            session_id=cast(UUID | None, getattr(session, "session_id", None)),
            created_at=datetime.now(UTC),
            usage=None,  # SDK doesn't provide per-tool usage
            value=tool_output if tool_error is None else None,
            rendered_output=str(tool_output) if tool_output else "",
            call_id=tool_use_id,
        )

        _ = session.event_bus.publish(event)

        return {}

    return post_tool_use_hook


def build_user_prompt_submit_hook(
    *,
    session: SessionProtocol,
    adapter_name: AdapterName,
    prompt_name: str,
) -> HookCallback:
    """Build a UserPromptSubmit hook that augments prompts with session context.

    Args:
        session: The session to query for context.
        adapter_name: Name of the adapter for logging.
        prompt_name: Name of the prompt being evaluated.

    Returns:
        An async hook callback function.
    """

    async def user_prompt_submit_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> HookOutput:
        """Augment user prompts with session context."""
        del tool_use_id, context  # Unused

        logger.debug(
            "user_prompt_submit_hook.start",
            event="sdk.hook.user_prompt_submit.start",
            context={"prompt_name": prompt_name},
        )

        # Currently we don't inject additional context
        # This hook is available for future extensions that might need to
        # query session state and inject it into prompts
        return {}

    return user_prompt_submit_hook


def build_stop_hook(
    *,
    session: SessionProtocol,
    adapter_name: AdapterName,
    prompt_name: str,
    on_stop: Callable[[str], None] | None = None,
) -> HookCallback:
    """Build a Stop hook that handles execution completion.

    Args:
        session: The session for state tracking.
        adapter_name: Name of the adapter for logging.
        prompt_name: Name of the prompt being evaluated.
        on_stop: Optional callback invoked with stop reason.

    Returns:
        An async hook callback function.
    """

    async def stop_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> HookOutput:
        """Handle execution completion."""
        del tool_use_id, context  # Unused

        stop_reason = input_data.get("stopReason", "end_turn")

        logger.debug(
            "stop_hook.complete",
            event="sdk.hook.stop.complete",
            context={
                "stop_reason": stop_reason,
                "prompt_name": prompt_name,
            },
        )

        if on_stop is not None:
            on_stop(stop_reason)

        return {}

    return stop_hook


async def safe_hook_wrapper(
    hook_fn: HookCallback,
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: dict[str, Any],
) -> HookOutput:
    """Wrap hook to catch exceptions and convert to responses.

    Args:
        hook_fn: The hook function to wrap.
        input_data: Hook input data.
        tool_use_id: Optional tool use identifier.
        context: Hook context.

    Returns:
        Hook output, converting exceptions to deny responses.
    """
    hook_event_name = input_data.get("hookEventName", "PreToolUse")

    try:
        return await hook_fn(input_data, tool_use_id, context)
    except DeadlineExceededError:
        logger.warning(
            "safe_hook_wrapper.deadline_exceeded",
            event="sdk.hook.deadline_exceeded",
            context={"hook_event_name": hook_event_name},
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": hook_event_name,
                "permissionDecision": "deny",
                "permissionDecisionReason": "Deadline exceeded",
            }
        }
    except BudgetExceededError as exc:
        logger.warning(
            "safe_hook_wrapper.budget_exceeded",
            event="sdk.hook.budget_exceeded",
            context={
                "hook_event_name": hook_event_name,
                "exceeded_dimension": exc.exceeded_dimension,
            },
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": hook_event_name,
                "permissionDecision": "deny",
                "permissionDecisionReason": "Budget exhausted",
            }
        }
    except Exception as exc:
        logger.exception(
            "safe_hook_wrapper.error",
            event="sdk.hook.error",
            context={
                "hook_event_name": hook_event_name,
                "error": str(exc),
            },
        )
        # Allow execution to continue on unknown errors
        return {}


def _extract_message(tool_output: object) -> str:
    """Extract a message string from tool output."""
    if tool_output is None:
        return ""
    if isinstance(tool_output, str):
        return tool_output
    if isinstance(tool_output, dict):
        # Cast to dict[str, object] for type safety
        output_dict = cast(dict[str, object], tool_output)
        # Check common message field names
        for key in ("message", "text", "content", "result"):
            value = output_dict.get(key)
            if isinstance(value, str):
                return value
        # Return string representation of the dict
        return str(tool_output)
    return str(tool_output)
