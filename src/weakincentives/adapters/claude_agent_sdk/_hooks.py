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

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...runtime.events._types import ToolInvoked
from ...runtime.execution_state import ExecutionState
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol
from ._notifications import Notification

__all__ = [
    "AsyncHookCallback",
    "HookCallback",
    "HookContext",
    "PostToolUseInput",
    "create_notification_hook",
    "create_post_tool_use_hook",
    "create_pre_compact_hook",
    "create_pre_tool_use_hook",
    "create_stop_hook",
    "create_subagent_start_hook",
    "create_subagent_stop_hook",
    "create_user_prompt_submit_hook",
    "safe_hook_wrapper",
]


@dataclass(slots=True, frozen=True)
class PostToolUseInput:
    """Typed representation of PostToolUse hook input.

    Mirrors the SDK's PostToolUseHookInput TypedDict but as a frozen dataclass
    for immutability and better type safety. The tool_response is kept as a raw
    value (dict or string) to avoid creating dataclasses that could be added
    to session state.
    """

    session_id: str
    tool_name: str
    tool_input: dict[str, Any]
    tool_response: dict[str, Any] | str
    cwd: str = ""
    transcript_path: str = ""
    permission_mode: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PostToolUseInput | None:
        """Parse a dict into PostToolUseInput, returning None if required fields missing."""
        if data is None or not isinstance(data, dict):
            return None
        # Check required fields
        if "tool_name" not in data:
            return None
        raw_response = data.get("tool_response")
        # Keep tool_response as raw dict or string
        if isinstance(raw_response, dict):
            response: dict[str, Any] | str = raw_response
        else:
            response = str(raw_response) if raw_response is not None else ""
        return cls(
            session_id=str(data.get("session_id", "")),
            tool_name=str(data.get("tool_name", "")),
            tool_input=data.get("tool_input", {})
            if isinstance(data.get("tool_input"), dict)
            else {},
            tool_response=response,
            cwd=str(data.get("cwd", "")),
            transcript_path=str(data.get("transcript_path", "")),
            permission_mode=data.get("permission_mode"),
        )


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
    """Context passed to hook callbacks for state access.

    The execution_state provides unified access to session and resources.
    Session is accessed via execution_state.session.
    """

    def __init__(
        self,
        *,
        execution_state: ExecutionState,
        adapter_name: str,
        prompt_name: str,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        self.execution_state = execution_state
        self.adapter_name = adapter_name
        self.prompt_name = prompt_name
        self.deadline = deadline
        self.budget_tracker = budget_tracker
        self.stop_reason: str | None = None
        self._tool_count = 0

    @property
    def session(self) -> SessionProtocol:
        """Get session from execution state."""
        return self.execution_state.session


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _read_transcript_file(path_str: str) -> list[dict[str, Any]]:
    """Read a JSONL transcript file and return its contents as a list.

    Args:
        path_str: Path to the transcript file (JSONL format).

    Returns:
        List of parsed JSON objects from the file.
        Returns empty list if file doesn't exist or can't be read.
    """
    if not path_str:
        return []

    path = Path(path_str).expanduser()
    if not path.exists():
        return []

    try:
        entries: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    else:
        return entries


def _expand_transcript_paths(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand transcript_path and agent_transcript_path in payload.

    Reads the JSONL files at transcript_path and agent_transcript_path
    and replaces the path strings with the actual transcript content.

    Args:
        payload: The hook input payload dict.

    Returns:
        A new payload dict with transcript paths expanded to content.
    """
    result = dict(payload)

    for key in ("transcript_path", "agent_transcript_path"):
        if key in result and isinstance(result[key], str):
            path_str = result[key]
            result[key] = _read_transcript_file(path_str)

    return result


def create_pre_tool_use_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PreToolUse hook for constraint enforcement and state snapshots.

    The hook checks deadlines and budgets before tool execution, blocking
    tools that would violate constraints. It also takes a state snapshot
    for transactional rollback if execution_state is configured.

    Args:
        hook_context: Context with session, deadline, budget, and execution_state.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_tool_use_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
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

        # Take snapshot for transactional rollback on native tools
        # Skip MCP-bridged WINK tools - they handle their own transactions
        if tool_use_id is not None and not tool_name.startswith("mcp__wink__"):
            hook_context.execution_state.begin_tool_execution(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
            )
            logger.debug(
                "claude_agent_sdk.hook.snapshot_taken",
                event="hook.snapshot_taken",
                context={"tool_name": tool_name, "tool_use_id": tool_use_id},
            )

        return {}

    return pre_tool_use_hook


@dataclass(slots=True, frozen=True)
class _ParsedToolData:
    """Parsed tool data from PostToolUse hook input."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_error: str | None
    output_text: str
    result_raw: Any


def _parse_tool_data(input_data: Any) -> _ParsedToolData:  # noqa: ANN401
    """Parse tool data from PostToolUse hook input.

    Attempts typed parsing first, falling back to dict access.
    """
    parsed = PostToolUseInput.from_dict(input_data)

    if parsed is not None:
        response = parsed.tool_response
        if isinstance(response, dict):
            tool_error = response.get("stderr") if response.get("stderr") else None
            output_text = response.get("stdout", "") or str(response)
        else:
            tool_error = None
            output_text = response
        return _ParsedToolData(
            tool_name=parsed.tool_name,
            tool_input=parsed.tool_input,
            tool_error=tool_error,
            output_text=output_text,
            result_raw=response,
        )

    # Fallback to dict access for malformed input
    tool_name = input_data.get("tool_name", "") if isinstance(input_data, dict) else ""
    tool_input = (
        input_data.get("tool_input", {}) if isinstance(input_data, dict) else {}
    )
    tool_response_raw = (
        input_data.get("tool_response", {}) if isinstance(input_data, dict) else {}
    )
    tool_error = (
        tool_response_raw.get("stderr")
        if isinstance(tool_response_raw, dict) and tool_response_raw.get("stderr")
        else None
    )
    if isinstance(tool_response_raw, dict):
        output_text = tool_response_raw.get("stdout", "") or str(tool_response_raw)
    elif tool_response_raw is not None:
        output_text = str(tool_response_raw)
    else:
        output_text = ""

    return _ParsedToolData(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_error=tool_error,
        output_text=output_text,
        result_raw=tool_response_raw,
    )


def create_post_tool_use_hook(
    hook_context: HookContext,
    *,
    stop_on_structured_output: bool = True,
) -> AsyncHookCallback:
    """Create a PostToolUse hook for tool result recording and state rollback.

    The hook publishes ToolInvoked events to the session bus. It attempts to
    parse the input data into typed dataclasses (PostToolUseInput, ToolResponse)
    for better type safety, falling back to dict access if parsing fails.

    If execution_state is configured and the tool failed, the hook restores
    state from the pre-execution snapshot.

    Args:
        hook_context: Context with session, adapter, and execution_state.
        stop_on_structured_output: If True, return ``continue: false`` after
            the StructuredOutput tool to end the turn immediately.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def post_tool_use_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = sdk_context
        data = _parse_tool_data(input_data)

        # Skip logging for MCP-bridged WINK tools - they publish their own
        # ToolInvoked events via BridgedTool with richer context (typed values).
        if data.tool_name.startswith("mcp__wink__"):
            return {}

        hook_context._tool_count += 1

        event = ToolInvoked(
            prompt_name=hook_context.prompt_name,
            adapter=hook_context.adapter_name,
            name=data.tool_name,
            params=data.tool_input,
            result=data.result_raw,
            session_id=None,
            created_at=_utcnow(),
            usage=None,
            rendered_output=data.output_text[:1000] if data.output_text else "",
            call_id=tool_use_id,
        )
        hook_context.session.dispatcher.dispatch(event)

        logger.debug(
            "claude_agent_sdk.hook.tool_invoked",
            event="hook.tool_invoked",
            context={
                "tool_name": data.tool_name,
                "success": data.tool_error is None,
                "call_id": tool_use_id,
            },
        )

        # Complete tool transaction - restore state on failure
        if tool_use_id is not None:
            success = data.tool_error is None and not _is_tool_error_response(
                data.result_raw
            )
            restored = hook_context.execution_state.end_tool_execution(
                tool_use_id=tool_use_id,
                success=success,
            )
            if restored:
                logger.info(
                    "claude_agent_sdk.hook.state_restored",
                    event="hook.state_restored",
                    context={
                        "tool_name": data.tool_name,
                        "tool_use_id": tool_use_id,
                        "reason": "tool_failure",
                    },
                )

        # Stop execution after StructuredOutput tool to end turn cleanly
        if stop_on_structured_output and data.tool_name == "StructuredOutput":
            logger.debug(
                "claude_agent_sdk.hook.structured_output_stop",
                event="hook.structured_output_stop",
                context={"tool_name": data.tool_name},
            )
            return {"continue": False}

        return {}

    return post_tool_use_hook


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


def create_subagent_start_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a SubagentStart hook to capture subagent launch events.

    Records Notification events when subagents are spawned during execution.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def subagent_start_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}

        notification = Notification(
            source="subagent_start",
            payload=payload,
            prompt_name=hook_context.prompt_name,
            adapter_name=hook_context.adapter_name,
            created_at=_utcnow(),
        )

        hook_context.session.dispatch(notification)

        logger.debug(
            "claude_agent_sdk.hook.subagent_start",
            event="hook.subagent_start",
            context={"payload": payload},
        )

        return {}

    return subagent_start_hook


def create_subagent_stop_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a SubagentStop hook to capture subagent completion events.

    Records Notification events when subagents complete execution.
    Automatically expands transcript_path and agent_transcript_path fields
    by reading the JSONL files and replacing the paths with their content.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def subagent_stop_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        raw_payload = input_data if isinstance(input_data, dict) else {}
        payload = _expand_transcript_paths(raw_payload)

        notification = Notification(
            source="subagent_stop",
            payload=payload,
            prompt_name=hook_context.prompt_name,
            adapter_name=hook_context.adapter_name,
            created_at=_utcnow(),
        )

        hook_context.session.dispatch(notification)

        logger.debug(
            "claude_agent_sdk.hook.subagent_stop",
            event="hook.subagent_stop",
            context={"payload": payload},
        )

        return {}

    return subagent_stop_hook


def create_pre_compact_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PreCompact hook to capture context compaction events.

    Records Notification events before the SDK compacts conversation context.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_compact_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}

        notification = Notification(
            source="pre_compact",
            payload=payload,
            prompt_name=hook_context.prompt_name,
            adapter_name=hook_context.adapter_name,
            created_at=_utcnow(),
        )

        hook_context.session.dispatch(notification)

        logger.debug(
            "claude_agent_sdk.hook.pre_compact",
            event="hook.pre_compact",
            context={"payload": payload},
        )

        return {}

    return pre_compact_hook


def create_notification_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a Notification hook to capture user-facing notifications.

    Records Notification events from the SDK's notification system.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def notification_hook(  # noqa: RUF029
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}

        notification = Notification(
            source="notification",
            payload=payload,
            prompt_name=hook_context.prompt_name,
            adapter_name=hook_context.adapter_name,
            created_at=_utcnow(),
        )

        hook_context.session.dispatch(notification)

        logger.debug(
            "claude_agent_sdk.hook.notification",
            event="hook.notification",
            context={"payload": payload},
        )

        return {}

    return notification_hook


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
