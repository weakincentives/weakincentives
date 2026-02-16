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

"""Hook implementations for Claude Agent SDK state synchronization.

This module provides hook callbacks that integrate WINK's session state management
with the Claude Agent SDK's execution model. Hooks use SDK native types for
type-safe integration.

SDK Hook Types Used
-------------------
- ``PreToolUseHookInput``: Input for PreToolUse hooks
- ``PostToolUseHookInput``: Input for PostToolUse hooks
- ``StopHookInput``: Input for Stop hooks
- ``UserPromptSubmitHookInput``: Input for UserPromptSubmit hooks
- ``SubagentStopHookInput``: Input for SubagentStop hooks
- ``PreCompactHookInput``: Input for PreCompact hooks
- ``HookContext``: SDK hook execution context
- ``SyncHookJSONOutput``: Synchronous hook return type
- ``PreToolUseHookSpecificOutput``: PreToolUse-specific output fields
- ``PostToolUseHookSpecificOutput``: PostToolUse-specific output fields

Note
----
The SDK does not support SessionStart, SessionEnd, SubagentStart, or Notification
hooks in the Python SDK. Only the hook types listed above are available.
"""

from __future__ import annotations

from typing import Any, cast

from claude_agent_sdk.types import (
    HookCallback,
    HookContext as SdkHookContext,
    HookInput,
    PostToolUseHookInput,
    PreCompactHookInput,
    PreToolUseHookInput,
    PreToolUseHookSpecificOutput,
    StopHookInput,
    SubagentStopHookInput,
    SyncHookJSONOutput,
    UserPromptSubmitHookInput,
)

from ...filesystem import Filesystem
from ...runtime.logging import StructuredLogger, get_logger
from ._hook_context import HookConstraints, HookContext, HookStats
from ._hook_tools import (
    _check_budget_constraint,
    _check_deadline_constraint,
    _compute_budget_info,
    _dispatch_tool_invoked_event,
    _handle_mcp_tool_post,
    _handle_structured_output_completion,
    _handle_tool_transaction,
    _is_budget_exhausted,
    _is_deadline_exceeded,
    _is_tool_error_response,  # noqa: F401 - re-exported for tests
    _parse_tool_data,
    _run_feedback_providers,
    _setup_tool_execution_state,
)
from ._task_completion import (
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)

__all__ = [
    "HookCallback",
    "HookConstraints",
    "HookContext",
    "HookStats",
    "create_post_tool_use_hook",
    "create_pre_compact_hook",
    "create_pre_tool_use_hook",
    "create_stop_hook",
    "create_subagent_stop_hook",
    "create_task_completion_stop_hook",
    "create_user_prompt_submit_hook",
    "safe_hook_wrapper",
]


logger: StructuredLogger = get_logger(__name__, context={"component": "sdk_hooks"})


def create_pre_tool_use_hook(
    hook_context: HookContext,
) -> HookCallback:
    """Create a PreToolUse hook for constraint enforcement and state snapshots.

    The hook checks deadlines and budgets before tool execution, blocking
    tools that would violate constraints.

    Args:
        hook_context: Context with session, deadline, budget, and prompt.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_tool_use_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = sdk_context
        hook_start = hook_context.clock.monotonic()
        hook_context.beat()

        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "PreToolUse"
        ):
            return {}

        pre_input: PreToolUseHookInput = input_data  # type: ignore[assignment]
        tool_name = pre_input.get("tool_name", "")

        # Log with constraint status
        deadline_remaining_ms: int | None = None
        if hook_context.deadline:
            deadline_remaining_ms = int(
                hook_context.deadline.remaining().total_seconds() * 1000
            )
        logger.debug(
            "claude_agent_sdk.hook.pre_tool_use",
            event="hook.pre_tool_use",
            context={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_input": pre_input.get("tool_input", {}),
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
                "deadline_remaining_ms": deadline_remaining_ms,
                **_compute_budget_info(hook_context.budget_tracker),
            },
        )

        # Check constraints - return early if violated
        if (deny := _check_deadline_constraint(hook_context, tool_name)) is not None:
            return deny
        if (deny := _check_budget_constraint(hook_context, tool_name)) is not None:
            return deny

        # Set subagent flag when Task tool is called (subagent spawn)
        # This flag is cleared by SubagentStop hook when subagent completes
        if tool_name == "Task":
            hook_context.stats.in_subagent = True

        # Set up tool execution state (MCP tool_use_id or native snapshot)
        tool_input = pre_input.get("tool_input", {})
        _setup_tool_execution_state(
            hook_context, tool_name, tool_input, tool_use_id, hook_start
        )

        return {}

    return pre_tool_use_hook


def create_post_tool_use_hook(
    hook_context: HookContext,
    *,
    stop_on_structured_output: bool = True,
    task_completion_checker: TaskCompletionChecker | None = None,
) -> HookCallback:
    """Create a PostToolUse hook for tool result recording and state rollback.

    The hook dispatches ToolInvoked events to the session dispatcher using the
    SDK's native PostToolUseHookInput TypedDict for type-safe access.

    If the tool failed, the hook restores state from the pre-execution snapshot.

    When ``task_completion_checker`` is provided and StructuredOutput is called,
    task completion is verified. If incomplete, returns an error result to the
    model with feedback on remaining tasks. If complete, signals to stop.

    Args:
        hook_context: Context with session, adapter, and prompt.
        stop_on_structured_output: If True and no checker configured, return
            ``continue: false`` after StructuredOutput to end the turn.
        task_completion_checker: Optional checker for verifying task completion
            when StructuredOutput is called. When provided and tasks are
            incomplete, returns error result with feedback.

    Returns:
        An async hook callback function matching SDK signature.
    """

    def _get_filesystem() -> Filesystem | None:
        try:
            return hook_context.resources.get(Filesystem)
        except (LookupError, AttributeError):
            return None

    def _check_task_completion(tool_input: dict[str, Any]) -> TaskCompletionResult:
        context = TaskCompletionContext(
            session=hook_context.session,
            tentative_output=tool_input.get("output"),
            stop_reason="structured_output",
            filesystem=_get_filesystem(),
        )
        return task_completion_checker.check(context)  # type: ignore[union-attr]

    async def post_tool_use_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = sdk_context
        hook_start = hook_context.clock.monotonic()

        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "PostToolUse"
        ):
            return {}

        post_input: PostToolUseHookInput = input_data  # type: ignore[assignment]
        data = _parse_tool_data(post_input)
        hook_context.beat()
        hook_context._tool_count += 1
        hook_context.stats.tool_count += 1

        # MCP tools dispatch their own events - just clear state and run feedback
        if data.tool_name.startswith("mcp__wink__"):
            return _handle_mcp_tool_post(hook_context, data)

        # Native tools: dispatch event, handle transaction, check completion
        _dispatch_tool_invoked_event(hook_context, data, tool_use_id)
        _handle_tool_transaction(hook_context, data, tool_use_id, hook_start)

        # Handle StructuredOutput completion
        completion_response = _handle_structured_output_completion(
            hook_context,
            data,
            task_completion_checker,
            _check_task_completion,
            stop_on_structured_output,
        )
        if completion_response is not None:
            return completion_response

        feedback_result = _run_feedback_providers(hook_context, data)
        return feedback_result if feedback_result is not None else {}

    return post_tool_use_hook


def create_user_prompt_submit_hook(
    hook_context: HookContext,
) -> HookCallback:
    """Create a UserPromptSubmit hook for turn boundary tracking.

    Logs turn start events and tracks turn count for debugging multi-turn
    conversations. Each prompt submission represents the start of a new turn.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def user_prompt_submit_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = tool_use_id
        _ = sdk_context

        hook_context.stats.turn_count += 1

        # Type narrow to UserPromptSubmitHookInput
        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "UserPromptSubmit"
        ):
            return {}

        prompt_input: UserPromptSubmitHookInput = input_data  # type: ignore[assignment]
        prompt_content = prompt_input.get("prompt", "")
        session_id = prompt_input.get("session_id", "")

        # Calculate prompt preview (truncate for logging)
        prompt_preview = prompt_content[:200] if prompt_content else ""

        logger.debug(
            "claude_agent_sdk.hook.turn_start",
            event="hook.turn_start",
            context={
                "turn_number": hook_context.stats.turn_count,
                "session_id": session_id,
                "prompt_preview": prompt_preview,
                "prompt_length": len(prompt_content),
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
                "cumulative_input_tokens": hook_context.stats.total_input_tokens,
                "cumulative_output_tokens": hook_context.stats.total_output_tokens,
            },
        )

        return {}

    return user_prompt_submit_hook


def create_stop_hook(
    hook_context: HookContext,
) -> HookCallback:
    """Create a Stop hook for execution finalization.

    Records the stop reason and logs final execution statistics for debugging.

    Args:
        hook_context: Context to record stop reason.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def stop_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = tool_use_id
        _ = sdk_context

        # Type narrow to StopHookInput
        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "Stop"
        ):
            return {}

        stop_input: StopHookInput = input_data  # type: ignore[assignment]
        stop_reason = "end_turn"  # StopHookInput doesn't have stopReason field
        hook_context.stop_reason = stop_reason

        logger.debug(
            "claude_agent_sdk.hook.stop",
            event="hook.stop",
            context={
                "stop_reason": stop_reason,
                "stop_hook_active": stop_input.get("stop_hook_active", False),
                "elapsed_ms": hook_context.elapsed_ms,
                "stats_tool_count": hook_context.stats.tool_count,
                "stats_turn_count": hook_context.stats.turn_count,
                "stats_subagent_count": hook_context.stats.subagent_count,
                "stats_compact_count": hook_context.stats.compact_count,
                "stats_input_tokens": hook_context.stats.total_input_tokens,
                "stats_output_tokens": hook_context.stats.total_output_tokens,
                "stats_thinking_tokens": hook_context.stats.total_thinking_tokens,
                "stats_hook_errors": hook_context.stats.hook_errors,
            },
        )

        return {}

    return stop_hook


def _should_skip_task_completion(hook_context: HookContext, stop_reason: str) -> bool:
    """Check if task completion should be skipped due to constraints."""
    if _is_deadline_exceeded(hook_context.deadline):
        logger.debug(
            "claude_agent_sdk.hook.task_completion_stop.deadline_exceeded",
            event="hook.task_completion_stop.deadline_exceeded",
            context={"stop_reason": stop_reason},
        )
        return True
    if _is_budget_exhausted(hook_context.budget_tracker):
        logger.debug(
            "claude_agent_sdk.hook.task_completion_stop.budget_exhausted",
            event="hook.task_completion_stop.budget_exhausted",
            context={"stop_reason": stop_reason},
        )
        return True
    return False


def create_task_completion_stop_hook(
    hook_context: HookContext,
    *,
    checker: TaskCompletionChecker,
) -> HookCallback:
    """Create a Stop hook that checks task completion before allowing stop.

    This hook uses the provided TaskCompletionChecker to verify that all tasks
    are complete before allowing the agent to stop. When tasks are incomplete,
    it returns a signal to continue execution with feedback.

    Args:
        hook_context: Context with session for state access.
        checker: Task completion checker for verifying completion status.

    Returns:
        An async hook callback that enforces task completion.

    Example:
        >>> from weakincentives.prompt import FileOutputChecker
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     create_task_completion_stop_hook,
        ... )
        >>>
        >>> checker = FileOutputChecker(files=("output.txt",))
        >>> hook = create_task_completion_stop_hook(hook_context, checker=checker)
    """

    def _get_filesystem() -> Filesystem | None:
        try:
            return hook_context.resources.get(Filesystem)
        except (LookupError, AttributeError):
            return None

    async def task_completion_stop_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = tool_use_id
        _ = sdk_context

        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "Stop"
        ):
            return {}

        stop_reason = "end_turn"
        hook_context.stop_reason = stop_reason

        # Skip if constraints exceeded - can't do more work
        if _should_skip_task_completion(hook_context, stop_reason):
            return {}

        # Check task completion
        context = TaskCompletionContext(
            session=hook_context.session,
            tentative_output=None,
            stop_reason=stop_reason,
            filesystem=_get_filesystem(),
        )
        result = checker.check(context)

        if result.complete:
            logger.debug(
                "claude_agent_sdk.hook.task_completion_stop.allow",
                event="hook.task_completion_stop.allow",
                context={"stop_reason": stop_reason, "feedback": result.feedback},
            )
            return {}

        logger.info(
            "claude_agent_sdk.hook.task_completion_stop.incomplete",
            event="hook.task_completion_stop.incomplete",
            context={"stop_reason": stop_reason, "feedback": result.feedback},
        )
        return {"continue_": True, "reason": result.feedback or "Tasks are incomplete"}

    return task_completion_stop_hook


def create_subagent_stop_hook(
    hook_context: HookContext,
) -> HookCallback:
    """Create a SubagentStop hook to capture subagent completion events.

    Logs subagent completion details and clears the in_subagent flag.

    Note
    ----
    The SDK does not support SubagentStart hooks. Subagent tracking is done
    via SubagentStop only, which clears the in_subagent flag.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def subagent_stop_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = tool_use_id
        _ = sdk_context

        # Increment subagent count and clear the sub-agent flag
        hook_context.stats.subagent_count += 1
        hook_context.stats.in_subagent = False

        # Type narrow to SubagentStopHookInput
        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "SubagentStop"
        ):
            return {}

        subagent_input: SubagentStopHookInput = input_data  # type: ignore[assignment]

        logger.debug(
            "claude_agent_sdk.hook.subagent_stop",
            event="hook.subagent_stop",
            context={
                "session_id": subagent_input.get("session_id", ""),
                "stop_hook_active": subagent_input.get("stop_hook_active", False),
                "elapsed_ms": hook_context.elapsed_ms,
                "parent_tool_count": hook_context.stats.tool_count,
                "subagent_count": hook_context.stats.subagent_count,
            },
        )

        return {}

    return subagent_stop_hook


def create_pre_compact_hook(
    hook_context: HookContext,
) -> HookCallback:
    """Create a PreCompact hook to capture context compaction events.

    Tracks context window utilization for debugging memory-constrained scenarios
    before the SDK compacts conversation context.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_compact_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        sdk_context: SdkHookContext,
    ) -> SyncHookJSONOutput:
        _ = tool_use_id
        _ = sdk_context

        hook_context.stats.compact_count += 1

        # Type narrow to PreCompactHookInput
        if (
            not isinstance(input_data, dict)
            or input_data.get("hook_event_name") != "PreCompact"
        ):
            return {}

        compact_input: PreCompactHookInput = input_data  # type: ignore[assignment]

        logger.debug(
            "claude_agent_sdk.hook.pre_compact",
            event="hook.pre_compact",
            context={
                "compact_number": hook_context.stats.compact_count,
                "trigger": compact_input.get("trigger", ""),
                "custom_instructions": compact_input.get("custom_instructions"),
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
                "turn_count": hook_context.stats.turn_count,
            },
        )

        return {}

    return pre_compact_hook


def safe_hook_wrapper(
    hook_fn: HookCallback,
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """Wrap a hook to catch exceptions and convert to responses.

    Prevents hook errors from crashing the SDK execution by catching
    exceptions and returning appropriate denial responses for constraint
    violations. Tracks error statistics for debugging.

    Args:
        hook_fn: The hook function to wrap.
        input_data: Data passed to the hook.
        tool_use_id: Optional tool use identifier.
        context: Hook context.

    Returns:
        Hook response dict, potentially with denial for errors.
    """
    # Note: This function wraps async hooks but runs synchronously.
    # It's used for error handling at the boundary, not for actual hook execution.
    # The SDK handles async execution of hooks internally.
    try:
        import asyncio

        return cast(
            SyncHookJSONOutput,
            asyncio.get_event_loop().run_until_complete(
                hook_fn(input_data, tool_use_id, {"signal": None})
            ),
        )
    except Exception as error:
        error_name = type(error).__name__
        context.stats.hook_errors += 1

        if error_name in {"DeadlineExceededError", "DeadlineExpired"}:
            logger.debug(
                "claude_agent_sdk.hook.deadline_error_caught",
                event="hook.deadline_error_caught",
                context={
                    "error_type": error_name,
                    "hook_errors": context.stats.hook_errors,
                    "elapsed_ms": context.elapsed_ms,
                },
            )
            output: PreToolUseHookSpecificOutput = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Deadline exceeded",
            }
            return {"hookSpecificOutput": output}

        if error_name in {"BudgetExhaustedError", "BudgetExceeded"}:
            logger.debug(
                "claude_agent_sdk.hook.budget_error_caught",
                event="hook.budget_error_caught",
                context={
                    "error_type": error_name,
                    "hook_errors": context.stats.hook_errors,
                    "elapsed_ms": context.elapsed_ms,
                },
            )
            output_budget: PreToolUseHookSpecificOutput = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Budget exhausted",
            }
            return {"hookSpecificOutput": output_budget}

        logger.exception(
            "claude_agent_sdk.hook.error",
            event="hook.error",
            context={
                "error": str(error),
                "error_type": error_name,
                "hook_errors": context.stats.hook_errors,
                "elapsed_ms": context.elapsed_ms,
                "tool_count": context.stats.tool_count,
            },
        )

        return {}
