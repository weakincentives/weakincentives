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

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from claude_agent_sdk.types import (
    HookCallback,
    HookContext as SdkHookContext,
    HookInput,
    PostToolUseHookInput,
    PostToolUseHookSpecificOutput,
    PreCompactHookInput,
    PreToolUseHookInput,
    PreToolUseHookSpecificOutput,
    StopHookInput,
    SubagentStopHookInput,
    SyncHookJSONOutput,
    UserPromptSubmitHookInput,
)

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...filesystem import Filesystem
from ...prompt.feedback import collect_feedback
from ...prompt.protocols import PromptProtocol
from ...runtime.events.types import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.transactions import PendingToolTracker
from ...runtime.watchdog import Heartbeat
from ._bridge import MCPToolExecutionState
from ._task_completion import (
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)

if TYPE_CHECKING:
    from ...prompt.prompt import PromptResources

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


@dataclass(slots=True)
class HookStats:
    """Cumulative statistics tracked during hook execution.

    These metrics provide visibility into the execution flow for debugging.
    """

    tool_count: int = 0
    """Number of tools invoked during this execution."""

    turn_count: int = 0
    """Number of user prompt submissions (turns) during this execution."""

    subagent_count: int = 0
    """Number of subagents spawned during this execution."""

    in_subagent: bool = False
    """True when currently executing within a sub-agent context."""

    compact_count: int = 0
    """Number of context compaction events during this execution."""

    total_input_tokens: int = 0
    """Cumulative input tokens from all messages."""

    total_output_tokens: int = 0
    """Cumulative output tokens from all messages."""

    total_thinking_tokens: int = 0
    """Cumulative thinking tokens from extended thinking."""

    hook_errors: int = 0
    """Number of hook execution errors encountered."""


@dataclass(slots=True)
class HookConstraints:
    """Constraint configuration for hook execution.

    Groups optional deadline, budget, heartbeat, run context, and MCP tool state
    together to simplify HookContext construction.
    """

    deadline: Deadline | None = None
    """Optional deadline for constraint checking."""

    budget_tracker: BudgetTracker | None = None
    """Optional budget tracker for token limits."""

    heartbeat: Heartbeat | None = None
    """Optional heartbeat for liveness monitoring."""

    run_context: RunContext | None = None
    """Optional run context for tracing."""

    mcp_tool_state: MCPToolExecutionState | None = None
    """Shared state for passing tool_use_id from hooks to MCP bridge."""


class HookContext:
    """Context passed to hook callbacks for state access.

    Provides unified access to session, prompt resources, and tool transaction
    tracking for hook-based execution management.

    Note
    ----
    This is WINK's HookContext, distinct from the SDK's ``HookContext`` TypedDict
    which only contains a ``signal`` field for future abort support. WINK's
    HookContext provides richer functionality for session state management.
    """

    def __init__(
        self,
        *,
        session: SessionProtocol,
        prompt: PromptProtocol[object],
        adapter_name: str,
        prompt_name: str,
        constraints: HookConstraints | None = None,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self.adapter_name = adapter_name
        self.prompt_name = prompt_name
        # Unpack constraints or use defaults
        self.deadline = constraints.deadline if constraints else None
        self.budget_tracker = constraints.budget_tracker if constraints else None
        self.heartbeat = constraints.heartbeat if constraints else None
        self.run_context = constraints.run_context if constraints else None
        self.mcp_tool_state = constraints.mcp_tool_state if constraints else None
        self.stop_reason: str | None = None
        self._tool_count = 0
        self._tool_tracker: PendingToolTracker | None = None
        self.stats: HookStats = HookStats()
        self._start_time = time.monotonic()

    def beat(self) -> None:
        """Record a heartbeat to prove processing is active.

        Hooks should call this during native tool execution to extend
        the message visibility timeout. This is a no-op if heartbeat
        is not configured.
        """
        if self.heartbeat is not None:
            self.heartbeat.beat()

    @property
    def session(self) -> SessionProtocol:
        """Get session."""
        return self._session

    @property
    def resources(
        self,
    ) -> PromptResources:  # pragma: no cover - tested via integration
        """Get resources from prompt."""
        return self._prompt.resources

    @property
    def _tracker(self) -> PendingToolTracker:
        """Get or create tool tracker (lazy initialization)."""
        if self._tool_tracker is None:
            self._tool_tracker = PendingToolTracker(
                session=self._session,
                resources=self._prompt.resources.context,
            )
        return self._tool_tracker

    def begin_tool_execution(self, tool_use_id: str, tool_name: str) -> None:
        """Take snapshot before native tool execution."""
        self._tracker.begin_tool_execution(tool_use_id, tool_name)

    def end_tool_execution(self, tool_use_id: str, *, success: bool) -> bool:
        """Complete tool execution, restoring on failure."""
        return self._tracker.end_tool_execution(tool_use_id, success=success)

    def abort_tool_execution(
        self, tool_use_id: str
    ) -> bool:  # pragma: no cover - tested via integration
        """Abort tool execution and restore state."""
        return self._tracker.abort_tool_execution(tool_use_id)

    @property
    def elapsed_ms(self) -> int:
        """Return elapsed time in milliseconds since context creation."""
        return int((time.monotonic() - self._start_time) * 1000)


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
    tool_use_id: str | None,
    hook_start: float,
) -> None:
    """Set up tool execution state for MCP or native tools.

    For MCP tools: stores tool_use_id in mcp_tool_state for BridgedTool.
    For native tools: begins transactional execution with snapshot.
    """
    is_mcp_tool = tool_name.startswith("mcp__wink__")
    if is_mcp_tool and hook_context.mcp_tool_state and tool_use_id is not None:
        hook_context.mcp_tool_state.set_tool_use_id(tool_name, tool_use_id)
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
        hook_start = time.monotonic()
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
        _setup_tool_execution_state(hook_context, tool_name, tool_use_id, hook_start)

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

    Clears the tool_use_id from mcp_tool_state and runs feedback providers.
    MCP tools dispatch their own ToolInvoked events via the bridge.
    """
    if hook_context.mcp_tool_state:
        hook_context.mcp_tool_state.clear_tool_use_id(data.tool_name)
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
        hook_start = time.monotonic()

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
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     create_task_completion_stop_hook,
        ...     PlanBasedChecker,
        ... )
        >>>
        >>> checker = PlanBasedChecker()
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
