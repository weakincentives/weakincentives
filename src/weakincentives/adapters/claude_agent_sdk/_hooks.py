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

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
from ._task_completion import (
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)

if TYPE_CHECKING:
    from ...prompt.prompt import PromptResources

__all__ = [
    "AsyncHookCallback",
    "HookCallback",
    "HookContext",
    "HookStats",
    "PostToolUseInput",
    "create_notification_hook",
    "create_post_tool_use_hook",
    "create_pre_compact_hook",
    "create_pre_tool_use_hook",
    "create_stop_hook",
    "create_subagent_start_hook",
    "create_subagent_stop_hook",
    "create_task_completion_stop_hook",
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


class HookContext:
    """Context passed to hook callbacks for state access.

    Provides unified access to session, prompt resources, and tool transaction
    tracking for hook-based execution management.
    """

    def __init__(  # noqa: PLR0913 - context objects often need many parameters
        self,
        *,
        session: SessionProtocol,
        prompt: PromptProtocol[object],
        adapter_name: str,
        prompt_name: str,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self.adapter_name = adapter_name
        self.prompt_name = prompt_name
        self.deadline = deadline
        self.budget_tracker = budget_tracker
        self.heartbeat = heartbeat
        self.run_context = run_context
        self.stop_reason: str | None = None
        self._tool_count = 0
        self._tool_tracker: PendingToolTracker | None = None
        self.stats: HookStats = HookStats()
        self._start_time = time.monotonic()
        self._pending_call_ids: dict[str, str | None] = {}
        self._pending_tool_start_times: dict[str, float] = {}

    def set_pending_call_id(self, tool_name: str, call_id: str | None) -> None:
        """Store pending call_id for a tool (used by MCP-bridged tools)."""
        self._pending_call_ids[tool_name] = call_id

    def get_pending_call_id(self, tool_name: str) -> str | None:
        """Get pending call_id for a tool (used by BridgedTool)."""
        return self._pending_call_ids.get(tool_name)

    def clear_pending_call_id(self, tool_name: str) -> None:
        """Clear pending call_id for a tool after use."""
        self._pending_call_ids.pop(tool_name, None)

    def set_tool_start_time(self, tool_use_id: str) -> None:
        """Record start time for a native tool execution."""
        self._pending_tool_start_times[tool_use_id] = time.monotonic()

    def get_tool_duration_ms(self, tool_use_id: str) -> float | None:
        """Get duration in ms since tool execution started and clear the entry."""
        start_time = self._pending_tool_start_times.pop(tool_use_id, None)
        if start_time is None:
            return None
        return (time.monotonic() - start_time) * 1000

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
    """Compute budget information for logging."""
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


def _is_budget_exhausted(budget_tracker: BudgetTracker | None) -> bool:
    """Check if the token budget is exhausted."""
    if budget_tracker is None or not isinstance(budget_tracker, BudgetTracker):
        return False
    budget = budget_tracker.budget
    consumed = budget_tracker.consumed
    consumed_total = (consumed.input_tokens or 0) + (consumed.output_tokens or 0)
    return (
        budget.max_total_tokens is not None
        and consumed_total >= budget.max_total_tokens
    )


def create_pre_tool_use_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PreToolUse hook for constraint enforcement and state snapshots.

    The hook checks deadlines and budgets before tool execution, blocking
    tools that would violate constraints.

    Args:
        hook_context: Context with session, deadline, budget, and prompt.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_tool_use_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = sdk_context
        hook_start = time.monotonic()

        # Beat before tool execution to prove liveness
        hook_context.beat()

        tool_name = (
            input_data.get("tool_name", "") if isinstance(input_data, dict) else ""
        )

        # Compute constraint status for logging
        deadline_remaining_ms: int | None = None
        if hook_context.deadline:
            deadline_remaining_ms = int(
                hook_context.deadline.remaining().total_seconds() * 1000
            )

        budget_info = _compute_budget_info(hook_context.budget_tracker)

        logger.debug(
            "claude_agent_sdk.hook.pre_tool_use",
            event="hook.pre_tool_use",
            context={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "input_data": input_data if isinstance(input_data, dict) else {},
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
                "deadline_remaining_ms": deadline_remaining_ms,
                **budget_info,
            },
        )

        if (
            hook_context.deadline
            and hook_context.deadline.remaining().total_seconds() <= 0
        ):
            logger.warning(
                "claude_agent_sdk.hook.deadline_exceeded",
                event="hook.deadline_exceeded",
                context={
                    "tool_name": tool_name,
                    "elapsed_ms": hook_context.elapsed_ms,
                    "tool_count": hook_context.stats.tool_count,
                },
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Deadline exceeded",
                }
            }

        if _is_budget_exhausted(hook_context.budget_tracker):
            logger.warning(
                "claude_agent_sdk.hook.budget_exhausted",
                event="hook.budget_exhausted",
                context={
                    "tool_name": tool_name,
                    "elapsed_ms": hook_context.elapsed_ms,
                    **budget_info,
                },
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
            hook_context.begin_tool_execution(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
            )
            # Record start time for duration tracking in post_tool_use_hook
            hook_context.set_tool_start_time(tool_use_id)
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

        # Store tool_use_id for MCP-bridged tools so BridgedTool can include it
        # in ToolInvoked events. Cleared in post_tool_use_hook after dispatch.
        # Strip prefix to get the original tool name that BridgedTool uses.
        if tool_name.startswith("mcp__wink__"):
            original_name = tool_name[len("mcp__wink__") :]
            hook_context.set_pending_call_id(original_name, tool_use_id)

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


def create_post_tool_use_hook(  # noqa: C901 - complexity needed for task completion
    hook_context: HookContext,
    *,
    stop_on_structured_output: bool = True,
    task_completion_checker: TaskCompletionChecker | None = None,
) -> AsyncHookCallback:
    """Create a PostToolUse hook for tool result recording and state rollback.

    The hook dispatches ToolInvoked events to the session dispatcher. It attempts to
    parse the input data into typed dataclasses (PostToolUseInput, ToolResponse)
    for better type safety, falling back to dict access if parsing fails.

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
        """Get filesystem from hook context resources if available."""
        try:
            return hook_context.resources.get(Filesystem)
        except (LookupError, AttributeError):
            return None

    def _check_task_completion(
        tool_input: dict[str, Any],
    ) -> TaskCompletionResult:
        """Check task completion using the configured checker."""
        context = TaskCompletionContext(
            session=hook_context.session,
            tentative_output=tool_input.get("output"),
            stop_reason="structured_output",
            filesystem=_get_filesystem(),
        )
        return task_completion_checker.check(context)  # type: ignore[union-attr]

    def _run_feedback_providers(  # pragma: no cover - integration tested
        data: _ParsedToolData,
    ) -> dict[str, Any] | None:
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
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": feedback_text,
                }
            }
        return None

    async def post_tool_use_hook(  # noqa: C901 - observer dispatch
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = sdk_context
        hook_start = time.monotonic()
        data = _parse_tool_data(input_data)

        # Beat after tool execution to prove liveness
        hook_context.beat()

        # Increment tool count for ALL tools (needed for feedback triggers)
        hook_context._tool_count += 1
        hook_context.stats.tool_count += 1

        # MCP-bridged WINK tools dispatch their own ToolInvoked events via
        # BridgedTool with richer context (typed values). By the time this hook
        # runs, BridgedTool has already dispatched the event, so tool_call_count
        # in FeedbackContext is accurate. Run feedback providers and return.
        if data.tool_name.startswith("mcp__wink__"):
            # Clear the tool_use_id stored for BridgedTool
            original_name = data.tool_name[len("mcp__wink__") :]
            hook_context.clear_pending_call_id(original_name)
            feedback_response = _run_feedback_providers(data)
            return feedback_response if feedback_response else {}

        # For native tools, dispatch ToolInvoked BEFORE running feedback providers
        # so that FeedbackContext.tool_call_count includes this tool call.
        # Get duration from start time recorded in pre_tool_use_hook.
        duration_ms: float | None = None
        if tool_use_id is not None:
            duration_ms = hook_context.get_tool_duration_ms(tool_use_id)

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
            duration_ms=duration_ms,
        )
        hook_context.session.dispatcher.dispatch(event)

        # Determine success status
        success = data.tool_error is None and not _is_tool_error_response(
            data.result_raw
        )

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

        # Complete tool transaction - restore state on failure
        if tool_use_id is not None:
            restored = hook_context.end_tool_execution(
                tool_use_id=tool_use_id,
                success=success,
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

        # Handle StructuredOutput: check task completion BEFORE feedback providers.
        # This ensures completion logic (continue: false) always runs and final
        # outputs aren't ignored when a feedback trigger fires.
        if data.tool_name == "StructuredOutput":
            if task_completion_checker is not None:
                result = _check_task_completion(data.tool_input)
                if not result.complete:
                    # Tasks incomplete - provide feedback via additionalContext
                    # Don't return continue: False - let model continue working.
                    # When model calls StructuredOutput again after completing tasks,
                    # we'll return continue: False and the SDK will use that output.
                    logger.info(
                        "claude_agent_sdk.hook.structured_output_incomplete",
                        event="hook.structured_output_incomplete",
                        context={
                            "feedback": result.feedback,
                            "elapsed_ms": hook_context.elapsed_ms,
                        },
                    )
                    feedback_message = (
                        f"Tasks incomplete: {result.feedback}. "
                        "Please complete the remaining tasks, then call "
                        "StructuredOutput again with your final output."
                    )
                    return {
                        "continue_": True,
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": feedback_message,
                        },
                    }
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
                return {"continue": False}
            if stop_on_structured_output:
                # No checker - stop immediately
                logger.debug(
                    "claude_agent_sdk.hook.structured_output_stop",
                    event="hook.structured_output_stop",
                    context={
                        "tool_name": data.tool_name,
                        "elapsed_ms": hook_context.elapsed_ms,
                        "tool_count": hook_context.stats.tool_count,
                    },
                )
                return {"continue": False}

        # Run feedback providers AFTER ToolInvoked dispatch so tool_call_count
        # includes this tool. Return feedback response if triggered.
        feedback_response = _run_feedback_providers(data)
        if feedback_response is not None:
            return feedback_response

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
    """Create a UserPromptSubmit hook for turn boundary tracking.

    Logs turn start events and tracks turn count for debugging multi-turn
    conversations. Each prompt submission represents the start of a new turn.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def user_prompt_submit_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        hook_context.stats.turn_count += 1

        # Extract prompt info from input
        payload = input_data if isinstance(input_data, dict) else {}
        prompt_content = payload.get("prompt", "")
        session_id = payload.get("session_id", "")

        # Calculate prompt preview (truncate for logging)
        prompt_preview = ""
        if isinstance(prompt_content, str):
            prompt_preview = prompt_content[:200] if prompt_content else ""
        elif isinstance(prompt_content, dict):  # pragma: no cover
            content = prompt_content.get("content", "")
            if isinstance(content, str):
                prompt_preview = content[:200]

        logger.debug(
            "claude_agent_sdk.hook.turn_start",
            event="hook.turn_start",
            context={
                "turn_number": hook_context.stats.turn_count,
                "session_id": session_id,
                "prompt_preview": prompt_preview,
                "prompt_length": (
                    len(prompt_content)
                    if isinstance(prompt_content, str)
                    else len(str(prompt_content))
                ),
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
) -> AsyncHookCallback:
    """Create a Stop hook for execution finalization.

    Records the stop reason and logs final execution statistics for debugging.

    Args:
        hook_context: Context to record stop reason.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def stop_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}
        stop_reason = payload.get("stopReason", "end_turn")
        hook_context.stop_reason = stop_reason

        # Extract additional stop context if available
        num_turns = payload.get("numTurns")
        duration_ms = payload.get("durationMs")
        final_result = payload.get("result", "")
        result_preview = final_result[:200] if isinstance(final_result, str) else ""

        logger.debug(
            "claude_agent_sdk.hook.stop",
            event="hook.stop",
            context={
                "stop_reason": stop_reason,
                "sdk_num_turns": num_turns,
                "sdk_duration_ms": duration_ms,
                "result_preview": result_preview,
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


def create_task_completion_stop_hook(
    hook_context: HookContext,
    *,
    checker: TaskCompletionChecker,
) -> AsyncHookCallback:
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
        """Get filesystem from hook context resources if available."""
        try:
            return hook_context.resources.get(Filesystem)
        except (LookupError, AttributeError):
            return None

    async def task_completion_stop_hook(
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

        # Skip task completion check if deadline exceeded - can't do more work
        if (
            hook_context.deadline
            and hook_context.deadline.remaining().total_seconds() <= 0
        ):
            logger.debug(
                "claude_agent_sdk.hook.task_completion_stop.deadline_exceeded",
                event="hook.task_completion_stop.deadline_exceeded",
                context={"stop_reason": stop_reason},
            )
            return {}

        # Skip task completion check if budget exhausted - can't do more work
        budget_tracker = hook_context.budget_tracker
        if budget_tracker is not None and isinstance(budget_tracker, BudgetTracker):
            budget = budget_tracker.budget
            consumed = budget_tracker.consumed
            consumed_total = (consumed.input_tokens or 0) + (
                consumed.output_tokens or 0
            )
            if (  # pragma: no branch
                budget.max_total_tokens is not None
                and consumed_total >= budget.max_total_tokens
            ):
                logger.debug(
                    "claude_agent_sdk.hook.task_completion_stop.budget_exhausted",
                    event="hook.task_completion_stop.budget_exhausted",
                    context={"stop_reason": stop_reason},
                )
                return {}

        # Check task completion using the checker
        context = TaskCompletionContext(
            session=hook_context.session,
            tentative_output=None,
            stop_reason=stop_reason,
            filesystem=_get_filesystem(),
        )
        result = checker.check(context)

        if result.complete:
            # All tasks complete - allow stop
            logger.debug(
                "claude_agent_sdk.hook.task_completion_stop.allow",
                event="hook.task_completion_stop.allow",
                context={
                    "stop_reason": stop_reason,
                    "feedback": result.feedback,
                },
            )
            return {}

        # Tasks incomplete - signal to continue
        logger.info(
            "claude_agent_sdk.hook.task_completion_stop.incomplete",
            event="hook.task_completion_stop.incomplete",
            context={
                "stop_reason": stop_reason,
                "feedback": result.feedback,
            },
        )

        return {
            "continue_": True,
            "reason": result.feedback,
        }

    return task_completion_stop_hook


def create_subagent_start_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a SubagentStart hook to capture subagent launch events.

    Tracks subagent statistics for debugging when subagents are spawned.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def subagent_start_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        hook_context.stats.subagent_count += 1
        payload = input_data if isinstance(input_data, dict) else {}

        # Extract subagent details for logging
        subagent_type = payload.get("subagent_type", "")
        subagent_description = payload.get("description", "")
        subagent_id = payload.get("subagent_id", "")

        logger.debug(
            "claude_agent_sdk.hook.subagent_start",
            event="hook.subagent_start",
            context={
                "subagent_number": hook_context.stats.subagent_count,
                "subagent_type": subagent_type,
                "subagent_id": subagent_id,
                "description": subagent_description,
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
            },
        )

        return {}

    return subagent_start_hook


def create_subagent_stop_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a SubagentStop hook to capture subagent completion events.

    Logs subagent completion details for debugging.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def subagent_stop_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}

        # Extract subagent completion details for logging
        subagent_id = payload.get("subagent_id", "")
        subagent_result = payload.get("result", "")
        result_preview = (
            subagent_result[:200] if isinstance(subagent_result, str) else ""
        )
        subagent_duration_ms = payload.get("duration_ms")
        subagent_tool_count = payload.get("tool_count")

        logger.debug(
            "claude_agent_sdk.hook.subagent_stop",
            event="hook.subagent_stop",
            context={
                "subagent_id": subagent_id,
                "result_preview": result_preview,
                "subagent_duration_ms": subagent_duration_ms,
                "subagent_tool_count": subagent_tool_count,
                "elapsed_ms": hook_context.elapsed_ms,
                "parent_tool_count": hook_context.stats.tool_count,
                "subagent_count": hook_context.stats.subagent_count,
            },
        )

        return {}

    return subagent_stop_hook


def create_pre_compact_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a PreCompact hook to capture context compaction events.

    Tracks context window utilization for debugging memory-constrained scenarios
    before the SDK compacts conversation context.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def pre_compact_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        hook_context.stats.compact_count += 1
        payload = input_data if isinstance(input_data, dict) else {}

        # Extract context window details for logging
        context_tokens = payload.get("context_tokens")
        max_context_tokens = payload.get("max_context_tokens")
        message_count = payload.get("message_count")
        compaction_reason = payload.get("reason", "")

        # Calculate utilization percentage if available
        utilization_pct: float | None = None
        if context_tokens is not None and max_context_tokens:  # pragma: no cover
            utilization_pct = round((context_tokens / max_context_tokens) * 100, 1)

        logger.debug(
            "claude_agent_sdk.hook.pre_compact",
            event="hook.pre_compact",
            context={
                "compact_number": hook_context.stats.compact_count,
                "context_tokens": context_tokens,
                "max_context_tokens": max_context_tokens,
                "utilization_pct": utilization_pct,
                "message_count": message_count,
                "compaction_reason": compaction_reason,
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
                "turn_count": hook_context.stats.turn_count,
            },
        )

        return {}

    return pre_compact_hook


def create_notification_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create a Notification hook to capture user-facing notifications.

    Extracts notification type and content for structured logging from
    the SDK's notification system.

    Args:
        hook_context: Context with session references.

    Returns:
        An async hook callback function matching SDK signature.
    """

    async def notification_hook(
        input_data: Any,  # noqa: ANN401
        tool_use_id: str | None,
        sdk_context: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        _ = tool_use_id
        _ = sdk_context

        payload = input_data if isinstance(input_data, dict) else {}

        # Extract notification details for logging
        notification_type = payload.get("type", "")
        notification_message = payload.get("message", "")
        message_preview = (
            notification_message[:200] if isinstance(notification_message, str) else ""
        )
        notification_level = payload.get("level", "info")

        logger.debug(
            "claude_agent_sdk.hook.notification",
            event="hook.notification",
            context={
                "notification_type": notification_type,
                "notification_level": notification_level,
                "message_preview": message_preview,
                "elapsed_ms": hook_context.elapsed_ms,
                "tool_count": hook_context.stats.tool_count,
            },
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
    violations. Tracks error statistics for debugging.

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
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data.get("hookEventName", "PreToolUse"),
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Deadline exceeded",
                }
            }

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
            context={
                "error": str(error),
                "error_type": error_name,
                "hook_errors": context.stats.hook_errors,
                "elapsed_ms": context.elapsed_ms,
                "tool_count": context.stats.tool_count,
            },
        )

        return {}
