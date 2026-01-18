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

"""Tests for Claude Agent SDK hook implementations."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    PostToolUseInput,
    create_notification_hook,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_start_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
    safe_hook_wrapper,
)
from weakincentives.adapters.claude_agent_sdk._task_completion import PlanBasedChecker
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.contrib.tools.planning import Plan, PlanningToolsSection, PlanStep
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    Feedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.events.types import TokenUsage, ToolInvoked
from weakincentives.runtime.session import Session


def _make_prompt() -> Prompt[object]:
    """Create a prompt in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="hooks-test"))
    prompt.resources.__enter__()
    return prompt


def _make_prompt_with_fs(fs: InMemoryFilesystem) -> Prompt[object]:
    """Create a prompt with filesystem bound in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="hooks-test"))
    prompt = prompt.bind(resources={Filesystem: fs})
    prompt.resources.__enter__()
    return prompt


class _AlwaysTriggerProvider:
    """Test provider that always triggers and returns fixed feedback."""

    @property
    def name(self) -> str:
        return "AlwaysTrigger"

    def should_run(
        self,
        *,
        context: Any,  # noqa: ANN401 - test mock
    ) -> bool:
        return True

    def provide(
        self,
        *,
        context: Any,  # noqa: ANN401 - test mock
    ) -> Feedback:
        return Feedback(
            provider_name=self.name,
            summary="Test feedback triggered",
            severity="info",
        )


def _make_prompt_with_feedback_provider() -> Prompt[object]:
    """Create a prompt with a feedback provider that always triggers."""
    provider = _AlwaysTriggerProvider()
    config = FeedbackProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        trigger=FeedbackTrigger(every_n_calls=1),  # Trigger on every call
    )
    template: PromptTemplate[object] = PromptTemplate(
        ns="tests",
        key="hooks-test-feedback",
        name="test_prompt",  # Match the prompt_name used in HookContext
        feedback_providers=(config,),
    )
    prompt: Prompt[object] = Prompt(template)
    prompt.resources.__enter__()
    return prompt


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        prompt=cast("PromptProtocol[object]", _make_prompt()),
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )


class TestHookContext:
    def test_basic_construction(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        assert context.session is session
        assert context.adapter_name == "test_adapter"
        assert context.prompt_name == "test_prompt"
        assert context.deadline is None
        assert context.budget_tracker is None
        assert context.stop_reason is None
        assert context._tool_count == 0

    def test_with_deadline_and_budget(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(minutes=5), clock=clock)
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            deadline=deadline,
            budget_tracker=tracker,
        )
        assert context.deadline is deadline
        assert context.budget_tracker is tracker


class TestPreToolUseHook:
    def test_allows_tool_by_default(self, hook_context: HookContext) -> None:
        hook = create_pre_tool_use_hook(hook_context)
        input_data = {"tool_name": "Read", "tool_input": {"path": "/test"}}

        result = asyncio.run(hook(input_data, "call-123", hook_context))

        assert result == {}

    def test_denies_when_deadline_exceeded(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            deadline=deadline,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", context))

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Deadline exceeded" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_denies_when_budget_exhausted(self, session: Session) -> None:
        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative(
            "eval1", TokenUsage(input_tokens=100, output_tokens=50)
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", context))

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "budget exhausted" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_allows_with_remaining_budget(self, session: Session) -> None:
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative(
            "eval1", TokenUsage(input_tokens=50, output_tokens=50)
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", context))

        assert result == {}


class TestPostToolUseHook:
    def test_publishes_tool_invoked_event(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "file contents"},
        }

        asyncio.run(hook(input_data, "call-123", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        assert event.params == {"path": "/test.txt"}
        assert event.success is True
        assert "file contents" in event.rendered_output
        assert event.call_id == "call-123"
        assert event.adapter == "test_adapter"
        assert event.prompt_name == "test_prompt"

    def test_tracks_tool_count(self, hook_context: HookContext) -> None:
        hook = create_post_tool_use_hook(hook_context)

        assert hook_context._tool_count == 0

        asyncio.run(
            hook(
                {"tool_name": "Read", "tool_input": {}, "tool_response": {}},
                None,
                hook_context,
            )
        )
        assert hook_context._tool_count == 1

        asyncio.run(
            hook(
                {"tool_name": "Write", "tool_input": {}, "tool_response": {}},
                None,
                hook_context,
            )
        )
        assert hook_context._tool_count == 2

    def test_handles_tool_error(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/missing.txt"},
            "tool_response": {"stderr": "File not found"},
        }

        asyncio.run(hook(input_data, "call-456", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        assert event.call_id == "call-456"

    def test_handles_non_dict_tool_response(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # tool_response is a non-dict, non-None value (e.g., a string)
        input_data = {
            "tool_name": "Echo",
            "tool_input": {"message": "hello"},
            "tool_response": "hello world",  # Non-dict response
        }

        asyncio.run(hook(input_data, "call-789", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Echo"
        assert event.rendered_output == "hello world"

    def test_truncates_long_output(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        long_output = "x" * 2000
        input_data = {
            "tool_name": "Read",
            "tool_input": {},
            "tool_response": {"stdout": long_output},
        }

        asyncio.run(hook(input_data, None, context))

        assert len(events) == 1
        assert len(events[0].rendered_output) == 1000

    def test_stops_on_structured_output_by_default(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", context))

        assert result == {"continue": False}

    def test_does_not_stop_on_structured_output_when_disabled(
        self, session: Session
    ) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, stop_on_structured_output=False)
        input_data = {
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", context))

        assert result == {}

    def test_skips_mcp_wink_tools(self, session: Session) -> None:
        """MCP-bridged WINK tools should not publish ToolInvoked events (they do it themselves)."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "mcp__wink__planning_setup_plan",
            "tool_input": {"objective": "test plan"},
            "tool_response": {"stdout": "Plan created"},
        }

        result = asyncio.run(hook(input_data, "call-mcp", context))

        # Should return empty without publishing ToolInvoked event
        assert result == {}
        assert len(events) == 0
        # Tool count IS incremented (needed for feedback provider triggers)
        assert context._tool_count == 1

    def test_skips_mcp_wink_tools_with_parsed_input(self, session: Session) -> None:
        """MCP-bridged WINK tools should be skipped even with full SDK input format."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Full SDK-format input that will be parsed successfully
        input_data = {
            "session_id": "sess-123",
            "tool_name": "mcp__wink__open_sections",
            "tool_input": {"section_keys": ["reference-docs"]},
            "tool_response": {"stdout": "Sections opened"},
            "cwd": "/home",
        }

        result = asyncio.run(hook(input_data, "call-mcp-full", context))

        assert result == {}
        assert len(events) == 0

    def test_returns_context_when_structured_output_with_incomplete_tasks(
        self, session: Session
    ) -> None:
        """PostToolUse returns additionalContext for StructuredOutput when tasks incomplete."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Done task", status="done"),
                    PlanStep(step_id=2, title="Pending task", status="pending"),
                ),
            )
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", context))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "incomplete" in additional_context.lower()
        assert "Pending task" in additional_context

    def test_stops_when_structured_output_with_complete_tasks(
        self, session: Session
    ) -> None:
        """PostToolUse stops after StructuredOutput when all tasks complete."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="completed",
                steps=(
                    PlanStep(step_id=1, title="Task 1", status="done"),
                    PlanStep(step_id=2, title="Task 2", status="done"),
                ),
            )
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", context))

        # Should stop - all tasks complete
        assert result == {"continue": False}

    def test_stops_when_structured_output_without_plan(self, session: Session) -> None:
        """PostToolUse stops after StructuredOutput when no plan exists."""
        # Don't initialize Plan slice

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", context))

        # Should stop - no plan means nothing to enforce
        assert result == {"continue": False}

    def test_returns_feedback_when_provider_triggers(self, session: Session) -> None:
        """PostToolUse returns additionalContext when feedback provider triggers."""
        # Initialize ToolInvoked slice with empty tuple for tool_call_count
        session[ToolInvoked].seed(())

        prompt = _make_prompt_with_feedback_provider()
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        # Use a regular tool (not StructuredOutput) so we hit the feedback path
        input_data = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.txt"},
            "tool_response": {"stdout": "file content"},
        }

        result = asyncio.run(hook(input_data, "call-read", context))

        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "Test feedback triggered" in additional_context


class TestUserPromptSubmitHook:
    def test_returns_empty_by_default(self, hook_context: HookContext) -> None:
        hook = create_user_prompt_submit_hook(hook_context)
        input_data = {"prompt": "Do something"}

        result = asyncio.run(hook(input_data, None, hook_context))

        assert result == {}


class TestStopHook:
    def test_records_stop_reason(self, hook_context: HookContext) -> None:
        hook = create_stop_hook(hook_context)
        input_data = {"stopReason": "tool_use"}

        assert hook_context.stop_reason is None

        asyncio.run(hook(input_data, None, hook_context))

        assert hook_context.stop_reason == "tool_use"

    def test_defaults_to_end_turn(self, hook_context: HookContext) -> None:
        hook = create_stop_hook(hook_context)
        input_data = {}

        asyncio.run(hook(input_data, None, hook_context))

        assert hook_context.stop_reason == "end_turn"


class TestSafeHookWrapper:
    def test_passes_through_successful_result(self, hook_context: HookContext) -> None:
        def success_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: HookContext,
        ) -> dict[str, Any]:
            return {"result": "success"}

        result = safe_hook_wrapper(
            success_hook,
            {"tool_name": "test"},
            "call-123",
            hook_context,
        )

        assert result == {"result": "success"}

    def test_catches_deadline_exceeded(self, hook_context: HookContext) -> None:
        class DeadlineExceededError(Exception):
            pass

        def deadline_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: HookContext,
        ) -> dict[str, Any]:
            raise DeadlineExceededError("Deadline exceeded")

        result = safe_hook_wrapper(
            deadline_hook,
            {"hookEventName": "PreToolUse", "tool_name": "test"},
            "call-123",
            hook_context,
        )

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Deadline exceeded" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_catches_budget_exhausted(self, hook_context: HookContext) -> None:
        class BudgetExhaustedError(Exception):
            pass

        def budget_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: HookContext,
        ) -> dict[str, Any]:
            raise BudgetExhaustedError("Budget exhausted")

        result = safe_hook_wrapper(
            budget_hook,
            {"hookEventName": "PreToolUse", "tool_name": "test"},
            "call-123",
            hook_context,
        )

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Budget exhausted" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_catches_unknown_errors(self, hook_context: HookContext) -> None:
        def failing_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: HookContext,
        ) -> dict[str, Any]:
            raise RuntimeError("Unexpected error")

        result = safe_hook_wrapper(
            failing_hook,
            {"tool_name": "test"},
            "call-123",
            hook_context,
        )

        assert result == {}


class TestPostToolUseInput:
    def test_from_dict_with_full_input(self) -> None:
        data = {
            "session_id": "sess-123",
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "file contents", "stderr": ""},
            "cwd": "/home/user",
            "transcript_path": "/path/to/transcript",
            "permission_mode": "bypassPermissions",
        }
        parsed = PostToolUseInput.from_dict(data)

        assert parsed is not None
        assert parsed.session_id == "sess-123"
        assert parsed.tool_name == "Read"
        assert parsed.tool_input == {"path": "/test.txt"}
        # tool_response is now a raw dict
        assert parsed.tool_response == {"stdout": "file contents", "stderr": ""}
        assert parsed.cwd == "/home/user"
        assert parsed.permission_mode == "bypassPermissions"

    def test_from_dict_with_none(self) -> None:
        assert PostToolUseInput.from_dict(None) is None

    def test_from_dict_with_non_dict(self) -> None:
        assert PostToolUseInput.from_dict("not a dict") is None  # type: ignore[arg-type]

    def test_from_dict_missing_tool_name(self) -> None:
        data = {"session_id": "sess-123", "tool_input": {}}
        assert PostToolUseInput.from_dict(data) is None

    def test_from_dict_with_non_dict_tool_response(self) -> None:
        data = {
            "tool_name": "Echo",
            "tool_response": "plain string output",
        }
        parsed = PostToolUseInput.from_dict(data)

        assert parsed is not None
        # tool_response is kept as string when non-dict
        assert parsed.tool_response == "plain string output"

    def test_from_dict_with_none_tool_response(self) -> None:
        data = {
            "tool_name": "Echo",
            "tool_response": None,
        }
        parsed = PostToolUseInput.from_dict(data)

        assert parsed is not None
        # None becomes empty string
        assert parsed.tool_response == ""

    def test_is_frozen(self) -> None:
        parsed = PostToolUseInput(
            session_id="sess",
            tool_name="Test",
            tool_input={},
            tool_response={},
        )
        with pytest.raises(AttributeError):
            parsed.tool_name = "modified"  # type: ignore[misc]


class TestPostToolUseHookWithTypedParsing:
    def test_publishes_event_with_raw_result(self, session: Session) -> None:
        """Test that hook stores raw dict as result, not typed dataclass."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Full SDK-format input
        input_data = {
            "session_id": "sess-123",
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {
                "stdout": "file contents",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
            },
            "cwd": "/home",
            "transcript_path": "/transcript",
        }

        asyncio.run(hook(input_data, "call-typed", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        # SDK native tools publish success=True and rendered_output contains response text
        assert event.success is True
        assert "file contents" in event.rendered_output

    def test_fallback_when_missing_tool_name(self, session: Session) -> None:
        """Test fallback to dict access when parsing fails (missing required field)."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Malformed input missing tool_name
        input_data = {
            "tool_input": {"key": "value"},
            "tool_response": {"stdout": "output"},
        }

        asyncio.run(hook(input_data, "call-fallback", context))

        assert len(events) == 1
        event = events[0]
        # Falls back to dict access, tool_name defaults to ""
        assert event.name == ""

    def test_fallback_with_string_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is a string."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Missing tool_name causes parsing failure, and tool_response is a string
        input_data = {
            "tool_input": {"key": "value"},
            "tool_response": "string output",  # Non-dict
        }

        asyncio.run(hook(input_data, "call-string", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == ""
        assert event.rendered_output == "string output"

    def test_fallback_with_none_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is None."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Missing tool_name causes parsing failure, and tool_response is None
        input_data = {
            "tool_input": {"key": "value"},
            "tool_response": None,
        }

        asyncio.run(hook(input_data, "call-none", context))

        assert len(events) == 1
        event = events[0]
        assert event.name == ""
        assert event.rendered_output == ""


class TestSubagentStartHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_subagent_start_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}

    def test_increments_subagent_count(self, hook_context: HookContext) -> None:
        assert hook_context.stats.subagent_count == 0
        hook = create_subagent_start_hook(hook_context)

        asyncio.run(hook({"session_id": "test"}, None, hook_context))
        assert hook_context.stats.subagent_count == 1

        asyncio.run(hook({"session_id": "test2"}, None, hook_context))
        assert hook_context.stats.subagent_count == 2


class TestSubagentStopHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_subagent_stop_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}


class TestPreCompactHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_pre_compact_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}

    def test_increments_compact_count(self, hook_context: HookContext) -> None:
        assert hook_context.stats.compact_count == 0
        hook = create_pre_compact_hook(hook_context)

        asyncio.run(hook({"session_id": "test"}, None, hook_context))
        assert hook_context.stats.compact_count == 1

        asyncio.run(hook({"session_id": "test2"}, None, hook_context))
        assert hook_context.stats.compact_count == 2


class TestNotificationHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_notification_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}


class TestPreToolUseHookTransactional:
    """Tests for pre-tool hook transactional snapshot functionality."""

    def test_takes_snapshot_with_session_and_prompt(self, session: Session) -> None:
        """Pre-tool hook takes snapshot when session and prompt are present."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        hook = create_pre_tool_use_hook(context)
        result = asyncio.run(hook({"tool_name": "Edit"}, "tool-123", None))

        assert result == {}
        assert "tool-123" in context._tracker._pending_tools
        pending = context._tracker._pending_tools["tool-123"]
        assert pending.tool_name == "Edit"

    def test_skips_snapshot_for_mcp_wink_tools(self, session: Session) -> None:
        """Pre-tool hook skips snapshot for MCP WINK tools."""
        fs = InMemoryFilesystem()
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(hook({"tool_name": "mcp__wink__search"}, "tool-123", None))

        # Should NOT have pending execution for MCP tools (tracker not even initialized)
        assert (
            context._tool_tracker is None
            or "tool-123" not in context._tracker._pending_tools
        )


class TestPostToolUseHookTransactional:
    """Tests for post-tool hook transactional restore functionality."""

    def test_restores_state_on_tool_failure(self, session: Session) -> None:
        """Post-tool hook restores state when tool fails."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        # Take snapshot via pre-tool hook
        pre_hook = create_pre_tool_use_hook(context)
        asyncio.run(pre_hook({"tool_name": "Edit"}, "tool-123", None))

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with failure (stderr indicates error)
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stderr": "Error: file not found"},
                },
                "tool-123",
                None,
            )
        )

        # State should be restored
        assert fs.read("/test.txt").content == "initial"
        assert "tool-123" not in context._tracker._pending_tools

    def test_no_restore_on_success(self, session: Session) -> None:
        """Post-tool hook doesn't restore state on success."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        # Take snapshot via pre-tool hook
        pre_hook = create_pre_tool_use_hook(context)
        asyncio.run(pre_hook({"tool_name": "Edit"}, "tool-123", None))

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with success
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stdout": "File edited successfully"},
                },
                "tool-123",
                None,
            )
        )

        # State should NOT be restored
        assert fs.read("/test.txt").content == "modified"
        assert "tool-123" not in context._tracker._pending_tools


class TestIsToolErrorResponse:
    """Tests for _is_tool_error_response helper function."""

    def test_non_dict_returns_false(self) -> None:
        """Non-dict responses are not considered errors."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response("string") is False
        assert _is_tool_error_response(123) is False
        assert _is_tool_error_response(None) is False

    def test_is_error_flag(self) -> None:
        """is_error flag indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"is_error": True}) is True
        assert _is_tool_error_response({"is_error": False}) is False

    def test_isError_flag(self) -> None:
        """isError flag indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"isError": True}) is True
        assert _is_tool_error_response({"isError": False}) is False

    def test_error_in_content(self) -> None:
        """Error text in content indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "Error: something went wrong"}]}
            )
            is True
        )

        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "error - file not found"}]}
            )
            is True
        )

        # Normal content is not an error
        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "File created successfully"}]}
            )
            is False
        )

    def test_empty_content(self) -> None:
        """Empty content is not an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"content": []}) is False
        assert _is_tool_error_response({}) is False

    def test_content_with_non_dict_item(self) -> None:
        """Content with non-dict first item is not considered an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        # First item is a string instead of dict
        assert _is_tool_error_response({"content": ["not a dict"]}) is False
        # First item is a number instead of dict
        assert _is_tool_error_response({"content": [123]}) is False
        # First item is None instead of dict
        assert _is_tool_error_response({"content": [None]}) is False

    def test_content_with_non_string_text(self) -> None:
        """Content dict with non-string text field is not considered an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        # Text field is a number instead of string
        assert _is_tool_error_response({"content": [{"text": 123}]}) is False
        # Text field is None instead of string
        assert _is_tool_error_response({"content": [{"text": None}]}) is False
        # Text field is a list instead of string
        assert _is_tool_error_response({"content": [{"text": ["error"]}]}) is False


class TestTaskCompletionStopHook:
    """Tests for task completion stop hook functionality."""

    def test_allows_stop_when_no_plan_slice(self, hook_context: HookContext) -> None:
        """Stop is allowed when Plan slice is not installed in session."""
        hook = create_task_completion_stop_hook(
            hook_context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        assert result == {}
        assert hook_context.stop_reason == "end_turn"

    def test_allows_stop_when_no_plan_initialized(self, session: Session) -> None:
        """Stop is allowed when Plan slice exists but no plan is initialized."""
        # Install Plan slice but don't initialize a plan
        PlanningToolsSection._initialize_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_allows_stop_when_all_tasks_complete(self, session: Session) -> None:
        """Stop is allowed when all plan tasks are marked as done."""
        PlanningToolsSection._initialize_session(session)

        # Create a plan with all steps done
        completed_plan = Plan(
            objective="Test objective",
            status="completed",
            steps=(
                PlanStep(step_id=1, title="Task 1", status="done"),
                PlanStep(step_id=2, title="Task 2", status="done"),
            ),
        )
        session.dispatch(completed_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_signals_continue_when_tasks_incomplete(self, session: Session) -> None:
        """Stop hook signals continuation when tasks are incomplete."""
        PlanningToolsSection._initialize_session(session)

        # Create a plan with incomplete steps
        incomplete_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(
                PlanStep(step_id=1, title="Done task", status="done"),
                PlanStep(step_id=2, title="Pending task", status="pending"),
                PlanStep(step_id=3, title="In progress task", status="in_progress"),
            ),
        )
        session.dispatch(incomplete_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        assert "2 incomplete task(s)" in result.get("reason", "")
        assert "Pending task" in result.get("reason", "")
        assert context.stop_reason == "end_turn"

    def test_records_stop_reason(self, session: Session) -> None:
        """Stop hook records the stop reason regardless of task state."""
        PlanningToolsSection._initialize_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "max_turns_reached"}

        assert context.stop_reason is None
        asyncio.run(hook(input_data, None, None))
        assert context.stop_reason == "max_turns_reached"

    def test_defaults_stop_reason_to_end_turn(self, session: Session) -> None:
        """Stop hook defaults stopReason to end_turn when not provided."""
        PlanningToolsSection._initialize_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )

        asyncio.run(hook({}, None, None))

        assert context.stop_reason == "end_turn"

    def test_handles_empty_steps(self, session: Session) -> None:
        """Stop is allowed when plan has no steps."""
        PlanningToolsSection._initialize_session(session)

        # Create a plan with no steps
        empty_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(),
        )
        session.dispatch(empty_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        assert result == {}

    def test_reminder_message_truncates_long_task_list(self, session: Session) -> None:
        """Reminder message truncates task titles when there are many incomplete."""
        PlanningToolsSection._initialize_session(session)

        # Create a plan with many incomplete steps
        many_steps_plan = Plan(
            objective="Test objective",
            status="active",
            steps=tuple(
                PlanStep(step_id=i, title=f"Task {i}", status="pending")
                for i in range(1, 10)
            ),
        )
        session.dispatch(many_steps_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, None))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        reason = result.get("reason", "")
        # Should contain first 3 tasks and ellipsis
        assert "Task 1" in reason
        assert "Task 2" in reason
        assert "Task 3" in reason
        assert "..." in reason
        # Should not list all tasks
        assert "Task 9" not in reason

    def test_checker_protocol_with_plan_based_checker(self, session: Session) -> None:
        """Hook accepts any TaskCompletionChecker implementation."""
        PlanningToolsSection._initialize_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        # Create hook with PlanBasedChecker - verifies protocol compatibility
        checker = PlanBasedChecker(plan_type=Plan)
        hook = create_task_completion_stop_hook(context, checker=checker)
        input_data = {"stopReason": "end_turn"}

        # Should work with checker protocol
        result = asyncio.run(hook(input_data, None, None))

        # No plan initialized, so stop is allowed
        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_skips_check_when_deadline_exceeded(self, session: Session) -> None:
        """Stop hook skips task completion check when deadline expired."""
        from weakincentives.clock import FakeClock

        PlanningToolsSection._initialize_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create expired deadline using fake clock
        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        expired_deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)  # Now expired

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            deadline=expired_deadline,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to expired deadline
        result = asyncio.run(hook(input_data, None, None))

        assert result == {}

    def test_skips_check_when_budget_exhausted(self, session: Session) -> None:
        """Stop hook skips task completion check when budget exhausted."""
        PlanningToolsSection._initialize_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create exhausted budget tracker
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        tracker.record_cumulative(
            "eval-1", TokenUsage(input_tokens=500, output_tokens=600)
        )  # Over budget

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to exhausted budget
        result = asyncio.run(hook(input_data, None, None))

        assert result == {}
