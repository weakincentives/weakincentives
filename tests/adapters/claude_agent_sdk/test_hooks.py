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
from pathlib import Path
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    PostToolUseInput,
    _expand_transcript_paths,
    _read_transcript_file,
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
from weakincentives.adapters.claude_agent_sdk._notifications import Notification
from weakincentives.adapters.claude_agent_sdk._task_completion import PlanBasedChecker
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.contrib.tools.planning import Plan, PlanningToolsSection, PlanStep
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.events._types import TokenUsage, ToolInvoked
from weakincentives.runtime.session import Session, append_all


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


@pytest.fixture
def session() -> Session:
    bus = InProcessDispatcher()
    return Session(bus=bus)


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

    def test_with_deadline_and_budget(
        self, session: Session, frozen_utcnow: FrozenUtcNow
    ) -> None:
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        deadline = Deadline(anchor + timedelta(minutes=5))
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

    def test_denies_when_deadline_exceeded(
        self, session: Session, frozen_utcnow: FrozenUtcNow
    ) -> None:
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5))
        frozen_utcnow.advance(timedelta(seconds=10))

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
        assert event.result == {"stdout": "file contents"}
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
        """MCP-bridged WINK tools should not publish events (they do it themselves)."""
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

        # Should return empty without publishing event
        assert result == {}
        assert len(events) == 0
        # Tool count should not be incremented
        assert context._tool_count == 0

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

        # Should return continue: True to force continuation
        assert result.get("continue") is True
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
        # result should be the raw dict (SDK native tools)
        assert event.result == {
            "stdout": "file contents",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }

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


class TestNotification:
    def test_construction(self) -> None:
        payload = {"session_id": "sess-123", "subagent_type": "analyzer"}
        notification = Notification(
            source="subagent_start",
            payload=payload,
            prompt_name="test_prompt",
            adapter_name="test_adapter",
        )

        assert notification.source == "subagent_start"
        assert notification.payload == payload
        assert notification.prompt_name == "test_prompt"
        assert notification.adapter_name == "test_adapter"
        assert notification.notification_id is not None

    def test_render(self) -> None:
        notification = Notification(
            source="notification",
            payload={"message": "Test message"},
            prompt_name="test_prompt",
            adapter_name="test_adapter",
        )
        rendered = notification.render()

        assert "[notification]" in rendered
        assert "test_prompt" in rendered


class TestSubagentStartHook:
    def test_dispatches_notification_to_session(self, session: Session) -> None:
        # Register reducer for Notification
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_start_hook(context)
        input_data = {
            "session_id": "sess-123",
            "subagent_type": "code_reviewer",
            "prompt": "Review this code",
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.source == "subagent_start"
        assert notif.payload == input_data
        assert notif.prompt_name == "test_prompt"
        assert notif.adapter_name == "test_adapter"

    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        # Register reducer for Notification
        hook_context.session[Notification].register(Notification, append_all)

        hook = create_subagent_start_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}


class TestSubagentStopHook:
    def test_dispatches_notification_to_session(self, session: Session) -> None:
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_stop_hook(context)
        input_data = {
            "session_id": "sess-123",
            "stop_reason": "completed",
            "result": "Review complete",
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.source == "subagent_stop"
        assert notif.payload == input_data

    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook_context.session[Notification].register(Notification, append_all)

        hook = create_subagent_stop_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}

    def test_expands_transcript_path(self, session: Session, tmp_path: Path) -> None:
        session[Notification].register(Notification, append_all)

        # Create a temporary JSONL transcript file
        transcript_file = tmp_path / "transcript.jsonl"
        transcript_file.write_text(
            '{"role": "user", "content": "hello"}\n'
            '{"role": "assistant", "content": "hi there"}\n'
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_stop_hook(context)
        input_data = {
            "session_id": "sess-123",
            "transcript_path": str(transcript_file),
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.source == "subagent_stop"
        # transcript_path should now be a list of parsed entries
        assert notif.payload["transcript_path"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_expands_agent_transcript_path(
        self, session: Session, tmp_path: Path
    ) -> None:
        session[Notification].register(Notification, append_all)

        # Create a temporary JSONL transcript file
        transcript_file = tmp_path / "agent_transcript.jsonl"
        transcript_file.write_text('{"event": "tool_call", "name": "Read"}\n')

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_stop_hook(context)
        input_data = {
            "session_id": "sess-123",
            "agent_transcript_path": str(transcript_file),
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.payload["agent_transcript_path"] == [
            {"event": "tool_call", "name": "Read"},
        ]

    def test_expands_both_transcript_paths(
        self, session: Session, tmp_path: Path
    ) -> None:
        session[Notification].register(Notification, append_all)

        transcript_file = tmp_path / "transcript.jsonl"
        transcript_file.write_text('{"role": "user", "content": "test"}\n')

        agent_transcript_file = tmp_path / "agent.jsonl"
        agent_transcript_file.write_text('{"event": "start"}\n')

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_stop_hook(context)
        input_data = {
            "session_id": "sess-123",
            "transcript_path": str(transcript_file),
            "agent_transcript_path": str(agent_transcript_file),
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.payload["transcript_path"] == [
            {"role": "user", "content": "test"},
        ]
        assert notif.payload["agent_transcript_path"] == [{"event": "start"}]

    def test_handles_nonexistent_transcript_path(self, session: Session) -> None:
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_subagent_stop_hook(context)
        input_data = {
            "session_id": "sess-123",
            "transcript_path": "/nonexistent/path/transcript.jsonl",
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        # Returns empty list for nonexistent file
        assert notif.payload["transcript_path"] == []


class TestReadTranscriptFile:
    def test_reads_jsonl_file(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        result = _read_transcript_file(str(transcript_file))

        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_returns_empty_list_for_empty_path(self) -> None:
        assert _read_transcript_file("") == []

    def test_returns_empty_list_for_nonexistent_file(self) -> None:
        assert _read_transcript_file("/nonexistent/file.jsonl") == []

    def test_handles_blank_lines(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text('{"a": 1}\n\n{"b": 2}\n  \n')

        result = _read_transcript_file(str(transcript_file))

        assert result == [{"a": 1}, {"b": 2}]

    def test_handles_invalid_json_gracefully(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text('{"a": 1}\nnot json\n{"b": 2}\n')

        # Returns empty list on JSON decode error
        result = _read_transcript_file(str(transcript_file))

        assert result == []

    def test_expands_tilde_in_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Create file in tmp_path and mock expanduser
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text('{"key": "value"}\n')

        original_expanduser = Path.expanduser

        def mock_expanduser(path_self: Path) -> Path:
            if str(path_self).startswith("~"):
                return tmp_path / str(path_self)[2:]  # Remove "~/"
            return original_expanduser(path_self)

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        result = _read_transcript_file("~/test.jsonl")

        assert result == [{"key": "value"}]


class TestExpandTranscriptPaths:
    def test_expands_transcript_path(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "transcript.jsonl"
        transcript_file.write_text('{"role": "user"}\n')

        payload = {"session_id": "123", "transcript_path": str(transcript_file)}
        result = _expand_transcript_paths(payload)

        assert result["session_id"] == "123"
        assert result["transcript_path"] == [{"role": "user"}]

    def test_expands_agent_transcript_path(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "agent.jsonl"
        transcript_file.write_text('{"event": "start"}\n')

        payload = {"agent_transcript_path": str(transcript_file)}
        result = _expand_transcript_paths(payload)

        assert result["agent_transcript_path"] == [{"event": "start"}]

    def test_preserves_other_fields(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "transcript.jsonl"
        transcript_file.write_text('{"msg": 1}\n')

        payload = {
            "session_id": "abc",
            "transcript_path": str(transcript_file),
            "other_field": "value",
        }
        result = _expand_transcript_paths(payload)

        assert result["session_id"] == "abc"
        assert result["other_field"] == "value"

    def test_does_not_modify_original_payload(self, tmp_path: Path) -> None:
        transcript_file = tmp_path / "transcript.jsonl"
        transcript_file.write_text('{"data": 1}\n')

        original_path = str(transcript_file)
        payload = {"transcript_path": original_path}
        _expand_transcript_paths(payload)

        # Original should be unchanged
        assert payload["transcript_path"] == original_path

    def test_handles_non_string_transcript_path(self) -> None:
        # If transcript_path is not a string, leave it unchanged
        payload = {"transcript_path": 123}
        result = _expand_transcript_paths(payload)

        assert result["transcript_path"] == 123


class TestPreCompactHook:
    def test_dispatches_notification_to_session(self, session: Session) -> None:
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_pre_compact_hook(context)
        input_data = {
            "session_id": "sess-123",
            "message_count": 100,
            "token_count": 50000,
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.source == "pre_compact"
        assert notif.payload == input_data

    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook_context.session[Notification].register(Notification, append_all)

        hook = create_pre_compact_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}


class TestNotificationHook:
    def test_dispatches_notification_to_session(self, session: Session) -> None:
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_notification_hook(context)
        input_data = {
            "session_id": "sess-123",
            "message": "Operation completed",
            "level": "info",
        }

        asyncio.run(hook(input_data, None, context))

        notifications = session[Notification].all()
        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.source == "notification"
        assert notif.payload == input_data

    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook_context.session[Notification].register(Notification, append_all)

        hook = create_notification_hook(hook_context)
        result = asyncio.run(hook({"session_id": "test"}, None, hook_context))

        assert result == {}


class TestMultipleNotificationsAccumulate:
    def test_multiple_hooks_accumulate_notifications(self, session: Session) -> None:
        """Test that notifications from different hooks accumulate in session."""
        session[Notification].register(Notification, append_all)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        # Fire multiple different hooks
        subagent_start = create_subagent_start_hook(context)
        asyncio.run(
            subagent_start(
                {"session_id": "sess-1", "subagent_type": "analyzer"}, None, context
            )
        )

        pre_compact = create_pre_compact_hook(context)
        asyncio.run(
            pre_compact({"session_id": "sess-1", "message_count": 50}, None, context)
        )

        subagent_stop = create_subagent_stop_hook(context)
        asyncio.run(
            subagent_stop(
                {"session_id": "sess-1", "stop_reason": "completed"}, None, context
            )
        )

        notification = create_notification_hook(context)
        asyncio.run(
            notification({"session_id": "sess-1", "message": "Done"}, None, context)
        )

        notifications = session[Notification].all()
        assert len(notifications) == 4

        sources = [n.source for n in notifications]
        assert "subagent_start" in sources
        assert "pre_compact" in sources
        assert "subagent_stop" in sources
        assert "notification" in sources


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

        assert result.get("needsMoreTurns") is True
        assert result.get("decision") == "continue"
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

        assert result.get("needsMoreTurns") is True
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

    def test_skips_check_when_deadline_exceeded(
        self, session: Session, frozen_utcnow: FrozenUtcNow
    ) -> None:
        """Stop hook skips task completion check when deadline expired."""
        PlanningToolsSection._initialize_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create expired deadline using frozen time
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        expired_deadline = Deadline(anchor + timedelta(seconds=5))
        frozen_utcnow.advance(timedelta(seconds=10))  # Now expired

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
