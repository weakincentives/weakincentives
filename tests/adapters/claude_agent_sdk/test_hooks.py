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
from typing import Any

import pytest

from tests.helpers import FrozenUtcNow
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    PostToolUseInput,
    ToolResponse,
    create_post_tool_use_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_user_prompt_submit_hook,
    safe_hook_wrapper,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.events._types import TokenUsage, ToolInvoked
from weakincentives.runtime.session import Session


@pytest.fixture
def session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )


class TestHookContext:
    def test_basic_construction(self, session: Session) -> None:
        context = HookContext(
            session=session,
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
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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


class TestToolResponse:
    def test_from_dict_with_all_fields(self) -> None:
        data = {
            "stdout": "hello world",
            "stderr": "warning",
            "interrupted": True,
            "isImage": False,
        }
        response = ToolResponse.from_dict(data)

        assert response.stdout == "hello world"
        assert response.stderr == "warning"
        assert response.interrupted is True
        assert response.is_image is False

    def test_from_dict_with_none(self) -> None:
        response = ToolResponse.from_dict(None)

        assert response.stdout == ""
        assert response.stderr == ""
        assert response.interrupted is False
        assert response.is_image is False

    def test_from_dict_with_partial_fields(self) -> None:
        data = {"stdout": "output"}
        response = ToolResponse.from_dict(data)

        assert response.stdout == "output"
        assert response.stderr == ""
        assert response.interrupted is False
        assert response.is_image is False

    def test_is_frozen(self) -> None:
        response = ToolResponse(stdout="test")
        with pytest.raises(AttributeError):
            response.stdout = "modified"  # type: ignore[misc]


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
        assert parsed.tool_response.stdout == "file contents"
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
        assert parsed.tool_response.stdout == "plain string output"

    def test_from_dict_with_none_tool_response(self) -> None:
        data = {
            "tool_name": "Echo",
            "tool_response": None,
        }
        parsed = PostToolUseInput.from_dict(data)

        assert parsed is not None
        assert parsed.tool_response.stdout == ""

    def test_is_frozen(self) -> None:
        parsed = PostToolUseInput(
            session_id="sess",
            tool_name="Test",
            tool_input={},
            tool_response=ToolResponse(),
        )
        with pytest.raises(AttributeError):
            parsed.tool_name = "modified"  # type: ignore[misc]


class TestPostToolUseHookWithTypedParsing:
    def test_publishes_event_with_typed_value(self, session: Session) -> None:
        """Test that hook parses input into typed ToolResponse and stores as value."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        # value should be a ToolResponse dataclass
        assert isinstance(event.value, ToolResponse)
        assert event.value.stdout == "file contents"
        assert event.value.stderr == ""
        # result should be the raw dict
        assert event.result == {
            "stdout": "file contents",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }

    def test_fallback_when_missing_tool_name(self, session: Session) -> None:
        """Test fallback to dict access when parsing fails (missing required field)."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        # value should be None when parsing fails
        assert event.value is None

    def test_fallback_with_string_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is a string."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        assert event.value is None

    def test_fallback_with_none_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is None."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
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
        assert event.value is None


class TestToolInvokedSuccessField:
    """Tests for ToolInvoked.success field in PostToolUse hook."""

    def test_success_defaults_to_true(self, session: Session) -> None:
        """SDK native tools should default to success=True."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "file contents", "stderr": ""},
        }

        asyncio.run(hook(input_data, "call-success", context))

        assert len(events) == 1
        assert events[0].success is True

    def test_success_true_even_with_stderr(self, session: Session) -> None:
        """SDK native tools should be success=True even with stderr output.

        Stderr is not a reliable failure indicator - many tools write warnings
        or debug info to stderr even on success.
        """
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "contents", "stderr": "Warning: file is large"},
        }

        asyncio.run(hook(input_data, "call-with-stderr", context))

        assert len(events) == 1
        assert events[0].success is True

    def test_success_true_when_empty_response(self, session: Session) -> None:
        """Tool execution with empty response should default to success=True."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "tool_name": "NoOp",
            "tool_input": {},
            "tool_response": {},
        }

        asyncio.run(hook(input_data, "call-empty", context))

        assert len(events) == 1
        assert events[0].success is True

    def test_is_failure_hook_sets_success_false(self, session: Session) -> None:
        """is_failure hook returning True should set success=False."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        def is_failure(input_data: PostToolUseInput) -> bool:
            return "error" in input_data.tool_response.stderr.lower()

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, is_failure=is_failure)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "", "stderr": "Error: file not found"},
        }

        asyncio.run(hook(input_data, "call-failure", context))

        assert len(events) == 1
        assert events[0].success is False

    def test_is_failure_hook_returns_false_keeps_success_true(
        self, session: Session
    ) -> None:
        """is_failure hook returning False should keep success=True."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        def is_failure(input_data: PostToolUseInput) -> bool:
            return "error" in input_data.tool_response.stderr.lower()

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, is_failure=is_failure)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "contents", "stderr": "Warning: deprecated"},
        }

        asyncio.run(hook(input_data, "call-success", context))

        assert len(events) == 1
        assert events[0].success is True

    def test_is_failure_hook_exception_keeps_success_true(
        self, session: Session
    ) -> None:
        """Exception in is_failure hook should be caught and keep success=True."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        def is_failure(input_data: PostToolUseInput) -> bool:
            raise ValueError("Hook error")

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, is_failure=is_failure)
        input_data = {
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "contents"},
        }

        # Should not raise
        asyncio.run(hook(input_data, "call-exception", context))

        assert len(events) == 1
        assert events[0].success is True

    def test_is_failure_hook_not_called_when_parsing_fails(
        self, session: Session
    ) -> None:
        """is_failure hook should not be called when input parsing fails."""
        events: list[ToolInvoked] = []
        session.event_bus.subscribe(ToolInvoked, lambda e: events.append(e))

        call_count = 0

        def is_failure(input_data: PostToolUseInput) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, is_failure=is_failure)
        # Missing tool_name causes parsing to fail
        input_data = {
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "contents"},
        }

        asyncio.run(hook(input_data, "call-no-parse", context))

        assert len(events) == 1
        assert events[0].success is True  # Default when hook not called
        assert call_count == 0  # Hook was not called
