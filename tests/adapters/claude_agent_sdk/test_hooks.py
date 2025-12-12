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
