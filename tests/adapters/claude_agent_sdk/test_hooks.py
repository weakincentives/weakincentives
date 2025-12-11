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

"""Tests for Claude Agent SDK hooks."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from weakincentives.adapters.claude_agent_sdk._hooks import (
    build_post_tool_use_hook,
    build_pre_tool_use_hook,
    build_stop_hook,
    build_user_prompt_submit_hook,
    safe_hook_wrapper,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.runtime.events import InProcessEventBus, TokenUsage, ToolInvoked
from weakincentives.runtime.session import Session


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine synchronously."""
    return asyncio.run(coro)


class TestPreToolUseHook:
    """Tests for PreToolUse hook."""

    def test_allows_tool_when_no_constraints(self) -> None:
        """Test that hook allows tool execution with no constraints."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        hook = build_pre_tool_use_hook(
            session=session,
            deadline=None,
            budget_tracker=None,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        result = _run_async(
            hook(
                {"tool_name": "Read", "tool_input": {}},
                "tool_123",
                {},
            )
        )

        # Empty result means allow
        assert result == {}

    def test_denies_tool_when_deadline_expired(self) -> None:
        """Test that hook denies tool execution when deadline expired."""
        import time

        bus = InProcessEventBus()
        session = Session(bus=bus)

        # Create deadline that will expire in 1.1 seconds
        expiring_deadline = Deadline(
            expires_at=datetime.now(UTC) + timedelta(seconds=1.1)
        )

        hook = build_pre_tool_use_hook(
            session=session,
            deadline=expiring_deadline,
            budget_tracker=None,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        # Wait for deadline to expire
        time.sleep(1.2)

        result = _run_async(
            hook(
                {"tool_name": "Read", "tool_input": {}},
                "tool_123",
                {},
            )
        )

        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["permissionDecision"] == "deny"
        assert "Deadline exceeded" in output["permissionDecisionReason"]

    def test_denies_tool_when_budget_exhausted(self) -> None:
        """Test that hook denies tool execution when budget exhausted."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        # Create budget tracker with exhausted budget
        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget=budget)
        # Consume all tokens
        tracker.record_cumulative(
            "eval_1",
            TokenUsage(input_tokens=100, output_tokens=50),
        )

        hook = build_pre_tool_use_hook(
            session=session,
            deadline=None,
            budget_tracker=tracker,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        result = _run_async(
            hook(
                {"tool_name": "Read", "tool_input": {}},
                "tool_123",
                {},
            )
        )

        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["permissionDecision"] == "deny"
        assert "Budget exhausted" in output["permissionDecisionReason"]

    def test_allows_tool_with_sufficient_budget(self) -> None:
        """Test that hook allows tool execution with sufficient budget."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        tracker.record_cumulative(
            "eval_1",
            TokenUsage(input_tokens=100, output_tokens=50),
        )

        hook = build_pre_tool_use_hook(
            session=session,
            deadline=None,
            budget_tracker=tracker,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        result = _run_async(
            hook(
                {"tool_name": "Read", "tool_input": {}},
                "tool_123",
                {},
            )
        )

        assert result == {}


class TestPostToolUseHook:
    """Tests for PostToolUse hook."""

    def test_publishes_tool_invoked_event(self) -> None:
        """Test that hook publishes ToolInvoked event."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        received_events: list[ToolInvoked] = []

        def handler(event: object) -> None:
            if isinstance(event, ToolInvoked):
                received_events.append(event)

        bus.subscribe(ToolInvoked, handler)

        hook = build_post_tool_use_hook(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        _run_async(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/test.txt"},
                    "tool_output": {"content": "file contents"},
                    "tool_error": None,
                },
                "tool_123",
                {},
            )
        )

        assert len(received_events) == 1
        event = received_events[0]
        assert event.name == "Read"
        assert event.params == {"file_path": "/test.txt"}
        assert event.call_id == "tool_123"
        assert event.adapter == "test_adapter"
        assert event.prompt_name == "test_prompt"

    def test_handles_tool_error(self) -> None:
        """Test that hook handles tool errors correctly."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        received_events: list[ToolInvoked] = []

        def handler(event: object) -> None:
            if isinstance(event, ToolInvoked):
                received_events.append(event)

        bus.subscribe(ToolInvoked, handler)

        hook = build_post_tool_use_hook(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        _run_async(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/nonexistent.txt"},
                    "tool_output": {},
                    "tool_error": "File not found",
                },
                "tool_123",
                {},
            )
        )

        assert len(received_events) == 1
        event = received_events[0]
        assert "Error:" in event.result


class TestUserPromptSubmitHook:
    """Tests for UserPromptSubmit hook."""

    def test_returns_empty_by_default(self) -> None:
        """Test that hook returns empty by default."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        hook = build_user_prompt_submit_hook(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        result = _run_async(
            hook(
                {"prompt": "Do something"},
                None,
                {},
            )
        )

        assert result == {}


class TestStopHook:
    """Tests for Stop hook."""

    def test_calls_on_stop_callback(self) -> None:
        """Test that hook calls on_stop callback."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        stop_reasons: list[str] = []

        def on_stop(reason: str) -> None:
            stop_reasons.append(reason)

        hook = build_stop_hook(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            on_stop=on_stop,
        )

        _run_async(
            hook(
                {"stopReason": "end_turn"},
                None,
                {},
            )
        )

        assert stop_reasons == ["end_turn"]

    def test_handles_missing_stop_reason(self) -> None:
        """Test that hook handles missing stop reason."""
        bus = InProcessEventBus()
        session = Session(bus=bus)

        stop_reasons: list[str] = []

        def on_stop(reason: str) -> None:
            stop_reasons.append(reason)

        hook = build_stop_hook(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            on_stop=on_stop,
        )

        _run_async(
            hook(
                {},  # No stopReason
                None,
                {},
            )
        )

        assert stop_reasons == ["end_turn"]  # Default value


class TestSafeHookWrapper:
    """Tests for safe_hook_wrapper function."""

    def test_passes_through_successful_result(self) -> None:
        """Test that wrapper passes through successful results."""

        async def hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: dict[str, Any],
        ) -> dict[str, Any]:
            await asyncio.sleep(0)  # Make it actually async
            return {"result": "success"}

        result = _run_async(
            safe_hook_wrapper(
                hook,
                {"hookEventName": "PreToolUse"},
                "tool_123",
                {},
            )
        )

        assert result == {"result": "success"}

    def test_catches_generic_exception(self) -> None:
        """Test that wrapper catches generic exceptions."""

        async def hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: dict[str, Any],
        ) -> dict[str, Any]:
            await asyncio.sleep(0)  # Make it actually async
            msg = "Something went wrong"
            raise RuntimeError(msg)

        result = _run_async(
            safe_hook_wrapper(
                hook,
                {"hookEventName": "PreToolUse"},
                "tool_123",
                {},
            )
        )

        # Should return empty dict on generic errors
        assert result == {}
