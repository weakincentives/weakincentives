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

"""Tests for pre-tool-use hook, including transactional snapshot functionality."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast

from weakincentives.adapters.claude_agent_sdk._bridge import MCPToolExecutionState
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookConstraints,
    HookContext,
    create_pre_tool_use_hook,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.deadlines import Deadline
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events.types import TokenUsage
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt, _make_prompt_with_fs


class TestPreToolUseHook:
    def test_allows_tool_by_default(self, hook_context: HookContext) -> None:
        hook = create_pre_tool_use_hook(hook_context)
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/test"},
        }

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert result == {}

    def test_denies_when_deadline_exceeded(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        constraints = HookConstraints(deadline=deadline)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

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

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

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

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

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
        result = asyncio.run(
            hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

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
        asyncio.run(
            hook(
                {"hook_event_name": "PreToolUse", "tool_name": "mcp__wink__search"},
                "tool-123",
                {"signal": None},
            )
        )

        # Should NOT have pending execution for MCP tools (tracker not even initialized)
        assert (
            context._tool_tracker is None
            or "tool-123" not in context._tracker._pending_tools
        )

    def test_enqueues_mcp_tool_state_for_mcp_tools(self, session: Session) -> None:
        """Pre-tool hook enqueues tool_use_id on mcp_tool_state for MCP tools."""
        mcp_state = MCPToolExecutionState()
        tool_input = {"objective": "test plan"}

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(
            hook(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__wink__planning",
                    "tool_input": tool_input,
                },
                "call-mcp-123",
                {"signal": None},
            )
        )

        # mcp_tool_state should have the tool_use_id enqueued for this tool+params
        assert mcp_state.dequeue("planning", tool_input) == "call-mcp-123"

    def test_does_not_enqueue_mcp_tool_state_for_native_tools(
        self, session: Session
    ) -> None:
        """Pre-tool hook does not enqueue tool_use_id for native SDK tools."""
        mcp_state = MCPToolExecutionState()
        tool_input = {"file_path": "/test.txt"}

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(
            hook(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Read",
                    "tool_input": tool_input,
                },
                "call-native-123",
                {"signal": None},
            )
        )

        # mcp_tool_state should remain empty for native tools
        assert mcp_state.dequeue("Read", tool_input) is None
