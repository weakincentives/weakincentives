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

"""Tests for hook event name validation and subagent lifecycle flag management."""

from __future__ import annotations

import asyncio
from typing import cast

from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.prompt.task_completion import FileOutputChecker
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt


class TestHookEventNameValidation:
    """Tests for early return when hooks receive wrong event names."""

    def _make_context(self) -> HookContext:
        """Create a basic hook context for testing."""
        prompt = _make_prompt()
        session = Session(dispatcher=InProcessDispatcher())
        return HookContext(
            prompt=cast("PromptProtocol[object]", prompt),
            prompt_name="test-prompt",
            session=session,
            adapter_name="claude_agent_sdk",
        )

    def test_pre_tool_use_hook_wrong_event_name(self) -> None:
        """PreToolUse hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PostToolUse", "tool_name": "Read"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_post_tool_use_hook_wrong_event_name(self) -> None:
        """PostToolUse hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_post_tool_use_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_user_prompt_submit_hook_wrong_event_name(self) -> None:
        """UserPromptSubmit hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_user_prompt_submit_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "prompt": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_stop_hook_wrong_event_name(self) -> None:
        """Stop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_stop_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_task_completion_stop_hook_wrong_event_name(self) -> None:
        """TaskCompletion stop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_task_completion_stop_hook(
            context, checker=FileOutputChecker(files=("output.txt",))
        )

        # Call with wrong event name
        input_data = {"hook_event_name": "PreCompact", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_subagent_stop_hook_wrong_event_name(self) -> None:
        """SubagentStop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_subagent_stop_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_pre_compact_hook_wrong_event_name(self) -> None:
        """PreCompact hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_pre_compact_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_pre_tool_use_sets_in_subagent_for_task_tool(self) -> None:
        """PreToolUse hook sets in_subagent flag when Task tool is called."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Verify flag is initially False
        assert context.stats.in_subagent is False

        # Call with Task tool
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"prompt": "test subagent"},
        }
        asyncio.run(hook(input_data, "tool-123", {"signal": None}))

        # Flag should be set
        assert context.stats.in_subagent is True

    def test_pre_tool_use_does_not_set_in_subagent_for_other_tools(self) -> None:
        """PreToolUse hook does not set in_subagent for non-Task tools."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Verify flag is initially False
        assert context.stats.in_subagent is False

        # Call with regular tool
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test"},
        }
        asyncio.run(hook(input_data, "tool-123", {"signal": None}))

        # Flag should remain False
        assert context.stats.in_subagent is False

    def test_subagent_stop_increments_count_and_clears_flag(self) -> None:
        """SubagentStop hook increments subagent_count and clears in_subagent."""
        context = self._make_context()
        hook = create_subagent_stop_hook(context)

        # Set up initial state as if subagent was running
        context.stats.in_subagent = True
        context.stats.subagent_count = 0

        # Call SubagentStop
        input_data = {
            "hook_event_name": "SubagentStop",
            "session_id": "subagent-123",
            "stop_hook_active": False,
        }
        asyncio.run(hook(input_data, None, {"signal": None}))

        # Flag should be cleared and count incremented
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 1

    def test_subagent_lifecycle_flag_management(self) -> None:
        """Test full subagent lifecycle: PreToolUse(Task) -> SubagentStop."""
        context = self._make_context()
        pre_hook = create_pre_tool_use_hook(context)
        subagent_stop_hook = create_subagent_stop_hook(context)

        # Initially not in subagent
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 0

        # PreToolUse for Task sets flag
        task_input = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"prompt": "run subagent"},
        }
        asyncio.run(pre_hook(task_input, "task-tool-1", {"signal": None}))
        assert context.stats.in_subagent is True

        # SubagentStop clears flag and increments count
        stop_input = {
            "hook_event_name": "SubagentStop",
            "session_id": "subagent-1",
            "stop_hook_active": False,
        }
        asyncio.run(subagent_stop_hook(stop_input, None, {"signal": None}))
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 1
