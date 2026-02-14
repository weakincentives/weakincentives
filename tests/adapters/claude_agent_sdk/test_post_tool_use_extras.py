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

"""Tests for post-tool-use typed parsing, transactional restore, and error detection."""

from __future__ import annotations

import asyncio
from typing import cast

from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    create_post_tool_use_hook,
    create_pre_tool_use_hook,
)
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt, _make_prompt_with_fs


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
            "hook_event_name": "PostToolUse",
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

        asyncio.run(hook(input_data, "call-typed", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": {"stdout": "output"},
        }

        asyncio.run(hook(input_data, "call-fallback", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": "string output",  # Non-dict
        }

        asyncio.run(hook(input_data, "call-string", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": None,
        }

        asyncio.run(hook(input_data, "call-none", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == ""
        assert event.rendered_output == ""


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
        asyncio.run(
            pre_hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with failure (stderr indicates error)
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stderr": "Error: file not found"},
                },
                "tool-123",
                {"signal": None},
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
        asyncio.run(
            pre_hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with success
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stdout": "File edited successfully"},
                },
                "tool-123",
                {"signal": None},
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
