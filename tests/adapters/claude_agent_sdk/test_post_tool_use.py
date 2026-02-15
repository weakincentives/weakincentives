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

"""Tests for post-tool-use hook."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast

from weakincentives.adapters.claude_agent_sdk._bridge import MCPToolExecutionState
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookConstraints,
    HookContext,
    create_post_tool_use_hook,
)
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.prompt.task_completion import FileOutputChecker
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from ._hook_helpers import (
    _make_prompt,
    _make_prompt_with_feedback_provider,
    _make_prompt_with_fs,
)


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
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "file contents"},
        }

        asyncio.run(hook(input_data, "call-123", {"signal": None}))

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
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Read",
                    "tool_input": {},
                    "tool_response": {},
                },
                None,
                {"signal": None},
            )
        )
        assert hook_context._tool_count == 1

        asyncio.run(
            hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Write",
                    "tool_input": {},
                    "tool_response": {},
                },
                None,
                {"signal": None},
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
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/missing.txt"},
            "tool_response": {"stderr": "File not found"},
        }

        asyncio.run(hook(input_data, "call-456", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_name": "Echo",
            "tool_input": {"message": "hello"},
            "tool_response": "hello world",  # Non-dict response
        }

        asyncio.run(hook(input_data, "call-789", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {},
            "tool_response": {"stdout": long_output},
        }

        asyncio.run(hook(input_data, None, {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # SDK uses continue_ to avoid Python keyword conflict
        assert result == {"continue_": False}

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
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_setup_plan",
            "tool_input": {"objective": "test plan"},
            "tool_response": {"stdout": "Plan created"},
        }

        result = asyncio.run(hook(input_data, "call-mcp", {"signal": None}))

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
            "hook_event_name": "PostToolUse",
            "session_id": "sess-123",
            "tool_name": "mcp__wink__open_sections",
            "tool_input": {"section_keys": ["reference-docs"]},
            "tool_response": {"stdout": "Sections opened"},
            "cwd": "/home",
        }

        result = asyncio.run(hook(input_data, "call-mcp-full", {"signal": None}))

        assert result == {}
        assert len(events) == 0

    def test_handles_mcp_tool_post_without_error(self, session: Session) -> None:
        """PostToolUse handles MCP tools without error."""
        mcp_state = MCPToolExecutionState()
        # Note: With queue-based approach, PreToolUse enqueues and BridgedTool dequeues.
        # PostToolUse just runs feedback providers - no state management needed.

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_update",
            "tool_input": {"step_id": 1},
            "tool_response": {"stdout": "Updated"},
        }

        # Should complete without error
        result = asyncio.run(hook(input_data, "call-mcp-456", {"signal": None}))
        assert result == {}

    def test_returns_context_when_structured_output_with_missing_files(
        self, session: Session
    ) -> None:
        """PostToolUse returns additionalContext for StructuredOutput when files missing."""
        fs = InMemoryFilesystem()
        # Don't create the required file

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt_with_fs(fs)),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context,
            task_completion_checker=FileOutputChecker(files=("output.txt",)),
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should return continue_: True to force continuation
        assert result.get("continue_") is True
        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "output.txt" in additional_context

    def test_stops_when_structured_output_with_all_files_present(
        self, session: Session
    ) -> None:
        """PostToolUse stops after StructuredOutput when all required files exist."""
        fs = InMemoryFilesystem()
        fs.write("output.txt", "done")

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt_with_fs(fs)),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context,
            task_completion_checker=FileOutputChecker(files=("output.txt",)),
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should stop - all required files exist
        assert result == {"continue_": False}

    def test_stops_when_structured_output_without_filesystem(
        self, session: Session
    ) -> None:
        """PostToolUse stops after StructuredOutput when no filesystem available."""
        # Prompt has no filesystem bound - checker fails open
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context,
            task_completion_checker=FileOutputChecker(files=("output.txt",)),
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should stop - no filesystem means fail-open
        assert result == {"continue_": False}

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
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.txt"},
            "tool_response": {"stdout": "file content"},
        }

        result = asyncio.run(hook(input_data, "call-read", {"signal": None}))

        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "Test feedback triggered" in additional_context

    def test_returns_feedback_for_mcp_tool_when_provider_triggers(
        self, session: Session
    ) -> None:
        """PostToolUse returns additionalContext for MCP tools when feedback triggers."""
        # Seed ToolInvoked with an existing event so feedback provider triggers.
        # MCP tools dispatch their ToolInvoked via the bridge, so we simulate that
        # by pre-seeding an event. The feedback trigger checks tool_call_count.
        existing_event = ToolInvoked(
            prompt_name="test_prompt",
            adapter="claude_agent_sdk",
            name="mcp__wink__some_tool",
            params={},
            result={},
            session_id=None,
            created_at=datetime.now(UTC),
            call_id="prev-call",
        )
        session[ToolInvoked].seed((existing_event,))

        prompt = _make_prompt_with_feedback_provider()
        mcp_tool_state = MCPToolExecutionState()
        constraints = HookConstraints(mcp_tool_state=mcp_tool_state)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_post_tool_use_hook(context)

        # Use an MCP tool name to hit the _handle_mcp_tool_post path
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_setup_plan",
            "tool_input": {"plan": "test plan"},
            "tool_response": {"success": True},
        }

        result = asyncio.run(hook(input_data, "call-mcp", {"signal": None}))

        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "Test feedback triggered" in additional_context
