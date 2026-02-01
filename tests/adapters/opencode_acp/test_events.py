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

"""Tests for OpenCode ACP event mapping."""

from __future__ import annotations

from uuid import uuid4

from weakincentives.adapters.opencode_acp._events import (
    WINK_MCP_SERVER_PREFIX,
    is_wink_tool_call,
    map_tool_call_to_event,
)
from weakincentives.runtime.events import ToolInvoked


class TestIsWinkToolCall:
    def test_identifies_wink_server_by_name(self) -> None:
        assert is_wink_tool_call("some_tool", mcp_server_name="wink")
        assert is_wink_tool_call("some_tool", mcp_server_name="wink-tools")

    def test_identifies_wink_tool_by_prefix(self) -> None:
        prefixed_name = f"{WINK_MCP_SERVER_PREFIX}my_tool"
        assert is_wink_tool_call(prefixed_name)

    def test_rejects_non_wink_tools(self) -> None:
        assert not is_wink_tool_call("regular_tool")
        assert not is_wink_tool_call("regular_tool", mcp_server_name="other-server")
        assert not is_wink_tool_call("mcp_tool")  # Similar but not WINK prefix

    def test_handles_none_server_name(self) -> None:
        assert not is_wink_tool_call("regular_tool", mcp_server_name=None)


class TestMapToolCallToEvent:
    def test_maps_successful_tool_call(self) -> None:
        session_id = uuid4()

        event = map_tool_call_to_event(
            tool_name="my_tool",
            tool_use_id="call-123",
            params={"arg": "value"},
            result="Tool executed successfully",
            success=True,
            prompt_name="test_prompt",
            adapter_name="opencode_acp",
            session_id=session_id,
        )

        assert isinstance(event, ToolInvoked)
        assert event.name == "my_tool"
        assert event.params == {"arg": "value"}
        assert event.result.success
        assert event.call_id == "call-123"
        assert event.prompt_name == "test_prompt"
        assert event.adapter == "opencode_acp"
        assert event.session_id == session_id

    def test_maps_failed_tool_call(self) -> None:
        event = map_tool_call_to_event(
            tool_name="failing_tool",
            tool_use_id="call-456",
            params=None,
            result="Error: something went wrong",
            success=False,
            prompt_name="test_prompt",
            adapter_name="opencode_acp",
            session_id=None,
        )

        assert isinstance(event, ToolInvoked)
        assert event.name == "failing_tool"
        assert not event.result.success
        assert event.params == {}  # None converted to empty dict
        assert event.session_id is None

    def test_truncates_long_result(self) -> None:
        long_result = "x" * 2000

        event = map_tool_call_to_event(
            tool_name="tool",
            tool_use_id="call-789",
            params={},
            result=long_result,
            success=True,
            prompt_name="test",
            adapter_name="opencode_acp",
            session_id=None,
        )

        # Rendered output should be truncated
        assert len(event.rendered_output) == 1000

    def test_handles_none_result(self) -> None:
        event = map_tool_call_to_event(
            tool_name="tool",
            tool_use_id="call-000",
            params={},
            result=None,
            success=True,
            prompt_name="test",
            adapter_name="opencode_acp",
            session_id=None,
        )

        assert event.rendered_output == ""

    def test_includes_run_context(self) -> None:
        from weakincentives.runtime.run_context import RunContext

        run_context = RunContext(worker_id="test-run")

        event = map_tool_call_to_event(
            tool_name="tool",
            tool_use_id="call-111",
            params={},
            result="ok",
            success=True,
            prompt_name="test",
            adapter_name="opencode_acp",
            session_id=None,
            run_context=run_context,
        )

        assert event.run_context is run_context
