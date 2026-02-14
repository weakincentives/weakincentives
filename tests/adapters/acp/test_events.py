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

"""Tests for ACP event mapping helpers."""

from __future__ import annotations

from weakincentives.adapters.acp._events import (
    WINK_MCP_SERVER_PREFIX,
    dispatch_tool_invoked,
    extract_token_usage,
)
from weakincentives.runtime.events import InProcessDispatcher, ToolInvoked
from weakincentives.runtime.session import Session


def _make_session() -> tuple[Session, list[ToolInvoked]]:
    """Create a session with ToolInvoked event capture."""
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    captured: list[ToolInvoked] = []
    dispatcher.subscribe(ToolInvoked, lambda e: captured.append(e))
    return session, captured


class TestDispatchToolInvoked:
    def test_completed_tool(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="test-prompt",
            run_context=None,
            tool_call_id="call-1",
            title="file_edit",
            status="completed",
        )
        assert len(captured) == 1
        event = captured[0]
        assert event.name == "acp:file_edit"
        assert event.result.success is True
        assert event.call_id == "call-1"
        assert event.adapter == "acp"
        assert event.prompt_name == "test-prompt"

    def test_failed_tool(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="p",
            run_context=None,
            tool_call_id="call-2",
            title="bash",
            status="failed",
        )
        assert len(captured) == 1
        assert captured[0].result.success is False
        assert "bash" in captured[0].result.message

    def test_skips_wink_bridged_tools(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="p",
            run_context=None,
            tool_call_id="call-3",
            title=f"{WINK_MCP_SERVER_PREFIX}my_tool",
            status="completed",
        )
        assert len(captured) == 0

    def test_custom_mcp_prefix(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="p",
            run_context=None,
            tool_call_id="call-4",
            title="custom-prefix_tool",
            status="completed",
            mcp_server_prefix="custom-prefix_",
        )
        assert len(captured) == 0

    def test_rendered_output_passed_through(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="p",
            run_context=None,
            tool_call_id="call-r",
            title="file_read",
            status="completed",
            rendered_output="file contents here",
        )
        assert len(captured) == 1
        assert captured[0].rendered_output == "file contents here"

    def test_none_call_id(self) -> None:
        session, captured = _make_session()
        dispatch_tool_invoked(
            session=session,
            adapter_name="acp",
            prompt_name="p",
            run_context=None,
            tool_call_id=None,
            title="search",
            status="completed",
        )
        assert len(captured) == 1
        assert captured[0].call_id is None


class TestExtractTokenUsage:
    def test_valid_usage(self) -> None:
        class MockUsage:
            input_tokens = 100
            output_tokens = 50
            cached_read_tokens = 20

        usage = extract_token_usage(MockUsage())
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 20

    def test_none_usage(self) -> None:
        assert extract_token_usage(None) is None

    def test_partial_usage(self) -> None:
        class MockUsage:
            input_tokens = 42
            output_tokens = None
            cached_read_tokens = None

        usage = extract_token_usage(MockUsage())
        assert usage is not None
        assert usage.input_tokens == 42
        assert usage.output_tokens is None
        assert usage.cached_tokens is None

    def test_no_tokens_returns_none(self) -> None:
        class MockUsage:
            input_tokens = None
            output_tokens = None
            cached_read_tokens = None

        assert extract_token_usage(MockUsage()) is None

    def test_object_without_attributes(self) -> None:
        """Object missing token attributes returns None."""
        assert extract_token_usage(object()) is None
