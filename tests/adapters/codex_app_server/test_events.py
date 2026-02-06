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

"""Tests for Codex event mapping helpers."""

from __future__ import annotations

from typing import Any

from weakincentives.adapters.codex_app_server._events import (
    _extract_mcp_output,
    dispatch_item_tool_invoked,
    extract_token_usage,
    map_codex_error_phase,
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


class TestDispatchItemToolInvoked:
    def test_command_execution_completed(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "commandExecution",
            "status": "completed",
            "command": "ls -la",
            "cwd": "/tmp",
            "aggregatedOutput": "total 0\ndrwxr-xr-x  2 user user 64 Jan 1 00:00 .",
            "id": "call-1",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="test-prompt",
            run_context=None,
        )
        assert len(captured) == 1
        event = captured[0]
        assert event.name == "codex:command"
        assert event.params["command"] == "ls -la"
        assert event.params["cwd"] == "/tmp"
        assert event.result.success is True
        assert event.call_id == "call-1"

    def test_command_execution_failed(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "commandExecution",
            "status": "failed",
            "command": "bad-cmd",
            "cwd": "/tmp",
            "aggregatedOutput": "not found",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].result.success is False

    def test_file_change(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "fileChange",
            "status": "completed",
            "file": "/tmp/foo.py",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].name == "codex:file_change"
        assert captured[0].params["file"] == "/tmp/foo.py"

    def test_mcp_tool_call(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "mcpToolCall",
            "status": "completed",
            "tool": "search",
            "server": "my-server",
            "result": {
                "content": [
                    {"type": "text", "text": "found it"},
                    {"type": "image", "data": "..."},
                ]
            },
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].name == "codex:mcp:search"
        assert captured[0].params["server"] == "my-server"
        assert "found it" in captured[0].rendered_output

    def test_web_search(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "webSearch",
            "status": "completed",
            "query": "python asyncio",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].name == "codex:web_search"
        assert captured[0].params["query"] == "python asyncio"

    def test_unknown_item_type(self) -> None:
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "customAction",
            "status": "completed",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].name == "codex:customAction"

    def test_failed_item_with_no_output(self) -> None:
        """Failed item with empty aggregated output uses status in message."""
        session, captured = _make_session()
        item: dict[str, Any] = {
            "type": "commandExecution",
            "status": "error",
            "command": "fail",
            "cwd": "/tmp",
            "aggregatedOutput": "",
        }
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name="codex_app_server",
            prompt_name="p",
            run_context=None,
        )
        assert len(captured) == 1
        assert captured[0].result.success is False
        assert "error" in captured[0].result.message.lower()


class TestExtractMcpOutput:
    def test_dict_with_text_content(self) -> None:
        item: dict[str, Any] = {
            "result": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ]
            }
        }
        assert _extract_mcp_output(item) == "hello\nworld"

    def test_dict_without_text(self) -> None:
        item: dict[str, Any] = {
            "result": {"content": [{"type": "image", "data": "abc"}]}
        }
        assert _extract_mcp_output(item) == ""

    def test_non_dict_result(self) -> None:
        item: dict[str, Any] = {"result": "plain string result"}
        assert _extract_mcp_output(item) == "plain string result"

    def test_empty_result(self) -> None:
        item: dict[str, Any] = {}
        result = _extract_mcp_output(item)
        assert isinstance(result, str)

    def test_non_dict_content_entries(self) -> None:
        """Non-dict items in the content list are skipped."""
        item: dict[str, Any] = {
            "result": {"content": ["not-a-dict", 42, {"type": "text", "text": "ok"}]}
        }
        assert _extract_mcp_output(item) == "ok"

    def test_truncation_at_1000(self) -> None:
        item: dict[str, Any] = {
            "result": {"content": [{"type": "text", "text": "x" * 2000}]}
        }
        assert len(_extract_mcp_output(item)) == 1000


class TestExtractTokenUsage:
    def test_valid_usage(self) -> None:
        params: dict[str, Any] = {
            "tokenUsage": {
                "last": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cachedInputTokens": 20,
                }
            }
        }
        usage = extract_token_usage(params)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 20

    def test_missing_last(self) -> None:
        params: dict[str, Any] = {"tokenUsage": {}}
        assert extract_token_usage(params) is None

    def test_non_dict_last(self) -> None:
        params: dict[str, Any] = {"tokenUsage": {"last": "invalid"}}
        assert extract_token_usage(params) is None

    def test_missing_token_usage(self) -> None:
        assert extract_token_usage({}) is None

    def test_partial_fields(self) -> None:
        params: dict[str, Any] = {"tokenUsage": {"last": {"inputTokens": 42}}}
        usage = extract_token_usage(params)
        assert usage is not None
        assert usage.input_tokens == 42
        assert usage.output_tokens is None


class TestMapCodexErrorPhase:
    def test_string_known_error(self) -> None:
        assert map_codex_error_phase("contextWindowExceeded") == "response"
        assert map_codex_error_phase("usageLimitExceeded") == "budget"
        assert map_codex_error_phase("httpConnectionFailed") == "request"
        assert map_codex_error_phase("sandboxError") == "tool"
        assert map_codex_error_phase("modelCap") == "budget"

    def test_string_unknown_error(self) -> None:
        assert map_codex_error_phase("someNewError") == "response"

    def test_dict_with_type(self) -> None:
        assert map_codex_error_phase({"type": "unauthorized"}) == "request"
        assert map_codex_error_phase({"type": "sandboxError"}) == "tool"

    def test_dict_unknown_type(self) -> None:
        assert map_codex_error_phase({"type": "novel"}) == "response"

    def test_dict_missing_type(self) -> None:
        assert map_codex_error_phase({}) == "response"

    def test_none(self) -> None:
        assert map_codex_error_phase(None) == "response"
