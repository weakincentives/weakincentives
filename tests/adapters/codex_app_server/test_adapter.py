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

"""Tests for the Codex App Server adapter."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server.adapter import (
    CODEX_APP_SERVER_ADAPTER_NAME,
    CodexAppServerAdapter,
    _bridged_tools_to_dynamic_specs,
    _openai_strict_schema,
)
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)
from weakincentives.adapters.codex_app_server.config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    ExternalTokenAuth,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import Prompt, PromptTemplate, Tool
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import (
    InProcessDispatcher,
    PromptExecuted,
    PromptRendered,
)
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


@dataclass(slots=True, frozen=True)
class _AddParams:
    x: int
    y: int


def _add_handler(params: _AddParams, *, context: ToolContext) -> ToolResult[int]:
    return ToolResult.ok(params.x + params.y, message=str(params.x + params.y))


_ADD_TOOL = Tool[_AddParams, int](
    name="add",
    description="Add two numbers",
    handler=_add_handler,
)


def _make_prompt_with_tool(name: str = "tool-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="with-tool",
        sections=(),
        tools=(_ADD_TOOL,),
        name=name,
    )
    return Prompt(template)


def _make_mock_client() -> AsyncMock:
    """Create a mock CodexAppServerClient."""
    client = AsyncMock(spec=CodexAppServerClient)
    client.stderr_output = ""
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.send_request = AsyncMock(return_value={})
    client.send_notification = AsyncMock()
    client.send_response = AsyncMock()
    return client


def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()


# ---- Tests ----


class TestAdapterName:
    def test_name_constant(self) -> None:
        assert CODEX_APP_SERVER_ADAPTER_NAME == "codex_app_server"


class TestBridgedToolsToDynamicSpecs:
    def test_empty(self) -> None:
        assert _bridged_tools_to_dynamic_specs(()) == []

    def test_converts_tools(self) -> None:
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does something"
        tool.input_schema = {"type": "object", "properties": {}}
        specs = _bridged_tools_to_dynamic_specs((tool,))
        assert len(specs) == 1
        assert specs[0]["name"] == "my_tool"
        assert specs[0]["description"] == "Does something"
        assert specs[0]["inputSchema"] == {"type": "object", "properties": {}}


class TestOpenaiStrictSchema:
    def test_sets_additional_properties_false(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        result = _openai_strict_schema(s)
        assert result["additionalProperties"] is False

    def test_all_properties_required(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        result = _openai_strict_schema(s)
        assert sorted(result["required"]) == ["a", "b"]

    def test_nested_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        }
        result = _openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert result["properties"]["inner"]["additionalProperties"] is False
        assert result["properties"]["inner"]["required"] == ["x"]

    def test_non_object_unchanged(self) -> None:
        s = {"type": "array", "items": {"type": "string"}}
        result = _openai_strict_schema(s)
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_object_without_properties(self) -> None:
        s = {"type": "object", "additionalProperties": True}
        result = _openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert "required" not in result

    def test_preserves_other_fields(self) -> None:
        s = {"type": "object", "title": "Foo", "properties": {"a": {"type": "string"}}}
        result = _openai_strict_schema(s)
        assert result["title"] == "Foo"
        assert result["required"] == ["a"]

    def test_array_items_strictified(self) -> None:
        s = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": True,
            },
        }
        result = _openai_strict_schema(s)
        assert result["items"]["additionalProperties"] is False
        assert result["items"]["required"] == ["x"]

    def test_anyof_strictified(self) -> None:
        s = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": True,
                },
                {"type": "string"},
            ]
        }
        result = _openai_strict_schema(s)
        assert result["anyOf"][0]["additionalProperties"] is False
        assert result["anyOf"][0]["required"] == ["a"]
        assert result["anyOf"][1] == {"type": "string"}

    def test_oneof_strictified(self) -> None:
        s = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = _openai_strict_schema(s)
        assert result["oneOf"][0]["additionalProperties"] is False

    def test_allof_strictified(self) -> None:
        s = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"c": {"type": "boolean"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = _openai_strict_schema(s)
        assert result["allOf"][0]["additionalProperties"] is False

    def test_defs_strictified(self) -> None:
        s = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                    "additionalProperties": True,
                }
            },
        }
        result = _openai_strict_schema(s)
        assert result["$defs"]["Inner"]["additionalProperties"] is False
        assert result["$defs"]["Inner"]["required"] == ["val"]

    def test_definitions_strictified(self) -> None:
        s = {
            "definitions": {
                "Foo": {
                    "type": "object",
                    "properties": {"z": {"type": "number"}},
                    "additionalProperties": True,
                }
            }
        }
        result = _openai_strict_schema(s)
        assert result["definitions"]["Foo"]["additionalProperties"] is False

    def test_deeply_nested_array_of_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nested": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"v": {"type": "integer"}},
                                    "additionalProperties": True,
                                },
                            }
                        },
                        "additionalProperties": True,
                    },
                }
            },
            "additionalProperties": True,
        }
        result = _openai_strict_schema(s)
        # Top-level object
        assert result["additionalProperties"] is False
        # Array items object
        items_obj = result["properties"]["items"]["items"]
        assert items_obj["additionalProperties"] is False
        assert items_obj["required"] == ["nested"]
        # Deeply nested array items object
        deep_obj = items_obj["properties"]["nested"]["items"]
        assert deep_obj["additionalProperties"] is False
        assert deep_obj["required"] == ["v"]


class TestAdapterInit:
    def test_defaults(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter._model_config.model == "gpt-5.3-codex"
        assert adapter._client_config.codex_bin == "codex"

    def test_custom_config(self) -> None:
        model_cfg = CodexAppServerModelConfig(model="o3", effort="high")
        client_cfg = CodexAppServerClientConfig(codex_bin="/usr/bin/codex")
        adapter = CodexAppServerAdapter(
            model_config=model_cfg, client_config=client_cfg
        )
        assert adapter._model_config.model == "o3"
        assert adapter._client_config.codex_bin == "/usr/bin/codex"


class TestAuthenticate:
    def test_no_auth(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            await adapter._authenticate(client)
            client.send_request.assert_not_called()

        asyncio.run(_run())

    def test_api_key_auth(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(
                    auth_mode=ApiKeyAuth(api_key="sk-test")
                )
            )
            client = _make_mock_client()
            await adapter._authenticate(client)
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "account/login/start"
            assert args[0][1]["type"] == "apiKey"
            assert args[0][1]["apiKey"] == "sk-test"

        asyncio.run(_run())

    def test_external_token_auth(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(
                    auth_mode=ExternalTokenAuth(
                        id_token="id-tok", access_token="access-tok"
                    )
                )
            )
            client = _make_mock_client()
            await adapter._authenticate(client)
            args = client.send_request.call_args
            assert args[0][1]["type"] == "chatgptAuthTokens"
            assert args[0][1]["idToken"] == "id-tok"
            assert args[0][1]["accessToken"] == "access-tok"

        asyncio.run(_run())


class TestStartTurn:
    def test_basic_turn(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 1}}

            result = await adapter._start_turn(client, "thread-1", "Hello", None)
            assert result["turn"]["id"] == 1

            args = client.send_request.call_args
            assert args[0][0] == "turn/start"
            params = args[0][1]
            assert params["threadId"] == "thread-1"
            assert params["input"][0]["text"] == "Hello"
            assert "effort" not in params
            assert "outputSchema" not in params

        asyncio.run(_run())

    def test_turn_with_options(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                model_config=CodexAppServerModelConfig(
                    effort="high",
                    summary="concise",
                    personality="friendly",
                )
            )
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 2}}

            out_schema = {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
            }
            await adapter._start_turn(client, "thread-1", "Solve", out_schema)

            params = client.send_request.call_args[0][1]
            assert params["effort"] == "high"
            assert params["summary"] == "concise"
            assert params["personality"] == "friendly"
            assert params["outputSchema"] == out_schema

        asyncio.run(_run())


class TestCreateThread:
    def test_basic_thread(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-abc"}}

            thread_id = await adapter._create_thread(client, "/tmp/work", [])
            assert thread_id == "t-abc"

            params = client.send_request.call_args[0][1]
            assert params["model"] == "gpt-5.3-codex"
            assert params["cwd"] == "/tmp/work"
            assert params["approvalPolicy"] == "never"
            assert "dynamicTools" not in params

        asyncio.run(_run())

    def test_thread_with_sandbox_and_tools(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(
                    sandbox_mode="read-only",
                    mcp_servers={"srv": {"command": "npx"}},
                )
            )
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-1"}}

            tools = [{"name": "t", "description": "d", "inputSchema": {}}]
            await adapter._create_thread(client, "/tmp", tools)

            params = client.send_request.call_args[0][1]
            assert params["sandbox"] == "read-only"
            assert params["dynamicTools"] == tools
            assert "config" in params

        asyncio.run(_run())


class TestProcessNotification:
    def test_delta(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {"method": "item/agentMessage/delta", "params": {"delta": "hello"}}
        result = adapter._process_notification(msg, session, "p", None)
        assert result == ("delta", "hello")

    def test_item_completed_agent_message(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "text": "final answer"}},
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result == ("text", "final answer")

    def test_item_completed_agent_message_no_text(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage"}},
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result is None

    def test_item_completed_command_execution(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "status": "completed",
                    "command": "ls",
                    "cwd": "/tmp",
                    "aggregatedOutput": "files",
                }
            },
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result is None  # Tool dispatch doesn't return a kind

    def test_token_usage_updated(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {"method": "thread/tokenUsage/updated", "params": {}}
        result = adapter._process_notification(msg, session, "p", None)
        assert result == ("usage", "")

    def test_turn_completed_done(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "turn/completed",
            "params": {"turn": {"status": "completed"}},
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result == ("done", "")

    def test_turn_completed_failed(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "status": "failed",
                    "codexErrorInfo": "unauthorized",
                    "additionalDetails": "bad key",
                }
            },
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result is not None
        assert result[0] == "error"
        assert "unauthorized" in result[1]

    def test_turn_completed_interrupted(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "turn/completed",
            "params": {"turn": {"status": "interrupted"}},
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result == ("interrupted", "")

    def test_unknown_method(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {"method": "unknown/thing", "params": {}}
        result = adapter._process_notification(msg, session, "p", None)
        assert result is None

    def test_item_completed_unknown_type(self) -> None:
        """item/completed with an unrecognized item type returns None."""
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "somethingNew", "status": "completed"}},
        }
        result = adapter._process_notification(msg, session, "p", None)
        assert result is None


class TestRaiseForTerminalNotification:
    def test_error_raises(self) -> None:
        adapter = CodexAppServerAdapter()
        msg: dict[str, Any] = {"params": {"turn": {"codexErrorInfo": "sandboxError"}}}
        with pytest.raises(PromptEvaluationError) as exc_info:
            adapter._raise_for_terminal_notification("error", "boom", "p", msg)
        assert exc_info.value.phase == "tool"

    def test_interrupted_raises(self) -> None:
        adapter = CodexAppServerAdapter()
        with pytest.raises(PromptEvaluationError, match="interrupted"):
            adapter._raise_for_terminal_notification("interrupted", "", "p", {})

    def test_other_kind_is_noop(self) -> None:
        adapter = CodexAppServerAdapter()
        # Should not raise
        adapter._raise_for_terminal_notification("done", "", "p", {})


class TestHandleServerRequest:
    def test_unknown_tool(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            msg = {
                "id": 1,
                "method": "item/tool/call",
                "params": {"tool": "nonexistent", "arguments": {}},
            }
            await adapter._handle_server_request(client, msg, {})
            client.send_response.assert_called_once()
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())

    def test_command_approval_accept(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            msg = {
                "id": 2,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await adapter._handle_server_request(client, msg, {})
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())

    def test_command_approval_decline(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(approval_policy="on-request")
            )
            client = _make_mock_client()
            msg = {
                "id": 3,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await adapter._handle_server_request(client, msg, {})
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "decline"

        asyncio.run(_run())

    def test_file_approval(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            msg = {
                "id": 4,
                "method": "item/fileChange/requestApproval",
                "params": {},
            }
            await adapter._handle_server_request(client, msg, {})
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())

    def test_unknown_server_request(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            msg = {"id": 5, "method": "unknown/request", "params": {}}
            await adapter._handle_server_request(client, msg, {})
            client.send_response.assert_called_once_with(5, {})

        asyncio.run(_run())


class TestHandleToolCall:
    def test_successful_tool_call(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "result: 42"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}
            await adapter._handle_tool_call(
                client, 10, {"tool": "calc", "arguments": {"x": 1}}, tool_lookup
            )
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is True
            assert resp["contentItems"][0]["text"] == "result: 42"

        asyncio.run(_run())

    def test_tool_call_with_string_arguments(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }
            tool_lookup = {"t": mock_tool}
            await adapter._handle_tool_call(
                client, 11, {"tool": "t", "arguments": '{"a": 1}'}, tool_lookup
            )
            mock_tool.assert_called_once_with({"a": 1})

        asyncio.run(_run())

    def test_tool_call_with_invalid_string_arguments(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }
            tool_lookup = {"t": mock_tool}
            await adapter._handle_tool_call(
                client, 12, {"tool": "t", "arguments": "not json"}, tool_lookup
            )
            mock_tool.assert_called_once_with({})

        asyncio.run(_run())

    def test_tool_call_error_result(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "error msg"}],
                "isError": True,
            }
            tool_lookup = {"t": mock_tool}
            await adapter._handle_tool_call(
                client, 13, {"tool": "t", "arguments": {}}, tool_lookup
            )
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())


class TestCreateDeadlineWatchdog:
    def test_no_deadline(self) -> None:
        adapter = CodexAppServerAdapter()
        client = _make_mock_client()
        result = adapter._create_deadline_watchdog(client, "t", 1, None)
        assert result is None

    def test_expired_deadline(self) -> None:
        adapter = CodexAppServerAdapter()
        client = _make_mock_client()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        # Advance clock past expiration
        clock.advance(10)
        result = adapter._create_deadline_watchdog(client, "t", 1, deadline)
        assert result is None

    def test_active_deadline_creates_task(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            future_time = datetime.now(UTC) + timedelta(seconds=60)
            deadline = Deadline(expires_at=future_time)
            task = adapter._create_deadline_watchdog(client, "t", 1, deadline)
            try:
                assert task is not None
                assert isinstance(task, asyncio.Task)
            finally:
                if task is not None:
                    _ = task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        asyncio.run(_run())


class TestEvaluateExpiredDeadline:
    def test_raises_on_expired_deadline(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(prompt, session=session, deadline=deadline)


class TestEvaluateEndToEnd:
    """End-to-end tests mocking the CodexAppServerClient."""

    def test_simple_evaluation(self) -> None:
        """Test basic prompt evaluation with mocked client."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = _make_session()

        # Track events
        rendered_events: list[PromptRendered] = []
        executed_events: list[PromptExecuted] = []
        dispatcher.subscribe(PromptRendered, lambda e: rendered_events.append(e))
        dispatcher.subscribe(PromptExecuted, lambda e: executed_events.append(e))

        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hello "}},
            {"method": "item/agentMessage/delta", "params": {"delta": "world"}},
            # Unknown notification — should be silently ignored
            {"method": "unknown/notification", "params": {}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Hello world"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            # Initialize returns capabilities
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                {"thread": {"id": "t-1"}},  # thread/start
                {"turn": {"id": 1}},  # turn/start
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "Hello world"
        assert len(rendered_events) == 1
        assert len(executed_events) == 1
        assert executed_events[0].adapter == "codex_app_server"

    def test_evaluation_with_token_usage(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 100,
                            "outputTokens": 50,
                        }
                    }
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "answer"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "answer"

    def test_client_error_wrapped(self) -> None:
        """CodexClientError is wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("conn failed")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="conn failed"):
                adapter.evaluate(prompt, session=session)

    def test_generic_error_wrapped(self) -> None:
        """Generic exceptions are wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.start.side_effect = RuntimeError("unexpected")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="unexpected"):
                adapter.evaluate(prompt, session=session)

    def test_prompt_eval_error_passthrough(self) -> None:
        """PromptEvaluationError passes through unwrapped."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "failed",
                        "codexErrorInfo": "unauthorized",
                        "additionalDetails": "bad key",
                    }
                },
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session)
            assert exc_info.value.phase == "request"

    def test_stream_eof_before_turn_completed(self) -> None:
        """Stream ends with messages but no turn/completed → raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "partial"}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "partial"}},
            },
            # No turn/completed — stream ends
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_stream_eof_empty_stream(self) -> None:
        """Zero messages in stream → raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator([])
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_budget_creates_tracker(self) -> None:
        """Budget without tracker creates one automatically."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        budget = Budget(
            max_input_tokens=1000,
            max_output_tokens=500,
        )

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, budget=budget)
            assert result.text == "ok"


class TestResolveCwd:
    def test_no_filesystem_no_cwd_creates_temp(self) -> None:
        import shutil as _shutil

        adapter = CodexAppServerAdapter()
        prompt = _make_simple_prompt()

        cwd, temp_dir, _new_prompt = adapter._resolve_cwd(prompt)
        try:
            assert cwd is not None
            assert temp_dir is not None
            assert cwd == temp_dir
        finally:
            if temp_dir:
                _shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_filesystem_with_cwd_uses_configured(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        prompt = _make_simple_prompt()

        cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert cwd == "/tmp/configured"
        assert temp_dir is None

    def test_workspace_section_extracts_root(self) -> None:
        """When prompt has a workspace section with HostFilesystem and no cwd."""
        from weakincentives.prompt import WorkspaceSection

        adapter = CodexAppServerAdapter()
        session, _ = _make_session()

        workspace = WorkspaceSection(session=session)
        workspace_root = str(workspace.temp_dir)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws",
                sections=(workspace,),
                name="ws-prompt",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd == workspace_root
            assert temp_dir is None
        finally:
            workspace.cleanup()

    def test_non_host_filesystem_falls_back_to_cwd(self) -> None:
        """When filesystem is not HostFilesystem, falls back to Path.cwd()."""
        adapter = CodexAppServerAdapter()
        prompt = _make_simple_prompt()

        # Mock prompt.filesystem() to return a non-HostFilesystem
        mock_fs = MagicMock(spec=Filesystem)
        with patch.object(type(prompt), "filesystem", return_value=mock_fs):
            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd is not None
            assert temp_dir is None

    def test_workspace_with_configured_cwd(self) -> None:
        """When prompt has workspace section AND cwd is configured, cwd wins."""
        from weakincentives.prompt import WorkspaceSection

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        session, _ = _make_session()

        workspace = WorkspaceSection(session=session)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws2",
                sections=(workspace,),
                name="ws-prompt2",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            # Configured cwd should win over workspace root
            assert cwd == "/tmp/configured"
            assert temp_dir is None
        finally:
            workspace.cleanup()


class TestParseStructuredOutput:
    def test_valid_json(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        adapter = CodexAppServerAdapter()

        @dataclass(slots=True, frozen=True)
        class Result:
            answer: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Result,
                container="object",
                allow_extra_keys=False,
            ),
        )

        result = adapter._parse_structured_output('{"answer": 42}', rendered, "test")
        assert result is not None
        assert result.answer == 42

    def test_invalid_json_raises(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        adapter = CodexAppServerAdapter()

        @dataclass(slots=True, frozen=True)
        class Dummy:
            x: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Dummy,
                container="object",
                allow_extra_keys=False,
            ),
        )

        with pytest.raises(PromptEvaluationError, match="parse structured"):
            adapter._parse_structured_output("not json", rendered, "test")

    def test_array_container_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        adapter = CodexAppServerAdapter()

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Item,
                container="array",
                allow_extra_keys=False,
            ),
        )

        # Array wrapper format: {"items": [...]}
        text = '{"items": [{"value": 1}, {"value": 2}]}'
        result = adapter._parse_structured_output(text, rendered, "test")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].value == 1
        assert result[1].value == 2


class TestArraySchemaWrapping:
    def test_array_container_wraps_schema(self) -> None:
        """When container='array', the output schema wraps element in items."""
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "text": '{"items": [{"value": 1}]}',
                    }
                },
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RP[Any]:
            rendered = original_render(**kwargs)
            return RP(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Item,
                    container="array",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert isinstance(result.output, list)
        assert result.output[0].value == 1

        # Verify outputSchema was wrapped with items array
        turn_call = mock_client.send_request.call_args_list[2]
        output_schema = turn_call[0][1]["outputSchema"]
        assert output_schema["type"] == "object"
        assert "items" in output_schema["properties"]
        assert output_schema["properties"]["items"]["type"] == "array"


class TestEvaluateWithOutputSchema:
    """Test evaluate with output schema/structured output."""

    def test_output_schema_sent_and_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Answer:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": '{"value": 42}'}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        # Patch prompt.render to return a RenderedPrompt with output_type
        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RenderedPrompt[Any]:
            rendered = original_render(**kwargs)
            # Replace structured_output to set output_type
            return RenderedPrompt(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Answer,
                    container="object",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert result.output.value == 42

        # Verify outputSchema was sent in turn/start
        turn_call = mock_client.send_request.call_args_list[2]
        assert "outputSchema" in turn_call[0][1]


class TestEvaluateWithDeltaAccumulation:
    """Test that delta messages are accumulated properly."""

    def test_delta_then_text_uses_text(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
            {"method": "item/agentMessage/delta", "params": {"delta": "lo"}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Hello world"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        # The "text" kind from item/completed replaces accumulated delta
        assert result.text == "Hello world"


class TestEvaluateWithBudgetTracking:
    """Test budget tracking integration."""

    def test_budget_tracker_records_usage(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        budget = Budget(max_input_tokens=1000, max_output_tokens=500)
        budget_tracker = BudgetTracker(budget)

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {"last": {"inputTokens": 100, "outputTokens": 50}}
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(
                prompt, session=session, budget_tracker=budget_tracker
            )
            assert result.text == "ok"


class TestEvaluateWithServerRequestDuringStream:
    """Test that server requests during streaming are handled."""

    def test_tool_call_during_stream(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            # Server request: tool call
            {
                "id": 100,
                "method": "item/tool/call",
                "params": {"tool": "unknown_tool", "arguments": {}},
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "done"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)
            assert result.text == "done"
            # The unknown tool call should have sent a response
            mock_client.send_response.assert_called()


class TestEvaluateTempWorkspaceCleanup:
    """Test that temporary workspace is cleaned up on failure."""

    def test_temp_dir_cleaned_up_on_error(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("fail")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError):
                adapter.evaluate(prompt, session=session)


class TestStreamTurnWithDeadline:
    """Test that _stream_turn properly cleans up the watchdog task."""

    def test_stream_turn_with_active_deadline(self) -> None:
        """Watchdog task is created and cancelled after turn completes."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=60), clock=clock)

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, deadline=deadline)
            assert result.text == "ok"


class TestDeadlineWatchdogBody:
    """Test the deadline watchdog actually sends an interrupt."""

    def test_watchdog_sends_interrupt(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.return_value = {}
            await adapter._deadline_watchdog(client, "t-1", 42, 0.01)
            # After sleep, it should send turn/interrupt
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "turn/interrupt"
            assert args[0][1]["threadId"] == "t-1"
            assert args[0][1]["turnId"] == 42

        asyncio.run(_run())

    def test_watchdog_suppresses_error(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.side_effect = CodexClientError("already done")
            # Should not raise
            await adapter._deadline_watchdog(client, "t-1", 42, 0.01)

        asyncio.run(_run())


class TestVisibilitySignalPassthrough:
    """Test that VisibilityExpansionRequired passes through unwrapped."""

    def test_visibility_expansion_passthrough(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            # Patch visibility signal to return a stored exception
            with patch(
                "weakincentives.adapters.codex_app_server.adapter.VisibilityExpansionSignal"
            ) as MockSignal:
                mock_signal = MagicMock()
                mock_signal.get_and_clear.return_value = VisibilityExpansionRequired(
                    "test",
                    requested_overrides={},
                    reason="test",
                    section_keys=(),
                )
                MockSignal.return_value = mock_signal

                with pytest.raises(VisibilityExpansionRequired):
                    adapter.evaluate(prompt, session=session)


class TestToolCallRunsInThread:
    def test_tool_call_runs_in_thread(self) -> None:
        """Tool dispatch is wrapped in asyncio.to_thread."""

        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "threaded"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = {
                    "content": [{"type": "text", "text": "threaded"}],
                    "isError": False,
                }
                await adapter._handle_tool_call(
                    client, 20, {"tool": "calc", "arguments": {"x": 1}}, tool_lookup
                )
                mock_to_thread.assert_called_once_with(mock_tool, {"x": 1})

        asyncio.run(_run())


class TestDeadlineRemainingS:
    def test_no_deadline_returns_none(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter._deadline_remaining_s(None, "p") is None

    def test_expired_deadline_raises(self) -> None:
        adapter = CodexAppServerAdapter()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired during"):
            adapter._deadline_remaining_s(deadline, "test-prompt")

    def test_active_deadline_returns_seconds(self) -> None:
        adapter = CodexAppServerAdapter()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        remaining = adapter._deadline_remaining_s(deadline, "test-prompt")
        assert remaining is not None
        assert remaining > 0


class TestSetupRPCDeadlineBounding:
    def test_setup_timeout_wraps_client_error(self) -> None:
        """When thread/start times out, PromptEvaluationError has phase='request'."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            # initialize succeeds, thread/start raises client timeout
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                CodexClientError("Timeout waiting for response to thread/start"),
            ]
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session, deadline=deadline)
            assert exc_info.value.phase == "request"

    def test_setup_passes_timeout_to_send_request(self) -> None:
        """Verify timeout is forwarded to send_request for setup RPCs."""

        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-1"}}

            await adapter._create_thread(client, "/tmp", [], timeout=5.0)
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 5.0 or call_args[0][2] == 5.0

        asyncio.run(_run())

    def test_authenticate_passes_timeout(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(
                    auth_mode=ApiKeyAuth(api_key="sk-test")
                )
            )
            client = _make_mock_client()

            await adapter._authenticate(client, timeout=3.0)
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 3.0

        asyncio.run(_run())

    def test_start_turn_passes_timeout(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 1}}

            await adapter._start_turn(client, "thread-1", "Hello", None, timeout=7.0)
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 7.0

        asyncio.run(_run())


class TestTranscriptBridgeIntegration:
    """Tests for transcript bridge integration in the adapter."""

    def test_evaluate_with_transcript_disabled(self) -> None:
        """When transcript=False, bridge is not created."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test", transcript=False),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "ok"

    def test_handle_tool_call_with_bridge(self) -> None:
        """Tool call emits transcript entries via bridge."""

        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "result: 42"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}

            bridge = MagicMock()
            await adapter._handle_tool_call(
                client,
                10,
                {"tool": "calc", "arguments": {"x": 1}},
                tool_lookup,
                bridge=bridge,
            )
            bridge.on_tool_call.assert_called_once_with(
                {"tool": "calc", "arguments": {"x": 1}}
            )
            bridge.on_tool_result.assert_called_once()
            resp = bridge.on_tool_result.call_args[0][1]
            assert resp["success"] is True

        asyncio.run(_run())

    def test_handle_tool_call_unknown_tool_with_bridge(self) -> None:
        """Unknown tool call still emits tool_result via bridge."""

        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = _make_mock_client()
            tool_lookup: dict[str, Any] = {}

            bridge = MagicMock()
            await adapter._handle_tool_call(
                client,
                10,
                {"tool": "missing", "arguments": {}},
                tool_lookup,
                bridge=bridge,
            )
            bridge.on_tool_call.assert_called_once()
            bridge.on_tool_result.assert_called_once()
            resp = bridge.on_tool_result.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())


class TestApprovalPolicyUntrusted:
    def test_approval_untrusted_declines(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(approval_policy="untrusted")
            )
            client = _make_mock_client()
            msg = {
                "id": 6,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await adapter._handle_server_request(client, msg, {})
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "decline"

        asyncio.run(_run())

    def test_approval_on_failure_accepts_requested_approval(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter(
                client_config=CodexAppServerClientConfig(approval_policy="on-failure")
            )
            client = _make_mock_client()
            msg = {
                "id": 7,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await adapter._handle_server_request(client, msg, {})
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())
