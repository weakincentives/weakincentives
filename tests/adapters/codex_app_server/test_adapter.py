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
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from weakincentives.adapters.codex_app_server._protocol import (
    authenticate,
    create_thread,
    handle_item_completed_notification,
    handle_tool_call,
    handle_turn_completed_notification,
    start_turn,
)
from weakincentives.adapters.codex_app_server._schema import (
    bridged_tools_to_dynamic_specs,
)
from weakincentives.adapters.codex_app_server.adapter import (
    CODEX_APP_SERVER_ADAPTER_NAME,
    CodexAppServerAdapter,
)
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
)
from weakincentives.adapters.codex_app_server.config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    ExternalTokenAuth,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate, Tool
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate.create(
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


@dataclass(slots=True, frozen=True)
class _AddResult:
    sum: int


def _add_handler(params: _AddParams, *, context: ToolContext) -> ToolResult[_AddResult]:
    return ToolResult.ok(
        _AddResult(sum=params.x + params.y), message=str(params.x + params.y)
    )


_ADD_TOOL = Tool[_AddParams, _AddResult].create(
    name="add",
    description="Add two numbers",
    handler=_add_handler,
)


def _make_prompt_with_tool(name: str = "tool-prompt") -> Prompt[object]:
    section = MarkdownSection(
        title="Tools",
        template="Use the tools below.",
        key="tools",
        tools=[_ADD_TOOL],
    )
    template: PromptTemplate[object] = PromptTemplate.create(
        ns="test",
        key="with-tool",
        sections=(section,),
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
        assert bridged_tools_to_dynamic_specs(()) == []

    def test_converts_tools(self) -> None:
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does something"
        tool.input_schema = {"type": "object", "properties": {}}
        specs = bridged_tools_to_dynamic_specs((tool,))
        assert len(specs) == 1
        assert specs[0]["name"] == "my_tool"
        assert specs[0]["description"] == "Does something"
        assert specs[0]["inputSchema"] == {"type": "object", "properties": {}}


class TestAdapterInit:
    def test_defaults(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter._model_config.model == "gpt-5.4"
        assert adapter._codex_client_config.codex_bin == "codex"

    def test_custom_config(self) -> None:
        model_cfg = CodexAppServerModelConfig(model="o3", effort="high")
        client_cfg = CodexAppServerClientConfig(codex_bin="/usr/bin/codex")
        adapter = CodexAppServerAdapter(
            model_config=model_cfg, client_config=client_cfg
        )
        assert adapter._model_config.model == "o3"
        assert adapter._codex_client_config.codex_bin == "/usr/bin/codex"

    def test_adapter_name_property(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter.adapter_name == "codex_app_server"


class TestAuthenticate:
    def test_no_auth(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            await authenticate(client, None)
            client.send_request.assert_not_called()

        asyncio.run(_run())

    def test_api_key_auth(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            await authenticate(client, ApiKeyAuth(api_key="sk-test"))
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "account/login/start"
            assert args[0][1]["type"] == "apiKey"
            assert args[0][1]["apiKey"] == "sk-test"

        asyncio.run(_run())

    def test_external_token_auth(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            await authenticate(
                client,
                ExternalTokenAuth(id_token="id-tok", access_token="access-tok"),
            )
            args = client.send_request.call_args
            assert args[0][1]["type"] == "chatgptAuthTokens"
            assert args[0][1]["idToken"] == "id-tok"
            assert args[0][1]["accessToken"] == "access-tok"

        asyncio.run(_run())


class TestStartTurn:
    def test_basic_turn(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 1}}

            result = await start_turn(
                client,
                "thread-1",
                "Hello",
                None,
                model_config=CodexAppServerModelConfig(),
            )
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
            model_config = CodexAppServerModelConfig(
                effort="high",
                summary="concise",
                personality="friendly",
            )
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 2}}

            out_schema = {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
            }
            await start_turn(
                client,
                "thread-1",
                "Solve",
                out_schema,
                model_config=model_config,
            )

            params = client.send_request.call_args[0][1]
            assert params["effort"] == "high"
            assert params["summary"] == "concise"
            assert params["personality"] == "friendly"
            assert params["outputSchema"] == out_schema

        asyncio.run(_run())


class TestCreateThread:
    def test_basic_thread(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-abc"}}

            thread_id = await create_thread(
                client,
                "/tmp/work",
                [],
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
            )
            assert thread_id == "t-abc"

            params = client.send_request.call_args[0][1]
            assert params["model"] == "gpt-5.4"
            assert params["cwd"] == "/tmp/work"
            assert params["approvalPolicy"] == "never"
            assert "dynamicTools" not in params

        asyncio.run(_run())

    def test_thread_with_sandbox_and_tools(self) -> None:
        async def _run() -> None:
            client_config = CodexAppServerClientConfig(
                sandbox_mode="read-only",
                mcp_servers={"srv": {"command": "npx"}},
            )
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-1"}}

            tools = [{"name": "t", "description": "d", "inputSchema": {}}]
            await create_thread(
                client,
                "/tmp",
                tools,
                client_config=client_config,
                model_config=CodexAppServerModelConfig(),
            )

            params = client.send_request.call_args[0][1]
            assert params["sandbox"] == "read-only"
            assert params["dynamicTools"] == tools
            assert "config" in params

        asyncio.run(_run())


class TestHandleToolCall:
    def test_successful_tool_call(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "result: 42"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}
            await handle_tool_call(
                client, 10, {"tool": "calc", "arguments": {"x": 1}}, tool_lookup
            )
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is True
            assert resp["contentItems"][0]["text"] == "result: 42"

        asyncio.run(_run())

    def test_tool_call_with_string_arguments(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }
            tool_lookup = {"t": mock_tool}
            await handle_tool_call(
                client, 11, {"tool": "t", "arguments": '{"a": 1}'}, tool_lookup
            )
            mock_tool.assert_called_once_with({"a": 1})

        asyncio.run(_run())

    def test_tool_call_with_invalid_string_arguments(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }
            tool_lookup = {"t": mock_tool}
            await handle_tool_call(
                client, 12, {"tool": "t", "arguments": "not json"}, tool_lookup
            )
            mock_tool.assert_called_once_with({})

        asyncio.run(_run())

    def test_tool_call_error_result(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "error msg"}],
                "isError": True,
            }
            tool_lookup = {"t": mock_tool}
            await handle_tool_call(
                client, 13, {"tool": "t", "arguments": {}}, tool_lookup
            )
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())


class TestNotificationHandlers:
    """Test the registry notification handlers directly."""

    def test_item_completed_agent_message_with_text(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {"item": {"type": "agentMessage", "text": "hello"}}
        result = handle_item_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        assert result == ("text", "hello")

    def test_item_completed_agent_message_without_text(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {"item": {"type": "agentMessage"}}
        result = handle_item_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        assert result is None

    def test_item_completed_command_execution(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {
            "item": {
                "type": "commandExecution",
                "id": "c-1",
                "status": "completed",
                "command": "ls",
                "cwd": "/tmp",
                "aggregatedOutput": "file1\n",
            }
        }
        result = handle_item_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        # Tool items return None; ToolInvoked is dispatched as side effect
        assert result is None

    def test_item_completed_unknown_type(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {"item": {"type": "unknownKind"}}
        result = handle_item_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        assert result is None

    def test_turn_completed_interrupted(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {"turn": {"status": "interrupted"}}
        result = handle_turn_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        assert result == ("interrupted", "")

    def test_turn_completed_failed(self) -> None:
        session, _ = _make_session()
        params: dict[str, Any] = {
            "turn": {
                "status": "failed",
                "codexErrorInfo": "badRequest",
                "additionalDetails": "oops",
            }
        }
        result = handle_turn_completed_notification(
            params, session, "codex_app_server", "p", None
        )
        assert result[0] == "error"
        assert "badRequest" in result[1]


class TestCodexSendInterrupt:
    """Exercise the real _send_interrupt body on CodexAppServerAdapter."""

    def test_send_interrupt_calls_turn_interrupt(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = AsyncMock()
            client.send_request = AsyncMock(return_value={})

            await adapter._send_interrupt(
                client,
                session_state="thread-1",
                turn_state={"turn": {"id": 42}},
            )
            client.send_request.assert_called_once()
            call_args = client.send_request.call_args
            assert call_args[0][0] == "turn/interrupt"
            assert call_args[0][1] == {"threadId": "thread-1", "turnId": 42}

        asyncio.run(_run())


class TestHandleToolCallUnknownWithBridge:
    """Covers bridge branch when tool is not found."""

    def test_unknown_tool_emits_transcript(self) -> None:
        async def _run() -> None:
            client = AsyncMock()
            client.send_response = AsyncMock()
            bridge = MagicMock()
            await handle_tool_call(
                client,
                99,
                {"tool": "missing", "arguments": {}},
                {},
                bridge=bridge,
            )
            bridge.on_tool_call.assert_called_once()
            bridge.on_tool_result.assert_called_once()

        asyncio.run(_run())

    def test_unknown_tool_no_bridge(self) -> None:
        """Covers branch where bridge is None in the unknown-tool path."""

        async def _run() -> None:
            client = AsyncMock()
            client.send_response = AsyncMock()
            await handle_tool_call(
                client,
                100,
                {"tool": "missing", "arguments": {}},
                {},
                bridge=None,
            )
            client.send_response.assert_called_once()
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())
