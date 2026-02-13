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

"""Tests for ACP MCP HTTP server wrapper."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp._mcp_http import (
    MCPHttpServer,
    _get_header,
    _make_asgi_app,
    _send_401,
    create_mcp_tool_server,
)

from .conftest import MockHttpHeader, MockHttpMcpServer


class TestMakeAsgiApp:
    _TOKEN = "test-token-abc"

    def test_delegates_http_to_transport(self) -> None:
        transport = MagicMock()
        transport.handle_request = AsyncMock()

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)
            scope: dict[str, Any] = {
                "type": "http",
                "path": "/mcp",
                "headers": [(b"authorization", b"Bearer test-token-abc")],
            }
            receive = MagicMock()
            send = MagicMock()
            await app(scope, receive, send)
            transport.handle_request.assert_called_once_with(scope, receive, send)

        asyncio.run(_run())

    def test_handles_lifespan_startup_and_shutdown(self) -> None:
        transport = MagicMock()
        messages = iter(
            [
                {"type": "lifespan.startup"},
                {"type": "lifespan.shutdown"},
            ]
        )
        sent: list[dict[str, str]] = []

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)

            async def receive() -> dict[str, str]:
                return next(messages)

            async def send(msg: dict[str, str]) -> None:
                sent.append(msg)

            await app({"type": "lifespan"}, receive, send)

        asyncio.run(_run())
        assert sent == [
            {"type": "lifespan.startup.complete"},
            {"type": "lifespan.shutdown.complete"},
        ]

    def test_skips_unknown_lifespan_messages(self) -> None:
        transport = MagicMock()
        messages = iter(
            [
                {"type": "lifespan.unknown_event"},
                {"type": "lifespan.startup"},
                {"type": "lifespan.shutdown"},
            ]
        )
        sent: list[dict[str, str]] = []

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)

            async def receive() -> dict[str, str]:
                return next(messages)

            async def send(msg: dict[str, str]) -> None:
                sent.append(msg)

            await app({"type": "lifespan"}, receive, send)

        asyncio.run(_run())
        assert sent == [
            {"type": "lifespan.startup.complete"},
            {"type": "lifespan.shutdown.complete"},
        ]

    def test_ignores_unknown_scope_types(self) -> None:
        transport = MagicMock()

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)
            await app({"type": "websocket"}, MagicMock(), MagicMock())
            transport.handle_request.assert_not_called()

        asyncio.run(_run())

    def test_rejects_missing_auth_header(self) -> None:
        transport = MagicMock()
        transport.handle_request = AsyncMock()
        sent: list[dict[str, Any]] = []

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)
            scope: dict[str, Any] = {"type": "http", "path": "/mcp", "headers": []}

            async def send(msg: dict[str, Any]) -> None:
                sent.append(msg)

            await app(scope, MagicMock(), send)
            transport.handle_request.assert_not_called()

        asyncio.run(_run())
        assert sent[0]["status"] == 401

    def test_rejects_wrong_token(self) -> None:
        transport = MagicMock()
        transport.handle_request = AsyncMock()
        sent: list[dict[str, Any]] = []

        async def _run() -> None:
            app = _make_asgi_app(transport, self._TOKEN)
            scope: dict[str, Any] = {
                "type": "http",
                "path": "/mcp",
                "headers": [(b"authorization", b"Bearer wrong-token")],
            }

            async def send(msg: dict[str, Any]) -> None:
                sent.append(msg)

            await app(scope, MagicMock(), send)
            transport.handle_request.assert_not_called()

        asyncio.run(_run())
        assert sent[0]["status"] == 401


class TestGetHeader:
    def test_finds_header(self) -> None:
        headers = [(b"content-type", b"text/plain"), (b"authorization", b"Bearer x")]
        assert _get_header(headers, b"authorization") == b"Bearer x"

    def test_case_insensitive(self) -> None:
        headers = [(b"Authorization", b"Bearer y")]
        assert _get_header(headers, b"authorization") == b"Bearer y"

    def test_missing_header(self) -> None:
        headers = [(b"content-type", b"text/plain")]
        assert _get_header(headers, b"authorization") is None


class TestSend401:
    def test_sends_401_response(self) -> None:
        sent: list[dict[str, Any]] = []

        async def _run() -> None:
            async def send(msg: dict[str, Any]) -> None:
                sent.append(msg)

            await _send_401(send)

        asyncio.run(_run())
        assert sent[0]["status"] == 401
        assert sent[1]["body"] == b"Unauthorized"


class TestMCPHttpServerProperties:
    def test_server_name_default(self) -> None:
        server = MCPHttpServer(MagicMock())
        assert server.server_name == "wink-tools"

    def test_server_name_custom(self) -> None:
        server = MCPHttpServer(MagicMock(), server_name="custom")
        assert server.server_name == "custom"

    def test_port_raises_before_start(self) -> None:
        server = MCPHttpServer(MagicMock())
        with pytest.raises(RuntimeError, match="Server not started"):
            _ = server.port

    def test_url_uses_port(self) -> None:
        server = MCPHttpServer(MagicMock())
        server._port = 12345
        assert server.url == "http://127.0.0.1:12345/mcp"

    def test_bearer_token_nonempty(self) -> None:
        server = MCPHttpServer(MagicMock())
        assert isinstance(server.bearer_token, str)
        assert len(server.bearer_token) > 0

    def test_two_instances_different_tokens(self) -> None:
        s1 = MCPHttpServer(MagicMock())
        s2 = MCPHttpServer(MagicMock())
        assert s1.bearer_token != s2.bearer_token


class TestMCPHttpServerToHttpMcpServer:
    def test_constructs_config_with_auth_header(self) -> None:
        server = MCPHttpServer(MagicMock(), server_name="test-srv")
        server._port = 9999

        mock_acp_schema = MagicMock()
        mock_acp_schema.HttpMcpServer = MockHttpMcpServer
        mock_acp_schema.HttpHeader = MockHttpHeader
        with patch.dict(
            sys.modules, {"acp.schema": mock_acp_schema, "acp": MagicMock()}
        ):
            result = server.to_http_mcp_server()

        assert result.url == "http://127.0.0.1:9999/mcp"
        assert result.name == "test-srv"
        assert result.type == "http"
        assert len(result.headers) == 1
        header = result.headers[0]
        assert header.name == "Authorization"
        assert header.value == f"Bearer {server.bearer_token}"


def _mock_server_deps() -> dict[str, MagicMock]:
    """Build mock modules for start() dependencies."""
    mock_transport = MagicMock()

    mock_streamable = MagicMock()
    mock_streamable.StreamableHTTPServerTransport.return_value = mock_transport

    mock_uvicorn_server = MagicMock()
    mock_uvicorn_server.run = MagicMock()
    mock_uvicorn = MagicMock()
    mock_uvicorn.Config.return_value = MagicMock()
    mock_uvicorn.Server.return_value = mock_uvicorn_server

    return {
        "mcp": MagicMock(),
        "mcp.server": MagicMock(),
        "mcp.server.streamable_http": mock_streamable,
        "mcp.types": MagicMock(),
        "uvicorn": mock_uvicorn,
    }


class TestMCPHttpServerLifecycle:
    def test_start_and_stop(self) -> None:
        async def _run() -> None:
            server = MCPHttpServer(MagicMock(), server_name="test")
            mock_thread = MagicMock()
            # Simulate _run_server setting the port after uvicorn binds
            mock_thread.start = MagicMock(
                side_effect=lambda: setattr(server, "_port", 54321)
            )

            async def fake_to_thread(fn: Any, *args: Any) -> None:
                return None

            with (
                patch.dict(sys.modules, _mock_server_deps()),
                patch("threading.Thread", return_value=mock_thread),
                patch("asyncio.to_thread", fake_to_thread),
            ):
                await server.start()
                assert server._port == 54321
                assert server._thread is mock_thread
                assert server.url == "http://127.0.0.1:54321/mcp"
                assert server._startup_error is None
            await server.stop()
            assert server._port is None
            assert server._thread is None

        asyncio.run(_run())

    def test_stop_without_start(self) -> None:
        async def _run() -> None:
            server = MCPHttpServer(MagicMock())
            await server.stop()

        asyncio.run(_run())

    def test_context_manager(self) -> None:
        async def _run() -> None:
            server = MCPHttpServer(MagicMock(), server_name="ctx")
            mock_thread = MagicMock()
            # Simulate _run_server setting the port after uvicorn binds
            mock_thread.start = MagicMock(
                side_effect=lambda: setattr(server, "_port", 54321)
            )

            async def fake_to_thread(fn: Any, *args: Any) -> None:
                return None

            with (
                patch.dict(sys.modules, _mock_server_deps()),
                patch("threading.Thread", return_value=mock_thread),
                patch("asyncio.to_thread", fake_to_thread),
            ):
                async with server as srv:
                    assert srv is server
                    assert srv._port == 54321
            assert server._port is None

        asyncio.run(_run())

    def test_startup_error_propagated(self) -> None:
        """If the background thread fails during startup, start() re-raises."""

        async def _run() -> None:
            server = MCPHttpServer(MagicMock(), server_name="fail")
            mock_thread = MagicMock()
            mock_thread.start.side_effect = lambda: setattr(
                server, "_startup_error", RuntimeError("bind failed")
            )

            async def fake_to_thread(fn: Any, *args: Any) -> None:
                return None

            with (
                patch.dict(sys.modules, _mock_server_deps()),
                patch("threading.Thread", return_value=mock_thread),
                patch("asyncio.to_thread", fake_to_thread),
            ):
                with pytest.raises(RuntimeError, match="bind failed"):
                    await server.start()

        asyncio.run(_run())


class TestCreateMcpToolServer:
    def test_creates_server_with_tools(self) -> None:
        """create_mcp_tool_server registers list_tools and call_tool handlers."""
        bt = MagicMock()
        bt.name = "my_tool"
        bt.description = "A test tool"
        bt.input_schema = {"type": "object", "properties": {}}

        mock_server = MagicMock()
        mock_tool_cls = MagicMock()
        mock_text_content_cls = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(Server=MagicMock(return_value=mock_server)),
                "mcp.types": MagicMock(
                    Tool=mock_tool_cls, TextContent=mock_text_content_cls
                ),
            },
        ):
            result = create_mcp_tool_server((bt,))

        assert result is mock_server
        # list_tools and call_tool decorators should have been called
        mock_server.list_tools.assert_called_once()
        mock_server.call_tool.assert_called_once()

    def test_call_tool_dispatches_to_bridged_tool(self) -> None:
        """call_tool handler invokes the correct BridgedTool."""
        bt = MagicMock()
        bt.name = "greet"
        bt.description = "Greet user"
        bt.input_schema = {"type": "object", "properties": {}}
        bt.return_value = {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }

        # Capture the handlers registered via decorators
        registered_handlers: dict[str, Any] = {}

        mock_server = MagicMock()

        def _capture_list_tools() -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers["list_tools"] = fn
                return fn

            return decorator

        def _capture_call_tool() -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers["call_tool"] = fn
                return fn

            return decorator

        mock_server.list_tools = _capture_list_tools
        mock_server.call_tool = _capture_call_tool

        mock_tool_cls = MagicMock(side_effect=lambda **kw: kw)
        mock_text_content_cls = MagicMock(side_effect=lambda **kw: kw)

        with patch.dict(
            sys.modules,
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(Server=MagicMock(return_value=mock_server)),
                "mcp.types": MagicMock(
                    Tool=mock_tool_cls, TextContent=mock_text_content_cls
                ),
            },
        ):
            create_mcp_tool_server((bt,))

        # Verify list_tools returns the tool
        tools = asyncio.run(registered_handlers["list_tools"]())
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"

        # Verify call_tool dispatches to BridgedTool
        result = asyncio.run(
            registered_handlers["call_tool"]("greet", {"name": "world"})
        )
        bt.assert_called_once_with({"name": "world"})
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_call_tool_unknown_returns_error(self) -> None:
        """call_tool returns error text for unknown tool name."""
        registered_handlers: dict[str, Any] = {}
        mock_server = MagicMock()

        def _capture_call_tool() -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers["call_tool"] = fn
                return fn

            return decorator

        mock_server.list_tools = lambda: lambda fn: fn
        mock_server.call_tool = _capture_call_tool

        mock_text_content_cls = MagicMock(side_effect=lambda **kw: kw)

        with patch.dict(
            sys.modules,
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(Server=MagicMock(return_value=mock_server)),
                "mcp.types": MagicMock(
                    Tool=MagicMock(), TextContent=mock_text_content_cls
                ),
            },
        ):
            create_mcp_tool_server(())

        result = asyncio.run(registered_handlers["call_tool"]("nonexistent", {}))
        assert len(result) == 1
        assert "Unknown tool" in result[0]["text"]

    def test_call_tool_none_arguments(self) -> None:
        """call_tool passes empty dict when arguments is None."""
        bt = MagicMock()
        bt.name = "no_args"
        bt.description = "No args tool"
        bt.input_schema = {"type": "object", "properties": {}}
        bt.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        }

        registered_handlers: dict[str, Any] = {}
        mock_server = MagicMock()

        def _capture_call_tool() -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers["call_tool"] = fn
                return fn

            return decorator

        mock_server.list_tools = lambda: lambda fn: fn
        mock_server.call_tool = _capture_call_tool

        mock_text_content_cls = MagicMock(side_effect=lambda **kw: kw)

        with patch.dict(
            sys.modules,
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(Server=MagicMock(return_value=mock_server)),
                "mcp.types": MagicMock(
                    Tool=MagicMock(), TextContent=mock_text_content_cls
                ),
            },
        ):
            create_mcp_tool_server((bt,))

        asyncio.run(registered_handlers["call_tool"]("no_args", None))
        bt.assert_called_once_with({})
