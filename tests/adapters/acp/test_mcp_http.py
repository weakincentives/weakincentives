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
    _find_free_port,
    _make_asgi_app,
)

from .conftest import MockHttpMcpServer


class TestFindFreePort:
    def test_returns_positive_integer(self) -> None:
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_returns_different_ports(self) -> None:
        p1 = _find_free_port()
        p2 = _find_free_port()
        assert isinstance(p1, int)
        assert isinstance(p2, int)


class TestMakeAsgiApp:
    def test_delegates_http_to_transport(self) -> None:
        transport = MagicMock()
        transport.handle_request = AsyncMock()

        async def _run() -> None:
            app = _make_asgi_app(transport)
            scope: dict[str, Any] = {"type": "http", "path": "/mcp"}
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
            app = _make_asgi_app(transport)

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
            app = _make_asgi_app(transport)

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
            app = _make_asgi_app(transport)
            await app({"type": "websocket"}, MagicMock(), MagicMock())
            transport.handle_request.assert_not_called()

        asyncio.run(_run())


class TestMCPHttpServerProperties:
    def test_server_name_default(self) -> None:
        server = MCPHttpServer(mcp_server_config={})
        assert server.server_name == "wink-tools"

    def test_server_name_custom(self) -> None:
        server = MCPHttpServer(mcp_server_config={}, server_name="custom")
        assert server.server_name == "custom"

    def test_port_raises_before_start(self) -> None:
        server = MCPHttpServer(mcp_server_config={})
        with pytest.raises(RuntimeError, match="Server not started"):
            _ = server.port

    def test_url_uses_port(self) -> None:
        server = MCPHttpServer(mcp_server_config={})
        server._port = 12345
        assert server.url == "http://127.0.0.1:12345/mcp"


class TestMCPHttpServerToHttpMcpServer:
    def test_constructs_config(self) -> None:
        server = MCPHttpServer(mcp_server_config={}, server_name="test-srv")
        server._port = 9999

        mock_acp_schema = MagicMock()
        mock_acp_schema.HttpMcpServer = MockHttpMcpServer
        with patch.dict(
            sys.modules, {"acp.schema": mock_acp_schema, "acp": MagicMock()}
        ):
            result = server.to_http_mcp_server()

        assert result.url == "http://127.0.0.1:9999/mcp"
        assert result.name == "test-srv"
        assert result.headers == []
        assert result.type == "http"


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
            config: dict[str, Any] = {"instance": MagicMock()}
            server = MCPHttpServer(mcp_server_config=config, server_name="test")
            await server.start()
            assert server._port is not None
            assert server._thread is not None
            assert server.url.startswith("http://127.0.0.1:")
            await server.stop()
            assert server._port is None
            assert server._thread is None

        with patch.dict(sys.modules, _mock_server_deps()):
            asyncio.run(_run())

    def test_stop_without_start(self) -> None:
        async def _run() -> None:
            server = MCPHttpServer(mcp_server_config={})
            await server.stop()

        asyncio.run(_run())

    def test_context_manager(self) -> None:
        async def _run() -> None:
            config: dict[str, Any] = {"instance": MagicMock()}
            server = MCPHttpServer(mcp_server_config=config, server_name="ctx")
            async with server as srv:
                assert srv is server
                assert srv._port is not None
            assert server._port is None

        with patch.dict(sys.modules, _mock_server_deps()):
            asyncio.run(_run())
