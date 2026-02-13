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

"""HTTP transport for exposing an in-process MCP server."""

from __future__ import annotations

import asyncio
import secrets
import threading
from typing import TYPE_CHECKING, Any

from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from .._shared._bridge import BridgedTool

__all__ = ["MCPHttpServer", "create_mcp_tool_server"]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_mcp_http"})


def _get_header(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes | None:
    """Extract a header value by lowercase name from ASGI headers."""
    for k, v in headers:
        if k.lower() == name:
            return v
    return None


async def _send_401(send: Any) -> None:
    """Send a 401 Unauthorized ASGI response."""
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"Unauthorized"})


def _make_asgi_app(transport: Any, bearer_token: str) -> Any:
    """Build a minimal ASGI app that delegates HTTP requests to the transport.

    Starlette's ``Route`` wraps bound methods with ``request_response()``,
    converting them into ``(request) -> response`` endpoints.  Since
    ``handle_request`` has an ASGI ``(scope, receive, send)`` signature, we
    mount it directly as a raw ASGI application with a minimal lifespan
    handler for uvicorn.

    HTTP requests must include a valid ``Authorization: Bearer <token>``
    header.  Requests with missing or invalid tokens receive a 401 response.
    """
    expected = f"Bearer {bearer_token}".encode()

    async def asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            auth = _get_header(scope.get("headers", []), b"authorization")
            if auth != expected:
                await _send_401(send)
                return
            await transport.handle_request(scope, receive, send)

    return asgi_app


def create_mcp_tool_server(
    bridged_tools: tuple[BridgedTool, ...],
) -> Any:
    """Create an MCP ``Server`` with tool handlers registered directly.

    Uses the standard ``mcp`` library decorators (``@server.list_tools()`` and
    ``@server.call_tool()``) instead of routing through ``claude-agent-sdk``.

    Args:
        bridged_tools: Tuple of BridgedTool instances to register.

    Returns:
        An ``mcp.server.Server`` ready to be passed to ``MCPHttpServer``.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server(name="wink-tools")
    tools_by_name: dict[str, BridgedTool] = {bt.name: bt for bt in bridged_tools}

    @server.list_tools()
    async def _list_tools() -> list[Tool]:  # noqa: RUF029
        return [
            Tool(
                name=bt.name,
                description=bt.description,
                inputSchema=bt.input_schema,
            )
            for bt in bridged_tools
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        bt = tools_by_name.get(name)
        if bt is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = await asyncio.to_thread(bt, arguments or {})
        return [
            TextContent(type="text", text=item.get("text", ""))
            for item in result.get("content", [])
        ]

    return server


class MCPHttpServer:
    """Wraps an MCP server instance with HTTP transport.

    Exposes the MCP server over HTTP on localhost using
    ``StreamableHTTPServerTransport``.  Runs uvicorn in a daemon thread for the
    lifecycle of the adapter call.
    """

    def __init__(
        self,
        mcp_server: Any,
        *,
        server_name: str = "wink-tools",
    ) -> None:
        self._mcp_server = mcp_server
        self._server_name = server_name
        self._bearer_token = secrets.token_urlsafe(32)
        self._port: int | None = None
        self._thread: threading.Thread | None = None
        self._uvicorn_server: Any = None
        self._ready_event = threading.Event()
        self._startup_error: BaseException | None = None

    @property
    def port(self) -> int:
        """TCP port the server is listening on."""
        if self._port is None:
            msg = "Server not started"
            raise RuntimeError(msg)
        return self._port

    @property
    def url(self) -> str:
        """Full URL of the MCP endpoint."""
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def server_name(self) -> str:
        """Name passed to ACP ``HttpMcpServer``."""
        return self._server_name

    @property
    def bearer_token(self) -> str:
        """Bearer token required for HTTP requests."""
        return self._bearer_token

    def to_http_mcp_server(self) -> Any:
        """Construct an ``HttpMcpServer`` config for ACP ``new_session``."""
        from acp.schema import HttpHeader, HttpMcpServer

        return HttpMcpServer(
            url=self.url,
            name=self._server_name,
            headers=[
                HttpHeader(
                    name="Authorization",
                    value=f"Bearer {self._bearer_token}",
                ),
            ],
            type="http",
        )

    async def start(self) -> None:
        """Start the HTTP server in a background thread."""
        import uvicorn
        from mcp.server import InitializationOptions, Server
        from mcp.server.streamable_http import StreamableHTTPServerTransport
        from mcp.types import ServerCapabilities, ToolsCapability

        self._port = None  # Set by _run_server after uvicorn binds
        self._ready_event.clear()
        self._startup_error = None

        mcp_server: Server = self._mcp_server
        transport = StreamableHTTPServerTransport(mcp_session_id=None)
        ready_event = self._ready_event

        config = uvicorn.Config(
            app=_make_asgi_app(transport, self._bearer_token),
            host="127.0.0.1",
            port=0,  # Let OS assign a free port atomically
            log_level="warning",
        )
        uv_server = uvicorn.Server(config)
        self._uvicorn_server = uv_server

        init_options = InitializationOptions(
            server_name=self._server_name,
            server_version="0.1.0",
            capabilities=ServerCapabilities(tools=ToolsCapability()),
        )

        async def _run_server() -> None:  # pragma: no cover - runs in bg thread
            """Run both the MCP server and uvicorn together."""
            async with transport.connect() as (read_stream, write_stream):
                mcp_task = asyncio.create_task(
                    mcp_server.run(read_stream, write_stream, init_options)
                )
                uv_task = asyncio.create_task(uv_server.serve())
                # Wait until uvicorn has actually bound the socket.
                while not uv_server.started:
                    await asyncio.sleep(0.01)
                # Read real port from bound socket.
                self._port = uv_server.servers[0].sockets[0].getsockname()[1]
                ready_event.set()
                await uv_task
                mcp_task.cancel()

        def _thread_target() -> None:  # pragma: no cover - runs in bg thread
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_run_server())
            except BaseException as exc:
                self._startup_error = exc
            finally:
                ready_event.set()
                loop.close()

        self._thread = threading.Thread(
            target=_thread_target,
            daemon=True,
            name=f"mcp-http-{self._server_name}",
        )
        self._thread.start()

        # Wait for the server to be ready or fail
        started = await asyncio.to_thread(self._ready_event.wait, 5.0)
        if self._startup_error is not None:
            raise self._startup_error
        if not started:
            msg = "MCP HTTP server failed to start within 5 seconds"
            raise RuntimeError(msg)

        logger.info(
            "acp.mcp_http.started",
            event="mcp_http.started",
            context={"port": self._port, "url": self.url},
        )

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._port = None
        logger.info("acp.mcp_http.stopped", event="mcp_http.stopped")

    async def __aenter__(self) -> MCPHttpServer:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
