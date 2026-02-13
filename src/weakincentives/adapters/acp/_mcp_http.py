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
import socket
import threading
from typing import Any

from ...runtime.logging import StructuredLogger, get_logger

__all__ = ["MCPHttpServer"]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_mcp_http"})


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_asgi_app(transport: Any) -> Any:
    """Build a minimal ASGI app that delegates HTTP requests to the transport.

    Starlette's ``Route`` wraps bound methods with ``request_response()``,
    converting them into ``(request) -> response`` endpoints.  Since
    ``handle_request`` has an ASGI ``(scope, receive, send)`` signature, we
    mount it directly as a raw ASGI application with a minimal lifespan
    handler for uvicorn.
    """

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
            await transport.handle_request(scope, receive, send)

    return asgi_app


class MCPHttpServer:
    """Wraps an MCP server instance with HTTP transport.

    Exposes the MCP server over HTTP on localhost using
    ``StreamableHTTPServerTransport``.  Runs uvicorn in a daemon thread for the
    lifecycle of the adapter call.
    """

    def __init__(
        self,
        mcp_server_config: Any,
        *,
        server_name: str = "wink-tools",
    ) -> None:
        self._mcp_server_config = mcp_server_config
        self._server_name = server_name
        self._port: int | None = None
        self._thread: threading.Thread | None = None
        self._uvicorn_server: Any = None

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

    def to_http_mcp_server(self) -> Any:
        """Construct an ``HttpMcpServer`` config for ACP ``new_session``."""
        from acp.schema import HttpMcpServer

        return HttpMcpServer(
            url=self.url,
            name=self._server_name,
            headers=[],
            type="http",
        )

    async def start(self) -> None:
        """Start the HTTP server in a background thread."""
        import uvicorn
        from mcp.server import InitializationOptions, Server
        from mcp.server.streamable_http import StreamableHTTPServerTransport
        from mcp.types import ServerCapabilities, ToolsCapability

        self._port = _find_free_port()

        mcp_server: Server = self._mcp_server_config["instance"]
        transport = StreamableHTTPServerTransport(mcp_session_id=None)

        config = uvicorn.Config(
            app=_make_asgi_app(transport),
            host="127.0.0.1",
            port=self._port,
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
                await uv_task
                mcp_task.cancel()

        def _thread_target() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_run_server())
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_thread_target,
            daemon=True,
            name=f"mcp-http-{self._server_name}",
        )
        self._thread.start()

        # Brief wait for server to be ready
        await asyncio.sleep(0.1)

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
