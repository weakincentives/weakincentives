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

"""Bidirectional JSON-RPC client for the Codex app-server.

Supports three transport modes:

- **stdio** (default): spawns ``codex app-server`` as a subprocess and
  communicates via newline-delimited JSON over stdin/stdout pipes.
- **managed WebSocket**: spawns ``codex app-server --listen ws://…`` as a
  subprocess and connects to it via WebSocket.
- **external WebSocket**: connects to an already-running Codex app-server at
  a ``ws://`` or ``wss://`` URL.  No subprocess is spawned or managed.

The public API (``send_request``, ``send_notification``, ``send_response``,
``read_messages``) is identical across all transports.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from collections import deque
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, Protocol, cast

from ...runtime.logging import StructuredLogger, get_logger

__all__ = [
    "CodexAppServerClient",
    "CodexClientError",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "codex_client"})

_SENTINEL: dict[str, Any] = {"_sentinel": True}

_WS_CONNECT_MAX_RETRIES = 50
_WS_CONNECT_RETRY_DELAY = 0.1


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _WebSocketProtocol(Protocol):
    """Narrow protocol for the websockets client connection."""

    async def send(self, data: str) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> _WebSocketProtocol: ...
    async def __anext__(self) -> str | bytes: ...


class CodexClientError(Exception):
    """Error from the Codex app-server client."""


class CodexAppServerClient:
    """Bidirectional JSON-RPC client for the Codex app-server.

    *transport* selects the wire protocol (``"stdio"`` or ``"websocket"``).
    When *remote_url* is provided, the client connects to an external
    WebSocket server and no subprocess is spawned.  When *remote_url* is
    ``None`` and *transport* is ``"websocket"``, the client spawns a local
    ``codex app-server --listen ws://…`` subprocess and connects to it.
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        env: Mapping[str, str] | None = None,
        suppress_stderr: bool = True,
        *,
        transport: Literal["stdio", "websocket"] = "stdio",
        remote_url: str | None = None,
        ws_auth_token: str | None = None,
    ) -> None:
        super().__init__()
        self._codex_bin = codex_bin
        self._extra_env = dict(env) if env else {}
        self._suppress_stderr = suppress_stderr
        self._transport: Literal["stdio", "websocket"] = (
            "websocket" if remote_url is not None else transport
        )
        self._remote_url = remote_url
        self._ws_auth_token = ws_auth_token

        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=1000)
        self._ws: _WebSocketProtocol | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._read_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Establish the transport and begin reading messages."""
        if self._transport == "stdio":
            await self._start_stdio()
        elif self._remote_url is not None:
            await self._start_ws_external()
        else:
            await self._start_ws_managed()

    async def stop(self) -> None:
        """Tear down the transport."""
        if self._read_task is not None:
            _ = self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None

        if self._stderr_task is not None:
            _ = self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        if self._proc is not None:
            if self._proc.stdin is not None:  # pragma: no branch
                with contextlib.suppress(OSError):
                    self._proc.stdin.close()
            try:
                _ = await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
                _ = await self._proc.wait()
            self._proc = None

        self._fail_pending("Client stopped with pending requests")

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and await the response.

        Returns the ``result`` field from the response.  Raises
        :class:`CodexClientError` if the response contains an ``error`` field.
        """
        self._next_id += 1
        req_id = self._next_id
        msg: dict[str, Any] = {"id": req_id, "method": method, "params": params}

        read_task = self._read_task
        if read_task is None:
            raise CodexClientError("Client not started")
        if read_task.done():
            raise CodexClientError("Transport disconnected unexpectedly")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = future

        try:
            await self._write(msg)
        except Exception as exc:
            _ = self._pending.pop(req_id, None)
            if isinstance(exc, CodexClientError):
                raise
            raise CodexClientError(
                f"Failed to send request {method} (id={req_id})"
            ) from exc

        # If read loop exited between pending registration and write completion,
        # fail fast instead of waiting forever when timeout=None.
        if read_task.done() and not future.done():
            _ = self._pending.pop(req_id, None)
            future.set_exception(
                CodexClientError("Transport disconnected unexpectedly")
            )

        try:
            resp = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            _ = self._pending.pop(req_id, None)
            raise CodexClientError(
                f"Timeout waiting for response to {method} (id={req_id})"
            ) from None

        if "error" in resp:
            raise CodexClientError(f"{method} failed: {resp['error']}")
        result: dict[str, Any] = resp.get("result", {})
        return result

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def send_response(
        self, request_id: int, result: Mapping[str, object]
    ) -> None:
        """Send a response to a server-initiated request."""
        msg: dict[str, object] = {"id": request_id, "result": dict(result)}
        await self._write(msg)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield notifications and server requests from the message queue.

        Responses are consumed internally by :meth:`send_request` and do
        not appear here.  Yields until the transport disconnects.
        """
        while True:
            msg = await self._message_queue.get()
            if msg is _SENTINEL:
                break
            yield msg

    @property
    def stderr_output(self) -> str:
        """Return captured stderr output (empty in external-WS mode)."""
        return "\n".join(self._stderr_lines)

    # ---- transport setup ----

    async def _start_stdio(self) -> None:
        """Spawn the codex app-server subprocess and begin reading."""
        self._proc = await self._spawn("app-server", stdin=asyncio.subprocess.PIPE)
        self._read_task = asyncio.create_task(self._read_loop())
        self._start_stderr_capture()

    async def _start_ws_managed(self) -> None:
        """Spawn ``codex app-server --listen ws://…`` and connect."""
        port = _find_free_port()
        ws_url = f"ws://127.0.0.1:{port}"

        self._proc = await self._spawn(
            "app-server", "--listen", ws_url, stdout=asyncio.subprocess.DEVNULL
        )
        self._start_stderr_capture()

        try:
            await self._await_tcp_ready("127.0.0.1", port, ws_url)
            await self._connect_ws(ws_url)
        except BaseException:
            await self.stop()
            raise

    async def _start_ws_external(self) -> None:
        """Connect to an external Codex app-server via WebSocket."""
        url = self._remote_url
        if url is None:  # pragma: no cover — guarded by start()
            raise CodexClientError("remote_url is required for external WS mode")
        await self._connect_ws(url)

    async def _spawn(
        self,
        *args: str,
        stdin: int | None = None,
        stdout: int | None = asyncio.subprocess.PIPE,
    ) -> asyncio.subprocess.Process:
        """Spawn ``codex <args>`` as a subprocess."""
        merged_env = {**os.environ, **self._extra_env}
        return await asyncio.create_subprocess_exec(
            self._codex_bin,
            *args,
            stdin=stdin,
            stdout=stdout,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

    def _start_stderr_capture(self) -> None:
        """Start the stderr capture loop if a subprocess is running."""
        if (
            self._proc is not None and self._proc.stderr is not None
        ):  # pragma: no branch
            self._stderr_task = asyncio.create_task(self._stderr_loop())

    async def _await_tcp_ready(self, host: str, port: int, label: str) -> None:
        """Block until a TCP server is accepting connections."""
        for _ in range(_WS_CONNECT_MAX_RETRIES):
            if self._proc is not None and self._proc.returncode is not None:
                raise CodexClientError(
                    f"codex app-server exited during startup (code={self._proc.returncode}): {self.stderr_output[-2000:]}"
                )
            try:
                _reader, writer = await asyncio.open_connection(host, port)
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(_WS_CONNECT_RETRY_DELAY)
            else:
                writer.close()
                await writer.wait_closed()
                return

        raise CodexClientError(
            f"codex app-server did not start listening on {label} within {_WS_CONNECT_MAX_RETRIES * _WS_CONNECT_RETRY_DELAY}s"
        )

    async def _connect_ws(self, url: str) -> None:
        """Open a WebSocket connection and start the read loop."""
        try:
            import websockets
        except ModuleNotFoundError as exc:
            raise CodexClientError(
                "WebSocket transport requires the 'websockets' package — install via: pip install weakincentives[codex-ws]"
            ) from exc

        additional_headers: dict[str, str] = {}
        if self._ws_auth_token is not None:
            additional_headers["Authorization"] = f"Bearer {self._ws_auth_token}"

        try:
            self._ws = cast(
                _WebSocketProtocol,
                await websockets.connect(
                    url,
                    additional_headers=additional_headers or None,
                ),
            )
        except Exception as exc:
            raise CodexClientError(f"Failed to connect to {url}: {exc}") from exc

        self._read_task = asyncio.create_task(self._read_loop())

    # ---- message I/O ----

    async def _write(self, msg: dict[str, Any]) -> None:
        """Write a JSON message to the active transport."""
        data = json.dumps(msg, separators=(",", ":"))
        if self._ws is not None:
            try:
                await self._ws.send(data)
            except Exception as exc:
                raise CodexClientError(f"WebSocket send failed: {exc}") from exc
        elif self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.write((data + "\n").encode())
            await self._proc.stdin.drain()
        else:
            raise CodexClientError("Client not started")  # pragma: no cover

    async def _read_loop(self) -> None:
        """Read messages from the transport and route them."""
        try:
            await self._read_messages()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "codex_client.read_loop_error",
                event="client.read_loop_error",
                exc_info=True,
            )
        finally:
            await self._message_queue.put(_SENTINEL)
            self._fail_pending("Transport disconnected unexpectedly")

    async def _read_messages(self) -> None:
        """Read and route messages until the transport disconnects."""
        if self._ws is not None:
            await self._read_messages_ws()
        else:
            await self._read_messages_stdio()

    async def _read_messages_stdio(self) -> None:
        """Read NDJSON lines from stdout until EOF."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return  # pragma: no cover
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode().strip()
            if not line:
                continue
            parsed = self._try_parse(line)
            if parsed is not None:
                self._route_message(parsed)

    async def _read_messages_ws(self) -> None:
        """Read WebSocket text frames until the connection closes."""
        ws = self._ws
        if ws is None:
            return  # pragma: no cover
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                parsed = self._try_parse(raw)
                if parsed is not None:
                    self._route_message(parsed)
        except Exception:
            logger.debug(
                "codex_client.ws_read_eof",
                event="client.ws_read_eof",
            )

    def _fail_pending(self, message: str) -> None:
        """Resolve all pending futures with an error and clear them."""
        for future in self._pending.values():  # pragma: no branch
            if not future.done():  # pragma: no branch
                future.set_exception(CodexClientError(message))
        self._pending.clear()

    @staticmethod
    def _try_parse(line: str) -> dict[str, Any] | None:
        """Try to parse a JSON line, returning None on failure."""
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "codex_client.invalid_json",
                event="client.invalid_json",
                context={"line": line[:200]},
            )
            return None

        if not isinstance(parsed, dict):
            return None
        return cast(dict[str, Any], parsed)

    def _route_message(self, parsed: dict[str, Any]) -> None:
        """Route a parsed message to futures or queue."""
        if "id" in parsed and "method" not in parsed:
            req_id: int = parsed["id"]
            future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(parsed)
            return
        self._message_queue.put_nowait(parsed)

    async def _stderr_loop(self) -> None:
        """Read stderr lines and buffer them."""
        if self._proc is None or self._proc.stderr is None:
            return  # pragma: no cover

        try:
            while True:
                raw = await self._proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode().rstrip()
                self._stderr_lines.append(line)
                if not self._suppress_stderr:
                    logger.debug(
                        "codex_client.stderr",
                        event="client.stderr",
                        context={"line": line},
                    )
        except asyncio.CancelledError:
            pass
