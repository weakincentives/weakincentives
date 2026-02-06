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

"""Bidirectional JSON-RPC client for the Codex app-server over NDJSON stdio."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from ...runtime.logging import StructuredLogger, get_logger

__all__ = [
    "CodexAppServerClient",
    "CodexClientError",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "codex_client"})

_SENTINEL: dict[str, Any] = {"_sentinel": True}


class CodexClientError(Exception):
    """Error from the Codex app-server client."""


class CodexAppServerClient:
    """Bidirectional JSON-RPC client for the Codex app-server.

    Manages the ``codex app-server`` subprocess and provides typed methods
    for the NDJSON stdio protocol. Messages are demultiplexed into three
    streams: responses (matched by id), notifications, and server requests.
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        env: Mapping[str, str] | None = None,
        suppress_stderr: bool = True,
    ) -> None:
        super().__init__()
        self._codex_bin = codex_bin
        self._extra_env = dict(env) if env else {}
        self._suppress_stderr = suppress_stderr
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._read_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: list[str] = []

    async def start(self) -> None:
        """Spawn the codex app-server subprocess and begin reading."""
        merged_env = {**os.environ, **self._extra_env}
        self._proc = await asyncio.create_subprocess_exec(
            self._codex_bin,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        if self._proc.stderr is not None:  # pragma: no branch
            self._stderr_task = asyncio.create_task(self._stderr_loop())

    async def stop(self) -> None:
        """Terminate the subprocess gracefully."""
        if self._read_task is not None:
            _ = self._read_task.cancel()
            self._read_task = None

        if self._stderr_task is not None:
            _ = self._stderr_task.cancel()
            self._stderr_task = None

        if self._proc is not None:
            if self._proc.stdin is not None:  # pragma: no branch
                self._proc.stdin.close()
            try:
                _ = await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
                _ = await self._proc.wait()
            self._proc = None

        # Resolve any pending futures with errors
        for future in self._pending.values():  # pragma: no branch
            if not future.done():  # pragma: no branch
                future.set_exception(
                    CodexClientError("Client stopped with pending requests")
                )
        self._pending.clear()

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and await the response.

        Returns the ``result`` field from the response. Raises
        :class:`CodexClientError` if the response contains an ``error`` field.
        """
        self._next_id += 1
        req_id = self._next_id
        msg: dict[str, Any] = {"id": req_id, "method": method, "params": params}

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = future

        await self._write(msg)

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

    async def send_response(self, request_id: int, result: dict[str, Any]) -> None:
        """Send a response to a server-initiated request."""
        msg: dict[str, Any] = {"id": request_id, "result": result}
        await self._write(msg)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield notifications and server requests from the message queue.

        Responses are consumed internally by :meth:`send_request` and do
        not appear here. Yields until the subprocess exits.
        """
        while True:
            msg = await self._message_queue.get()
            if msg is _SENTINEL:
                break
            yield msg

    @property
    def stderr_output(self) -> str:
        """Return captured stderr output."""
        return "\n".join(self._stderr_lines)

    # ---- internal ----

    async def _write(self, msg: dict[str, Any]) -> None:
        """Write a JSON message to stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise CodexClientError("Client not started")  # pragma: no cover
        data = json.dumps(msg, separators=(",", ":")) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        """Read lines from stdout and route them."""
        if self._proc is None or self._proc.stdout is None:
            return  # pragma: no cover

        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    break  # EOF — process exited

                line = raw.decode().strip()
                if not line:
                    continue

                parsed = self._try_parse(line)
                if parsed is None:
                    continue

                self._route_message(parsed)

        except asyncio.CancelledError:
            pass
        finally:
            # Signal end of messages
            await self._message_queue.put(_SENTINEL)
            # Resolve any remaining pending futures
            for future in self._pending.values():  # pragma: no branch
                if not future.done():  # pragma: no branch
                    future.set_exception(
                        CodexClientError("Subprocess exited unexpectedly")
                    )
            self._pending.clear()

    def _try_parse(self, line: str) -> dict[str, Any] | None:
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
        # Response (has id, no method) → resolve pending future
        if "id" in parsed and "method" not in parsed:
            req_id: int = parsed["id"]
            future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(parsed)
            return

        # Server request (has both id and method) or notification → queue
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
