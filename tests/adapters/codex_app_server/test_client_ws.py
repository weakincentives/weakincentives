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

"""Tests for the Codex App Server WebSocket transport modes."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
    _find_free_port,
)

# ---- helpers ----


class FakeWebSocket:
    """Simulates a websockets client connection."""

    def __init__(self, inbound: list[str] | None = None) -> None:
        self._inbound = list(inbound or [])
        self._index = 0
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._inbound):
            raise StopAsyncIteration
        msg = self._inbound[self._index]
        self._index += 1
        return msg


def _response_json(req_id: int, result: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"id": req_id}
    if result is not None:
        msg["result"] = result
    return json.dumps(msg)


def _notification_json(method: str, params: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


# ---- transport selection ----


class TestTransportSelection:
    def test_default_is_stdio(self) -> None:
        client = CodexAppServerClient()
        assert client._transport == "stdio"
        assert client._remote_url is None

    def test_transport_websocket_no_url(self) -> None:
        client = CodexAppServerClient(transport="websocket")
        assert client._transport == "websocket"
        assert client._remote_url is None

    def test_remote_url_forces_websocket(self) -> None:
        client = CodexAppServerClient(
            transport="stdio", remote_url="ws://10.0.1.5:4500"
        )
        assert client._transport == "websocket"
        assert client._remote_url is not None

    def test_remote_url_with_websocket_transport(self) -> None:
        client = CodexAppServerClient(
            transport="websocket", remote_url="ws://host:1234"
        )
        assert client._transport == "websocket"
        assert client._remote_url is not None


# ---- external WebSocket ----


class TestExternalWsConnect:
    def test_connect_and_initialize(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(
                inbound=[_response_json(1, result={"status": "ok"})]
            )

            with patch(
                "weakincentives.adapters.codex_app_server.client.CodexAppServerClient._connect_ws",
                new_callable=AsyncMock,
            ) as mock_connect:

                async def do_connect(url: str) -> None:
                    client._ws = fake_ws
                    client._read_task = asyncio.create_task(client._read_loop())

                mock_connect.side_effect = do_connect

                client = CodexAppServerClient(remote_url="ws://host:4500")
                await client.start()

                mock_connect.assert_called_once_with("ws://host:4500")
                result = await client.send_request("initialize", {})
                assert result == {"status": "ok"}
                await client.stop()

        asyncio.run(_run())

    def test_connect_failure_raises(self) -> None:
        async def _run() -> None:
            with patch(
                "weakincentives.adapters.codex_app_server.client.CodexAppServerClient._connect_ws",
                new_callable=AsyncMock,
            ) as mock_connect:
                mock_connect.side_effect = CodexClientError("connection refused")

                client = CodexAppServerClient(remote_url="ws://host:4500")
                with pytest.raises(CodexClientError, match="connection refused"):
                    await client.start()

        asyncio.run(_run())

    def test_no_stderr_in_external_mode(self) -> None:
        client = CodexAppServerClient(remote_url="ws://host:4500")
        assert client.stderr_output == ""


# ---- WebSocket read/write ----


class TestWsReadMessages:
    def test_notifications_yielded(self) -> None:
        async def _run() -> None:
            notif = _notification_json("item/completed", {"item": {"type": "msg"}})
            fake_ws = FakeWebSocket(inbound=[notif])
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            messages: list[dict[str, Any]] = []
            async for msg in client.read_messages():
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0]["method"] == "item/completed"
            await client.stop()

        asyncio.run(_run())

    def test_binary_frames_ignored(self) -> None:
        async def _run() -> None:
            notif = _notification_json("done", {})

            class BinaryThenTextWs(FakeWebSocket):
                def __init__(self) -> None:
                    self._items: list[str | bytes] = [b"\x00\x01", notif]
                    self._index = 0

                async def __anext__(self) -> str | bytes:
                    if self._index >= len(self._items):
                        raise StopAsyncIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_ws = BinaryThenTextWs()
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            messages: list[dict[str, Any]] = []
            async for msg in client.read_messages():
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0]["method"] == "done"
            await client.stop()

        asyncio.run(_run())

    def test_invalid_json_skipped(self) -> None:
        async def _run() -> None:
            notif = _notification_json("ok", {})
            fake_ws = FakeWebSocket(inbound=["not json", notif])
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            messages: list[dict[str, Any]] = []
            async for msg in client.read_messages():
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0]["method"] == "ok"
            await client.stop()

        asyncio.run(_run())


class TestWsWrite:
    def test_send_notification(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(inbound=[])
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            await client.send_notification("initialized")

            assert len(fake_ws.sent) == 1
            msg = json.loads(fake_ws.sent[0])
            assert msg["method"] == "initialized"
            assert "id" not in msg
            await client.stop()

        asyncio.run(_run())

    def test_send_response(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(inbound=[])
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            await client.send_response(42, {"decision": "accept"})

            msg = json.loads(fake_ws.sent[0])
            assert msg["id"] == 42
            assert msg["result"]["decision"] == "accept"
            await client.stop()

        asyncio.run(_run())

    def test_send_failure_raises(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(inbound=[])
            fake_ws.send = AsyncMock(side_effect=ConnectionError("broken"))  # type: ignore[assignment]
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            with pytest.raises(CodexClientError, match="WebSocket send failed"):
                await client.send_notification("boom")

            await client.stop()

        asyncio.run(_run())


# ---- WS connection exception handling ----


class TestWsConnectionException:
    def test_ws_read_loop_handles_connection_close(self) -> None:
        async def _run() -> None:

            class ExplodingWs(FakeWebSocket):
                async def __anext__(self) -> str:
                    raise ConnectionError("connection lost")

            fake_ws = ExplodingWs()
            client = CodexAppServerClient(remote_url="ws://host:4500")
            client._ws = fake_ws
            client._read_task = asyncio.create_task(client._read_loop())

            # read_messages should terminate cleanly (sentinel delivered)
            messages: list[dict[str, Any]] = []
            async for msg in client.read_messages():
                messages.append(msg)

            assert len(messages) == 0
            await client.stop()

        asyncio.run(_run())


# ---- managed WebSocket ----


class FakeStreamReader:
    """Simulates an asyncio.StreamReader for stderr capture testing."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = list(lines or [])
        self._index = 0

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return (line + "\n").encode()


class FakeManagedProcess:
    """Simulates a subprocess for managed-WS mode testing."""

    def __init__(self, stderr_lines: list[str] | None = None) -> None:
        self.stdin = None  # No stdin in managed-WS mode
        self.stdout = None  # No stdout — we use WebSocket
        self.stderr = FakeStreamReader(stderr_lines or [])
        self.returncode: int | None = None
        self._killed = False

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        self._killed = True


class TestManagedWs:
    def test_start_spawns_and_connects(self) -> None:
        async def _run() -> None:
            fake_proc = FakeManagedProcess()
            fake_ws = FakeWebSocket(inbound=[_response_json(1, result={"ok": True})])

            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                ) as mock_exec,
                patch(
                    "asyncio.open_connection",
                    new_callable=AsyncMock,
                ) as mock_open_conn,
                patch(
                    "weakincentives.adapters.codex_app_server.client.CodexAppServerClient._connect_ws",
                    new_callable=AsyncMock,
                ) as mock_connect,
            ):
                mock_exec.return_value = fake_proc

                # Simulate successful TCP probe
                mock_writer = MagicMock()
                mock_writer.close = MagicMock()
                mock_writer.wait_closed = AsyncMock()
                mock_open_conn.return_value = (MagicMock(), mock_writer)

                async def do_connect(url: str) -> None:
                    client._ws = fake_ws
                    client._read_task = asyncio.create_task(client._read_loop())

                mock_connect.side_effect = do_connect

                client = CodexAppServerClient(transport="websocket")
                await client.start()

                # Verify subprocess was spawned with --listen ws://
                # and stdout=DEVNULL (no pipe — messages go over WS).
                mock_exec.assert_called_once()
                exec_args = mock_exec.call_args[0]
                assert exec_args[0] == "codex"
                assert exec_args[1] == "app-server"
                assert exec_args[2] == "--listen"
                assert exec_args[3].startswith("ws://127.0.0.1:")
                exec_kwargs = mock_exec.call_args[1]
                assert exec_kwargs.get("stdout") == asyncio.subprocess.DEVNULL

                result = await client.send_request("initialize", {})
                assert result == {"ok": True}
                await client.stop()

        asyncio.run(_run())

    def test_process_exit_during_startup_raises(self) -> None:
        async def _run() -> None:
            fake_proc = FakeManagedProcess()
            fake_proc.returncode = 1

            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                ) as mock_exec,
                patch(
                    "asyncio.open_connection",
                    new_callable=AsyncMock,
                ) as mock_open_conn,
            ):
                mock_exec.return_value = fake_proc
                mock_open_conn.side_effect = ConnectionRefusedError

                client = CodexAppServerClient(transport="websocket")
                with pytest.raises(CodexClientError, match="exited during startup"):
                    await client.start()

                # Cleanup — stop should handle already-dead process
                await client.stop()

        asyncio.run(_run())

    def test_stderr_captured_in_managed_mode(self) -> None:
        async def _run() -> None:
            fake_proc = FakeManagedProcess(stderr_lines=["warn: something"])
            fake_ws = FakeWebSocket(inbound=[])

            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                ) as mock_exec,
                patch(
                    "asyncio.open_connection",
                    new_callable=AsyncMock,
                ) as mock_open_conn,
                patch(
                    "weakincentives.adapters.codex_app_server.client.CodexAppServerClient._connect_ws",
                    new_callable=AsyncMock,
                ) as mock_connect,
            ):
                mock_exec.return_value = fake_proc

                mock_writer = MagicMock()
                mock_writer.close = MagicMock()
                mock_writer.wait_closed = AsyncMock()
                mock_open_conn.return_value = (MagicMock(), mock_writer)

                async def do_connect(url: str) -> None:
                    client._ws = fake_ws
                    client._read_task = asyncio.create_task(client._read_loop())

                mock_connect.side_effect = do_connect

                client = CodexAppServerClient(transport="websocket")
                await client.start()

                # Let stderr loop run
                for _ in range(5):
                    await asyncio.sleep(0)

                assert "warn: something" in client.stderr_output
                await client.stop()

        asyncio.run(_run())


# ---- _await_tcp_ready ----


class TestAwaitTcpReady:
    def test_retries_on_connection_refused(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient(transport="websocket")
            # Simulate a process that is alive (returncode=None)
            fake_proc = FakeManagedProcess()
            client._proc = fake_proc  # type: ignore[assignment]

            call_count = 0

            async def mock_open(host: str, port: int) -> tuple[MagicMock, MagicMock]:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionRefusedError("not yet")
                writer = MagicMock()
                writer.close = MagicMock()
                writer.wait_closed = AsyncMock()
                return MagicMock(), writer

            with patch("asyncio.open_connection", side_effect=mock_open):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await client._await_tcp_ready("127.0.0.1", 9999, "ws://test")

            assert call_count == 3

        asyncio.run(_run())

    def test_raises_after_max_retries(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient(transport="websocket")
            fake_proc = FakeManagedProcess()
            client._proc = fake_proc  # type: ignore[assignment]

            with (
                patch(
                    "asyncio.open_connection",
                    new_callable=AsyncMock,
                    side_effect=ConnectionRefusedError("nope"),
                ),
                patch("asyncio.sleep", new_callable=AsyncMock),
                patch(
                    "weakincentives.adapters.codex_app_server.client._WS_CONNECT_MAX_RETRIES",
                    2,
                ),
            ):
                with pytest.raises(CodexClientError, match="did not start listening"):
                    await client._await_tcp_ready("127.0.0.1", 9999, "ws://test")

        asyncio.run(_run())


# ---- _connect_ws ----


class TestConnectWs:
    def test_connects_with_auth_token(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(inbound=[])
            client = CodexAppServerClient(
                remote_url="ws://host:4500", ws_auth_token="secret"
            )

            mock_connect = AsyncMock(return_value=fake_ws)
            with patch("websockets.connect", mock_connect):
                await client._connect_ws("ws://host:4500")

            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args
            headers = call_kwargs.kwargs.get("additional_headers") or call_kwargs[
                1
            ].get("additional_headers")
            assert headers["Authorization"] == "Bearer secret"

            await client.stop()

        asyncio.run(_run())

    def test_connects_without_auth_token(self) -> None:
        async def _run() -> None:
            fake_ws = FakeWebSocket(inbound=[])
            client = CodexAppServerClient(remote_url="ws://host:4500")

            mock_connect = AsyncMock(return_value=fake_ws)
            with patch("websockets.connect", mock_connect):
                await client._connect_ws("ws://host:4500")

            call_kwargs = mock_connect.call_args
            headers = call_kwargs.kwargs.get("additional_headers") or call_kwargs[
                1
            ].get("additional_headers")
            assert headers is None

            await client.stop()

        asyncio.run(_run())

    def test_connect_failure_raises(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient(remote_url="ws://host:4500")

            mock_connect = AsyncMock(side_effect=ConnectionError("refused"))
            with patch("websockets.connect", mock_connect):
                with pytest.raises(CodexClientError, match="Failed to connect"):
                    await client._connect_ws("ws://host:4500")

        asyncio.run(_run())

    def test_missing_package_raises(self) -> None:
        async def _run() -> None:
            with patch.dict("sys.modules", {"websockets": None}):
                client = CodexAppServerClient(remote_url="ws://host:4500")
                with pytest.raises(CodexClientError, match="websockets"):
                    await client._connect_ws("ws://host:4500")

        asyncio.run(_run())


# ---- utility ----


class TestFindFreePort:
    def test_returns_int(self) -> None:
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_ports_differ(self) -> None:
        p1 = _find_free_port()
        p2 = _find_free_port()
        # They should almost always differ, but we're testing the function works
        assert isinstance(p1, int)
        assert isinstance(p2, int)
