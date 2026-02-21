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

"""Tests for Codex App Server NDJSON client: init, start/stop, send request/notification/response."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.adapters.codex_app_server.conftest import (
    FakeProcess,
    _response_line,
    _StubReadTask,
)
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)


class TestCodexClientError:
    def test_is_exception(self) -> None:
        err = CodexClientError("boom")
        assert str(err) == "boom"
        assert isinstance(err, Exception)


class TestCodexAppServerClientInit:
    def test_defaults(self) -> None:
        client = CodexAppServerClient()
        assert client.stderr_output == ""

    def test_custom_params(self) -> None:
        client = CodexAppServerClient(
            codex_bin="/custom/codex",
            env={"KEY": "VAL"},
            suppress_stderr=False,
        )
        assert client.stderr_output == ""


class TestClientStartStop:
    def test_start_spawns_process(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])
            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                mock_exec.assert_called_once()
                args = mock_exec.call_args
                assert args[0][0] == "codex"
                assert args[0][1] == "app-server"
                await client.stop()

        asyncio.run(_run())

    def test_stop_without_start(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()
            await client.stop()

        asyncio.run(_run())

    def test_stop_resolves_pending(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])
            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                loop = asyncio.get_running_loop()
                future: asyncio.Future[dict[str, Any]] = loop.create_future()
                client._pending[999] = future

                await client.stop()

                assert future.done()
                with pytest.raises(CodexClientError, match="pending"):
                    future.result()

        asyncio.run(_run())


class TestSendRequest:
    def test_request_before_start_raises(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()

            with pytest.raises(CodexClientError, match="Client not started"):
                await client.send_request("initialize", {})

        asyncio.run(_run())

    def test_send_and_receive_response(self) -> None:
        async def _run() -> None:
            response = _response_line(1, result={"status": "ok"})
            proc = FakeProcess(stdout_lines=[response])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                result = await client.send_request("initialize", {"clientInfo": {}})
                assert result == {"status": "ok"}
                await client.stop()

        asyncio.run(_run())

    def test_error_response_raises(self) -> None:
        async def _run() -> None:
            response = json.dumps({"id": 1, "error": {"code": -1, "message": "bad"}})
            proc = FakeProcess(stdout_lines=[response])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                with pytest.raises(CodexClientError, match="failed"):
                    await client.send_request("bad-method", {})
                await client.stop()

        asyncio.run(_run())

    def test_timeout_raises(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            # Override stdout to block instead of immediately returning EOF
            async def blocking_readline() -> bytes:
                await asyncio.sleep(10)
                return b""

            proc.stdout.readline = blocking_readline  # type: ignore[assignment]

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                with pytest.raises(CodexClientError, match="Timeout"):
                    await client.send_request("initialize", {}, timeout=0.01)
                await client.stop()

        asyncio.run(_run())

    def test_subprocess_exit_before_request_fails_fast(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                # Let read loop observe EOF and terminate before request.
                await asyncio.sleep(0)

                with pytest.raises(CodexClientError, match="exited"):
                    await client.send_request("initialize", {})
                await client.stop()

        asyncio.run(_run())

    def test_write_failure_clears_pending(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            # Keep read loop alive so send_request reaches _write path.
            async def blocking_readline() -> bytes:
                await asyncio.sleep(10)
                return b""

            proc.stdout.readline = blocking_readline  # type: ignore[assignment]

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                with patch.object(
                    client, "_write", new_callable=AsyncMock
                ) as mock_write:
                    mock_write.side_effect = BrokenPipeError("broken pipe")
                    with pytest.raises(
                        CodexClientError, match="Failed to send request"
                    ):
                        await client.send_request("initialize", {})

                assert client._pending == {}
                await client.stop()

        asyncio.run(_run())

    def test_codex_client_error_from_write_is_reraised(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()
            client._read_task = _StubReadTask([False])  # type: ignore[assignment]

            with patch.object(client, "_write", new_callable=AsyncMock) as mock_write:
                mock_write.side_effect = CodexClientError("write failed")
                with pytest.raises(CodexClientError, match="write failed"):
                    await client.send_request("initialize", {})

            assert client._pending == {}

        asyncio.run(_run())

    def test_read_loop_exit_after_write_sets_future_exception(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()
            # First done() check passes entry guard, second check triggers fast-fail.
            client._read_task = _StubReadTask([False, True])  # type: ignore[assignment]

            with patch.object(client, "_write", new_callable=AsyncMock):
                with pytest.raises(CodexClientError, match="exited unexpectedly"):
                    await client.send_request("initialize", {})

            assert client._pending == {}

        asyncio.run(_run())


class TestSendNotification:
    def test_sends_notification_without_params(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                await client.send_notification("initialized")

                written = proc.stdin.written
                assert len(written) == 1
                msg = json.loads(written[0].decode())
                assert msg["method"] == "initialized"
                assert "id" not in msg
                assert "params" not in msg
                await client.stop()

        asyncio.run(_run())

    def test_sends_notification_with_params(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                await client.send_notification("event", {"key": "val"})

                written = proc.stdin.written
                msg = json.loads(written[0].decode())
                assert msg["method"] == "event"
                assert msg["params"] == {"key": "val"}
                await client.stop()

        asyncio.run(_run())


class TestSendResponse:
    def test_sends_response(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                await client.send_response(42, {"decision": "accept"})

                written = proc.stdin.written
                msg = json.loads(written[0].decode())
                assert msg["id"] == 42
                assert msg["result"]["decision"] == "accept"
                await client.stop()

        asyncio.run(_run())
