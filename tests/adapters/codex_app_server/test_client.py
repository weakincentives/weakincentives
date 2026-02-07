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

"""Tests for the Codex App Server NDJSON client."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)


class FakeStreamReader:
    """Simulates an asyncio.StreamReader for testing."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._index = 0

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""  # EOF
        line = self._lines[self._index]
        self._index += 1
        return (line + "\n").encode()


class FakeStreamWriter:
    """Simulates an asyncio.StreamWriter for testing."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Simulates an asyncio.subprocess.Process for testing."""

    def __init__(
        self,
        stdout_lines: list[str] | None = None,
        stderr_lines: list[str] | None = None,
    ) -> None:
        self.stdin = FakeStreamWriter()
        self.stdout = FakeStreamReader(stdout_lines or [])
        self.stderr = FakeStreamReader(stderr_lines or [])
        self._killed = False
        self._waited = False

    async def wait(self) -> int:
        self._waited = True
        return 0

    def kill(self) -> None:
        self._killed = True


class _StubReadTask:
    """Minimal task-like object exposing ``done()`` for send_request tests."""

    def __init__(self, done_values: list[bool]) -> None:
        self._done_values = done_values
        self._index = 0

    def done(self) -> bool:
        if self._index >= len(self._done_values):
            return self._done_values[-1]
        value = self._done_values[self._index]
        self._index += 1
        return value


def _response_line(req_id: int, result: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"id": req_id}
    if result is not None:
        msg["result"] = result
    return json.dumps(msg)


def _notification_line(method: str, params: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def _server_request_line(
    req_id: int, method: str, params: dict[str, Any] | None = None
) -> str:
    msg: dict[str, Any] = {"id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


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

    def test_replace_env_default_false(self) -> None:
        client = CodexAppServerClient()
        assert client._replace_env is False

    def test_replace_env_true(self) -> None:
        client = CodexAppServerClient(
            env={"HOME": "/tmp/fake", "PATH": "/usr/bin"},
            replace_env=True,
        )
        assert client._replace_env is True


class TestClientStartReplaceEnv:
    def test_start_uses_only_provided_env_when_replace_env(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])
            custom_env = {"HOME": "/tmp/fake", "PATH": "/usr/bin"}
            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient(env=custom_env, replace_env=True)
                await client.start()
                args = mock_exec.call_args
                passed_env = args[1]["env"]
                # Only the provided env vars should be present
                assert passed_env["HOME"] == "/tmp/fake"
                assert passed_env["PATH"] == "/usr/bin"
                # Should NOT include arbitrary host env vars
                assert len(passed_env) == 2
                await client.stop()

        asyncio.run(_run())

    def test_start_merges_env_when_not_replace_env(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])
            custom_env = {"MY_CUSTOM_VAR": "value"}
            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient(env=custom_env)
                await client.start()
                args = mock_exec.call_args
                passed_env = args[1]["env"]
                # Should include both host env and custom vars
                assert passed_env["MY_CUSTOM_VAR"] == "value"
                # Should also include PATH from host env
                assert "PATH" in passed_env
                await client.stop()

        asyncio.run(_run())


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


class TestReadMessages:
    def test_yields_notifications(self) -> None:
        async def _run() -> None:
            notif = _notification_line(
                "item/completed", {"item": {"type": "agentMessage"}}
            )
            proc = FakeProcess(stdout_lines=[notif])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                assert len(messages) == 1
                assert messages[0]["method"] == "item/completed"
                await client.stop()

        asyncio.run(_run())

    def test_yields_server_requests(self) -> None:
        async def _run() -> None:
            req = _server_request_line(10, "item/tool/call", {"tool": "my_tool"})
            proc = FakeProcess(stdout_lines=[req])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                assert len(messages) == 1
                assert messages[0]["method"] == "item/tool/call"
                assert messages[0]["id"] == 10
                await client.stop()

        asyncio.run(_run())


class TestStderrCapture:
    def test_stderr_lines_captured(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[], stderr_lines=["warning: something"])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                await asyncio.sleep(0.05)
                assert "warning: something" in client.stderr_output
                await client.stop()

        asyncio.run(_run())


class TestTryParse:
    def test_valid_json(self) -> None:
        client = CodexAppServerClient()
        result = client._try_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self) -> None:
        client = CodexAppServerClient()
        result = client._try_parse("not json")
        assert result is None

    def test_non_dict_json(self) -> None:
        client = CodexAppServerClient()
        result = client._try_parse("[1, 2, 3]")
        assert result is None


class TestRouteMessage:
    def test_routes_response_to_future(self) -> None:
        client = CodexAppServerClient()
        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        client._pending[1] = future

        client._route_message({"id": 1, "result": {"ok": True}})

        assert future.done()
        assert future.result() == {"id": 1, "result": {"ok": True}}
        loop.close()

    def test_routes_notification_to_queue(self) -> None:
        client = CodexAppServerClient()
        client._route_message({"method": "item/completed"})
        assert client._message_queue.qsize() == 1

    def test_routes_server_request_to_queue(self) -> None:
        client = CodexAppServerClient()
        client._route_message({"id": 5, "method": "item/tool/call"})
        assert client._message_queue.qsize() == 1

    def test_unknown_response_id_ignored(self) -> None:
        client = CodexAppServerClient()
        client._route_message({"id": 999})
        assert client._message_queue.qsize() == 0


class TestStopWithTimeout:
    def test_stop_kills_on_timeout(self) -> None:
        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])
            original_wait = proc.wait

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                proc.wait = original_wait
                original_wait_for = asyncio.wait_for
                call_count = 0

                async def mock_wait_for(coro: Any, timeout: Any = None) -> Any:
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        coro.close()
                        raise TimeoutError
                    return await original_wait_for(coro, timeout=timeout)

                with patch("asyncio.wait_for", side_effect=mock_wait_for):
                    await client.stop()

                assert proc._killed

        asyncio.run(_run())


class TestStopBrokenPipe:
    def test_stop_completes_when_stdin_close_raises_broken_pipe(self) -> None:
        """stop() still completes when stdin.close() raises BrokenPipeError."""

        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                def _raise_broken_pipe() -> None:
                    raise BrokenPipeError("broken pipe")

                proc.stdin.close = _raise_broken_pipe  # type: ignore[assignment,method-assign]

                await client.stop()

                # Process should still be cleaned up
                assert client._proc is None
                assert proc._waited

        asyncio.run(_run())


class TestReadLoopEdgeCases:
    def test_empty_lines_skipped(self) -> None:
        async def _run() -> None:
            notif = _notification_line("item/completed", {})
            proc = FakeProcess(stdout_lines=["", "  ", notif])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                assert len(messages) == 1
                await client.stop()

        asyncio.run(_run())

    def test_eof_resolves_pending(self) -> None:
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
                client._pending[99] = future

                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                assert len(messages) == 0
                assert future.done()
                with pytest.raises(CodexClientError, match="exited"):
                    future.result()

                await client.stop()

        asyncio.run(_run())

    def test_response_missing_result_returns_empty(self) -> None:
        async def _run() -> None:
            response = json.dumps({"id": 1})
            proc = FakeProcess(stdout_lines=[response])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                result = await client.send_request("test", {})
                assert result == {}
                await client.stop()

        asyncio.run(_run())


class TestInvalidJsonInStream:
    def test_invalid_json_skipped(self) -> None:
        """Invalid JSON lines in stdout are silently skipped."""

        async def _run() -> None:
            notif = _notification_line("item/completed", {})
            proc = FakeProcess(stdout_lines=["not valid json", notif])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                # Only the valid notification should appear
                assert len(messages) == 1
                assert messages[0]["method"] == "item/completed"
                await client.stop()

        asyncio.run(_run())


class TestStderrLogging:
    def test_stderr_logged_when_not_suppressed(self) -> None:
        """When suppress_stderr=False, stderr lines are logged."""

        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[], stderr_lines=["line1", "line2"])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient(suppress_stderr=False)
                await client.start()
                await asyncio.sleep(0.05)
                assert "line1" in client.stderr_output
                assert "line2" in client.stderr_output
                await client.stop()

        asyncio.run(_run())

    def test_stderr_cancelled_during_read(self) -> None:
        """Stderr task cancellation is handled gracefully."""

        async def _run() -> None:
            # Create a FakeStreamReader that blocks forever on readline
            class BlockingStderrReader:
                async def readline(self) -> bytes:
                    # This will be interrupted by cancellation
                    await asyncio.sleep(100)
                    return b""  # pragma: no cover

            proc = FakeProcess(stdout_lines=[])
            proc.stderr = BlockingStderrReader()  # type: ignore[assignment]

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                # Give the stderr task a moment to start blocking
                await asyncio.sleep(0.01)
                assert client._stderr_task is not None

                # Directly cancel the stderr task
                _ = client._stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client._stderr_task
                client._stderr_task = None

                await client.stop()

        asyncio.run(_run())


class TestStopAwaitsTasksClient:
    def test_stop_awaits_cancelled_tasks(self) -> None:
        """stop() awaits cancelled read and stderr tasks."""

        async def _run() -> None:
            proc = FakeProcess(stdout_lines=[], stderr_lines=[])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()

                # Wait for read_loop and stderr_loop to finish (EOF)
                await asyncio.sleep(0.05)

                # Verify tasks exist before stop
                assert client._read_task is not None
                assert client._stderr_task is not None

                await client.stop()

                # Tasks should be cleaned up
                assert client._read_task is None
                assert client._stderr_task is None

        asyncio.run(_run())


class TestStderrBufferBounded:
    def test_stderr_buffer_bounded(self) -> None:
        """Stderr buffer only keeps last 1000 lines."""

        async def _run() -> None:
            lines = [f"line-{i}" for i in range(1200)]
            proc = FakeProcess(stdout_lines=[], stderr_lines=lines)

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()
                await client.start()
                await asyncio.sleep(0.1)

                output = client.stderr_output
                output_lines = output.split("\n")
                assert len(output_lines) == 1000
                assert output_lines[0] == "line-200"
                assert output_lines[-1] == "line-1199"
                await client.stop()

        asyncio.run(_run())


class TestReadLoopBroadException:
    def test_read_loop_handles_unexpected_error(self) -> None:
        """Unexpected exceptions in _read_loop are caught, sentinel delivered."""

        async def _run() -> None:
            notif = _notification_line("item/completed", {})
            proc = FakeProcess(stdout_lines=[notif])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()

                # Patch _route_message to raise an unexpected error
                def exploding_route(parsed: dict[str, Any]) -> None:
                    raise RuntimeError("unexpected boom")

                client._route_message = exploding_route  # type: ignore[assignment]
                await client.start()

                # read_messages should terminate (sentinel delivered)
                messages: list[dict[str, Any]] = []
                async for msg in client.read_messages():
                    messages.append(msg)

                # No messages yielded because _route_message exploded
                assert len(messages) == 0
                await client.stop()

        asyncio.run(_run())

    def test_read_loop_logs_exception_context(self) -> None:
        """The _read_loop exception handler includes exc_info for diagnostics."""

        async def _run() -> None:
            notif = _notification_line("item/completed", {})
            proc = FakeProcess(stdout_lines=[notif])

            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = proc
                client = CodexAppServerClient()

                # Patch _route_message to raise an unexpected error
                def exploding_route(parsed: dict[str, Any]) -> None:
                    raise RuntimeError("unexpected boom")

                client._route_message = exploding_route  # type: ignore[assignment]

                with patch(
                    "weakincentives.adapters.codex_app_server.client.logger"
                ) as mock_logger:
                    await client.start()

                    # Drain messages to let read_loop complete
                    async for _ in client.read_messages():
                        pass  # pragma: no cover

                    # Verify warning was called with exc_info=True
                    mock_logger.warning.assert_called_once()
                    _, kwargs = mock_logger.warning.call_args
                    assert kwargs.get("exc_info") is True

                await client.stop()

        asyncio.run(_run())
