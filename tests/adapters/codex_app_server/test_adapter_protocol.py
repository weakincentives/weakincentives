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

"""Tests for protocol functions: authenticate, thread, turn, notifications, etc."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server._protocol import (
    authenticate,
    create_deadline_watchdog,
    create_thread,
    deadline_remaining_s,
    deadline_watchdog,
    handle_server_request,
    handle_tool_call,
    process_notification,
    raise_for_terminal_notification,
    start_turn,
)
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import CodexClientError
from weakincentives.adapters.codex_app_server.config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    ExternalTokenAuth,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline

from .conftest import (
    make_mock_client,
    make_session,
    make_simple_prompt,
)


class TestAuthenticate:
    def test_no_auth(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            await authenticate(client, None)
            client.send_request.assert_not_called()

        asyncio.run(_run())

    def test_api_key_auth(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            await authenticate(client, ApiKeyAuth(api_key="sk-test"))
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "account/login/start"
            assert args[0][1]["type"] == "apiKey"
            assert args[0][1]["apiKey"] == "sk-test"

        asyncio.run(_run())

    def test_external_token_auth(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
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
            client = make_mock_client()
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
            client = make_mock_client()
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
            client = make_mock_client()
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
            assert params["model"] == "gpt-5.3-codex"
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
            client = make_mock_client()
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


class TestProcessNotification:
    def test_delta(self) -> None:
        session, _ = make_session()
        msg = {"method": "item/agentMessage/delta", "params": {"delta": "hello"}}
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result == ("delta", "hello")

    def test_item_completed_agent_message(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "text": "final answer"}},
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result == ("text", "final answer")

    def test_item_completed_agent_message_no_text(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage"}},
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result is None

    def test_item_completed_command_execution(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "status": "completed",
                    "command": "ls",
                    "cwd": "/tmp",
                    "aggregatedOutput": "files",
                }
            },
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result is None  # Tool dispatch doesn't return a kind

    def test_token_usage_updated(self) -> None:
        session, _ = make_session()
        msg = {"method": "thread/tokenUsage/updated", "params": {}}
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result == ("usage", "")

    def test_turn_completed_done(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "turn/completed",
            "params": {"turn": {"status": "completed"}},
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result == ("done", "")

    def test_turn_completed_failed(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "status": "failed",
                    "codexErrorInfo": "unauthorized",
                    "additionalDetails": "bad key",
                }
            },
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result is not None
        assert result[0] == "error"
        assert "unauthorized" in result[1]

    def test_turn_completed_interrupted(self) -> None:
        session, _ = make_session()
        msg = {
            "method": "turn/completed",
            "params": {"turn": {"status": "interrupted"}},
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result == ("interrupted", "")

    def test_unknown_method(self) -> None:
        session, _ = make_session()
        msg = {"method": "unknown/thing", "params": {}}
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result is None

    def test_item_completed_unknown_type(self) -> None:
        """item/completed with an unrecognized item type returns None."""
        session, _ = make_session()
        msg = {
            "method": "item/completed",
            "params": {"item": {"type": "somethingNew", "status": "completed"}},
        }
        result = process_notification(msg, session, "codex_app_server", "p", None)
        assert result is None


class TestRaiseForTerminalNotification:
    def test_error_raises(self) -> None:
        msg: dict[str, Any] = {"params": {"turn": {"codexErrorInfo": "sandboxError"}}}
        with pytest.raises(PromptEvaluationError) as exc_info:
            raise_for_terminal_notification("error", "boom", "p", msg)
        assert exc_info.value.phase == "tool"

    def test_interrupted_raises(self) -> None:
        with pytest.raises(PromptEvaluationError, match="interrupted"):
            raise_for_terminal_notification("interrupted", "", "p", {})

    def test_other_kind_is_noop(self) -> None:
        # Should not raise
        raise_for_terminal_notification("done", "", "p", {})


class TestHandleServerRequest:
    def test_unknown_tool(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 1,
                "method": "item/tool/call",
                "params": {"tool": "nonexistent", "arguments": {}},
            }
            await handle_server_request(client, msg, {}, approval_policy="never")
            client.send_response.assert_called_once()
            resp = client.send_response.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())

    def test_command_approval_accept(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 2,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="never")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())

    def test_command_approval_decline(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 3,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="on-request")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "decline"

        asyncio.run(_run())

    def test_file_approval(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 4,
                "method": "item/fileChange/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="never")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())

    def test_unknown_server_request(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {"id": 5, "method": "unknown/request", "params": {}}
            await handle_server_request(client, msg, {}, approval_policy="never")
            client.send_response.assert_called_once_with(5, {})

        asyncio.run(_run())


class TestHandleToolCall:
    def test_successful_tool_call(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
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
            client = make_mock_client()
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
            client = make_mock_client()
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
            client = make_mock_client()
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


class TestCreateDeadlineWatchdog:
    def test_no_deadline(self) -> None:
        client = make_mock_client()
        result = create_deadline_watchdog(client, "t", 1, None)
        assert result is None

    def test_expired_deadline(self) -> None:
        client = make_mock_client()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        # Advance clock past expiration
        clock.advance(10)
        result = create_deadline_watchdog(client, "t", 1, deadline)
        assert result is None

    def test_active_deadline_creates_task(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            future_time = datetime.now(UTC) + timedelta(seconds=60)
            deadline = Deadline(expires_at=future_time)
            task = create_deadline_watchdog(client, "t", 1, deadline)
            try:
                assert task is not None
                assert isinstance(task, asyncio.Task)
            finally:
                if task is not None:
                    _ = task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        asyncio.run(_run())


class TestDeadlineWatchdogBody:
    """Test the deadline watchdog actually sends an interrupt."""

    def test_watchdog_sends_interrupt(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            client.send_request.return_value = {}
            await deadline_watchdog(client, "t-1", 42, 0.01)
            # After sleep, it should send turn/interrupt
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "turn/interrupt"
            assert args[0][1]["threadId"] == "t-1"
            assert args[0][1]["turnId"] == 42

        asyncio.run(_run())

    def test_watchdog_suppresses_error(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            client.send_request.side_effect = CodexClientError("already done")
            # Should not raise
            await deadline_watchdog(client, "t-1", 42, 0.01)

        asyncio.run(_run())


class TestToolCallRunsInThread:
    def test_tool_call_runs_in_thread(self) -> None:
        """Tool dispatch is wrapped in asyncio.to_thread."""

        async def _run() -> None:
            client = make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "threaded"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = {
                    "content": [{"type": "text", "text": "threaded"}],
                    "isError": False,
                }
                await handle_tool_call(
                    client, 20, {"tool": "calc", "arguments": {"x": 1}}, tool_lookup
                )
                mock_to_thread.assert_called_once_with(mock_tool, {"x": 1})

        asyncio.run(_run())


class TestDeadlineRemainingS:
    def test_no_deadline_returns_none(self) -> None:
        assert deadline_remaining_s(None, "p") is None

    def test_expired_deadline_raises(self) -> None:
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired during"):
            deadline_remaining_s(deadline, "test-prompt")

    def test_active_deadline_returns_seconds(self) -> None:
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        remaining = deadline_remaining_s(deadline, "test-prompt")
        assert remaining is not None
        assert remaining > 0


class TestSetupRPCDeadlineBounding:
    def test_setup_timeout_wraps_client_error(self) -> None:
        """When thread/start times out, PromptEvaluationError has phase='request'."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            # initialize succeeds, thread/start raises client timeout
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                CodexClientError("Timeout waiting for response to thread/start"),
            ]
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session, deadline=deadline)
            assert exc_info.value.phase == "request"

    def test_setup_passes_timeout_to_send_request(self) -> None:
        """Verify timeout is forwarded to send_request for setup RPCs."""

        async def _run() -> None:
            client = make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-1"}}

            await create_thread(
                client,
                "/tmp",
                [],
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                timeout=5.0,
            )
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 5.0 or call_args[0][2] == 5.0

        asyncio.run(_run())

    def test_authenticate_passes_timeout(self) -> None:
        async def _run() -> None:
            client = make_mock_client()

            await authenticate(client, ApiKeyAuth(api_key="sk-test"), timeout=3.0)
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 3.0

        asyncio.run(_run())

    def test_start_turn_passes_timeout(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            client.send_request.return_value = {"turn": {"id": 1}}

            await start_turn(
                client,
                "thread-1",
                "Hello",
                None,
                model_config=CodexAppServerModelConfig(),
                timeout=7.0,
            )
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 7.0

        asyncio.run(_run())


class TestApprovalPolicyUntrusted:
    def test_approval_untrusted_declines(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 6,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="untrusted")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "decline"

        asyncio.run(_run())

    def test_approval_on_failure_accepts_requested_approval(self) -> None:
        async def _run() -> None:
            client = make_mock_client()
            msg = {
                "id": 7,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="on-failure")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())
