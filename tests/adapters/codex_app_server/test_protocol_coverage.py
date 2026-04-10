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

"""Coverage tests for Codex _protocol and adapter internal branches."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from weakincentives.adapters._shared._visibility_signal import VisibilityExpansionSignal
from weakincentives.adapters.codex_app_server._protocol import (
    _create_bridge,
    _initialize_session,
    apply_notification,
    consume_messages,
    execute_protocol,
    raise_for_terminal_notification,
)
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.jsonrpc import JsonRpcClientError


async def _messages_iterator(messages: list[dict[str, Any]]) -> Any:
    for msg in messages:
        yield msg


class TestCreateBridgeDisabled:
    """Covers codex_app_server/_protocol.py:82 — transcript disabled path."""

    def test_returns_none_when_transcript_disabled(self) -> None:
        config = CodexAppServerClientConfig(transcript=False)
        result = _create_bridge(config, MagicMock(), "p")
        assert result is None


class TestInitializeSessionError:
    """Covers codex_app_server/_protocol.py:133-134 — JsonRpcClientError wrap."""

    def test_client_error_wrapped(self) -> None:
        async def _run() -> None:
            client = AsyncMock()
            client.send_notification = AsyncMock()
            config = CodexAppServerClientConfig(cwd="/tmp/test")
            model_config = MagicMock()
            model_config.model = "gpt-5"

            # Make authenticate call raise JsonRpcClientError (via send_request)
            call_count = 0

            def side_effect(*args: object, **kwargs: object) -> dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # initialize
                    return {}
                # Second call is thread/start (no auth since auth_mode=None)
                raise JsonRpcClientError("thread creation failed")

            client.send_request = AsyncMock(side_effect=side_effect)

            with pytest.raises(PromptEvaluationError, match="thread creation failed"):
                await _initialize_session(
                    client=client,
                    client_config=config,
                    model_config=model_config,
                    effective_cwd="/tmp/test",
                    dynamic_tool_specs=[],
                    deadline=None,
                    prompt_name="p",
                )

        asyncio.run(_run())


class TestApplyNotificationTextKind:
    """Covers codex_app_server/_protocol.py:493 — text kind replaces accumulated."""

    def test_text_kind_replaces(self) -> None:
        text, usage, done = apply_notification(
            ("text", "final"), {}, "partial", None, "p"
        )
        assert text == "final"
        assert usage is None
        assert not done

    def test_delta_kind_appends(self) -> None:
        text, _usage, done = apply_notification(
            ("delta", " more"), {}, "hello", None, "p"
        )
        assert text == "hello more"
        assert not done


class TestRaiseForTerminalNotification:
    """Covers codex_app_server/_protocol.py:504 — terminal notification raises."""

    def test_error_raises(self) -> None:
        with pytest.raises(PromptEvaluationError, match="boom"):
            raise_for_terminal_notification(
                "error", "boom", "p", {"params": {"turn": {}}}
            )

    def test_interrupted_raises(self) -> None:
        with pytest.raises(PromptEvaluationError, match="interrupted"):
            raise_for_terminal_notification("interrupted", "", "p", {"params": {}})


class TestConsumeMessagesStreamEof:
    """Covers codex_app_server/_protocol.py:475 — stream EOF raises."""

    def test_stream_eof_raises(self) -> None:
        async def _run() -> None:
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator([])
            with pytest.raises(
                PromptEvaluationError,
                match="Codex stream ended before turn completion",
            ):
                await consume_messages(
                    client=client,
                    session=MagicMock(),
                    adapter_name="codex_app_server",
                    prompt_name="p",
                    tool_lookup={},
                    approval_policy="never",
                    run_context=None,
                    accumulated_text="",
                    usage=None,
                )

        asyncio.run(_run())


class TestConsumeMessagesUnknownNotification:
    """Covers codex_app_server/_protocol.py:466 — unknown notification returns None."""

    def test_unknown_notification_continues(self) -> None:
        async def _run() -> None:
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator(
                [
                    # Unknown notification — process_notification returns None
                    {"method": "unknown/method", "params": {}},
                    # Then turn complete to exit the loop
                    {"method": "turn/completed", "params": {"turn": {}}},
                ]
            )
            text, _usage = await consume_messages(
                client=client,
                session=MagicMock(),
                adapter_name="codex_app_server",
                prompt_name="p",
                tool_lookup={},
                approval_policy="never",
                run_context=None,
                accumulated_text="",
                usage=None,
            )
            assert text == ""

        asyncio.run(_run())


class TestConsumeMessagesServerRequestContinues:
    """Covers codex_app_server/_protocol.py:454 — server request continue branch."""

    def test_server_request_without_visibility_continues(self) -> None:
        """After handling a server request with no visibility signal, loop continues."""

        async def _run() -> None:
            client = AsyncMock()
            client.send_response = AsyncMock()
            client.read_messages = lambda: _messages_iterator(
                [
                    # Unknown server request (gets empty response); no visibility signal
                    {"id": 1, "method": "unknown/request", "params": {}},
                    # Turn complete to exit the loop
                    {"method": "turn/completed", "params": {"turn": {}}},
                ]
            )
            text, _usage = await consume_messages(
                client=client,
                session=MagicMock(),
                adapter_name="codex_app_server",
                prompt_name="p",
                tool_lookup={},
                approval_policy="never",
                run_context=None,
                accumulated_text="",
                usage=None,
            )
            assert text == ""
            # Unknown server request received an empty response
            client.send_response.assert_called_once_with(1, {})

        asyncio.run(_run())


class TestApplyNotificationTerminalFallthrough:
    """Covers codex_app_server/_protocol.py:504 — terminal kind raises."""

    def test_error_kind_raises_via_apply(self) -> None:
        with pytest.raises(PromptEvaluationError, match="boom"):
            apply_notification(
                ("error", "boom"),
                {"params": {"turn": {}}},
                "",
                None,
                "p",
            )

    def test_interrupted_kind_raises_via_apply(self) -> None:
        with pytest.raises(PromptEvaluationError, match="interrupted"):
            apply_notification(
                ("interrupted", ""),
                {"params": {}},
                "",
                None,
                "p",
            )


class TestExecuteProtocolNoBridge:
    """Covers codex_app_server/_protocol.py:188->190 and 246->250 —
    execute_protocol branches when bridge is None (transcript=False).
    """

    def test_execute_protocol_with_transcript_disabled(self) -> None:
        async def _run() -> None:
            from weakincentives.adapters.codex_app_server.config import (
                CodexAppServerModelConfig,
            )

            client = AsyncMock()
            # initialize, thread/start, turn/start responses
            send_request_results: list[dict[str, Any]] = [
                {},  # initialize
                {"thread": {"id": "t-1"}},  # thread/start
                {"turn": {"id": 1}},  # turn/start
            ]
            client.send_request = AsyncMock(side_effect=send_request_results)
            client.send_notification = AsyncMock()
            client.start = AsyncMock()
            client.read_messages = lambda: _messages_iterator(
                [{"method": "turn/completed", "params": {"turn": {}}}]
            )

            client_config = CodexAppServerClientConfig(
                cwd="/tmp/test", transcript=False
            )
            model_config = CodexAppServerModelConfig()

            text, _usage = await execute_protocol(
                client_config=client_config,
                model_config=model_config,
                client=client,
                session=MagicMock(),
                adapter_name="codex_app_server",
                prompt_name="p",
                prompt_text="hello",
                effective_cwd="/tmp/test",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=VisibilityExpansionSignal(),
            )
            # Completes without bridge calls
            assert text is None or text == ""

        asyncio.run(_run())


class TestJsonRpcRaiseForTerminalNotification:
    """Covers jsonrpc/_protocol.py:277->exit — raise_for_terminal fall-through branch."""

    def test_non_terminal_kind_no_raise(self) -> None:
        """kind that is neither error nor interrupted returns without raising."""
        from weakincentives.adapters.jsonrpc._protocol import (
            raise_for_terminal_notification as jsonrpc_raise,
        )

        adapter = MagicMock()
        adapter._map_error_phase.return_value = "response"
        # Passing a non-terminal kind falls through both if branches
        jsonrpc_raise("other", "value", "p", {"params": {}}, adapter)


class TestCodexSendInterrupt:
    """Covers codex_app_server/adapter.py:239-240 — _send_interrupt body."""

    def test_send_interrupt_calls_turn_interrupt(self) -> None:
        async def _run() -> None:
            adapter = CodexAppServerAdapter()
            client = AsyncMock()
            client.send_request = AsyncMock(return_value={})

            await adapter._send_interrupt(
                client,
                session_state="thread-1",
                turn_state={"turn": {"id": 42}},
            )
            client.send_request.assert_called_once()
            args = client.send_request.call_args
            assert args[0][0] == "turn/interrupt"
            assert args[0][1] == {"threadId": "thread-1", "turnId": 42}

        asyncio.run(_run())
