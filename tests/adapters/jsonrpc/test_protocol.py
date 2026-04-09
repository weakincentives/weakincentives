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

"""Tests for the generic JSON-RPC adapter base class and protocol."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.jsonrpc._protocol import (
    apply_notification,
    consume_messages,
    create_deadline_watchdog,
    deadline_remaining_s,
    raise_for_terminal_notification,
)
from weakincentives.adapters.jsonrpc._types import (
    JsonRpcMessage,
    ProtocolContext,
    ServerRequestContext,
)
from weakincentives.adapters.jsonrpc.adapter import JsonRpcAdapter
from weakincentives.adapters.jsonrpc.client import JsonRpcClient
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate.create(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


async def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    for msg in messages:
        yield msg


# ---- Protocol unit tests ----


class TestDeadlineRemainingS:
    def test_none_deadline(self) -> None:
        assert deadline_remaining_s(None, "p") is None

    def test_expired_deadline(self) -> None:
        clock = FakeClock(_wall=datetime(2025, 1, 1, tzinfo=UTC))
        deadline = Deadline.create(
            expires_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
            clock=clock,
        )
        # Advance past expiry
        clock.advance(seconds=10)
        with pytest.raises(PromptEvaluationError, match="expired"):
            deadline_remaining_s(deadline, "p")

    def test_valid_deadline(self) -> None:
        clock = FakeClock(_wall=datetime(2025, 1, 1, tzinfo=UTC))
        deadline = Deadline.create(
            expires_at=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
            clock=clock,
        )
        result = deadline_remaining_s(deadline, "p")
        assert result is not None
        assert result > 0


class TestRaiseForTerminalNotification:
    def test_error_raises(self) -> None:
        adapter = MagicMock()
        adapter._map_error_phase.return_value = "response"
        msg: JsonRpcMessage = {"params": {}}
        with pytest.raises(PromptEvaluationError, match="boom"):
            raise_for_terminal_notification("error", "boom", "p", msg, adapter)

    def test_interrupted_raises(self) -> None:
        adapter = MagicMock()
        msg: JsonRpcMessage = {}
        with pytest.raises(PromptEvaluationError, match="interrupted"):
            raise_for_terminal_notification("interrupted", "", "p", msg, adapter)


class TestApplyNotification:
    def test_text_replaces(self) -> None:
        adapter = MagicMock()
        text, _usage, done = apply_notification(
            ("text", "new"), {}, "old", None, "p", adapter
        )
        assert text == "new"
        assert not done

    def test_delta_appends(self) -> None:
        adapter = MagicMock()
        text, _usage, done = apply_notification(
            ("delta", " world"), {}, "hello", None, "p", adapter
        )
        assert text == "hello world"
        assert not done

    def test_usage_extracts(self) -> None:
        adapter = MagicMock()
        adapter._extract_token_usage.return_value = "mock_usage"
        _text, usage, done = apply_notification(
            ("usage", ""), {"params": {}}, "t", None, "p", adapter
        )
        assert usage == "mock_usage"
        assert not done

    def test_done_signals_completion(self) -> None:
        adapter = MagicMock()
        _text, _usage, done = apply_notification(
            ("done", ""), {}, "t", None, "p", adapter
        )
        assert done


class TestCreateDeadlineWatchdog:
    def test_no_deadline(self) -> None:
        adapter = MagicMock()
        client = MagicMock()
        result = create_deadline_watchdog(adapter, client, "sess", "turn", None)
        assert result is None

    def test_expired_deadline(self) -> None:
        adapter = MagicMock()
        client = MagicMock()
        clock = FakeClock(_wall=datetime(2025, 1, 1, tzinfo=UTC))
        deadline = Deadline.create(
            expires_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
            clock=clock,
        )
        clock.advance(seconds=10)
        result = create_deadline_watchdog(adapter, client, "sess", "turn", deadline)
        assert result is None


class TestConsumeMessages:
    def test_stream_eof_raises(self) -> None:
        """Empty stream raises PromptEvaluationError."""

        async def _run() -> None:
            adapter = MagicMock()
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator([])
            with pytest.raises(
                PromptEvaluationError,
                match=r"(?i)stream ended before",
            ):
                await consume_messages(
                    adapter=adapter,
                    client=client,
                    session=MagicMock(),
                    adapter_name="test",
                    prompt_name="p",
                    tool_lookup={},
                    run_context=None,
                    accumulated_text="",
                    usage=None,
                )

        asyncio.run(_run())

    def test_notification_dispatches(self) -> None:
        """Notifications flow through adapter._process_notification."""

        async def _run() -> None:
            adapter = MagicMock()
            adapter._process_notification.return_value = ("done", "")
            adapter._on_notification_for_transcript = MagicMock()
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator(
                [{"method": "turn/completed", "params": {}}]
            )
            _text, _usage = await consume_messages(
                adapter=adapter,
                client=client,
                session=MagicMock(),
                adapter_name="test",
                prompt_name="p",
                tool_lookup={},
                run_context=None,
                accumulated_text="",
                usage=None,
            )
            adapter._process_notification.assert_called_once()
            adapter._on_notification_for_transcript.assert_called_once()

        asyncio.run(_run())

    def test_server_request_dispatches(self) -> None:
        """Server requests flow through adapter._handle_server_request."""

        async def _run() -> None:
            adapter = MagicMock()
            adapter._handle_server_request = AsyncMock()
            # First message is a server request, second is turn complete
            adapter._process_notification.return_value = ("done", "")
            adapter._on_notification_for_transcript = MagicMock()
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator(
                [
                    {"id": 1, "method": "item/tool/call", "params": {}},
                    {"method": "turn/completed", "params": {}},
                ]
            )
            await consume_messages(
                adapter=adapter,
                client=client,
                session=MagicMock(),
                adapter_name="test",
                prompt_name="p",
                tool_lookup={},
                run_context=None,
                accumulated_text="",
                usage=None,
            )
            adapter._handle_server_request.assert_called_once()

        asyncio.run(_run())


class TestProtocolContext:
    def test_frozen(self) -> None:
        ctx = ProtocolContext(
            effective_cwd="/tmp",
            dynamic_tool_specs=[],
            output_schema=None,
        )
        assert ctx.effective_cwd == "/tmp"
        with pytest.raises(AttributeError):
            ctx.effective_cwd = "/other"  # type: ignore[misc]


class TestServerRequestContext:
    def test_fields(self) -> None:
        ctx = ServerRequestContext(
            client=None,
            request_id=1,
            method="test",
            params={},
            tool_lookup={},
            bridge=None,
            prompt=None,
            session=None,
            deadline=None,
        )
        assert ctx.method == "test"
        assert ctx.request_id == 1


class TestJsonRpcAdapterDispatch:
    """Test the concrete dispatch methods on JsonRpcAdapter."""

    def test_process_notification_dispatches_to_handler(self) -> None:
        """_process_notification looks up handler from registry."""

        class FakeAdapter(JsonRpcAdapter[Any]):
            def _adapter_name(self) -> str:
                return "fake"

            def _create_client(self, env: dict[str, str] | None) -> JsonRpcClient:
                raise NotImplementedError

            async def _initialize_session(
                self,
                client: Any,
                *,
                deadline: Any,
                prompt_name: str,
                protocol_context: Any,
            ) -> object:
                raise NotImplementedError

            async def _start_turn(
                self,
                client: Any,
                session_state: Any,
                prompt_text: str,
                *,
                deadline: Any,
                prompt_name: str,
                timeout: Any,
                protocol_context: Any,
            ) -> object:
                raise NotImplementedError

            async def _send_interrupt(
                self, client: Any, session_state: Any, turn_state: Any
            ) -> None:
                raise NotImplementedError

            def _notification_handlers(self) -> dict[str, Any]:
                return {"test/method": lambda p, s, a, pn, rc: ("delta", "hi")}

            def _server_request_handlers(self) -> dict[str, Any]:
                return {}

            def _build_tool_specs(self, bridged_tools: Any) -> list[dict[str, object]]:
                return []

            def _build_output_schema(self, rendered: Any) -> dict[str, Any] | None:
                return None

            def _extract_token_usage(self, params: dict[str, object]) -> Any:
                return None

            def _map_error_phase(self, message: Any) -> str:
                return "response"

        adapter = FakeAdapter()
        # Known method -> dispatches to handler
        result = adapter._process_notification(
            {"method": "test/method", "params": {}},
            MagicMock(),
            "fake",
            "p",
            None,
        )
        assert result == ("delta", "hi")

        # Unknown method -> None
        result = adapter._process_notification(
            {"method": "unknown", "params": {}},
            MagicMock(),
            "fake",
            "p",
            None,
        )
        assert result is None

    def test_handle_server_request_dispatches_to_handler(self) -> None:
        """_handle_server_request looks up handler from registry."""

        called_with: list[ServerRequestContext] = []

        async def fake_handler(ctx: ServerRequestContext) -> None:
            called_with.append(ctx)

        class FakeAdapter(JsonRpcAdapter[Any]):
            def _adapter_name(self) -> str:
                return "fake"

            def _create_client(self, env: dict[str, str] | None) -> JsonRpcClient:
                raise NotImplementedError

            async def _initialize_session(
                self,
                client: Any,
                *,
                deadline: Any,
                prompt_name: str,
                protocol_context: Any,
            ) -> object:
                raise NotImplementedError

            async def _start_turn(
                self,
                client: Any,
                session_state: Any,
                prompt_text: str,
                *,
                deadline: Any,
                prompt_name: str,
                timeout: Any,
                protocol_context: Any,
            ) -> object:
                raise NotImplementedError

            async def _send_interrupt(
                self, client: Any, session_state: Any, turn_state: Any
            ) -> None:
                raise NotImplementedError

            def _notification_handlers(self) -> dict[str, Any]:
                return {}

            def _server_request_handlers(self) -> dict[str, Any]:
                return {"test/call": fake_handler}

            def _build_tool_specs(self, bridged_tools: Any) -> list[dict[str, object]]:
                return []

            def _build_output_schema(self, rendered: Any) -> dict[str, Any] | None:
                return None

            def _extract_token_usage(self, params: dict[str, object]) -> Any:
                return None

            def _map_error_phase(self, message: Any) -> str:
                return "response"

        adapter = FakeAdapter()
        mock_client = AsyncMock()

        async def _run() -> None:
            # Known method -> dispatches to handler
            await adapter._handle_server_request(
                mock_client,
                {"id": 1, "method": "test/call", "params": {"x": 1}},
                {},
            )
            assert len(called_with) == 1
            assert called_with[0].method == "test/call"

            # Unknown method -> empty response
            await adapter._handle_server_request(
                mock_client,
                {"id": 2, "method": "unknown", "params": {}},
                {},
            )
            mock_client.send_response.assert_called_once_with(2, {})

        asyncio.run(_run())
