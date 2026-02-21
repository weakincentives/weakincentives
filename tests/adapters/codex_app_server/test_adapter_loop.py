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

"""Tests for the Codex App Server adapter — evaluation loop, deadline, and visibility tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters._shared._visibility_signal import VisibilityExpansionSignal
from weakincentives.adapters.codex_app_server._protocol import (
    consume_messages,
    deadline_watchdog,
)
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.budget import Budget
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate, Tool
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import (
    InProcessDispatcher,
    PromptExecuted,
    PromptRendered,
)
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


def _make_mock_client() -> AsyncMock:
    """Create a mock CodexAppServerClient."""
    client = AsyncMock(spec=CodexAppServerClient)
    client.stderr_output = ""
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.send_request = AsyncMock(return_value={})
    client.send_notification = AsyncMock()
    client.send_response = AsyncMock()
    return client


def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()


# ---- Tests ----


class TestEvaluateExpiredDeadline:
    def test_raises_on_expired_deadline(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(prompt, session=session, deadline=deadline)


class TestEvaluateEndToEnd:
    """End-to-end tests mocking the CodexAppServerClient."""

    def test_simple_evaluation(self) -> None:
        """Test basic prompt evaluation with mocked client."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = _make_session()

        # Track events
        rendered_events: list[PromptRendered] = []
        executed_events: list[PromptExecuted] = []
        dispatcher.subscribe(PromptRendered, lambda e: rendered_events.append(e))
        dispatcher.subscribe(PromptExecuted, lambda e: executed_events.append(e))

        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hello "}},
            {"method": "item/agentMessage/delta", "params": {"delta": "world"}},
            # Unknown notification — should be silently ignored
            {"method": "unknown/notification", "params": {}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Hello world"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            # Initialize returns capabilities
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                {"thread": {"id": "t-1"}},  # thread/start
                {"turn": {"id": 1}},  # turn/start
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "Hello world"
        assert len(rendered_events) == 1
        assert len(executed_events) == 1
        assert executed_events[0].adapter == "codex_app_server"

    def test_evaluation_with_token_usage(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 100,
                            "outputTokens": 50,
                        }
                    }
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "answer"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "answer"

    def test_client_error_wrapped(self) -> None:
        """CodexClientError is wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("conn failed")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="conn failed"):
                adapter.evaluate(prompt, session=session)

    def test_generic_error_wrapped(self) -> None:
        """Generic exceptions are wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.start.side_effect = RuntimeError("unexpected")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="unexpected"):
                adapter.evaluate(prompt, session=session)

    def test_prompt_eval_error_passthrough(self) -> None:
        """PromptEvaluationError passes through unwrapped."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "failed",
                        "codexErrorInfo": "unauthorized",
                        "additionalDetails": "bad key",
                    }
                },
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session)
            assert exc_info.value.phase == "request"

    def test_stream_eof_before_turn_completed(self) -> None:
        """Stream ends with messages but no turn/completed -> raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "partial"}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "partial"}},
            },
            # No turn/completed — stream ends
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_stream_eof_empty_stream(self) -> None:
        """Zero messages in stream -> raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator([])
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_budget_creates_tracker(self) -> None:
        """Budget without tracker creates one automatically."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        budget = Budget(
            max_input_tokens=1000,
            max_output_tokens=500,
        )

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, budget=budget)
            assert result.text == "ok"

    def test_rendered_tools_dispatch_failure_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that RenderedTools dispatch failures are logged."""
        import logging

        from weakincentives.runtime.session.rendered_tools import RenderedTools

        def failing_handler(event: RenderedTools) -> None:
            raise RuntimeError("Subscriber error")

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = _make_session()
        dispatcher.subscribe(RenderedTools, failing_handler)

        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Done"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        caplog.set_level(logging.ERROR)

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "Done"
        assert any(
            "rendered_tools_dispatch_failed" in record.message
            for record in caplog.records
        )


@dataclass(slots=True, frozen=True)
class _AddParams:
    x: int
    y: int


@dataclass(slots=True, frozen=True)
class _AddResult:
    sum: int


def _add_handler(params: _AddParams, *, context: ToolContext) -> ToolResult[_AddResult]:
    return ToolResult.ok(_AddResult(sum=params.x + params.y))


_ADD_TOOL = Tool[_AddParams, _AddResult](
    name="add",
    description="Add two numbers",
    handler=_add_handler,
)


class TestEvaluateWithTools:
    """Test that prompt tools are extracted into RenderedTools schemas."""

    def test_tool_schemas_dispatched(self) -> None:
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        section = MarkdownSection(
            title="Tools",
            template="Use the tools below.",
            key="tools",
            tools=[_ADD_TOOL],
        )
        template: PromptTemplate[object] = PromptTemplate(
            ns="test",
            key="with-tool",
            sections=(section,),
            name="tool-prompt",
        )
        prompt = Prompt(template)

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = _make_session()

        dispatched: list[RenderedTools] = []
        dispatcher.subscribe(RenderedTools, lambda e: dispatched.append(e))

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "ok"
        assert len(dispatched) == 1
        assert len(dispatched[0].tools) == 1
        assert dispatched[0].tools[0].name == "add"


class TestStreamTurnWithDeadline:
    """Test that _stream_turn properly cleans up the watchdog task."""

    def test_stream_turn_with_active_deadline(self) -> None:
        """Watchdog task is created and cancelled after turn completes."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=60), clock=clock)

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, deadline=deadline)
            assert result.text == "ok"


class TestDeadlineWatchdogBody:
    """Test the deadline watchdog actually sends an interrupt."""

    def test_watchdog_sends_interrupt(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
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
            client = _make_mock_client()
            client.send_request.side_effect = CodexClientError("already done")
            # Should not raise
            await deadline_watchdog(client, "t-1", 42, 0.01)

        asyncio.run(_run())


class TestVisibilitySignalPassthrough:
    """Test that VisibilityExpansionRequired passes through unwrapped."""

    def test_visibility_expansion_passthrough(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            # Patch visibility signal to return a stored exception
            with patch(
                "weakincentives.adapters.codex_app_server.adapter.VisibilityExpansionSignal"
            ) as MockSignal:
                mock_signal = MagicMock()
                mock_signal.get_and_clear.return_value = VisibilityExpansionRequired(
                    "test",
                    requested_overrides={},
                    reason="test",
                    section_keys=(),
                )
                MockSignal.return_value = mock_signal

                with pytest.raises(VisibilityExpansionRequired):
                    adapter.evaluate(prompt, session=session)


class TestConsumeMessagesVisibilitySignal:
    """Test that consume_messages breaks early when visibility signal is set."""

    def test_early_break_on_visibility_signal(self) -> None:
        """consume_messages exits after tool call sets visibility signal."""
        session, _ = _make_session()
        signal = VisibilityExpansionSignal()

        # Simulate: tool call sets signal, then more messages that shouldn't be read
        messages = [
            {
                "id": 1,
                "method": "item/tool/call",
                "params": {"tool": "t", "arguments": {}},
            },
            # These should never be reached after early break:
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "x"}},
            },
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]

        client = _make_mock_client()
        client.read_messages.return_value = _messages_iterator(messages)
        # Respond to tool call, then set the signal to simulate open_sections
        client.send_response = AsyncMock(
            side_effect=lambda *a, **kw: signal.set(
                VisibilityExpansionRequired(
                    "expand", requested_overrides={}, reason="test", section_keys=()
                )
            )
        )

        async def _run() -> None:
            text, _ = await consume_messages(
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                tool_lookup={},
                approval_policy="never",
                run_context=None,
                accumulated_text="",
                usage=None,
                visibility_signal=signal,
            )
            # Should return empty text (broke early, no turn/completed text)
            assert text == ""

        asyncio.run(_run())
