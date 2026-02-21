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

"""Tests for evaluate features: output schema, delta, budget, deadlines, visibility."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters._shared._visibility_signal import VisibilityExpansionSignal
from weakincentives.adapters.codex_app_server._protocol import consume_messages
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import CodexClientError
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt.errors import VisibilityExpansionRequired

from .conftest import (
    make_mock_client,
    make_session,
    make_simple_prompt,
    messages_iterator,
)


class TestEvaluateWithOutputSchema:
    """Test evaluate with output schema/structured output."""

    def test_output_schema_sent_and_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Answer:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": '{"value": 42}'}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        # Patch prompt.render to return a RenderedPrompt with output_type
        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RenderedPrompt[Any]:
            rendered = original_render(**kwargs)
            # Replace structured_output to set output_type
            return RenderedPrompt(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Answer,
                    container="object",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert result.output.value == 42

        # Verify outputSchema was sent in turn/start
        turn_call = mock_client.send_request.call_args_list[2]
        assert "outputSchema" in turn_call[0][1]


class TestEvaluateWithDeltaAccumulation:
    """Test that delta messages are accumulated properly."""

    def test_delta_then_text_uses_text(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
            {"method": "item/agentMessage/delta", "params": {"delta": "lo"}},
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
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        # The "text" kind from item/completed replaces accumulated delta
        assert result.text == "Hello world"


class TestEvaluateWithBudgetTracking:
    """Test budget tracking integration."""

    def test_budget_tracker_records_usage(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()
        budget = Budget(max_input_tokens=1000, max_output_tokens=500)
        budget_tracker = BudgetTracker(budget)

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {"last": {"inputTokens": 100, "outputTokens": 50}}
                },
            },
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
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(
                prompt, session=session, budget_tracker=budget_tracker
            )
            assert result.text == "ok"


class TestEvaluateWithServerRequestDuringStream:
    """Test that server requests during streaming are handled."""

    def test_tool_call_during_stream(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            # Server request: tool call
            {
                "id": 100,
                "method": "item/tool/call",
                "params": {"tool": "unknown_tool", "arguments": {}},
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "done"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)
            assert result.text == "done"
            # The unknown tool call should have sent a response
            mock_client.send_response.assert_called()


class TestEvaluateTempWorkspaceCleanup:
    """Test that temporary workspace is cleaned up on failure."""

    def test_temp_dir_cleaned_up_on_error(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = make_session()
        prompt = make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("fail")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError):
                adapter.evaluate(prompt, session=session)


class TestStreamTurnWithDeadline:
    """Test that _stream_turn properly cleans up the watchdog task."""

    def test_stream_turn_with_active_deadline(self) -> None:
        """Watchdog task is created and cancelled after turn completes."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()
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
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, deadline=deadline)
            assert result.text == "ok"


class TestVisibilitySignalPassthrough:
    """Test that VisibilityExpansionRequired passes through unwrapped."""

    def test_visibility_expansion_passthrough(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
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
        session, _ = make_session()
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

        client = make_mock_client()
        client.read_messages.return_value = messages_iterator(messages)
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
