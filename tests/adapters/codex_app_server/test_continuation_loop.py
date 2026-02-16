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

"""Tests for task completion continuation loop in execute_protocol."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from weakincentives.adapters._shared._visibility_signal import VisibilityExpansionSignal
from weakincentives.adapters.codex_app_server._guardrails import accumulate_usage
from weakincentives.adapters.codex_app_server._protocol import execute_protocol
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt import VisibilityExpansionRequired
from weakincentives.prompt.task_completion import (
    TaskCompletionChecker,
    TaskCompletionResult,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.events.types import TokenUsage
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


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


def _messages_iterator(messages: list[dict[str, Any]]) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()


# ---- Tests ----


class TestContinuationLoop:
    """Tests for the task completion continuation loop in execute_protocol."""

    def _make_turn_messages(self, text: str = "response") -> list[dict[str, Any]]:
        return [
            {"method": "item/agentMessage/delta", "params": {"delta": text}},
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

    def test_single_turn_no_continuation(self) -> None:
        """No continuation when task completion checker returns complete."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                {"thread": {"id": "t-1"}},  # thread/start
                {"turn": {"id": 1}},  # turn/start
            ]
            client.read_messages.return_value = _messages_iterator(
                self._make_turn_messages("hello")
            )

            signal = VisibilityExpansionSignal()
            text, _usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=None,
            )
            assert text == "hello"

        asyncio.run(_run())

    def test_continuation_on_incomplete(self) -> None:
        """Continues with additional turns when task completion returns incomplete."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            # initialize, thread/start, turn/start (1st), turn/start (2nd)
            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
                {"turn": {"id": 2}},
            ]

            call_count = 0

            def _read_messages() -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _messages_iterator(
                        self._make_turn_messages("first response")
                    )
                return _messages_iterator(self._make_turn_messages("second response"))

            client.read_messages = _read_messages

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            # First call: incomplete, second call: complete
            mock_checker.check.side_effect = [
                TaskCompletionResult.incomplete("Missing file"),
                TaskCompletionResult.ok("Done"),
            ]

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            signal = VisibilityExpansionSignal()
            text, _usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=mock_prompt,
            )
            assert text == "second response"
            assert mock_checker.check.call_count == 2
            # Second turn should have been started with feedback text
            second_turn_call = client.send_request.call_args_list[3]
            assert second_turn_call[0][0] == "turn/start"
            turn_params = second_turn_call[0][1]
            assert turn_params["input"][0]["text"] == "Missing file"

        asyncio.run(_run())

    def test_max_continuation_rounds(self) -> None:
        """Stops after max continuation rounds even if still incomplete."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            # initialize + thread/start + 11 turn/start calls
            turn_responses = [{"turn": {"id": i}} for i in range(11)]
            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                *turn_responses,
            ]

            call_count = 0

            def _read_messages() -> Any:
                nonlocal call_count
                call_count += 1
                return _messages_iterator(self._make_turn_messages(f"r{call_count}"))

            client.read_messages = _read_messages

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            mock_checker.check.return_value = TaskCompletionResult.incomplete(
                "Still missing"
            )

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            signal = VisibilityExpansionSignal()
            _text, _usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=mock_prompt,
            )
            # 1 initial + 10 continuations = 11 turns
            assert mock_checker.check.call_count == 11

        asyncio.run(_run())

    def test_no_continuation_when_feedback_is_none(self) -> None:
        """No continuation when incomplete result has None feedback."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            client.read_messages.return_value = _messages_iterator(
                self._make_turn_messages("response")
            )

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            mock_checker.check.return_value = TaskCompletionResult(
                complete=False, feedback=None
            )

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            signal = VisibilityExpansionSignal()
            text, _usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=mock_prompt,
            )
            assert text == "response"
            assert mock_checker.check.call_count == 1

        asyncio.run(_run())

    def test_deadline_abort_stops_continuation(self) -> None:
        """Continuation stops when deadline is exhausted after first turn."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            clock = FakeClock()
            anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
            clock.set_wall(anchor)
            deadline = Deadline(
                expires_at=anchor + timedelta(seconds=60),
                clock=clock,
            )

            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]

            # Build message stream that advances clock past deadline
            # before the turn/completed message, so when check_task_completion
            # runs, the deadline is already expired.
            async def _expiring_messages() -> Any:
                yield {
                    "method": "item/agentMessage/delta",
                    "params": {"delta": "response"},
                }
                # Advance clock past deadline before turn completes
                clock.set_wall(anchor + timedelta(seconds=120))
                yield {
                    "method": "turn/completed",
                    "params": {"turn": {"status": "completed"}},
                }

            client.read_messages.return_value = _expiring_messages()

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            mock_checker.check.return_value = TaskCompletionResult.incomplete(
                "Missing file"
            )

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            signal = VisibilityExpansionSignal()
            text, _usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=deadline,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=mock_prompt,
                async_sleeper=clock,
            )
            assert text == "response"
            # Checker should not have been called because deadline was exhausted
            mock_checker.check.assert_not_called()

        asyncio.run(_run())

    def test_usage_accumulated_across_turns(self) -> None:
        """Token usage is summed across continuation rounds."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
                {"turn": {"id": 2}},
            ]

            call_count = 0

            def _read_messages() -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _messages_iterator(
                        [
                            {
                                "method": "thread/tokenUsage/updated",
                                "params": {
                                    "tokenUsage": {
                                        "last": {
                                            "inputTokens": 10,
                                            "outputTokens": 5,
                                        }
                                    }
                                },
                            },
                            {
                                "method": "turn/completed",
                                "params": {"turn": {"status": "completed"}},
                            },
                        ]
                    )
                return _messages_iterator(
                    [
                        {
                            "method": "thread/tokenUsage/updated",
                            "params": {
                                "tokenUsage": {
                                    "last": {
                                        "inputTokens": 30,
                                        "outputTokens": 15,
                                    }
                                }
                            },
                        },
                        {
                            "method": "turn/completed",
                            "params": {"turn": {"status": "completed"}},
                        },
                    ]
                )

            client.read_messages = _read_messages

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            mock_checker.check.side_effect = [
                TaskCompletionResult.incomplete("Not done"),
                TaskCompletionResult.ok("Done"),
            ]

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            signal = VisibilityExpansionSignal()
            _text, usage = await execute_protocol(
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                client=client,
                session=session,
                adapter_name="codex_app_server",
                prompt_name="test",
                prompt_text="do something",
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                tool_lookup={},
                output_schema=None,
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=signal,
                prompt=mock_prompt,
            )
            # Usage should be accumulated: 10+30=40, 5+15=20
            assert usage is not None
            assert usage.input_tokens == 40
            assert usage.output_tokens == 20

        asyncio.run(_run())

    def test_start_turn_error_wrapped(self) -> None:
        """CodexClientError from start_turn is wrapped in PromptEvaluationError."""

        async def _run() -> None:
            import pytest

            client = _make_mock_client()
            session, _ = _make_session()

            # initialize and thread/start succeed, turn/start fails
            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                CodexClientError("turn failed"),
            ]

            signal = VisibilityExpansionSignal()
            with pytest.raises(PromptEvaluationError, match="turn failed"):
                await execute_protocol(
                    client_config=CodexAppServerClientConfig(),
                    model_config=CodexAppServerModelConfig(),
                    client=client,
                    session=session,
                    adapter_name="codex_app_server",
                    prompt_name="test",
                    prompt_text="do something",
                    effective_cwd="/tmp",
                    dynamic_tool_specs=[],
                    tool_lookup={},
                    output_schema=None,
                    deadline=None,
                    budget_tracker=None,
                    run_context=None,
                    visibility_signal=signal,
                    prompt=None,
                )

        asyncio.run(_run())

    def test_visibility_signal_skips_continuation(self) -> None:
        """Continuation stops when visibility expansion signal is set."""

        async def _run() -> None:
            client = _make_mock_client()
            session, _ = _make_session()

            client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]

            # Set the visibility signal during the message stream.
            signal = VisibilityExpansionSignal()

            async def _messages_with_signal() -> Any:
                yield {
                    "method": "item/agentMessage/delta",
                    "params": {"delta": "response"},
                }
                signal.set(
                    VisibilityExpansionRequired(
                        "expand",
                        requested_overrides={},
                        reason="expand",
                        section_keys=(),
                    )
                )
                yield {
                    "method": "turn/completed",
                    "params": {"turn": {"status": "completed"}},
                }

            client.read_messages.return_value = _messages_with_signal()

            mock_checker = MagicMock(spec=TaskCompletionChecker)
            mock_checker.check.return_value = TaskCompletionResult.incomplete(
                "Missing file"
            )

            mock_prompt = MagicMock()
            mock_prompt.task_completion_checker = mock_checker
            mock_prompt.resources.get_optional.return_value = None

            import pytest

            with pytest.raises(VisibilityExpansionRequired):
                await execute_protocol(
                    client_config=CodexAppServerClientConfig(),
                    model_config=CodexAppServerModelConfig(),
                    client=client,
                    session=session,
                    adapter_name="codex_app_server",
                    prompt_name="test",
                    prompt_text="do something",
                    effective_cwd="/tmp",
                    dynamic_tool_specs=[],
                    tool_lookup={},
                    output_schema=None,
                    deadline=None,
                    budget_tracker=None,
                    run_context=None,
                    visibility_signal=signal,
                    prompt=mock_prompt,
                )
            # Checker should not have been called — loop broke early.
            mock_checker.check.assert_not_called()

        asyncio.run(_run())


class TestAccumulateUsage:
    """Tests for accumulate_usage helper."""

    def test_none_current(self) -> None:
        new = TokenUsage(input_tokens=10, output_tokens=5)
        result = accumulate_usage(None, new)
        assert result is new

    def test_sums_all_fields(self) -> None:
        current = TokenUsage(input_tokens=10, output_tokens=5, cached_tokens=2)
        new = TokenUsage(input_tokens=30, output_tokens=15, cached_tokens=3)
        result = accumulate_usage(current, new)
        assert result.input_tokens == 40
        assert result.output_tokens == 20
        assert result.cached_tokens == 5

    def test_none_fields_treated_as_zero(self) -> None:
        current = TokenUsage(input_tokens=None, output_tokens=5)
        new = TokenUsage(input_tokens=10, output_tokens=None)
        result = accumulate_usage(current, new)
        assert result.input_tokens == 10
        assert result.output_tokens == 5
