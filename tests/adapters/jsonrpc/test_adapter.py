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

"""Tests for JsonRpcAdapter default hooks and protocol edge paths."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.jsonrpc._types import (
    ProtocolContext,
)
from weakincentives.adapters.jsonrpc.adapter import JsonRpcAdapter
from weakincentives.adapters.jsonrpc.client import JsonRpcClient, JsonRpcClientError
from weakincentives.adapters.jsonrpc.config import JsonRpcClientConfig
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline

# ---- Helpers ----

_TEST_CONFIG = JsonRpcClientConfig(
    bin_path="test-bin",
    bin_args=("serve",),
    bin_ws_args=("serve", "--listen"),
)


async def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    for msg in messages:
        yield msg


class TestDefaultHooks:
    """Test default hook implementations on JsonRpcAdapter base class."""

    def _make_fake_adapter(self) -> JsonRpcAdapter[Any]:
        class Fake(JsonRpcAdapter[Any]):
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
                return {}

            def _build_tool_specs(self, bridged_tools: Any) -> list[dict[str, object]]:
                return []

            def _build_output_schema(self, rendered: Any) -> dict[str, Any] | None:
                return None

            def _extract_token_usage(self, params: dict[str, object]) -> Any:
                return None

            def _map_error_phase(self, message: Any) -> str:
                return "response"

        return Fake(client_config=_TEST_CONFIG)

    def test_setup_environment_default(self) -> None:
        adapter = self._make_fake_adapter()
        state, env = adapter._setup_environment(MagicMock(), None)
        assert state is None
        assert env is None

    def test_setup_environment_with_env(self) -> None:
        adapter = self._make_fake_adapter()
        state, env = adapter._setup_environment(MagicMock(), {"K": "V"})
        assert state is None
        assert env == {"K": "V"}

    def test_cleanup_environment_noop(self) -> None:
        adapter = self._make_fake_adapter()
        adapter._cleanup_environment(None)  # Should not raise

    def test_create_transcript_bridge_default(self) -> None:
        adapter = self._make_fake_adapter()
        bridge = adapter._create_transcript_bridge(MagicMock(), "p")
        assert bridge is None

    def test_stop_transcript_bridge_none(self) -> None:
        adapter = self._make_fake_adapter()
        adapter._stop_transcript_bridge(None)  # Should not raise

    def test_stop_transcript_bridge_with_emitter(self) -> None:
        adapter = self._make_fake_adapter()
        bridge = MagicMock()
        adapter._stop_transcript_bridge(bridge)
        bridge.emitter.stop.assert_called_once()

    def test_check_task_completion_no_prompt(self) -> None:
        adapter = self._make_fake_adapter()
        cont, _fb = adapter._check_task_completion(
            prompt=None,
            session=MagicMock(),
            accumulated_text=None,
            deadline=None,
            budget_tracker=None,
        )
        assert not cont

    def test_check_task_completion_no_checker(self) -> None:
        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        prompt.task_completion_checker = None
        cont, _fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text=None,
            deadline=None,
            budget_tracker=None,
        )
        assert not cont

    def test_check_task_completion_deadline_exhausted(self) -> None:
        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        prompt.task_completion_checker = MagicMock()
        clock = FakeClock(_wall=datetime(2025, 1, 1, tzinfo=UTC))
        deadline = Deadline.create(
            expires_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
            clock=clock,
        )
        clock.advance(seconds=10)
        cont, _fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text="text",
            deadline=deadline,
            budget_tracker=None,
        )
        assert not cont

    def test_check_task_completion_budget_exhausted(self) -> None:
        from weakincentives.budget import Budget, BudgetTracker

        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        prompt.task_completion_checker = MagicMock()
        budget = Budget(max_total_tokens=1)
        tracker = BudgetTracker(budget)
        # Record usage to exceed
        from weakincentives.runtime.events.types import TokenUsage

        tracker.record_cumulative("p", TokenUsage(input_tokens=100))
        cont, _fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text="text",
            deadline=None,
            budget_tracker=tracker,
        )
        assert not cont

    def test_check_task_completion_complete(self) -> None:
        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        checker = MagicMock()
        result = MagicMock()
        result.complete = True
        result.feedback = None
        checker.check.return_value = result
        prompt.task_completion_checker = checker
        prompt.resources = MagicMock()
        prompt.resources.get_optional.return_value = None
        cont, fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text="done",
            deadline=None,
            budget_tracker=None,
        )
        assert not cont
        assert fb is None

    def test_check_task_completion_incomplete_no_feedback(self) -> None:
        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        checker = MagicMock()
        result = MagicMock()
        result.complete = False
        result.feedback = None
        checker.check.return_value = result
        prompt.task_completion_checker = checker
        prompt.resources = MagicMock()
        prompt.resources.get_optional.return_value = None
        cont, fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text="partial",
            deadline=None,
            budget_tracker=None,
        )
        assert not cont
        assert fb is None

    def test_check_task_completion_incomplete_with_feedback(self) -> None:
        adapter = self._make_fake_adapter()
        prompt = MagicMock()
        checker = MagicMock()
        result = MagicMock()
        result.complete = False
        result.feedback = "try again"
        checker.check.return_value = result
        prompt.task_completion_checker = checker
        prompt.resources = MagicMock()
        prompt.resources.get_optional.return_value = None
        cont, fb = adapter._check_task_completion(
            prompt=prompt,
            session=MagicMock(),
            accumulated_text="partial",
            deadline=None,
            budget_tracker=None,
        )
        assert cont
        assert fb == "try again"


class TestProtocolEdgePaths:
    """Cover remaining protocol branches: start_turn error, continuation,
    deadline watchdog, and visibility signal during server requests."""

    def test_start_turn_client_error_wrapped(self) -> None:
        """JsonRpcClientError from _start_turn is wrapped in PromptEvaluationError."""
        from weakincentives.adapters.jsonrpc._protocol import execute_protocol

        async def _run() -> None:
            adapter = MagicMock()
            adapter._create_transcript_bridge.return_value = None
            adapter._initialize_session = AsyncMock(return_value="sess")
            adapter._on_user_message_for_transcript = MagicMock()
            adapter._start_turn = AsyncMock(
                side_effect=JsonRpcClientError("connection lost")
            )
            adapter._stop_transcript_bridge = MagicMock()

            client = AsyncMock()
            client.start = AsyncMock()

            ctx = ProtocolContext(
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                output_schema=None,
            )

            with pytest.raises(PromptEvaluationError, match="connection lost"):
                await execute_protocol(
                    adapter=adapter,
                    client=client,
                    session=MagicMock(),
                    adapter_name="test",
                    prompt_name="p",
                    prompt_text="hello",
                    tool_lookup={},
                    deadline=None,
                    budget_tracker=None,
                    run_context=None,
                    visibility_signal=MagicMock(is_set=MagicMock(return_value=False)),
                    protocol_context=ctx,
                )

        asyncio.run(_run())

    def test_continuation_feedback_loop(self) -> None:
        """Task completion feedback drives a continuation turn."""
        from weakincentives.adapters.jsonrpc._protocol import (
            execute_protocol,
        )

        round_count = 0

        async def _run() -> None:
            nonlocal round_count
            adapter = MagicMock()
            adapter._create_transcript_bridge.return_value = None
            adapter._initialize_session = AsyncMock(return_value="sess")
            adapter._on_user_message_for_transcript = MagicMock()
            adapter._start_turn = AsyncMock(return_value="turn")
            adapter._stop_transcript_bridge = MagicMock()
            adapter._on_notification_for_transcript = MagicMock()
            adapter._process_notification.return_value = ("done", "")

            # First call: incomplete with feedback; second: complete
            call_count = 0

            def check_completion(**kwargs: object) -> tuple[bool, str | None]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return True, "try harder"
                return False, None

            adapter._check_task_completion.side_effect = check_completion

            client = AsyncMock()
            client.start = AsyncMock()
            client.read_messages = lambda: _messages_iterator(
                [{"method": "turn/completed", "params": {}}]
            )

            ctx = ProtocolContext(
                effective_cwd="/tmp",
                dynamic_tool_specs=[],
                output_schema=None,
            )
            vis = MagicMock()
            vis.is_set.return_value = False
            vis.get_and_clear.return_value = None

            _text, _usage = await execute_protocol(
                adapter=adapter,
                client=client,
                session=MagicMock(),
                adapter_name="test",
                prompt_name="p",
                prompt_text="hello",
                tool_lookup={},
                deadline=None,
                budget_tracker=None,
                run_context=None,
                visibility_signal=vis,
                protocol_context=ctx,
            )
            # Should have been called twice (initial + continuation)
            assert adapter._start_turn.call_count == 2

        asyncio.run(_run())

    def test_visibility_signal_breaks_server_request_loop(self) -> None:
        """Visibility signal during server request breaks consume_messages."""
        from weakincentives.adapters._shared._visibility_signal import (
            VisibilityExpansionSignal,
        )
        from weakincentives.adapters.jsonrpc._protocol import consume_messages

        async def _run() -> None:
            signal = VisibilityExpansionSignal()
            adapter = MagicMock()

            async def set_signal_on_request(*args: object, **kwargs: object) -> None:
                from weakincentives.prompt.errors import VisibilityExpansionRequired

                signal.set(
                    VisibilityExpansionRequired(
                        "test", requested_overrides={}, reason="r", section_keys=()
                    )
                )

            adapter._handle_server_request = AsyncMock(
                side_effect=set_signal_on_request
            )
            client = MagicMock()
            client.read_messages = lambda: _messages_iterator(
                [
                    {"id": 1, "method": "item/tool/call", "params": {}},
                    # This should NOT be reached:
                    {"method": "turn/completed", "params": {}},
                ]
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
                visibility_signal=signal,
            )
            # Visibility signal was set, so we broke early
            assert signal.is_set()

        asyncio.run(_run())

    def test_deadline_watchdog_sends_interrupt(self) -> None:
        """Deadline watchdog calls _send_interrupt after sleep."""
        from weakincentives.adapters.jsonrpc._protocol import _deadline_watchdog

        async def _run() -> None:
            adapter = MagicMock()
            adapter._send_interrupt = AsyncMock()
            sleeper = MagicMock()
            sleeper.async_sleep = AsyncMock()

            await _deadline_watchdog(adapter, MagicMock(), "sess", "turn", 1.0, sleeper)
            adapter._send_interrupt.assert_called_once()

        asyncio.run(_run())

    def test_make_approval_handler(self) -> None:
        """make_approval_handler creates a working handler."""
        from weakincentives.adapters.codex_app_server._protocol import (
            make_approval_handler,
        )
        from weakincentives.adapters.jsonrpc._types import ServerRequestContext

        async def _run() -> None:
            handler = make_approval_handler("never")
            client = AsyncMock()
            ctx = ServerRequestContext(
                client=client,
                request_id=42,
                method="item/commandExecution/requestApproval",
                params={},
                tool_lookup={},
                bridge=None,
                prompt=None,
                session=None,
                deadline=None,
            )
            await handler(ctx)
            client.send_response.assert_called_once_with(42, {"decision": "accept"})

        asyncio.run(_run())
