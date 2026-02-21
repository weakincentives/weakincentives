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

"""Protocol-level tests for the generic ACP adapter - evaluate flow."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.runtime.events import PromptExecuted, PromptRendered

from .conftest import (
    MockNewSessionResponse,
    MockPromptResponse,
    MockUsage,
    cleanup_acp_mocks,
    make_mcp_mock,
    make_mock_connection,
    make_mock_process,
    make_mock_prompt,
    make_mock_session,
    patch_mcp,
    setup_acp_mocks,
)


class TestEvaluateProtocol:
    """Test the full evaluate flow with mocked ACP."""

    def setup_method(self) -> None:
        self._mocks = setup_acp_mocks()

    def teardown_method(self) -> None:
        cleanup_acp_mocks()

    def test_evaluate_simple(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt(text="Hello ACP")
        session = make_mock_session()

        rendered_events: list[PromptRendered] = []
        executed_events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptRendered, rendered_events.append)
        session.dispatcher.subscribe(PromptExecuted, executed_events.append)

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result.prompt_name == "test-prompt"
        assert len(rendered_events) == 1
        assert len(executed_events) == 1
        assert rendered_events[0].adapter == "acp"
        assert executed_events[0].adapter == "acp"

    def test_feedback_hook_passed_to_mcp_server(self) -> None:
        """The feedback hook closure is passed to create_mcp_tool_server."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt(text="Hello ACP")
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            adapter.evaluate(prompt, session=session)

        # Verify post_call_hook was passed
        _, kwargs = mock_create.call_args
        hook = kwargs["post_call_hook"]
        assert hook is not None

        # Exercise the hook closure to cover _feedback_hook body
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": "output"}],
            "isError": False,
        }
        with patch(
            "weakincentives.adapters.acp.adapter._append_feedback",
        ) as mock_append:
            hook("tool_name", {"arg": 1}, result)
            mock_append.assert_called_once_with(
                result["content"],
                is_error=False,
                prompt=prompt,
                session=session,
                deadline=None,
            )

    def test_evaluate_with_token_usage(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_with_usage(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(
                return_value=MockPromptResponse(
                    usage=MockUsage(input_tokens=100, output_tokens=50)
                )
            )
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_with_usage

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()
        executed_events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, executed_events.append)

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None
        assert len(executed_events) == 1
        usage = executed_events[0].usage
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_evaluate_wraps_unexpected_error(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_error(*args: Any, **kwargs: Any) -> Any:
            msg = "Connection refused"
            raise OSError(msg)
            yield  # pragma: no cover

        mock_acp.spawn_agent_process = spawn_error

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(PromptEvaluationError, match="ACP execution failed"):
                adapter.evaluate(prompt, session=session)

    def test_evaluate_passthrough_visibility_expansion(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.prompt.errors import VisibilityExpansionRequired

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_vis(*args: Any, **kwargs: Any) -> Any:
            raise VisibilityExpansionRequired(
                "Expand section",
                requested_overrides={("test",): "full"},
                reason="model requested",
                section_keys=("test",),
            )
            yield  # pragma: no cover

        mock_acp.spawn_agent_process = spawn_vis

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(VisibilityExpansionRequired):
                adapter.evaluate(prompt, session=session)

    def test_evaluate_passthrough_prompt_evaluation_error(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_eval_err(*args: Any, **kwargs: Any) -> Any:
            raise PromptEvaluationError(
                message="Model rate limit", prompt_name="p", phase="request"
            )
            yield  # pragma: no cover

        mock_acp.spawn_agent_process = spawn_eval_err

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(PromptEvaluationError, match="Model rate limit"):
                adapter.evaluate(prompt, session=session)

    def test_evaluate_with_budget_tracker(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.deadlines import Deadline

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_budget(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(
                return_value=MockPromptResponse(
                    usage=MockUsage(input_tokens=10, output_tokens=5)
                )
            )
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_budget

        budget = Budget(
            max_input_tokens=1000,
            deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1)),
        )
        tracker = BudgetTracker(budget)

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(
                prompt, session=session, budget=budget, budget_tracker=tracker
            )

        assert result is not None

    def test_evaluate_creates_budget_tracker_from_budget(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.budget import Budget
        from weakincentives.deadlines import Deadline

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_bt(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_bt

        budget = Budget(
            max_input_tokens=1000,
            deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1)),
        )

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session, budget=budget)

        assert result is not None

    def test_evaluate_with_tool_call_tracker(self) -> None:
        """Covers tool_call_tracker dispatch in _execute_protocol."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_with_tracker(*args: Any, **kwargs: Any) -> Any:
            # The first arg (after self) is the client
            client = args[0]
            client.tool_call_tracker["tc-1"] = {
                "title": "file_read",
                "status": "completed",
                "output": "file contents here",
            }
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_with_tracker

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None

    def test_evaluate_skips_empty_check_when_capture_called(self) -> None:
        """When structured_capture.called is True, skip empty-response check."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_capture(*args: Any, **kwargs: Any) -> Any:
            # Client will have no message_chunks by default, which
            # would trigger empty-response detection if capture_ok is False.
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_capture

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        # Create a mock structured_capture that was called
        mock_capture = MagicMock()
        mock_capture.called = True
        mock_capture.data = {"summary": "test"}

        mcp_cls_p, create_p = patch_mcp()
        with (
            mcp_cls_p as mock_mcp_cls,
            create_p as mock_create,
            patch.object(
                adapter,
                "_prepare_tools",
                return_value=([], mock_capture),
            ),
        ):
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        # Should succeed without raising empty-response error
        assert result is not None

    def test_evaluate_visibility_signal_in_protocol(self) -> None:
        """Covers visibility signal raise inside _execute_protocol."""
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.prompt.errors import VisibilityExpansionRequired

        mock_acp = self._mocks["acp"]

        vis_exc = VisibilityExpansionRequired(
            "Expand section",
            requested_overrides={("test",): "full"},
            reason="model requested",
            section_keys=("test",),
        )

        @asynccontextmanager
        async def spawn_vis_signal(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_vis_signal

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with (
            mcp_cls_p as mock_mcp_cls,
            create_p as mock_create,
            patch(
                "weakincentives.adapters.acp.adapter.VisibilityExpansionSignal"
            ) as mock_signal_cls,
        ):
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            mock_signal = MagicMock()
            mock_signal.get_and_clear.return_value = vis_exc
            mock_signal_cls.return_value = mock_signal

            with pytest.raises(VisibilityExpansionRequired):
                adapter.evaluate(prompt, session=session)

    def test_evaluate_with_temp_workspace_cleanup(self) -> None:
        """Covers temp_workspace_dir cleanup in _evaluate_async."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd=None),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None

    def test_evaluate_with_tools_dispatches_schemas(self) -> None:
        """Tool schemas are extracted and dispatched via RenderedTools."""
        from dataclasses import dataclass

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.prompt import Tool
        from weakincentives.prompt.tool import ToolContext, ToolResult
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        @dataclass(slots=True, frozen=True)
        class _Params:
            x: int

        @dataclass(slots=True, frozen=True)
        class _Result:
            value: int

        def _handler(params: _Params, *, context: ToolContext) -> ToolResult[_Result]:
            return ToolResult.ok(_Result(value=params.x))

        tool = Tool[_Params, _Result](
            name="calc", description="Calculate", handler=_handler
        )

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        prompt.render.return_value.tools = (tool,)
        session = make_mock_session()

        dispatched: list[RenderedTools] = []
        session.dispatcher.subscribe(RenderedTools, dispatched.append)

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None
        assert len(dispatched) == 1
        assert len(dispatched[0].tools) == 1
        assert dispatched[0].tools[0].name == "calc"

    def test_rendered_tools_dispatch_failure_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RenderedTools dispatch failure is logged, not raised."""
        import logging

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        def _failing_handler(event: RenderedTools) -> None:
            raise RuntimeError("boom")

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = make_mock_prompt()
        session = make_mock_session()
        session.dispatcher.subscribe(RenderedTools, _failing_handler)

        caplog.set_level(logging.ERROR)

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None
        assert any(
            "rendered_tools_dispatch_failed" in record.message
            for record in caplog.records
        )
