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

"""Protocol-level tests for the generic ACP adapter."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.runtime.events import (
    InProcessDispatcher,
    PromptExecuted,
    PromptRendered,
)

from .conftest import (
    MockModelInfo,
    MockNewSessionResponse,
    MockPromptResponse,
    MockSessionModelState,
    MockUsage,
    make_mock_connection,
    make_mock_process,
)


def _make_mock_session() -> MagicMock:
    """Create a mock session with dispatcher."""
    dispatcher = InProcessDispatcher()
    session = MagicMock()
    session.dispatcher = dispatcher
    session.session_id = "test-session"
    return session


def _make_mock_prompt(
    *,
    text: str = "Hello",
    output_type: type[Any] | None = None,
    container: str | None = None,
) -> MagicMock:
    """Create a mock prompt."""
    prompt = MagicMock()
    prompt.ns = "test"
    prompt.key = "prompt"
    prompt.name = "test-prompt"

    rendered = MagicMock()
    rendered.text = text
    rendered.tools = ()
    rendered.output_type = output_type
    rendered.container = container
    rendered.allow_extra_keys = False

    prompt.render.return_value = rendered
    prompt.filesystem.return_value = None
    prompt.bind.return_value = prompt
    prompt.resources = MagicMock()
    prompt.resources.__enter__ = MagicMock(return_value=None)
    prompt.resources.__exit__ = MagicMock(return_value=False)
    return prompt


def _make_mock_deadline(*, remaining_s: float = 60.0) -> MagicMock:
    """Create a mock deadline with remaining time."""
    deadline = MagicMock()
    deadline.remaining.return_value = timedelta(seconds=remaining_s)
    return deadline


def _setup_acp_mocks() -> dict[str, Any]:
    """Set up mock ACP modules in sys.modules."""
    mock_acp = MagicMock()

    mock_acp.PROTOCOL_VERSION = 1

    @asynccontextmanager
    async def _spawn(*args: Any, **kwargs: Any) -> Any:
        conn = make_mock_connection()
        proc = make_mock_process()
        conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
        conn.prompt = AsyncMock(return_value=MockPromptResponse())
        yield conn, proc

    mock_acp.spawn_agent_process = _spawn

    mock_schema = MagicMock()
    mock_schema.ClientCapabilities = MagicMock
    mock_schema.FileSystemCapability = MagicMock
    mock_schema.Implementation = MagicMock
    mock_schema.HttpMcpServer = MagicMock
    mock_schema.TextBlock = MagicMock(side_effect=lambda **kw: kw)

    mock_acp.schema = mock_schema

    sys.modules["acp"] = mock_acp
    sys.modules["acp.schema"] = mock_schema

    return {"acp": mock_acp, "acp.schema": mock_schema}


def _cleanup_acp_mocks() -> None:
    """Remove mock ACP modules from sys.modules."""
    for key in ["acp", "acp.schema", "acp.contrib", "acp.contrib.session_state"]:
        sys.modules.pop(key, None)


def _patch_mcp() -> tuple[Any, Any]:
    """Return (mock_mcp_cls_patch, mock_create_patch) context managers."""
    return (
        patch("weakincentives.adapters.acp.adapter.MCPHttpServer"),
        patch("weakincentives.adapters.acp.adapter.create_mcp_tool_server"),
    )


def _make_mcp_mock(cls_mock: MagicMock) -> MagicMock:
    """Configure the MCPHttpServer class mock and return the instance mock."""
    mock_mcp = MagicMock()
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()
    mock_mcp.url = "http://127.0.0.1:9999/mcp"
    mock_mcp.server_name = "wink-tools"
    mock_mcp.to_http_mcp_server.return_value = MagicMock()
    cls_mock.return_value = mock_mcp
    return mock_mcp


class TestEvaluateProtocol:
    """Test the full evaluate flow with mocked ACP."""

    def setup_method(self) -> None:
        self._mocks = _setup_acp_mocks()

    def teardown_method(self) -> None:
        _cleanup_acp_mocks()

    def test_evaluate_simple(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = _make_mock_prompt(text="Hello ACP")
        session = _make_mock_session()

        rendered_events: list[PromptRendered] = []
        executed_events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptRendered, rendered_events.append)
        session.dispatcher.subscribe(PromptExecuted, executed_events.append)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result.prompt_name == "test-prompt"
        assert len(rendered_events) == 1
        assert len(executed_events) == 1
        assert rendered_events[0].adapter == "acp"
        assert executed_events[0].adapter == "acp"

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

        prompt = _make_mock_prompt()
        session = _make_mock_session()
        executed_events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, executed_events.append)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        # Create a mock structured_capture that was called
        mock_capture = MagicMock()
        mock_capture.called = True
        mock_capture.data = {"summary": "test"}

        mcp_cls_p, create_p = _patch_mcp()
        with (
            mcp_cls_p as mock_mcp_cls,
            create_p as mock_create,
            patch.object(
                adapter,
                "_prepare_tools",
                return_value=([], mock_capture),
            ),
        ):
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with (
            mcp_cls_p as mock_mcp_cls,
            create_p as mock_create,
            patch(
                "weakincentives.adapters.acp.adapter.VisibilityExpansionSignal"
            ) as mock_signal_cls,
        ):
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        prompt.render.return_value.tools = (tool,)
        session = _make_mock_session()

        dispatched: list[RenderedTools] = []
        session.dispatcher.subscribe(RenderedTools, dispatched.append)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
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

        prompt = _make_mock_prompt()
        session = _make_mock_session()
        session.dispatcher.subscribe(RenderedTools, _failing_handler)

        caplog.set_level(logging.ERROR)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None
        assert any(
            "rendered_tools_dispatch_failed" in record.message
            for record in caplog.records
        )


class TestHandshakeAndConfigure:
    """Test _handshake and _configure_session via full evaluate."""

    def setup_method(self) -> None:
        self._mocks = _setup_acp_mocks()

    def teardown_method(self) -> None:
        _cleanup_acp_mocks()

    def test_evaluate_with_model_and_mode(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_configured(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(
                return_value=MockNewSessionResponse(
                    models=MockSessionModelState(
                        available_models=[
                            MockModelInfo(model_id="test-model", name="Test")
                        ]
                    )
                )
            )
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_configured

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(
                model_id="test-model",
                mode_id="build",
                quiet_period_ms=0,
            ),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None

    def test_evaluate_with_model_set_failure(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_model_err(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.set_session_model = AsyncMock(side_effect=RuntimeError("fail"))
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_model_err

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(
                model_id="test-model",
                quiet_period_ms=0,
            ),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            # Should NOT raise — model set failure is non-fatal
            result = adapter.evaluate(prompt, session=session)

        assert result is not None

    def test_evaluate_with_mode_set_failure(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        @asynccontextmanager
        async def spawn_mode_err(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.set_session_mode = AsyncMock(
                side_effect=RuntimeError("Internal error")
            )
            conn.prompt = AsyncMock(return_value=MockPromptResponse())
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_mode_err

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(
                mode_id="plan",
                quiet_period_ms=0,
            ),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            # Should NOT raise — mode set failure is non-fatal
            result = adapter.evaluate(prompt, session=session)

        assert result is not None

    def test_evaluate_with_custom_env(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp", env={"FOO": "bar"}),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None


class TestDrainQuietPeriod:
    """Tests for _drain_quiet_period behaviour."""

    def test_exits_immediately_when_no_updates(self) -> None:
        """Drain returns immediately when no updates have been received."""
        import asyncio

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.adapters.acp.client import ACPClient

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=5000),
            client_config=ACPClientConfig(cwd="/tmp"),
        )
        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        assert client.last_update_time is None

        # Should return immediately — not block for 5 s.
        asyncio.run(adapter._drain_quiet_period(client, deadline=None))

    def test_respects_max_drain_cap(self) -> None:
        """Drain exits within the max cap when no deadline is set."""
        import asyncio
        import time

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.adapters.acp.client import ACPClient

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=60_000),
            client_config=ACPClientConfig(cwd="/tmp"),
        )
        adapter._MAX_DRAIN_S = 0.05  # 50 ms cap for test speed

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        start = time.monotonic()
        asyncio.run(adapter._drain_quiet_period(client, deadline=None))
        elapsed = time.monotonic() - start

        # Should finish well under 1 s (capped at ~50 ms, not 60 s).
        assert elapsed < 1.0

    def test_drain_snapshot_consistency(self) -> None:
        """Drain uses a snapshot of last_update_time per iteration."""
        import asyncio
        import time

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.adapters.acp.client import ACPClient

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=100),
            client_config=ACPClientConfig(cwd="/tmp"),
        )
        adapter._MAX_DRAIN_S = 1.0

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        # Set last_update_time to "just now" so drain waits for quiet period.
        client._last_update_time = time.monotonic()

        start = time.monotonic()
        asyncio.run(adapter._drain_quiet_period(client, deadline=None))
        elapsed = time.monotonic() - start

        # Should terminate after ~100 ms quiet period, not hang.
        assert elapsed < 1.0
        # Should have waited at least the quiet period.
        assert elapsed >= 0.05

    def test_drain_exits_when_snapshot_becomes_none(self) -> None:
        """Drain exits if last_update_time becomes None mid-loop."""
        import asyncio
        import time

        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.adapters.acp.client import ACPClient

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=60_000),
            client_config=ACPClientConfig(cwd="/tmp"),
        )
        adapter._MAX_DRAIN_S = 5.0

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        original_sleep = asyncio.sleep

        async def _clear_and_sleep(s: float) -> None:
            # Simulate last_update_time becoming None during the sleep.
            client._last_update_time = None
            await original_sleep(min(s, 0.01))

        start = time.monotonic()
        with patch(
            "weakincentives.adapters.acp.adapter.asyncio.sleep", _clear_and_sleep
        ):
            asyncio.run(adapter._drain_quiet_period(client, deadline=None))
        elapsed = time.monotonic() - start

        # Should exit quickly, not wait the full 60 s quiet period.
        assert elapsed < 1.0


class TestProtocolImportError:
    def test_raises_import_error(self) -> None:
        import asyncio
        import builtins

        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        original_import = builtins.__import__

        def _block_acp(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "acp" or name.startswith("acp."):
                raise ImportError("No module named 'acp'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_block_acp),
            pytest.raises(ImportError, match="agent-client-protocol"),
        ):
            asyncio.run(
                adapter._execute_protocol(
                    client=MagicMock(),
                    mcp_server=MagicMock(),
                    session=_make_mock_session(),
                    prompt_name="test",
                    prompt_text="hello",
                    rendered=MagicMock(),
                    effective_cwd="/tmp",
                    deadline=None,
                    run_context=None,
                    visibility_signal=MagicMock(),
                    structured_capture=None,
                )
            )


class TestHandshakeTimeout:
    """Test that startup_timeout_s is enforced on the handshake."""

    def setup_method(self) -> None:
        self._mocks = _setup_acp_mocks()

    def teardown_method(self) -> None:
        _cleanup_acp_mocks()

    def test_evaluate_prompt_timeout(self) -> None:
        """conn.prompt() respects the deadline and raises PromptEvaluationError."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        async def _hang_forever(*_args: Any, **_kwargs: Any) -> MockPromptResponse:
            await asyncio.sleep(999)
            return MockPromptResponse()  # pragma: no cover

        @asynccontextmanager
        async def spawn_hang_prompt(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
            conn.prompt = AsyncMock(side_effect=_hang_forever)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_hang_prompt

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()
        deadline = _make_mock_deadline(remaining_s=0.05)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(PromptEvaluationError, match="ACP prompt timed out"):
                adapter.evaluate(prompt, session=session, deadline=deadline)

    def test_handshake_timeout_raises_prompt_evaluation_error(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]

        async def _hang_forever(*_args: Any, **_kwargs: Any) -> MockNewSessionResponse:
            await asyncio.sleep(999)
            return MockNewSessionResponse()  # pragma: no cover

        @asynccontextmanager
        async def spawn_hang(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            # Make initialize hang indefinitely
            conn.initialize = AsyncMock(side_effect=_hang_forever)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = spawn_hang

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp", startup_timeout_s=0.05),
        )

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(PromptEvaluationError, match="ACP handshake timed out"):
                adapter.evaluate(prompt, session=session)
