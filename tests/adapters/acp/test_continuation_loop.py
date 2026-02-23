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

"""Tests for task completion continuation loop in ACP _execute_protocol."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.prompt import VisibilityExpansionRequired
from weakincentives.prompt.task_completion import (
    TaskCompletionChecker,
    TaskCompletionResult,
)
from weakincentives.runtime.events import InProcessDispatcher

from .conftest import (
    MockNewSessionResponse,
    MockPromptResponse,
    MockUsage,
    make_mock_connection,
    make_mock_process,
)

# ---- Helpers ----


def _make_mock_session() -> MagicMock:
    dispatcher = InProcessDispatcher()
    session = MagicMock()
    session.dispatcher = dispatcher
    session.session_id = "test-session"
    return session


def _make_mock_prompt(
    *,
    checker: Any = None,
) -> MagicMock:
    prompt = MagicMock()
    prompt.ns = "test"
    prompt.key = "prompt"
    prompt.name = "test-prompt"

    rendered = MagicMock()
    rendered.text = "Hello"
    rendered.tools = ()
    rendered.output_type = None
    rendered.container = None
    rendered.allow_extra_keys = False

    prompt.render.return_value = rendered
    prompt.filesystem.return_value = None
    prompt.bind.return_value = prompt
    prompt.resources = MagicMock()
    prompt.resource_scope = MagicMock(return_value=MagicMock())
    prompt.task_completion_checker = checker
    prompt.resources.get_optional.return_value = None
    return prompt


def _setup_acp_mocks(
    *,
    conn: AsyncMock | None = None,
    usage: MockUsage | None = None,
) -> dict[str, Any]:
    mock_acp = MagicMock()
    mock_acp.PROTOCOL_VERSION = 1

    effective_conn = conn

    @asynccontextmanager
    async def _spawn(*args: Any, **kwargs: Any) -> Any:
        c = effective_conn or make_mock_connection()
        c.new_session = AsyncMock(return_value=MockNewSessionResponse())
        if c.prompt.return_value is None or not hasattr(c.prompt, "side_effect"):
            c.prompt = AsyncMock(
                return_value=MockPromptResponse(usage=usage),
            )
        yield c, make_mock_process()

    mock_acp.spawn_agent_process = _spawn

    mock_schema = MagicMock()
    mock_schema.ClientCapabilities = MagicMock
    mock_schema.FileSystemCapability = MagicMock
    mock_schema.Implementation = MagicMock
    mock_schema.HttpMcpServer = MagicMock
    mock_schema.TextContentBlock = MagicMock(side_effect=lambda **kw: kw)

    mock_acp.schema = mock_schema

    sys.modules["acp"] = mock_acp
    sys.modules["acp.schema"] = mock_schema

    return {"acp": mock_acp, "acp.schema": mock_schema}


def _cleanup_acp_mocks() -> None:
    for key in ["acp", "acp.schema", "acp.contrib", "acp.contrib.session_state"]:
        sys.modules.pop(key, None)


def _patch_mcp() -> tuple[Any, Any]:
    return (
        patch("weakincentives.adapters.acp.adapter.MCPHttpServer"),
        patch("weakincentives.adapters.acp.adapter.create_mcp_tool_server"),
    )


def _make_mcp_mock(cls_mock: MagicMock) -> MagicMock:
    mock_mcp = MagicMock()
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()
    mock_mcp.url = "http://127.0.0.1:9999/mcp"
    mock_mcp.server_name = "wink-tools"
    mock_mcp.to_http_mcp_server.return_value = MagicMock()
    cls_mock.return_value = mock_mcp
    return mock_mcp


# ---- Tests ----


class TestContinuationLoop:
    """Tests for the task completion continuation loop in _execute_protocol."""

    def setup_method(self) -> None:
        self._mocks = _setup_acp_mocks()

    def teardown_method(self) -> None:
        _cleanup_acp_mocks()

    def test_single_turn_no_continuation(self) -> None:
        """No continuation when task completion checker returns complete."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt()

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result.prompt_name == "test-prompt"

    def test_continuation_on_incomplete(self) -> None:
        """Continues with additional turns when task completion returns incomplete."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]
        call_count = 0

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.side_effect = [
            TaskCompletionResult.incomplete("Missing file"),
            TaskCompletionResult.ok("Done"),
        ]

        @asynccontextmanager
        async def _spawn(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())

            nonlocal call_count

            async def _prompt(*a: Any, **kw: Any) -> MockPromptResponse:
                nonlocal call_count
                call_count += 1
                # Inject message chunks into client for text extraction
                client = args[0]
                client._message_chunks.append(
                    MagicMock(content=f"response {call_count}")
                )
                client._last_update_time = None
                return MockPromptResponse()

            conn.prompt = AsyncMock(side_effect=_prompt)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = _spawn

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            adapter.evaluate(prompt, session=session)

        assert mock_checker.check.call_count == 2
        assert call_count == 2

    def test_max_continuation_rounds(self) -> None:
        """Stops after max continuation rounds even if still incomplete."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]
        call_count = 0

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.incomplete(
            "Still missing"
        )

        @asynccontextmanager
        async def _spawn(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())

            nonlocal call_count

            async def _prompt(*a: Any, **kw: Any) -> MockPromptResponse:
                nonlocal call_count
                call_count += 1
                client = args[0]
                client._message_chunks.append(
                    MagicMock(content=f"response {call_count}")
                )
                client._last_update_time = None
                return MockPromptResponse()

            conn.prompt = AsyncMock(side_effect=_prompt)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = _spawn

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            adapter.evaluate(prompt, session=session)

        # 1 initial + 10 continuations = 11 turns
        assert mock_checker.check.call_count == 11
        assert call_count == 11

    def test_no_continuation_when_feedback_is_none(self) -> None:
        """No continuation when incomplete result has None feedback."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult(
            complete=False, feedback=None
        )

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None
        assert mock_checker.check.call_count == 1

    def test_usage_accumulated_across_turns(self) -> None:
        """Token usage is summed across continuation rounds."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_acp = self._mocks["acp"]
        call_count = 0

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.side_effect = [
            TaskCompletionResult.incomplete("Not done"),
            TaskCompletionResult.ok("Done"),
        ]

        @asynccontextmanager
        async def _spawn(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())

            nonlocal call_count

            async def _prompt(*a: Any, **kw: Any) -> MockPromptResponse:
                nonlocal call_count
                call_count += 1
                client = args[0]
                client._message_chunks.append(
                    MagicMock(content=f"response {call_count}")
                )
                client._last_update_time = None
                if call_count == 1:
                    return MockPromptResponse(
                        usage=MockUsage(input_tokens=10, output_tokens=5)
                    )
                return MockPromptResponse(
                    usage=MockUsage(input_tokens=30, output_tokens=15)
                )

            conn.prompt = AsyncMock(side_effect=_prompt)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = _spawn

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        from weakincentives.runtime.events import PromptExecuted

        executed_events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, executed_events.append)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            adapter.evaluate(prompt, session=session)

        assert len(executed_events) == 1
        usage = executed_events[0].usage
        assert usage is not None
        assert usage.input_tokens == 40
        assert usage.output_tokens == 20

    def test_visibility_signal_breaks_loop(self) -> None:
        """Continuation stops when visibility expansion signal is set."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.incomplete(
            "Missing file"
        )

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        vis_exc = VisibilityExpansionRequired(
            "Expand section",
            requested_overrides={("test",): "full"},
            reason="model requested",
            section_keys=("test",),
        )

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

        # Checker should not have been called — loop broke early.
        mock_checker.check.assert_not_called()

    def test_deadline_exhaustion_stops_continuation(self) -> None:
        """Continuation stops when deadline is exhausted."""
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.clock import FakeClock
        from weakincentives.deadlines import Deadline

        mock_acp = self._mocks["acp"]
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(
            expires_at=anchor + timedelta(seconds=60),
            clock=clock,
        )

        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.incomplete("Not done")

        @asynccontextmanager
        async def _spawn(*args: Any, **kwargs: Any) -> Any:
            conn = make_mock_connection()
            conn.new_session = AsyncMock(return_value=MockNewSessionResponse())

            async def _prompt(*a: Any, **kw: Any) -> MockPromptResponse:
                client = args[0]
                client._message_chunks.append(MagicMock(content="response"))
                client._last_update_time = None
                # Advance clock past deadline after first prompt
                clock.set_wall(anchor + timedelta(seconds=120))
                return MockPromptResponse()

            conn.prompt = AsyncMock(side_effect=_prompt)
            yield conn, make_mock_process()

        mock_acp.spawn_agent_process = _spawn

        adapter = ACPAdapter(
            adapter_config=ACPAdapterConfig(quiet_period_ms=0),
            client_config=ACPClientConfig(cwd="/tmp"),
            clock=clock,
        )

        session = _make_mock_session()
        prompt = _make_mock_prompt(checker=mock_checker)

        mcp_cls_p, create_p = _patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            _make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            adapter.evaluate(prompt, session=session, deadline=deadline)

        # Checker should not have been called because deadline was exhausted
        mock_checker.check.assert_not_called()
