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

"""Session-level protocol tests for the generic ACP adapter."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.runtime.events import (
    InProcessDispatcher,
)

from .conftest import (
    MockModelInfo,
    MockNewSessionResponse,
    MockPromptResponse,
    MockSessionModelState,
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

        from weakincentives.adapters.acp._prompt_loop import drain_quiet_period
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        assert client.last_update_time is None

        # Should return immediately — not block for 5 s.
        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=5000,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

    def test_respects_max_drain_cap(self) -> None:
        """Drain exits within the max cap when no deadline is set."""
        import asyncio
        import time

        from weakincentives.adapters.acp._prompt_loop import drain_quiet_period
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        start = time.monotonic()
        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=60_000,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
                max_drain_s=0.05,
            )
        )
        elapsed = time.monotonic() - start

        # Should finish well under 1 s (capped at ~50 ms, not 60 s).
        assert elapsed < 1.0

    def test_drain_snapshot_consistency(self) -> None:
        """Drain uses a snapshot of last_update_time per iteration."""
        import asyncio
        import time

        from weakincentives.adapters.acp._prompt_loop import drain_quiet_period
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        # Set last_update_time to "just now" so drain waits for quiet period.
        client._last_update_time = time.monotonic()

        start = time.monotonic()
        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=100,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
                max_drain_s=1.0,
            )
        )
        elapsed = time.monotonic() - start

        # Should terminate after ~100 ms quiet period, not hang.
        assert elapsed < 1.0
        # Should have waited at least the quiet period.
        assert elapsed >= 0.05

    def test_drain_exits_when_snapshot_becomes_none(self) -> None:
        """Drain exits if last_update_time becomes None mid-loop."""
        import asyncio
        import time

        from weakincentives.adapters.acp._prompt_loop import drain_quiet_period
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        async def _draining() -> None:
            original_sleep = asyncio.sleep

            class _CustomSleeper:
                async def async_sleep(self, s: float) -> None:
                    client._last_update_time = None
                    await original_sleep(min(s, 0.01))

            await drain_quiet_period(
                client,
                None,
                quiet_period_ms=60_000,
                clock=SYSTEM_CLOCK,
                async_sleeper=_CustomSleeper(),
                max_drain_s=5.0,
            )

        start = time.monotonic()
        asyncio.run(_draining())
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
