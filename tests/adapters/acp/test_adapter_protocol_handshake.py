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

"""Protocol-level tests for ACP adapter - handshake, configure, and import."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError

from .conftest import (
    MockModelInfo,
    MockNewSessionResponse,
    MockPromptResponse,
    MockSessionModelState,
    cleanup_acp_mocks,
    make_mcp_mock,
    make_mock_connection,
    make_mock_deadline,
    make_mock_process,
    make_mock_prompt,
    make_mock_session,
    patch_mcp,
    setup_acp_mocks,
)


class TestHandshakeAndConfigure:
    """Test _handshake and _configure_session via full evaluate."""

    def setup_method(self) -> None:
        self._mocks = setup_acp_mocks()

    def teardown_method(self) -> None:
        cleanup_acp_mocks()

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

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
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

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
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

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
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

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()
            result = adapter.evaluate(prompt, session=session)

        assert result is not None


class TestProtocolImportError:
    def test_raises_import_error(self) -> None:
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
                    session=make_mock_session(),
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
        self._mocks = setup_acp_mocks()

    def teardown_method(self) -> None:
        cleanup_acp_mocks()

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

        prompt = make_mock_prompt()
        session = make_mock_session()
        deadline = make_mock_deadline(remaining_s=0.05)

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
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

        prompt = make_mock_prompt()
        session = make_mock_session()

        mcp_cls_p, create_p = patch_mcp()
        with mcp_cls_p as mock_mcp_cls, create_p as mock_create:
            make_mcp_mock(mock_mcp_cls)
            mock_create.return_value = MagicMock()

            with pytest.raises(PromptEvaluationError, match="ACP handshake timed out"):
                adapter.evaluate(prompt, session=session)
