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

"""Tests for the Gemini CLI ACP adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.gemini_acp.adapter import GeminiACPAdapter
from weakincentives.adapters.gemini_acp.config import (
    GeminiACPAdapterConfig,
    GeminiACPClientConfig,
)

from ..acp.conftest import AgentMessageChunk


class TestGeminiAdapterName:
    def test_returns_gemini_acp(self) -> None:
        adapter = GeminiACPAdapter()
        assert adapter._adapter_name() == "gemini_acp"


class TestGeminiAdapterDefaults:
    def test_uses_gemini_configs(self) -> None:
        adapter = GeminiACPAdapter()
        assert isinstance(adapter._adapter_config, GeminiACPAdapterConfig)
        assert isinstance(adapter._client_config, GeminiACPClientConfig)

    def test_custom_configs(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id="gemini-2.5-flash"),
            client_config=GeminiACPClientConfig(cwd="/tmp/work"),
        )
        assert adapter._adapter_config.model_id == "gemini-2.5-flash"
        assert adapter._client_config.cwd == "/tmp/work"


class TestValidateModel:
    def test_noop_with_models(self) -> None:
        adapter = GeminiACPAdapter()
        # Should not raise even with a non-matching model
        adapter._validate_model("nonexistent", [MagicMock(model_id="other")])

    def test_noop_with_empty_models(self) -> None:
        adapter = GeminiACPAdapter()
        adapter._validate_model("any-model", [])


class TestDetectEmptyResponse:
    def test_raises_on_zero_chunks(self) -> None:
        adapter = GeminiACPAdapter()
        client = MagicMock()
        client.message_chunks = []

        with pytest.raises(PromptEvaluationError, match="empty response"):
            adapter._detect_empty_response(client, MagicMock())

    def test_passes_with_chunks(self) -> None:
        adapter = GeminiACPAdapter()
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("hello")]

        # Should not raise
        adapter._detect_empty_response(client, MagicMock())


class TestAgentSpawnArgs:
    def test_base_args_with_default_model(self) -> None:
        adapter = GeminiACPAdapter()
        assert adapter._agent_spawn_args() == (
            "--experimental-acp",
            "--model",
            "gemini-2.5-flash",
        )

    def test_base_args_no_model(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id=None),
        )
        assert adapter._agent_spawn_args() == ("--experimental-acp",)

    def test_with_model(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id="gemini-2.5-flash"),
        )
        args = adapter._agent_spawn_args()
        assert args == ("--experimental-acp", "--model", "gemini-2.5-flash")

    def test_with_approval_mode_only(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id=None, approval_mode="yolo"),
        )
        args = adapter._agent_spawn_args()
        assert args == ("--experimental-acp", "--approval-mode", "yolo")

    def test_with_model_and_approval_mode(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                model_id="gemini-2.5-pro",
                approval_mode="auto_edit",
            ),
        )
        args = adapter._agent_spawn_args()
        assert args == (
            "--experimental-acp",
            "--model",
            "gemini-2.5-pro",
            "--approval-mode",
            "auto_edit",
        )

    def test_with_sandbox(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id=None, sandbox=True),
        )
        args = adapter._agent_spawn_args()
        assert args == ("--experimental-acp", "--sandbox")

    def test_with_all_flags(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                model_id="gemini-2.5-pro",
                approval_mode="yolo",
                sandbox=True,
            ),
        )
        args = adapter._agent_spawn_args()
        assert args == (
            "--experimental-acp",
            "--model",
            "gemini-2.5-pro",
            "--approval-mode",
            "yolo",
            "--sandbox",
        )

    def test_with_base_adapter_config(self) -> None:
        """When _adapter_config is not GeminiACPAdapterConfig, skip Gemini flags."""
        from weakincentives.adapters.acp.config import ACPAdapterConfig

        adapter = GeminiACPAdapter()
        # Force a base config to exercise the isinstance guard
        adapter._adapter_config = ACPAdapterConfig(model_id="test")  # type: ignore[misc]
        args = adapter._agent_spawn_args()
        assert args == ("--experimental-acp", "--model", "test")


class TestPrepareExecutionEnv:
    def test_no_sandbox_profile(self) -> None:
        adapter = GeminiACPAdapter()
        env, cleanup = adapter._prepare_execution_env(
            rendered=MagicMock(), effective_cwd="/tmp"
        )
        cleanup()
        # No SEATBELT_PROFILE injected
        if env is not None:
            assert "SEATBELT_PROFILE" not in env

    def test_sandbox_profile_injected(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                sandbox=True, sandbox_profile="strict-open"
            ),
        )
        env, cleanup = adapter._prepare_execution_env(
            rendered=MagicMock(), effective_cwd="/tmp"
        )
        cleanup()
        assert env is not None
        assert env["SEATBELT_PROFILE"] == "strict-open"

    def test_sandbox_profile_with_existing_env(self) -> None:
        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                sandbox=True, sandbox_profile="permissive-open"
            ),
            client_config=GeminiACPClientConfig(env={"MY_VAR": "hello"}),
        )
        env, cleanup = adapter._prepare_execution_env(
            rendered=MagicMock(), effective_cwd="/tmp"
        )
        cleanup()
        assert env is not None
        assert env["SEATBELT_PROFILE"] == "permissive-open"
        assert env["MY_VAR"] == "hello"

    def test_with_base_adapter_config(self) -> None:
        """When _adapter_config is not GeminiACPAdapterConfig, no profile injection."""
        from weakincentives.adapters.acp.config import ACPAdapterConfig

        adapter = GeminiACPAdapter()
        adapter._adapter_config = ACPAdapterConfig()  # type: ignore[misc]
        env, cleanup = adapter._prepare_execution_env(
            rendered=MagicMock(), effective_cwd="/tmp"
        )
        cleanup()
        if env is not None:
            assert "SEATBELT_PROFILE" not in env


class TestConfigureSession:
    def test_is_noop(self) -> None:
        import asyncio

        adapter = GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(model_id="gemini-2.5-flash"),
        )
        conn = AsyncMock()
        asyncio.run(adapter._configure_session(conn, "session-123"))
        # Should not call set_session_model or set_session_mode
        conn.set_session_model.assert_not_called()
        conn.set_session_mode.assert_not_called()
