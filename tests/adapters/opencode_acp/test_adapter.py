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

"""Tests for the OpenCode ACP adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.opencode_acp.adapter import OpenCodeACPAdapter
from weakincentives.adapters.opencode_acp.config import (
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)

from ..acp.conftest import AgentMessageChunk, MockModelInfo


class TestOpenCodeAdapterName:
    def test_returns_opencode_acp(self) -> None:
        adapter = OpenCodeACPAdapter()
        assert adapter._adapter_name() == "opencode_acp"


class TestOpenCodeAdapterDefaults:
    def test_uses_opencode_configs(self) -> None:
        adapter = OpenCodeACPAdapter()
        assert isinstance(adapter._adapter_config, OpenCodeACPAdapterConfig)
        assert isinstance(adapter._client_config, OpenCodeACPClientConfig)

    def test_custom_configs(self) -> None:
        adapter = OpenCodeACPAdapter(
            adapter_config=OpenCodeACPAdapterConfig(model_id="test-model"),
            client_config=OpenCodeACPClientConfig(cwd="/tmp/work"),
        )
        assert adapter._adapter_config.model_id == "test-model"
        assert adapter._client_config.cwd == "/tmp/work"


class TestValidateModel:
    def test_valid_model_passes(self) -> None:
        adapter = OpenCodeACPAdapter()
        models = [MockModelInfo(model_id="claude-4", name="Claude 4")]
        # Should not raise
        adapter._validate_model("claude-4", models)

    def test_invalid_model_raises(self) -> None:
        adapter = OpenCodeACPAdapter()
        models = [
            MockModelInfo(model_id="claude-4", name="Claude 4"),
            MockModelInfo(model_id="gpt-5", name="GPT-5"),
        ]
        with pytest.raises(PromptEvaluationError, match="not found"):
            adapter._validate_model("nonexistent", models)

    def test_empty_models_skips_validation(self) -> None:
        adapter = OpenCodeACPAdapter()
        # No models → skip validation
        adapter._validate_model("any-model", [])


class TestDetectEmptyResponse:
    def test_raises_on_zero_chunks(self) -> None:
        adapter = OpenCodeACPAdapter()
        client = MagicMock()
        client.message_chunks = []

        with pytest.raises(PromptEvaluationError, match="empty response"):
            adapter._detect_empty_response(client, MagicMock())

    def test_passes_with_chunks(self) -> None:
        adapter = OpenCodeACPAdapter()
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("hello")]

        # Should not raise
        adapter._detect_empty_response(client, MagicMock())
