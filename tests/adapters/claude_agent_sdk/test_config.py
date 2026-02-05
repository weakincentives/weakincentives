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

"""Tests for Claude Agent SDK configuration dataclasses."""

from __future__ import annotations

import pytest

from weakincentives.adapters.claude_agent_sdk import TranscriptCollectorConfig
from weakincentives.adapters.claude_agent_sdk.config import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    IsolationConfig,
    NetworkPolicy,
)


class TestClaudeAgentSDKClientConfig:
    def test_defaults(self) -> None:
        config = ClaudeAgentSDKClientConfig()
        assert config.permission_mode == "bypassPermissions"
        assert config.cwd is None
        assert config.max_turns is None
        assert config.max_budget_usd is None
        assert config.suppress_stderr is True
        assert config.stop_on_structured_output is True
        assert config.isolation is None
        assert config.betas is None
        # Transcript collection is enabled by default
        assert config.transcript_collection is not None
        assert isinstance(config.transcript_collection, TranscriptCollectorConfig)

    def test_transcript_collection_disabled_with_none(self) -> None:
        config = ClaudeAgentSDKClientConfig(transcript_collection=None)
        assert config.transcript_collection is None

    def test_transcript_collection_custom_config(self) -> None:
        custom_config = TranscriptCollectorConfig(
            poll_interval=0.5,
            emit_raw_json=False,
        )
        config = ClaudeAgentSDKClientConfig(transcript_collection=custom_config)
        assert config.transcript_collection is custom_config
        assert config.transcript_collection.poll_interval == 0.5
        assert config.transcript_collection.emit_raw_json is False

    def test_with_all_values(self) -> None:
        isolation = IsolationConfig(network_policy=NetworkPolicy.no_network())
        config = ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            cwd="/home/user/project",
            max_turns=10,
            max_budget_usd=5.0,
            suppress_stderr=False,
            stop_on_structured_output=False,
            isolation=isolation,
            betas=("extended-thinking", "computer-use"),
        )
        assert config.permission_mode == "acceptEdits"
        assert config.cwd == "/home/user/project"
        assert config.max_turns == 10
        assert config.max_budget_usd == 5.0
        assert config.suppress_stderr is False
        assert config.stop_on_structured_output is False
        assert config.isolation is isolation
        assert config.betas == ("extended-thinking", "computer-use")


class TestClaudeAgentSDKModelConfig:
    def test_defaults(self) -> None:
        config = ClaudeAgentSDKModelConfig()
        assert config.model == "claude-opus-4-6"
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.max_thinking_tokens is None

    def test_with_model(self) -> None:
        config = ClaudeAgentSDKModelConfig(model="claude-opus-4-5-20250929")
        assert config.model == "claude-opus-4-5-20250929"

    def test_with_temperature(self) -> None:
        config = ClaudeAgentSDKModelConfig(temperature=0.7, max_tokens=1000)
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_with_max_thinking_tokens(self) -> None:
        config = ClaudeAgentSDKModelConfig(max_thinking_tokens=10000)
        assert config.max_thinking_tokens == 10000

    def test_rejects_unsupported_seed(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            ClaudeAgentSDKModelConfig(seed=42)

    def test_rejects_unsupported_stop(self) -> None:
        with pytest.raises(ValueError, match="stop"):
            ClaudeAgentSDKModelConfig(stop=("STOP",))

    def test_rejects_unsupported_presence_penalty(self) -> None:
        with pytest.raises(ValueError, match="presence_penalty"):
            ClaudeAgentSDKModelConfig(presence_penalty=0.5)

    def test_rejects_unsupported_frequency_penalty(self) -> None:
        with pytest.raises(ValueError, match="frequency_penalty"):
            ClaudeAgentSDKModelConfig(frequency_penalty=0.1)

    def test_rejects_multiple_unsupported(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            ClaudeAgentSDKModelConfig(seed=42, stop=("STOP",))

    def test_to_request_params_empty(self) -> None:
        config = ClaudeAgentSDKModelConfig()
        params = config.to_request_params()
        assert params == {}

    def test_to_request_params_with_values(self) -> None:
        config = ClaudeAgentSDKModelConfig(temperature=0.5, max_tokens=500, top_p=0.9)
        params = config.to_request_params()
        assert params == {"temperature": 0.5, "max_tokens": 500, "top_p": 0.9}
