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

from weakincentives.adapters.claude_agent_sdk.config import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    SandboxNetworkConfig,
    SandboxSettings,
)


class TestSandboxNetworkConfig:
    """Tests for SandboxNetworkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SandboxNetworkConfig()
        assert config.allow_local_binding is False
        assert config.allow_unix_sockets == ()

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SandboxNetworkConfig(
            allow_local_binding=True,
            allow_unix_sockets=("/tmp/sock1", "/tmp/sock2"),
        )
        assert config.allow_local_binding is True
        assert config.allow_unix_sockets == ("/tmp/sock1", "/tmp/sock2")

    def test_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = SandboxNetworkConfig()
        try:
            config.allow_local_binding = True  # noqa: B003
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestSandboxSettings:
    """Tests for SandboxSettings dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SandboxSettings()
        assert config.enabled is False
        assert config.auto_allow_bash_if_sandboxed is False
        assert config.excluded_commands == ()
        assert config.allow_unsandboxed_commands is False
        assert config.network is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        network = SandboxNetworkConfig(allow_local_binding=True)
        config = SandboxSettings(
            enabled=True,
            auto_allow_bash_if_sandboxed=True,
            excluded_commands=("ls", "cat"),
            allow_unsandboxed_commands=True,
            network=network,
        )
        assert config.enabled is True
        assert config.auto_allow_bash_if_sandboxed is True
        assert config.excluded_commands == ("ls", "cat")
        assert config.allow_unsandboxed_commands is True
        assert config.network is network


class TestClaudeAgentSDKClientConfig:
    """Tests for ClaudeAgentSDKClientConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClaudeAgentSDKClientConfig()
        assert config.permission_mode == "bypassPermissions"
        assert config.cwd is None
        assert config.add_dirs == ()
        assert config.env is None
        assert config.setting_sources == ()
        assert config.sandbox is None
        assert config.max_turns is None
        assert config.include_partial_messages is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            cwd="/home/user/project",
            add_dirs=("/home/user/libs",),
            env={"FOO": "bar"},
            setting_sources=("user", "project"),
            max_turns=10,
            include_partial_messages=True,
        )
        assert config.permission_mode == "acceptEdits"
        assert config.cwd == "/home/user/project"
        assert config.add_dirs == ("/home/user/libs",)
        assert config.env == {"FOO": "bar"}
        assert config.setting_sources == ("user", "project")
        assert config.max_turns == 10
        assert config.include_partial_messages is True

    def test_to_client_kwargs_minimal(self) -> None:
        """Test to_client_kwargs with minimal config."""
        config = ClaudeAgentSDKClientConfig()
        kwargs = config.to_client_kwargs()
        assert kwargs == {"permission_mode": "bypassPermissions"}

    def test_to_client_kwargs_full(self) -> None:
        """Test to_client_kwargs with all fields set."""
        sandbox = SandboxSettings(enabled=True)
        config = ClaudeAgentSDKClientConfig(
            permission_mode="plan",
            cwd="/home/user/project",
            add_dirs=("/extra",),
            env={"VAR": "value"},
            setting_sources=("local",),
            sandbox=sandbox,
            max_turns=5,
            include_partial_messages=True,
        )
        kwargs = config.to_client_kwargs()
        assert kwargs["permission_mode"] == "plan"
        assert kwargs["cwd"] == "/home/user/project"
        assert kwargs["add_dirs"] == ["/extra"]
        assert kwargs["env"] == {"VAR": "value"}
        assert kwargs["setting_sources"] == ["local"]
        assert kwargs["sandbox"]["enabled"] is True
        assert kwargs["max_turns"] == 5
        assert kwargs["include_partial_messages"] is True


class TestClaudeAgentSDKModelConfig:
    """Tests for ClaudeAgentSDKModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClaudeAgentSDKModelConfig()
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.temperature is None
        assert config.max_tokens is None

    def test_inherits_from_llm_config(self) -> None:
        """Test that model config inherits LLMConfig fields."""
        config = ClaudeAgentSDKModelConfig(
            model="claude-opus-4-20250514",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        assert config.model == "claude-opus-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
