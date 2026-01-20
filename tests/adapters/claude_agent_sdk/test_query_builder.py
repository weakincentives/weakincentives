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

"""Tests for SDK query builder module."""

from __future__ import annotations

from unittest.mock import MagicMock

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    SdkQueryBuilder,
    SdkQueryOptions,
)


class TestSdkQueryOptions:
    """Tests for SdkQueryOptions dataclass."""

    def test_default_values(self) -> None:
        """Test that SdkQueryOptions has sensible defaults."""
        options = SdkQueryOptions(model="test-model")

        assert options.model == "test-model"
        assert options.cwd is None
        assert options.permission_mode is None
        assert options.max_turns is None
        assert options.max_budget_usd is None
        assert options.betas == ()
        assert options.output_format is None
        assert options.allowed_tools is None
        assert options.disallowed_tools == ()
        assert options.env == {}
        assert options.setting_sources == ()
        assert options.max_thinking_tokens is None
        assert options.mcp_servers == {}
        assert options.stderr_handler is None
        assert options.hooks == {}

    def test_to_kwargs_minimal(self) -> None:
        """Test to_kwargs with only model set."""
        options = SdkQueryOptions(model="test-model")
        kwargs = options.to_kwargs()

        assert kwargs == {"model": "test-model"}

    def test_to_kwargs_full(self) -> None:
        """Test to_kwargs with all options set."""
        stderr_handler = MagicMock()
        options = SdkQueryOptions(
            model="test-model",
            cwd="/test/cwd",
            permission_mode="bypassPermissions",
            max_turns=10,
            max_budget_usd=5.0,
            betas=("beta1", "beta2"),
            output_format={"type": "json_schema"},
            allowed_tools=("tool1", "tool2"),
            disallowed_tools=("tool3",),
            env={"HOME": "/test"},
            setting_sources=("user",),
            max_thinking_tokens=1024,
            mcp_servers={"wink": {}},
            stderr_handler=stderr_handler,
            hooks={"PreToolUse": []},
        )
        kwargs = options.to_kwargs()

        assert kwargs["model"] == "test-model"
        assert kwargs["cwd"] == "/test/cwd"
        assert kwargs["permission_mode"] == "bypassPermissions"
        assert kwargs["max_turns"] == 10
        assert kwargs["max_budget_usd"] == 5.0
        assert kwargs["betas"] == ["beta1", "beta2"]
        assert kwargs["output_format"] == {"type": "json_schema"}
        assert kwargs["allowed_tools"] == ["tool1", "tool2"]
        assert kwargs["disallowed_tools"] == ["tool3"]
        assert kwargs["env"] == {"HOME": "/test"}
        assert kwargs["setting_sources"] == ["user"]
        assert kwargs["max_thinking_tokens"] == 1024
        assert kwargs["mcp_servers"] == {"wink": {}}
        assert kwargs["stderr"] is stderr_handler
        assert kwargs["hooks"] == {"PreToolUse": []}


class TestSdkQueryBuilder:
    """Tests for SdkQueryBuilder class."""

    def test_build_minimal(self) -> None:
        """Test building with minimal configuration."""
        builder = SdkQueryBuilder("test-model")
        options = builder.build()

        assert options.model == "test-model"
        assert options.cwd is None

    def test_with_cwd(self) -> None:
        """Test with_cwd method."""
        builder = SdkQueryBuilder("test-model")
        builder.with_cwd("/test/cwd")
        options = builder.build()

        assert options.cwd == "/test/cwd"

    def test_with_client_config(self) -> None:
        """Test with_client_config method."""
        config = ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            max_turns=5,
            max_budget_usd=2.0,
            betas=("beta1",),
        )
        builder = SdkQueryBuilder("test-model")
        builder.with_client_config(config)
        options = builder.build()

        assert options.permission_mode == "acceptEdits"
        assert options.max_turns == 5
        assert options.max_budget_usd == 2.0
        assert options.betas == ("beta1",)

    def test_with_client_config_defaults(self) -> None:
        """Test with_client_config with default values."""
        config = ClaudeAgentSDKClientConfig()
        builder = SdkQueryBuilder("test-model")
        builder.with_client_config(config)
        options = builder.build()

        # permission_mode is bypassPermissions by default but should still be set
        assert options.permission_mode == "bypassPermissions"
        assert options.max_turns is None
        assert options.max_budget_usd is None

    def test_with_model_config(self) -> None:
        """Test with_model_config method."""
        config = ClaudeAgentSDKModelConfig(max_thinking_tokens=2048)
        builder = SdkQueryBuilder("test-model")
        builder.with_model_config(config)
        options = builder.build()

        assert options.max_thinking_tokens == 2048

    def test_with_model_config_no_thinking_tokens(self) -> None:
        """Test with_model_config without thinking tokens."""
        config = ClaudeAgentSDKModelConfig()
        builder = SdkQueryBuilder("test-model")
        builder.with_model_config(config)
        options = builder.build()

        assert options.max_thinking_tokens is None

    def test_with_ephemeral_home(self) -> None:
        """Test with_ephemeral_home method."""
        mock_home = MagicMock()
        mock_home.get_env.return_value = {"HOME": "/ephemeral"}
        mock_home.get_setting_sources.return_value = ["user"]

        builder = SdkQueryBuilder("test-model")
        builder.with_ephemeral_home(mock_home)
        options = builder.build()

        assert options.env == {"HOME": "/ephemeral"}
        assert options.setting_sources == ("user",)

    def test_with_output_format(self) -> None:
        """Test with_output_format method."""
        output_format = {"type": "json_schema", "schema": {}}
        builder = SdkQueryBuilder("test-model")
        builder.with_output_format(output_format)
        options = builder.build()

        assert options.output_format == output_format

    def test_with_tool_constraints(self) -> None:
        """Test with_tool_constraints method."""
        builder = SdkQueryBuilder("test-model")
        builder.with_tool_constraints(
            allowed_tools=("tool1", "tool2"),
            disallowed_tools=("tool3",),
        )
        options = builder.build()

        assert options.allowed_tools == ("tool1", "tool2")
        assert options.disallowed_tools == ("tool3",)

    def test_with_mcp_server(self) -> None:
        """Test with_mcp_server method."""
        server_config = {"type": "stdio"}
        builder = SdkQueryBuilder("test-model")
        builder.with_mcp_server("wink", server_config)
        options = builder.build()

        assert options.mcp_servers == {"wink": server_config}

    def test_with_stderr_handler(self) -> None:
        """Test with_stderr_handler method."""
        handler = MagicMock()
        builder = SdkQueryBuilder("test-model")
        builder.with_stderr_handler(handler)
        options = builder.build()

        assert options.stderr_handler is handler

    def test_with_hooks(self) -> None:
        """Test with_hooks method."""
        hooks = {"PreToolUse": [], "PostToolUse": []}
        builder = SdkQueryBuilder("test-model")
        builder.with_hooks(hooks)
        options = builder.build()

        assert options.hooks == hooks

    def test_method_chaining(self) -> None:
        """Test that all methods return self for chaining."""
        mock_home = MagicMock()
        mock_home.get_env.return_value = {}
        mock_home.get_setting_sources.return_value = []

        builder = SdkQueryBuilder("test-model")
        result = (
            builder.with_cwd("/test")
            .with_client_config(ClaudeAgentSDKClientConfig())
            .with_model_config(ClaudeAgentSDKModelConfig())
            .with_ephemeral_home(mock_home)
            .with_output_format(None)
            .with_tool_constraints()
            .with_mcp_server("test", {})
            .with_stderr_handler(None)
            .with_hooks({})
        )

        assert result is builder

    def test_build_with_empty_options(self) -> None:
        """Test building with explicitly empty optional values.

        This test covers the branch cases where env, setting_sources,
        stderr_handler, and hooks are empty/None.
        """
        builder = SdkQueryBuilder("test-model")
        # Set empty values explicitly
        builder._env = {}
        builder._setting_sources = ()
        builder._stderr_handler = None
        builder._hooks = {}
        options = builder.build()

        # Verify to_kwargs doesn't include empty values
        kwargs = options.to_kwargs()
        assert "env" not in kwargs
        assert "setting_sources" not in kwargs
        assert "stderr" not in kwargs
        assert "hooks" not in kwargs
