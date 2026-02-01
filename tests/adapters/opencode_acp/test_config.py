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

"""Tests for OpenCode ACP adapter configuration."""

from __future__ import annotations

import weakincentives.adapters.opencode_acp as opencode_acp_module
from weakincentives.adapters.opencode_acp import (
    McpServerConfig,
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)


class TestOpenCodeACPClientConfig:
    def test_defaults(self) -> None:
        config = OpenCodeACPClientConfig()
        assert config.opencode_bin == "opencode"
        assert config.opencode_args == ("acp",)
        assert config.cwd is None
        assert config.env is None
        assert config.suppress_stderr is True
        assert config.startup_timeout_s == 10.0
        assert config.permission_mode == "auto"
        assert config.allow_file_reads is False
        assert config.allow_file_writes is False
        assert config.allow_terminal is False
        assert config.mcp_servers == ()
        assert config.reuse_session is False

    def test_custom_values(self) -> None:
        mcp_server = McpServerConfig(
            name="test-server",
            command="test-cmd",
            args=("--arg1", "--arg2"),
            env={"KEY": "value"},
        )
        config = OpenCodeACPClientConfig(
            opencode_bin="/custom/opencode",
            opencode_args=("acp", "--debug"),
            cwd="/path/to/workspace",
            env={"API_KEY": "secret"},
            suppress_stderr=False,
            startup_timeout_s=30.0,
            permission_mode="deny",
            allow_file_reads=True,
            allow_file_writes=True,
            allow_terminal=True,
            mcp_servers=(mcp_server,),
            reuse_session=True,
        )
        assert config.opencode_bin == "/custom/opencode"
        assert config.opencode_args == ("acp", "--debug")
        assert config.cwd == "/path/to/workspace"
        assert config.env == {"API_KEY": "secret"}
        assert config.suppress_stderr is False
        assert config.startup_timeout_s == 30.0
        assert config.permission_mode == "deny"
        assert config.allow_file_reads is True
        assert config.allow_file_writes is True
        assert config.allow_terminal is True
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].name == "test-server"
        assert config.reuse_session is True

    def test_permission_mode_options(self) -> None:
        for mode in ("auto", "deny", "prompt"):
            config = OpenCodeACPClientConfig(permission_mode=mode)  # type: ignore[arg-type]
            assert config.permission_mode == mode


class TestOpenCodeACPAdapterConfig:
    def test_defaults(self) -> None:
        config = OpenCodeACPAdapterConfig()
        assert config.mode_id is None
        assert config.model_id is None
        assert config.quiet_period_ms == 100
        assert config.emit_thought_chunks is False

    def test_custom_values(self) -> None:
        config = OpenCodeACPAdapterConfig(
            mode_id="agent",
            model_id="claude-sonnet",
            quiet_period_ms=500,
            emit_thought_chunks=True,
        )
        assert config.mode_id == "agent"
        assert config.model_id == "claude-sonnet"
        assert config.quiet_period_ms == 500
        assert config.emit_thought_chunks is True


class TestMcpServerConfig:
    def test_defaults(self) -> None:
        config = McpServerConfig(name="test", command="test-cmd")
        assert config.name == "test"
        assert config.command == "test-cmd"
        assert config.args == ()
        assert config.env is None

    def test_with_all_options(self) -> None:
        config = McpServerConfig(
            name="my-server",
            command="/path/to/server",
            args=("--port", "8080"),
            env={"HOST": "localhost"},
        )
        assert config.name == "my-server"
        assert config.command == "/path/to/server"
        assert config.args == ("--port", "8080")
        assert config.env == {"HOST": "localhost"}


class TestModuleDir:
    def test_dir_returns_sorted_all(self) -> None:
        result = dir(opencode_acp_module)
        # __dir__ should return sorted __all__
        assert result == sorted(opencode_acp_module.__all__)
