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

"""Tests for ACP configuration dataclasses."""

from __future__ import annotations

from weakincentives.adapters.acp.config import (
    ACPAdapterConfig,
    ACPClientConfig,
)


class TestACPClientConfig:
    def test_defaults(self) -> None:
        cfg = ACPClientConfig()
        assert cfg.agent_bin == "opencode"
        assert cfg.agent_args == ("acp",)
        assert cfg.cwd is None
        assert cfg.env is None
        assert cfg.suppress_stderr is True
        assert cfg.startup_timeout_s == 10.0
        assert cfg.permission_mode == "auto"
        assert cfg.allow_file_reads is False
        assert cfg.allow_file_writes is False
        assert cfg.allow_terminal is False
        assert cfg.mcp_servers == ()
        assert cfg.reuse_session is False

    def test_custom_values(self) -> None:
        cfg = ACPClientConfig(
            agent_bin="/usr/local/bin/myagent",
            agent_args=("serve", "--port", "8080"),
            cwd="/tmp/work",
            env={"API_KEY": "secret"},
            suppress_stderr=False,
            startup_timeout_s=30.0,
            permission_mode="deny",
            allow_file_reads=True,
            allow_file_writes=True,
            allow_terminal=True,
            mcp_servers=({"command": "npx", "args": ["mcp"]},),
            reuse_session=True,
        )
        assert cfg.agent_bin == "/usr/local/bin/myagent"
        assert cfg.agent_args == ("serve", "--port", "8080")
        assert cfg.cwd == "/tmp/work"
        assert cfg.env is not None
        assert cfg.env["API_KEY"] == "secret"
        assert cfg.suppress_stderr is False
        assert cfg.startup_timeout_s == 30.0
        assert cfg.permission_mode == "deny"
        assert cfg.allow_file_reads is True
        assert cfg.allow_file_writes is True
        assert cfg.allow_terminal is True
        assert len(cfg.mcp_servers) == 1
        assert cfg.reuse_session is True

    def test_frozen(self) -> None:
        cfg = ACPClientConfig()
        try:
            cfg.agent_bin = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestACPAdapterConfig:
    def test_defaults(self) -> None:
        cfg = ACPAdapterConfig()
        assert cfg.mode_id is None
        assert cfg.model_id is None
        assert cfg.quiet_period_ms == 500
        assert cfg.emit_thought_chunks is False

    def test_custom_values(self) -> None:
        cfg = ACPAdapterConfig(
            mode_id="fast",
            model_id="gpt-4o",
            quiet_period_ms=1000,
            emit_thought_chunks=True,
        )
        assert cfg.mode_id == "fast"
        assert cfg.model_id == "gpt-4o"
        assert cfg.quiet_period_ms == 1000
        assert cfg.emit_thought_chunks is True

    def test_frozen(self) -> None:
        cfg = ACPAdapterConfig()
        try:
            cfg.quiet_period_ms = 999  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
