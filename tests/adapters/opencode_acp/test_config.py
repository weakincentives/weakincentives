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

"""Tests for OpenCode ACP configuration dataclasses."""

from __future__ import annotations

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.opencode_acp.config import (
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)


class TestOpenCodeACPClientConfig:
    def test_defaults_match_base(self) -> None:
        base = ACPClientConfig()
        opencode = OpenCodeACPClientConfig()
        assert opencode.agent_bin == base.agent_bin
        assert opencode.agent_args == base.agent_args
        assert opencode.cwd == base.cwd
        assert opencode.env == base.env
        assert opencode.startup_timeout_s == base.startup_timeout_s
        assert opencode.permission_mode == base.permission_mode
        assert opencode.allow_file_reads == base.allow_file_reads
        assert opencode.allow_file_writes == base.allow_file_writes
        assert opencode.mcp_servers == base.mcp_servers

    def test_is_subtype(self) -> None:
        opencode = OpenCodeACPClientConfig()
        assert isinstance(opencode, ACPClientConfig)

    def test_frozen(self) -> None:
        cfg = OpenCodeACPClientConfig()
        try:
            cfg.agent_bin = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestOpenCodeACPAdapterConfig:
    def test_defaults_match_base(self) -> None:
        base = ACPAdapterConfig()
        opencode = OpenCodeACPAdapterConfig()
        assert opencode.mode_id == base.mode_id
        assert opencode.model_id == base.model_id
        assert opencode.quiet_period_ms == base.quiet_period_ms
        assert opencode.emit_thought_chunks == base.emit_thought_chunks

    def test_is_subtype(self) -> None:
        opencode = OpenCodeACPAdapterConfig()
        assert isinstance(opencode, ACPAdapterConfig)

    def test_frozen(self) -> None:
        cfg = OpenCodeACPAdapterConfig()
        try:
            cfg.quiet_period_ms = 999  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
