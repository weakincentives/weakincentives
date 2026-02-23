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

"""Tests for Gemini CLI ACP configuration dataclasses."""

from __future__ import annotations

from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.gemini_acp.config import (
    GeminiACPAdapterConfig,
    GeminiACPClientConfig,
)


class TestGeminiACPClientConfig:
    def test_defaults_differ_from_base(self) -> None:
        base = ACPClientConfig()
        gemini = GeminiACPClientConfig()
        assert gemini.agent_bin == "gemini"
        assert gemini.agent_bin != base.agent_bin
        assert gemini.agent_args == ("--experimental-acp",)
        assert gemini.agent_args != base.agent_args
        assert gemini.startup_timeout_s == 15.0
        assert gemini.startup_timeout_s != base.startup_timeout_s

    def test_inherited_defaults_match_base(self) -> None:
        base = ACPClientConfig()
        gemini = GeminiACPClientConfig()
        assert gemini.cwd == base.cwd
        assert gemini.env == base.env
        assert gemini.permission_mode == base.permission_mode
        assert gemini.allow_file_reads == base.allow_file_reads
        assert gemini.allow_file_writes == base.allow_file_writes
        assert gemini.mcp_servers == base.mcp_servers

    def test_is_subtype(self) -> None:
        gemini = GeminiACPClientConfig()
        assert isinstance(gemini, ACPClientConfig)

    def test_frozen(self) -> None:
        cfg = GeminiACPClientConfig()
        try:
            cfg.agent_bin = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestGeminiACPAdapterConfig:
    def test_defaults_differ_from_base(self) -> None:
        base = ACPAdapterConfig()
        gemini = GeminiACPAdapterConfig()
        assert gemini.model_id == "gemini-2.5-flash"
        assert gemini.model_id != base.model_id
        assert gemini.emit_thought_chunks is True
        assert gemini.emit_thought_chunks != base.emit_thought_chunks
        assert gemini.quiet_period_ms == 200
        assert gemini.quiet_period_ms != base.quiet_period_ms

    def test_approval_mode_field(self) -> None:
        cfg = GeminiACPAdapterConfig()
        assert cfg.approval_mode is None
        cfg_with = GeminiACPAdapterConfig(approval_mode="yolo")
        assert cfg_with.approval_mode == "yolo"

    def test_sandbox_defaults(self) -> None:
        cfg = GeminiACPAdapterConfig()
        assert cfg.sandbox is False
        assert cfg.sandbox_profile is None

    def test_sandbox_fields(self) -> None:
        cfg = GeminiACPAdapterConfig(sandbox=True, sandbox_profile="strict-open")
        assert cfg.sandbox is True
        assert cfg.sandbox_profile == "strict-open"

    def test_inherited_defaults_match_base(self) -> None:
        base = ACPAdapterConfig()
        gemini = GeminiACPAdapterConfig()
        assert gemini.mode_id == base.mode_id

    def test_is_subtype(self) -> None:
        gemini = GeminiACPAdapterConfig()
        assert isinstance(gemini, ACPAdapterConfig)

    def test_frozen(self) -> None:
        cfg = GeminiACPAdapterConfig()
        try:
            cfg.quiet_period_ms = 999  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
