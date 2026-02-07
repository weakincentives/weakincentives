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

"""Configuration dataclasses for the Codex App Server adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ...dataclasses import FrozenDataclass

__all__ = [
    "ApiKeyAuth",
    "ApprovalPolicy",
    "CodexAppServerClientConfig",
    "CodexAppServerModelConfig",
    "CodexAuthMode",
    "ExternalTokenAuth",
    "McpServerConfig",
    "Personality",
    "ReasoningEffort",
    "ReasoningSummary",
    "SandboxMode",
]

ApprovalPolicy = Literal["never", "untrusted", "on-failure", "on-request"]
"""How command/file approvals are handled."""

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
"""Sandbox mode for thread/start."""

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
"""Reasoning effort level for the model."""

ReasoningSummary = Literal["auto", "concise", "detailed", "none"]
"""Reasoning summary preference."""

Personality = Literal["none", "friendly", "pragmatic"]
"""Response personality."""

McpServerConfig = dict[str, Any]
"""MCP server configuration (command+args or url)."""

DEFAULT_MODEL = "gpt-5.3-codex"
"""Default Codex model identifier."""


@FrozenDataclass()
class ApiKeyAuth:
    """API key authentication for Codex."""

    api_key: str


@FrozenDataclass()
class ExternalTokenAuth:
    """External token authentication (ChatGPT tokens)."""

    id_token: str
    access_token: str


CodexAuthMode = ApiKeyAuth | ExternalTokenAuth
"""Union of supported authentication modes."""


@FrozenDataclass()
class CodexAppServerClientConfig:
    """Client-level configuration for Codex App Server.

    Attributes:
        codex_bin: Executable to spawn.
        cwd: Working directory (must be absolute; defaults to Path.cwd()).
        env: Extra environment variables merged into the subprocess env.
        suppress_stderr: Capture stderr for debugging instead of printing.
        startup_timeout_s: Max time for the initialize handshake.
        approval_policy: How to handle command/file approvals.
        sandbox_mode: Sandbox mode for thread/start.
        auth_mode: Authentication configuration. None inherits host credentials.
        mcp_servers: Additional external MCP server configurations.
        ephemeral: If true, thread is not persisted to disk.
        client_name: Client identifier for initialize.
        client_version: Client version for initialize.
    """

    codex_bin: str = "codex"
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    suppress_stderr: bool = True
    startup_timeout_s: float = 10.0
    approval_policy: ApprovalPolicy = "never"
    sandbox_mode: SandboxMode | None = None
    auth_mode: CodexAuthMode | None = None
    mcp_servers: dict[str, McpServerConfig] | None = None
    ephemeral: bool = False
    client_name: str = "wink"
    client_version: str = "0.1.0"
    transcript: bool = True
    """Emit transcript entries during evaluation."""
    transcript_emit_raw: bool = True
    """Include raw notification JSON in ``raw`` field."""


@FrozenDataclass()
class CodexAppServerModelConfig:
    """Model-level configuration for Codex App Server.

    Attributes:
        model: Codex model identifier.
        effort: Reasoning effort level.
        summary: Reasoning summary preference.
        personality: Response personality.
    """

    model: str = DEFAULT_MODEL
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None
    personality: Personality | None = None
