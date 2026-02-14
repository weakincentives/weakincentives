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

"""Configuration dataclasses for the ACP adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ...dataclasses import FrozenDataclass

__all__ = ["ACPAdapterConfig", "ACPClientConfig", "McpServerConfig"]

McpServerConfig = dict[str, Any]
"""MCP server configuration passed through to ACP new_session."""


@FrozenDataclass()
class ACPClientConfig:
    """Client-level configuration for ACP agents.

    Generic defaults that work with any ACP-compatible agent binary.

    Attributes:
        agent_bin: Executable to spawn.
        agent_args: Arguments passed to the agent binary.
        cwd: Working directory (must be absolute; defaults to Path.cwd()).
        env: Extra environment variables merged into the subprocess env.
        startup_timeout_s: Max time for the initialize handshake.
        permission_mode: How to handle permissions in the ACP agent.
        allow_file_reads: Allow the agent to read files.
        allow_file_writes: Allow the agent to write files.
        mcp_servers: Additional external MCP server configurations.
    """

    agent_bin: str = "opencode"
    agent_args: tuple[str, ...] = ("acp",)
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 10.0
    permission_mode: Literal["auto", "deny", "prompt"] = "auto"
    allow_file_reads: bool = False
    allow_file_writes: bool = False
    mcp_servers: tuple[McpServerConfig, ...] = ()


@FrozenDataclass()
class ACPAdapterConfig:
    """Adapter-level configuration for ACP evaluation.

    Attributes:
        mode_id: ACP mode identifier. None uses the agent default.
        model_id: ACP model identifier. None uses the agent default.
        quiet_period_ms: Milliseconds of silence before considering the
            agent idle. Used for streaming coalescing.
        emit_thought_chunks: If true, emit thought/reasoning chunks from
            the ACP agent as streaming events.
    """

    mode_id: str | None = None
    model_id: str | None = None
    quiet_period_ms: int = 500
    emit_thought_chunks: bool = False
