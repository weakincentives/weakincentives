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

"""Configuration dataclasses for the OpenCode ACP adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from ...dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from .workspace import McpServerConfig

__all__ = [
    "OpenCodeACPAdapterConfig",
    "OpenCodeACPClientConfig",
    "PermissionMode",
]

PermissionMode = Literal["auto", "deny", "prompt"]
"""Permission handling mode for OpenCode ACP tool execution.

- ``"auto"``: Automatically approve permission requests.
- ``"deny"``: Deny all permission requests.
- ``"prompt"``: Respond as deny (non-interactive; adapter cannot block).
"""


@FrozenDataclass()
class OpenCodeACPClientConfig:
    """Client-level configuration for OpenCode ACP.

    Attributes:
        opencode_bin: Executable to spawn. Defaults to ``"opencode"``.
        opencode_args: Arguments to pass to the executable. Must include
            ``"acp"`` to start the ACP server.
        cwd: Working directory for OpenCode operations. Must be an absolute
            path if provided. When None, defaults to ``Path.cwd().resolve()``.
        env: Extra environment variables to pass to the subprocess.
        suppress_stderr: If True, capture stderr for errors instead of
            printing to console. Useful for programmatic usage.
        startup_timeout_s: Maximum time to wait for initialize and session/new
            to complete.
        permission_mode: How to respond to ``session/request_permission``.
            - ``"auto"``: Approve all permission requests.
            - ``"deny"``: Deny all permission requests.
            - ``"prompt"``: Deny with explanation that interactive prompting
              is not supported (non-blocking).
        allow_file_reads: Advertise ``readTextFile`` capability. Only
            effective when a workspace section is provided.
        allow_file_writes: Advertise ``writeTextFile`` capability. Only
            effective when a workspace section is provided.
        allow_terminal: Advertise terminal capability. When True, must
            implement ``create_terminal``.
        mcp_servers: Additional MCP servers to register. The WINK MCP server
            is always injected; user-provided servers must not shadow the
            WINK tool namespace.
        reuse_session: If True, attempt to load a previous OpenCode session
            ID from session state. Only reuses if cwd and workspace fingerprint
            match the stored values.
    """

    opencode_bin: str = "opencode"
    opencode_args: tuple[str, ...] = ("acp",)
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    suppress_stderr: bool = True
    startup_timeout_s: float = 10.0
    permission_mode: PermissionMode = "auto"
    allow_file_reads: bool = False
    allow_file_writes: bool = False
    allow_terminal: bool = False
    mcp_servers: tuple[McpServerConfig, ...] = ()
    reuse_session: bool = False


@FrozenDataclass()
class OpenCodeACPAdapterConfig:
    """Adapter-level configuration for OpenCode ACP.

    Attributes:
        mode_id: ACP ``session/set_mode`` value. Best-effort; OpenCode may
            not implement this method. Ignored if None.
        model_id: ACP ``session/set_model`` value. Best-effort; OpenCode may
            not implement this method. Ignored if None.
        quiet_period_ms: Milliseconds to wait after prompt for trailing
            updates to drain. The timer resets on each update; total wait
            is capped by the deadline.
        emit_thought_chunks: If True, include thought chunks in the returned
            text response.
    """

    mode_id: str | None = None
    model_id: str | None = None
    quiet_period_ms: int = 100
    emit_thought_chunks: bool = False
