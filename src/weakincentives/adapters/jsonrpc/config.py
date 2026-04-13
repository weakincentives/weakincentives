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

"""Base configuration for JSON-RPC provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import Literal

from ...dataclasses import FrozenDataclass

__all__ = [
    "JsonRpcClientConfig",
    "Transport",
]

Transport = Literal["stdio", "websocket"]
"""Wire protocol for communicating with the agent server."""


@FrozenDataclass()
class JsonRpcClientConfig:
    """Base client-level configuration for JSON-RPC adapters.

    Provider-specific adapters may extend this with additional fields
    or use it directly.

    **Transport selection:**

    - ``transport="stdio"`` (default): spawns the binary and communicates
      via NDJSON over stdin/stdout pipes.
    - ``transport="websocket"`` with ``remote_url=None``: spawns the binary
      with WS listen args and connects via WebSocket.
    - ``remote_url`` set: connects to an external server via WebSocket.
      No subprocess is spawned.
    """

    transport: Transport = "stdio"
    bin_path: str = "codex"
    """Binary to spawn (managed modes only)."""
    bin_args: tuple[str, ...] = ("app-server",)
    """Arguments for stdio mode."""
    bin_ws_args: tuple[str, ...] = ("app-server", "--listen")
    """Arguments prefix for managed WebSocket mode (URL appended)."""
    remote_url: str | None = None
    """WebSocket URL of an external server.  When set, no subprocess is
    spawned."""
    ws_auth_token: str | None = field(default=None, repr=False)
    """Bearer token for WebSocket auth."""
    cwd: str | None = None
    """Working directory (must be absolute; defaults to Path.cwd())."""
    env: Mapping[str, str] | None = None
    """Extra environment variables for the subprocess."""
    suppress_stderr: bool = True
    """Capture stderr quietly (stdio mode only)."""
    startup_timeout_s: float = 10.0
    """Max time for the initialize handshake."""
    transcript: bool = True
    """Emit transcript entries during evaluation."""
    transcript_emit_raw: bool = True
    """Include raw notification JSON in ``raw`` field."""
