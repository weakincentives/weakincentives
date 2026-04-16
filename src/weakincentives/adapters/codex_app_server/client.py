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

"""Codex App Server client with Codex-specific defaults.

Extends :class:`~weakincentives.adapters.jsonrpc.client.JsonRpcClient`
with default ``bin_path``, ``bin_args``, and ``bin_ws_args`` for the
Codex ``app-server`` binary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ..jsonrpc.client import (
    _SENTINEL,  # pyright: ignore[reportPrivateUsage]
    _WS_CONNECT_MAX_RETRIES,  # pyright: ignore[reportPrivateUsage]
    _WS_CONNECT_RETRY_DELAY,  # pyright: ignore[reportPrivateUsage]
    _WS_MANAGED_PORT_RETRIES,  # pyright: ignore[reportPrivateUsage]
    JsonRpcClient,
    JsonRpcClientError as CodexClientError,
    _find_free_port,  # pyright: ignore[reportPrivateUsage]
    logger,
)

__all__ = [  # noqa: RUF022
    "CodexAppServerClient",
    "CodexClientError",
    "_SENTINEL",
    "_WS_CONNECT_MAX_RETRIES",
    "_WS_CONNECT_RETRY_DELAY",
    "_WS_MANAGED_PORT_RETRIES",
    "_find_free_port",
    "logger",
]


class CodexAppServerClient(JsonRpcClient):
    """JSON-RPC client pre-configured for the Codex ``app-server`` binary."""

    def __init__(
        self,
        bin_path: str = "codex",
        bin_args: tuple[str, ...] = ("app-server",),
        bin_ws_args: tuple[str, ...] = ("app-server", "--listen"),
        env: Mapping[str, str] | None = None,
        suppress_stderr: bool = True,
        *,
        transport: Literal["stdio", "websocket"] = "stdio",
        remote_url: str | None = None,
        ws_auth_token: str | None = None,
    ) -> None:
        super().__init__(
            bin_path=bin_path,
            bin_args=bin_args,
            bin_ws_args=bin_ws_args,
            env=env,
            suppress_stderr=suppress_stderr,
            transport=transport,
            remote_url=remote_url,
            ws_auth_token=ws_auth_token,
        )
