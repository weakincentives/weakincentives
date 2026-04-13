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

"""Re-exports for the Codex App Server client.

The bidirectional JSON-RPC client has been extracted to the generic
``weakincentives.adapters.jsonrpc.client`` module.  This module
re-exports ``JsonRpcClient`` as ``CodexAppServerClient`` and
``JsonRpcClientError`` as ``CodexClientError`` for semantic clarity
within the Codex adapter package.
"""

from __future__ import annotations

from ..jsonrpc.client import (
    _SENTINEL,  # pyright: ignore[reportPrivateUsage]
    _WS_CONNECT_MAX_RETRIES,  # pyright: ignore[reportPrivateUsage]
    _WS_CONNECT_RETRY_DELAY,  # pyright: ignore[reportPrivateUsage]
    _WS_MANAGED_PORT_RETRIES,  # pyright: ignore[reportPrivateUsage]
    JsonRpcClient as CodexAppServerClient,
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
