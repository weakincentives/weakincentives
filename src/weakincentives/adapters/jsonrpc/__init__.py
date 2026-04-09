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

"""Generic JSON-RPC provider adapter for weakincentives.

This package provides a reusable base for evaluating WINK prompts via any
agent that speaks a turn-based JSON-RPC protocol over stdio or WebSocket.
WINK tools are bridged as dynamic tools over the same transport channel.

The ``JsonRpcAdapter`` base class handles the common lifecycle (render,
bridge, protocol loop, response building) and delegates provider-specific
protocol details to subclass hooks.  ``JsonRpcClient`` provides the
transport-agnostic bidirectional JSON-RPC client.

Concrete subclasses:
- ``CodexAppServerAdapter`` (``weakincentives.adapters.codex_app_server``)
"""

from __future__ import annotations

from ._protocol import deadline_remaining_s, execute_protocol
from ._response import build_response
from ._types import (
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    NotificationHandler,
    ServerRequestContext,
    ServerRequestHandler,
)
from .adapter import JsonRpcAdapter
from .client import JsonRpcClient, JsonRpcClientError
from .config import JsonRpcClientConfig, Transport

__all__ = [
    "JsonRpcAdapter",
    "JsonRpcClient",
    "JsonRpcClientConfig",
    "JsonRpcClientError",
    "JsonRpcMessage",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "NotificationHandler",
    "ServerRequestContext",
    "ServerRequestHandler",
    "Transport",
    "build_response",
    "deadline_remaining_s",
    "execute_protocol",
]
