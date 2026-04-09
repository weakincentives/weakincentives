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

"""Generic TypedDict definitions for JSON-RPC messages.

These provide typed shapes for the wire-level JSON-RPC messages shared
across all JSON-RPC provider adapters.  Provider-specific payload types
(e.g. Codex items, turn info) remain in each adapter's own ``_types``
module.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ...runtime.run_context import RunContext
    from ...runtime.session.protocols import SessionProtocol


class JsonRpcRequest(TypedDict, total=False):
    """Outbound JSON-RPC request."""

    id: int
    method: str
    params: dict[str, object]


class JsonRpcNotification(TypedDict, total=False):
    """Outbound JSON-RPC notification (no ``id``)."""

    method: str
    params: dict[str, object]


class JsonRpcResponse(TypedDict, total=False):
    """JSON-RPC response from the server."""

    id: int
    result: dict[str, object]
    error: str


class JsonRpcMessage(TypedDict, total=False):
    """Any message on the wire (union of request, notification, response).

    Discriminated at runtime by presence of ``id`` and ``method`` keys.
    """

    id: int
    method: str
    params: dict[str, object]
    result: dict[str, object]
    error: str


# ---------------------------------------------------------------------------
# Notification result protocol
# ---------------------------------------------------------------------------

# Notification processing returns (kind, value) tuples.
# The kind determines how the value is applied.
NOTIFICATION_KIND_DELTA = "delta"
"""Append ``value`` to accumulated text."""

NOTIFICATION_KIND_TEXT = "text"
"""Replace accumulated text with ``value``."""

NOTIFICATION_KIND_USAGE = "usage"
"""Token usage update (value is ignored; extract from params)."""

NOTIFICATION_KIND_DONE = "done"
"""Turn completed successfully."""

NOTIFICATION_KIND_ERROR = "error"
"""Turn failed (value contains error message)."""

NOTIFICATION_KIND_INTERRUPTED = "interrupted"
"""Turn was interrupted (deadline or user)."""


# ---------------------------------------------------------------------------
# Handler callable protocols
# ---------------------------------------------------------------------------

# Notification handlers receive the method's params and contextual info.
# Return (kind, value) to drive the turn accumulator, or None to skip.
#
# Example (Codex delta handler):
#     def _handle_delta(params, session, adapter_name, prompt_name, run_context):
#         return ("delta", str(params.get("delta", "")))
type NotificationHandler = Callable[
    [
        dict[str, object],  # params
        SessionProtocol,  # session
        str,  # adapter_name
        str,  # prompt_name
        RunContext | None,  # run_context
    ],
    tuple[str, str] | None,
]
"""Callable that processes a single notification method.

Receives the notification ``params`` and contextual info.
Returns ``(kind, value)`` to drive the turn accumulator, or ``None``
to indicate the notification was handled as a side-effect (e.g.
dispatching a ``ToolInvoked`` event).
"""

# Server-request handlers receive the full request context.
# They MUST send a response via ``client.send_response(request_id, …)``.
type ServerRequestHandler = Callable[
    [ServerRequestContext],
    Coroutine[Any, Any, None],
]
"""Async callable that handles a server-initiated request.

The handler MUST respond via ``ctx.client.send_response(ctx.request_id, …)``.
"""


@dataclass(slots=True, frozen=True)
class ServerRequestContext:
    """Context bundle passed to server-request handlers.

    Provides everything a handler needs to process a server-initiated
    request and send back a response.
    """

    client: Any  # JsonRpcClient (Any to avoid circular import)
    request_id: int
    method: str
    params: dict[str, object]
    tool_lookup: dict[str, Any]  # dict[str, BridgedTool]
    bridge: object | None
    prompt: object | None  # PromptProtocol | None
    session: object | None  # SessionProtocol | None
    deadline: object | None  # Deadline | None
