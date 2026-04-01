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

"""TypedDict definitions for the Codex App Server JSON-RPC protocol.

These replace raw ``dict[str, Any]`` with narrow, documented shapes for
messages flowing between the client and the Codex app-server subprocess.
"""

from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------------
# JSON-RPC transport layer
# ---------------------------------------------------------------------------


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
# Codex items (item/completed notification payloads)
# ---------------------------------------------------------------------------


class CodexItem(TypedDict, total=False):
    """Payload of an ``item/completed`` notification.

    The ``type`` field discriminates the item kind.
    """

    type: str
    id: str
    status: str
    text: str
    # commandExecution
    command: str
    cwd: str
    aggregatedOutput: str
    # fileChange
    file: str
    # mcpToolCall
    tool: str
    server: str
    result: MCPResult
    # webSearch
    query: str


class MCPContentEntry(TypedDict, total=False):
    """Single content block inside an MCP tool result."""

    type: str
    text: str


class MCPResult(TypedDict, total=False):
    """Result dict attached to ``mcpToolCall`` items."""

    content: list[MCPContentEntry]
    isError: bool


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------


class TurnInfo(TypedDict, total=False):
    """Turn metadata from ``turn/completed`` notification."""

    status: str
    codexErrorInfo: str | dict[str, str]
    additionalDetails: str


class TurnStartTurn(TypedDict):
    """Turn metadata from ``turn/start`` response (``id`` always present)."""

    id: int


class TurnStartResult(TypedDict):
    """Response body of ``turn/start``."""

    turn: TurnStartTurn


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TokenUsageLast(TypedDict, total=False):
    """Last-turn token counters from ``thread/tokenUsage/updated``."""

    inputTokens: int
    outputTokens: int
    cachedInputTokens: int


class TokenUsagePayload(TypedDict, total=False):
    """Payload of ``thread/tokenUsage/updated`` notification."""

    tokenUsage: TokenUsageInfo


class TokenUsageInfo(TypedDict, total=False):
    """Inner token-usage wrapper."""

    last: TokenUsageLast


# ---------------------------------------------------------------------------
# Tool calls (server → client requests)
# ---------------------------------------------------------------------------


class ToolCallParams(TypedDict, total=False):
    """Parameters of an ``item/tool/call`` server request."""

    tool: str
    arguments: str | dict[str, object]


class ToolCallResponse(TypedDict):
    """Response to an ``item/tool/call`` server request."""

    success: bool
    contentItems: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Notification params (keyed by method)
# ---------------------------------------------------------------------------


class ItemCompletedParams(TypedDict, total=False):
    """Parameters of ``item/completed`` notification."""

    item: CodexItem


class TurnCompletedParams(TypedDict, total=False):
    """Parameters of ``turn/completed`` notification."""

    turn: TurnInfo


class DeltaParams(TypedDict):
    """Parameters of ``item/agentMessage/delta`` notification."""

    delta: str


# ---------------------------------------------------------------------------
# Thread creation
# ---------------------------------------------------------------------------


class ThreadStartResult(TypedDict):
    """Response body of ``thread/start``."""

    thread: ThreadInfo


class ThreadInfo(TypedDict):
    """Thread metadata from ``thread/start`` response."""

    id: str
