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

"""Codex App Server JSON-RPC protocol primitives.

Provides the Codex-specific helpers that plug into
:class:`weakincentives.adapters.jsonrpc.JsonRpcAdapter` via
``_initialize_session``, ``_start_turn``, ``_notification_handlers``,
and ``_server_request_handlers``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, cast

from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from .._shared._bridge import BridgedTool
from ..jsonrpc import JsonRpcClient
from ._events import dispatch_item_tool_invoked
from ._guardrails import append_feedback
from ._transcript import CodexTranscriptBridge
from ._types import (
    CodexItem,
    ToolCallResponse,
    TurnInfo,
    TurnStartResult,
)
from .config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    CodexAuthMode,
)

if TYPE_CHECKING:
    from ...deadlines import Deadline
    from ...prompt.protocols import PromptProtocol
    from ..jsonrpc import ServerRequestContext


async def authenticate(
    client: JsonRpcClient,
    auth_mode: CodexAuthMode | None,
    *,
    timeout: float | None = None,
) -> None:
    """Perform authentication if auth_mode is configured."""
    if auth_mode is None:
        return

    if isinstance(auth_mode, ApiKeyAuth):
        _ = await client.send_request(
            "account/login/start",
            {"type": "apiKey", "apiKey": auth_mode.api_key},
            timeout=timeout,
        )
    else:
        # ExternalTokenAuth
        _ = await client.send_request(
            "account/login/start",
            {
                "type": "chatgptAuthTokens",
                "idToken": auth_mode.id_token,
                "accessToken": auth_mode.access_token,
            },
            timeout=timeout,
        )


async def create_thread(  # noqa: PLR0913
    client: JsonRpcClient,
    effective_cwd: str,
    dynamic_tool_specs: list[dict[str, object]],
    *,
    client_config: CodexAppServerClientConfig,
    model_config: CodexAppServerModelConfig,
    timeout: float | None = None,
) -> str:
    """Create a new Codex thread. Returns the thread ID."""
    thread_params: dict[str, object] = {
        "model": model_config.model,
        "cwd": effective_cwd,
        "approvalPolicy": client_config.approval_policy,
        "ephemeral": client_config.ephemeral,
    }
    if client_config.sandbox_mode is not None:
        thread_params["sandbox"] = client_config.sandbox_mode
    if dynamic_tool_specs:
        thread_params["dynamicTools"] = dynamic_tool_specs
    if client_config.mcp_servers:
        thread_params["config"] = {"mcp_servers": client_config.mcp_servers}

    result = await client.send_request("thread/start", thread_params, timeout=timeout)
    return result["thread"]["id"]


async def start_turn(  # noqa: PLR0913
    client: JsonRpcClient,
    thread_id: str,
    prompt_text: str,
    output_schema: dict[str, object] | None,
    *,
    model_config: CodexAppServerModelConfig,
    timeout: float | None = None,
) -> TurnStartResult:
    """Start a turn and return the response."""
    turn_params: dict[str, object] = {
        "threadId": thread_id,
        "input": [{"type": "text", "text": prompt_text}],
    }
    if model_config.effort is not None:
        turn_params["effort"] = model_config.effort
    if model_config.summary is not None:
        turn_params["summary"] = model_config.summary
    if model_config.personality is not None:
        turn_params["personality"] = model_config.personality
    if output_schema is not None:
        turn_params["outputSchema"] = output_schema

    result = await client.send_request("turn/start", turn_params, timeout=timeout)
    return cast(TurnStartResult, result)


async def handle_tool_call(  # noqa: PLR0913
    client: JsonRpcClient,
    request_id: int,
    params: dict[str, object],
    tool_lookup: dict[str, BridgedTool],
    *,
    bridge: CodexTranscriptBridge | None = None,
    prompt: PromptProtocol[Any] | None = None,
    session: SessionProtocol | None = None,
    deadline: Deadline | None = None,
) -> None:
    """Handle an item/tool/call server request."""
    tool_name = str(params.get("tool", ""))
    arguments_raw = params.get("arguments", {})
    arguments: dict[str, object]
    if isinstance(arguments_raw, str):
        try:
            parsed = json.loads(arguments_raw)
            arguments = (
                cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
            )
        except json.JSONDecodeError:
            arguments = {}
    elif isinstance(arguments_raw, dict):
        arguments = cast("dict[str, object]", arguments_raw)
    else:  # pragma: no cover - default is {} (dict)
        arguments = {}

    # Emit tool_use transcript entry before execution.
    if bridge is not None:
        bridge.on_tool_call(params)

    bridged_tool = tool_lookup.get(tool_name)
    if bridged_tool is None:
        error_response: ToolCallResponse = {
            "success": False,
            "contentItems": [
                {"type": "inputText", "text": f"Unknown tool: {tool_name}"}
            ],
        }
        if bridge is not None:
            bridge.on_tool_result(params, error_response)
        await client.send_response(request_id, error_response)
        return

    mcp_result: dict[str, Any] = await asyncio.to_thread(bridged_tool, arguments)
    is_error: bool = mcp_result.get("isError", False)
    mcp_content: list[dict[str, str]] = mcp_result.get("content", [])
    content_items: list[dict[str, str]] = [
        {"type": "inputText", "text": str(c.get("text", ""))}
        for c in mcp_content
        if c.get("type") == "text"
    ]

    append_feedback(
        content_items,
        is_error=is_error,
        prompt=prompt,
        session=session,
        deadline=deadline,
    )

    response: ToolCallResponse = {
        "success": not is_error,
        "contentItems": content_items,
    }

    # Emit tool_result transcript entry after execution.
    if bridge is not None:
        bridge.on_tool_result(params, response)

    await client.send_response(request_id, response)


# ---------------------------------------------------------------------------
# Registry-compatible notification handlers
# ---------------------------------------------------------------------------

# These match the ``NotificationHandler`` signature:
#   (params, session, adapter_name, prompt_name, run_context) -> (kind, value) | None


def handle_delta_notification(
    params: dict[str, object],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str]:
    """Handle ``item/agentMessage/delta``."""
    return ("delta", str(params.get("delta", "")))


def handle_item_completed_notification(
    params: dict[str, object],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str] | None:
    """Handle ``item/completed``."""
    return _handle_item_completed(
        params, session, adapter_name, prompt_name, run_context
    )


def handle_token_usage_notification(
    params: dict[str, object],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str]:
    """Handle ``thread/tokenUsage/updated``."""
    return ("usage", "")


def handle_turn_completed_notification(
    params: dict[str, object],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str]:
    """Handle ``turn/completed``."""
    return _handle_turn_completed(params)


def _handle_item_completed(
    params: dict[str, object],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str] | None:
    """Handle item/completed notification."""
    item_raw = params.get("item", {})
    item = cast(CodexItem, item_raw if isinstance(item_raw, dict) else {})
    item_type = item.get("type", "")

    if item_type == "agentMessage":
        final_text = item.get("text")
        if final_text is not None:
            return ("text", final_text)
        return None

    if item_type in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "webSearch",
    }:
        dispatch_item_tool_invoked(
            item=item,
            session=session,
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            run_context=run_context,
        )
    return None


def _handle_turn_completed(params: dict[str, object]) -> tuple[str, str]:
    """Handle turn/completed notification."""
    turn_raw = params.get("turn", {})
    turn = cast(TurnInfo, turn_raw if isinstance(turn_raw, dict) else {})
    status = turn.get("status", "")

    if status == "failed":
        error_info = turn.get("codexErrorInfo")
        additional = turn.get("additionalDetails", "")
        return ("error", f"Turn failed: {error_info or 'unknown'} — {additional}")

    if status == "interrupted":
        return ("interrupted", "")

    return ("done", "")


# ---------------------------------------------------------------------------
# Registry-compatible server-request handlers
# ---------------------------------------------------------------------------

# These match the ``ServerRequestHandler`` signature:
#   async (ctx: ServerRequestContext) -> None


async def handle_tool_call_request(ctx: ServerRequestContext) -> None:
    """Handle ``item/tool/call`` server request."""
    await handle_tool_call(
        ctx.client,
        ctx.request_id,
        ctx.params,
        ctx.tool_lookup,
        bridge=ctx.bridge,  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        prompt=ctx.prompt,  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        session=ctx.session,  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        deadline=ctx.deadline,  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    )


def make_approval_handler(
    approval_policy: str,
) -> Callable[[ServerRequestContext], Coroutine[Any, Any, None]]:
    """Create an approval handler bound to a specific policy.

    Returns an async callable matching ``ServerRequestHandler``::

        handlers = {
            "item/commandExecution/requestApproval": make_approval_handler("never"),
            "item/fileChange/requestApproval":       make_approval_handler("never"),
        }
    """

    async def _handle(ctx: ServerRequestContext) -> None:
        decision = "accept" if approval_policy in {"never", "on-failure"} else "decline"
        await ctx.client.send_response(ctx.request_id, {"decision": decision})

    return _handle


__all__ = [
    "authenticate",
    "create_thread",
    "handle_delta_notification",
    "handle_item_completed_notification",
    "handle_token_usage_notification",
    "handle_tool_call",
    "handle_tool_call_request",
    "handle_turn_completed_notification",
    "make_approval_handler",
    "start_turn",
]
