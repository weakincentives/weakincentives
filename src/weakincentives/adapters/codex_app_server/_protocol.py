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

"""Codex App Server JSON-RPC protocol orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any, cast

from ...budget import BudgetTracker
from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...deadlines import Deadline
from ...runtime.events.types import TokenUsage
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.transcript import TranscriptEmitter
from .._shared._bridge import BridgedTool
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError
from ._events import (
    dispatch_item_tool_invoked,
    extract_token_usage,
    map_codex_error_phase,
)
from ._guardrails import accumulate_usage, append_feedback, check_task_completion
from ._transcript import CodexTranscriptBridge
from .client import CodexAppServerClient, CodexClientError
from .config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    CodexAuthMode,
)

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol


def deadline_remaining_s(deadline: Deadline | None, prompt_name: str) -> float | None:
    """Return remaining seconds from deadline, or None if no deadline.

    Raises PromptEvaluationError if the deadline has already expired.
    """
    if deadline is None:
        return None
    remaining = deadline.remaining().total_seconds()
    if remaining <= 0:
        raise PromptEvaluationError(
            message="Deadline expired during setup",
            prompt_name=prompt_name,
            phase="request",
        )
    return remaining


def _create_bridge(
    client_config: CodexAppServerClientConfig,
    session: SessionProtocol,
    prompt_name: str,
) -> CodexTranscriptBridge | None:
    """Create and start a transcript bridge if transcription is enabled."""
    if not client_config.transcript:
        return None
    session_id = getattr(session, "session_id", None)
    emitter = TranscriptEmitter(
        prompt_name=prompt_name,
        adapter="codex_app_server",
        session_id=str(session_id) if session_id else None,
        emit_raw=client_config.transcript_emit_raw,
    )
    bridge = CodexTranscriptBridge(emitter)
    emitter.start()
    return bridge


async def _initialize_session(  # noqa: PLR0913
    *,
    client: CodexAppServerClient,
    client_config: CodexAppServerClientConfig,
    model_config: CodexAppServerModelConfig,
    effective_cwd: str,
    dynamic_tool_specs: list[dict[str, Any]],
    deadline: Deadline | None,
    prompt_name: str,
) -> str:
    """Run initialize → authenticate → create_thread. Returns thread_id."""
    _ = await client.send_request(
        "initialize",
        {
            "clientInfo": {
                "name": client_config.client_name,
                "title": "WINK Agent",
                "version": client_config.client_version,
            },
            "capabilities": {"experimentalApi": True},
        },
        timeout=client_config.startup_timeout_s,
    )
    await client.send_notification("initialized")

    try:
        timeout = deadline_remaining_s(deadline, prompt_name)
        await authenticate(client, client_config.auth_mode, timeout=timeout)

        timeout = deadline_remaining_s(deadline, prompt_name)
        return await create_thread(
            client,
            effective_cwd,
            dynamic_tool_specs,
            client_config=client_config,
            model_config=model_config,
            timeout=timeout,
        )
    except CodexClientError as error:
        raise PromptEvaluationError(
            message=str(error),
            prompt_name=prompt_name,
            phase="request",
        ) from error


async def execute_protocol(  # noqa: C901, PLR0913, PLR0914
    *,
    client_config: CodexAppServerClientConfig,
    model_config: CodexAppServerModelConfig,
    client: CodexAppServerClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    prompt_text: str,
    effective_cwd: str,
    dynamic_tool_specs: list[dict[str, Any]],
    tool_lookup: dict[str, BridgedTool],
    output_schema: dict[str, Any] | None,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    run_context: RunContext | None,
    visibility_signal: VisibilityExpansionSignal,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    prompt: PromptProtocol[Any] | None = None,
) -> tuple[str | None, TokenUsage | None]:
    """Execute the Codex protocol (init -> thread -> turn -> stream).

    Returns (accumulated_text, usage).
    """
    await client.start()
    bridge = _create_bridge(client_config, session, prompt_name)

    try:
        thread_id = await _initialize_session(
            client=client,
            client_config=client_config,
            model_config=model_config,
            effective_cwd=effective_cwd,
            dynamic_tool_specs=dynamic_tool_specs,
            deadline=deadline,
            prompt_name=prompt_name,
        )

        # Turn + Stream with task completion continuation loop
        max_continuation_rounds = 10
        continuation_round = 0
        current_prompt_text: str = prompt_text
        accumulated_text: str | None = None
        usage: TokenUsage | None = None

        while True:
            try:
                if bridge is not None:
                    bridge.on_user_message(current_prompt_text)
                current_schema = output_schema
                timeout = deadline_remaining_s(deadline, prompt_name)
                turn_result = await start_turn(
                    client,
                    thread_id,
                    current_prompt_text,
                    current_schema,
                    model_config=model_config,
                    timeout=timeout,
                )
            except CodexClientError as error:
                raise PromptEvaluationError(
                    message=str(error),
                    prompt_name=prompt_name,
                    phase="request",
                ) from error
            turn_id: int = turn_result["turn"]["id"]

            turn_text, turn_usage = await stream_turn(
                client=client,
                session=session,
                adapter_name=adapter_name,
                prompt_name=prompt_name,
                thread_id=thread_id,
                turn_id=turn_id,
                tool_lookup=tool_lookup,
                approval_policy=client_config.approval_policy,
                deadline=deadline,
                run_context=run_context,
                bridge=bridge,
                visibility_signal=visibility_signal,
                async_sleeper=async_sleeper,
                prompt=prompt,
            )
            accumulated_text = turn_text
            if turn_usage is not None:
                usage = accumulate_usage(usage, turn_usage)

            # Skip continuation when visibility expansion is pending.
            if visibility_signal.is_set():
                break

            should_continue, feedback = check_task_completion(
                prompt=prompt,
                session=session,
                accumulated_text=accumulated_text,
                deadline=deadline,
                budget_tracker=budget_tracker,
            )
            if should_continue and continuation_round < max_continuation_rounds:
                current_prompt_text = feedback or current_prompt_text
                continuation_round += 1
                continue
            break
    finally:
        # Stop transcript emitter — must run even on exception.
        if bridge is not None:
            bridge.emitter.stop()

    # Visibility signal
    stored_exc = visibility_signal.get_and_clear()
    if stored_exc is not None:
        raise stored_exc

    return accumulated_text, usage


async def authenticate(
    client: CodexAppServerClient,
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
    client: CodexAppServerClient,
    effective_cwd: str,
    dynamic_tool_specs: list[dict[str, Any]],
    *,
    client_config: CodexAppServerClientConfig,
    model_config: CodexAppServerModelConfig,
    timeout: float | None = None,
) -> str:
    """Create a new Codex thread. Returns the thread ID."""
    thread_params: dict[str, Any] = {
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
    client: CodexAppServerClient,
    thread_id: str,
    prompt_text: str,
    output_schema: dict[str, Any] | None,
    *,
    model_config: CodexAppServerModelConfig,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Start a turn and return the response."""
    turn_params: dict[str, Any] = {
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

    return await client.send_request("turn/start", turn_params, timeout=timeout)


async def stream_turn(  # noqa: PLR0913
    *,
    client: CodexAppServerClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    thread_id: str,
    turn_id: int,
    tool_lookup: dict[str, BridgedTool],
    approval_policy: str,
    deadline: Deadline | None,
    run_context: RunContext | None,
    bridge: CodexTranscriptBridge | None = None,
    visibility_signal: VisibilityExpansionSignal | None = None,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    prompt: PromptProtocol[Any] | None = None,
) -> tuple[str | None, TokenUsage | None]:
    """Stream turn notifications until turn/completed.

    Returns (accumulated_text, token_usage).
    """
    accumulated_text = ""
    usage: TokenUsage | None = None

    watchdog_task = create_deadline_watchdog(
        client, thread_id, turn_id, deadline, async_sleeper
    )

    try:
        accumulated_text, usage = await consume_messages(
            client=client,
            session=session,
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            tool_lookup=tool_lookup,
            approval_policy=approval_policy,
            run_context=run_context,
            accumulated_text=accumulated_text,
            usage=usage,
            bridge=bridge,
            visibility_signal=visibility_signal,
            prompt=prompt,
            deadline=deadline,
        )
    finally:
        if watchdog_task is not None:
            _ = watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task

    return accumulated_text or None, usage


def raise_for_terminal_notification(
    kind: str,
    value: str,
    prompt_name: str,
    message: dict[str, Any],
) -> None:
    """Raise PromptEvaluationError for terminal notification kinds."""
    if kind == "error":
        params: dict[str, Any] = message.get("params", {})
        turn: dict[str, Any] = params.get("turn", {})
        phase = map_codex_error_phase(turn.get("codexErrorInfo"))
        raise PromptEvaluationError(
            message=value,
            prompt_name=prompt_name,
            phase=phase,
        )
    if kind == "interrupted":
        raise PromptEvaluationError(
            message="Turn was interrupted (deadline or user)",
            prompt_name=prompt_name,
            phase="request",
        )


async def consume_messages(  # noqa: PLR0913
    *,
    client: CodexAppServerClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    tool_lookup: dict[str, BridgedTool],
    approval_policy: str,
    run_context: RunContext | None,
    accumulated_text: str,
    usage: TokenUsage | None,
    bridge: CodexTranscriptBridge | None = None,
    visibility_signal: VisibilityExpansionSignal | None = None,
    prompt: PromptProtocol[Any] | None = None,
    deadline: Deadline | None = None,
) -> tuple[str, TokenUsage | None]:
    """Consume messages from the client until turn/completed."""
    turn_completed = False
    async for message in client.read_messages():
        if "id" in message and "method" in message:
            await handle_server_request(
                client,
                message,
                tool_lookup,
                approval_policy=approval_policy,
                bridge=bridge,
                prompt=prompt,
                session=session,
                deadline=deadline,
            )
            # Break early if a tool call triggered visibility expansion.
            # The caller will re-raise the stored exception after cleanup.
            if visibility_signal is not None and visibility_signal.is_set():
                turn_completed = True
                break
            continue

        # Emit transcript entry for this notification.
        if bridge is not None:
            method = message.get("method", "")
            params: dict[str, Any] = message.get("params", {})
            bridge.on_notification(method, params)

        result = process_notification(
            message, session, adapter_name, prompt_name, run_context
        )
        if result is None:
            continue

        accumulated_text, usage, done = apply_notification(
            result, message, accumulated_text, usage, prompt_name
        )
        if done:
            turn_completed = True
            break
    if not turn_completed:
        raise PromptEvaluationError(
            message="Codex stream ended before turn completion",
            prompt_name=prompt_name,
            phase="response",
        )
    return accumulated_text, usage


def apply_notification(
    result: tuple[str, str],
    message: dict[str, Any],
    accumulated_text: str,
    usage: TokenUsage | None,
    prompt_name: str,
) -> tuple[str, TokenUsage | None, bool]:
    """Apply a single notification result. Returns (text, usage, done)."""
    kind, value = result
    if kind == "text":
        return value, usage, False
    if kind == "delta":
        return accumulated_text + value, usage, False
    if kind == "usage":
        return (
            accumulated_text,
            extract_token_usage(message.get("params", {})),
            False,
        )
    if kind == "done":
        return accumulated_text, usage, True
    raise_for_terminal_notification(kind, value, prompt_name, message)
    return accumulated_text, usage, False  # pragma: no cover


def create_deadline_watchdog(
    client: CodexAppServerClient,
    thread_id: str,
    turn_id: int,
    deadline: Deadline | None,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
) -> asyncio.Task[None] | None:
    """Create a watchdog task if a deadline is set."""
    if deadline is None:
        return None
    remaining = deadline.remaining().total_seconds()
    if remaining <= 0:
        return None
    return asyncio.create_task(
        deadline_watchdog(client, thread_id, turn_id, remaining, async_sleeper)
    )


def process_notification(
    message: dict[str, Any],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str] | None:
    """Process a notification message.

    Returns (kind, value) where kind is one of:
    "text", "delta", "usage", "error", "interrupted", "done".
    Returns None for unhandled notification types.
    """
    method = message.get("method", "")
    params: dict[str, Any] = message.get("params", {})

    if method == "item/agentMessage/delta":
        return ("delta", params.get("delta", ""))

    if method == "item/completed":
        return _handle_item_completed(
            params, session, adapter_name, prompt_name, run_context
        )

    if method == "thread/tokenUsage/updated":
        return ("usage", "")

    if method == "turn/completed":
        return _handle_turn_completed(params)

    return None


def _handle_item_completed(
    params: dict[str, Any],
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> tuple[str, str] | None:
    """Handle item/completed notification."""
    item: dict[str, Any] = params.get("item", {})
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


def _handle_turn_completed(params: dict[str, Any]) -> tuple[str, str]:
    """Handle turn/completed notification."""
    turn: dict[str, Any] = params.get("turn", {})
    status = turn.get("status", "")

    if status == "failed":
        error_info = turn.get("codexErrorInfo")
        additional = turn.get("additionalDetails", "")
        return ("error", f"Turn failed: {error_info or 'unknown'} — {additional}")

    if status == "interrupted":
        return ("interrupted", "")

    return ("done", "")


async def handle_server_request(  # noqa: PLR0913
    client: CodexAppServerClient,
    message: dict[str, Any],
    tool_lookup: dict[str, BridgedTool],
    *,
    approval_policy: str,
    bridge: CodexTranscriptBridge | None = None,
    prompt: PromptProtocol[Any] | None = None,
    session: SessionProtocol | None = None,
    deadline: Deadline | None = None,
) -> None:
    """Handle a server-initiated request (tool call or approval)."""
    method: str = message["method"]
    params: dict[str, Any] = message.get("params", {})
    request_id: int = message["id"]

    if method == "item/tool/call":
        await handle_tool_call(
            client,
            request_id,
            params,
            tool_lookup,
            bridge=bridge,
            prompt=prompt,
            session=session,
            deadline=deadline,
        )
    elif method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        # In non-interactive execution, we deterministically accept only
        # policies intended to proceed without human gating when requested.
        decision = "accept" if approval_policy in {"never", "on-failure"} else "decline"
        await client.send_response(request_id, {"decision": decision})
    else:
        await client.send_response(request_id, {})


async def handle_tool_call(  # noqa: PLR0913
    client: CodexAppServerClient,
    request_id: int,
    params: dict[str, Any],
    tool_lookup: dict[str, BridgedTool],
    *,
    bridge: CodexTranscriptBridge | None = None,
    prompt: PromptProtocol[Any] | None = None,
    session: SessionProtocol | None = None,
    deadline: Deadline | None = None,
) -> None:
    """Handle an item/tool/call server request."""
    tool_name: str = params.get("tool", "")
    arguments = params.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    # Emit tool_use transcript entry before execution.
    if bridge is not None:
        bridge.on_tool_call(params)

    bridged_tool = tool_lookup.get(tool_name)
    if bridged_tool is None:
        error_response: dict[str, Any] = {
            "success": False,
            "contentItems": [
                {"type": "inputText", "text": f"Unknown tool: {tool_name}"}
            ],
        }
        if bridge is not None:
            bridge.on_tool_result(params, error_response)
        await client.send_response(request_id, error_response)
        return

    mcp_result: dict[str, Any] = await asyncio.to_thread(
        bridged_tool, cast(dict[str, Any], arguments)
    )
    is_error: bool = mcp_result.get("isError", False)
    mcp_content: list[dict[str, Any]] = mcp_result.get("content", [])
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

    response: dict[str, Any] = {
        "success": not is_error,
        "contentItems": content_items,
    }

    # Emit tool_result transcript entry after execution.
    if bridge is not None:
        bridge.on_tool_result(params, response)

    await client.send_response(request_id, response)


async def deadline_watchdog(
    client: CodexAppServerClient,
    thread_id: str,
    turn_id: int,
    remaining_seconds: float,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
) -> None:
    """Sleep until deadline, then interrupt the turn."""
    await async_sleeper.async_sleep(remaining_seconds)
    with contextlib.suppress(CodexClientError):
        _ = await client.send_request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
            timeout=5.0,
        )
