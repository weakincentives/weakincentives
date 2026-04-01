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

"""Map Codex item/turn notifications to WINK events."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import UUID

from ...clock import SYSTEM_CLOCK
from ...prompt.tool import ToolResult
from ...runtime.events import ToolInvoked
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ..core import PromptEvaluationPhase
from ._types import (
    CodexItem,
    MCPContentEntry,
    TokenUsageInfo,
    TokenUsageLast,
)

if TYPE_CHECKING:
    from ...runtime.run_context import RunContext
    from ...runtime.session.protocols import SessionProtocol

__all__ = [
    "dispatch_item_tool_invoked",
    "extract_token_usage",
    "map_codex_error_phase",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "codex_events"})


def dispatch_item_tool_invoked(
    *,
    item: CodexItem,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
) -> None:
    """Dispatch a ToolInvoked event for a completed Codex item.

    Handles commandExecution, fileChange, mcpToolCall, and webSearch items.
    """
    item_type = item.get("type", "unknown")
    status = item.get("status", "")
    success = status == "completed"

    # Build a name and params summary based on item type
    name: str
    event_params: dict[str, str]
    rendered_output: str

    if item_type == "commandExecution":
        name = "codex:command"
        event_params = {"command": item.get("command", ""), "cwd": item.get("cwd", "")}
        rendered_output = (item.get("aggregatedOutput") or "")[:1000]
    elif item_type == "fileChange":
        name = "codex:file_change"
        event_params = {"file": item.get("file", "")}
        rendered_output = str(item.get("status", ""))
    elif item_type == "mcpToolCall":
        tool_name = item.get("tool", "unknown")
        name = f"codex:mcp:{tool_name}"
        event_params = {"server": item.get("server", ""), "tool": item.get("tool", "")}
        rendered_output = _extract_mcp_output(item)
    elif item_type == "webSearch":
        name = "codex:web_search"
        event_params = {"query": item.get("query", "")}
        rendered_output = ""
    else:
        name = f"codex:{item_type}"
        event_params = {}
        rendered_output = ""

    result: ToolResult[None]
    if success:
        result = ToolResult[None].ok(None, message=rendered_output)
    else:
        result = ToolResult[None].error(rendered_output or f"Item {status}")

    session_id: UUID | None = getattr(session, "session_id", None)

    event = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=name,
        params=event_params,
        result=result,
        session_id=session_id,
        created_at=SYSTEM_CLOCK.utcnow(),
        usage=None,
        rendered_output=rendered_output,
        call_id=item.get("id"),
        run_context=run_context,
    )
    _ = session.dispatcher.dispatch(event)


def _extract_mcp_output(item: CodexItem) -> str:
    """Extract text output from an mcpToolCall item."""
    result_raw: object = item.get("result", {})
    if not isinstance(result_raw, dict):
        return str(result_raw)[:1000]
    mcp_result = cast("dict[str, object]", result_raw)
    content_list = cast("list[MCPContentEntry]", mcp_result.get("content", []))
    text_parts: list[str] = []
    for entry in content_list:
        if not isinstance(entry, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        if entry.get("type") == "text":
            text_parts.append(str(entry.get("text", "")))
    return "\n".join(text_parts)[:1000]


def extract_token_usage(params: dict[str, object]) -> TokenUsage | None:
    """Extract TokenUsage from a thread/tokenUsage/updated notification."""
    token_usage: TokenUsageInfo = params.get("tokenUsage", {})  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    last_raw = token_usage.get("last")
    if not isinstance(last_raw, dict):
        return None

    last: TokenUsageLast = last_raw
    return TokenUsage(
        input_tokens=last.get("inputTokens"),
        output_tokens=last.get("outputTokens"),
        cached_tokens=last.get("cachedInputTokens"),
    )


# Mapping from codexErrorInfo type to PromptEvaluationError phase.
_ERROR_PHASE_MAP: dict[str, PromptEvaluationPhase] = {
    "contextWindowExceeded": "response",
    "usageLimitExceeded": "budget",
    "httpConnectionFailed": "request",
    "unauthorized": "request",
    "badRequest": "request",
    "sandboxError": "tool",
    "responseTooManyFailedAttempts": "request",
    "responseStreamConnectionFailed": "request",
    "responseStreamDisconnected": "request",
    "threadRollbackFailed": "response",
    "internalServerError": "response",
    "modelCap": "budget",
}


def map_codex_error_phase(
    error_info: str | dict[str, str] | None,
) -> PromptEvaluationPhase:
    """Map a codexErrorInfo value to a PromptEvaluationError phase.

    Args:
        error_info: The codexErrorInfo from turn/completed params.

    Returns:
        One of "request", "response", "tool", "budget".
    """
    if isinstance(error_info, str):
        return _ERROR_PHASE_MAP.get(error_info, "response")
    if isinstance(error_info, dict):
        error_type: str = error_info.get("type", "")
        return _ERROR_PHASE_MAP.get(error_type, "response")
    return "response"
