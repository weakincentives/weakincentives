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

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from ...prompt.tool import ToolResult
from ...runtime.events import ToolInvoked
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ..core import PromptEvaluationPhase

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
    item: dict[str, Any],
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
    params: dict[str, Any]
    rendered_output: str

    if item_type == "commandExecution":
        name = "codex:command"
        params = {"command": item.get("command", ""), "cwd": item.get("cwd", "")}
        rendered_output = item.get("aggregatedOutput", "")[:1000]
    elif item_type == "fileChange":
        name = "codex:file_change"
        params = {"file": item.get("file", "")}
        rendered_output = str(item.get("status", ""))
    elif item_type == "mcpToolCall":
        tool_name = item.get("tool", "unknown")
        name = f"codex:mcp:{tool_name}"
        params = {"server": item.get("server", ""), "tool": item.get("tool", "")}
        rendered_output = _extract_mcp_output(item)
    elif item_type == "webSearch":
        name = "codex:web_search"
        params = {"query": item.get("query", "")}
        rendered_output = ""
    else:
        name = f"codex:{item_type}"
        params = {}
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
        params=params,
        result=result,
        session_id=session_id,
        created_at=datetime.now(UTC),
        usage=None,
        rendered_output=rendered_output,
        call_id=item.get("id"),
        run_context=run_context,
    )
    _ = session.dispatcher.dispatch(event)


def _extract_mcp_output(item: dict[str, Any]) -> str:
    """Extract text output from an mcpToolCall item."""
    content: Any = item.get("result", {})
    if isinstance(content, dict):
        content_dict = cast(dict[str, Any], content)
        content_list: list[Any] = content_dict.get("content", [])
        text_parts: list[str] = []
        for c in content_list:
            entry = cast(dict[str, Any], c) if isinstance(c, dict) else None
            if entry is not None and entry.get("type") == "text":
                text_parts.append(str(entry.get("text", "")))
        return "\n".join(text_parts)[:1000]
    return str(content)[:1000]


def extract_token_usage(params: dict[str, Any]) -> TokenUsage | None:
    """Extract TokenUsage from a thread/tokenUsage/updated notification."""
    token_usage: dict[str, Any] = params.get("tokenUsage", {})
    last_raw: Any = token_usage.get("last")
    if not isinstance(last_raw, dict):
        return None

    last = cast(dict[str, Any], last_raw)
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
    error_info: str | dict[str, Any] | None,
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
