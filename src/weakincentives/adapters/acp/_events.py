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

"""Map ACP session updates to WINK events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ...prompt.tool import ToolResult
from ...runtime.events import ToolInvoked
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from ...runtime.run_context import RunContext
    from ...runtime.session.protocols import SessionProtocol

__all__ = ["dispatch_tool_invoked", "extract_token_usage"]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_events"})

# MCP server name used for WINK bridged tools
WINK_MCP_SERVER_PREFIX = "wink-tools_"


def dispatch_tool_invoked(  # noqa: PLR0913
    *,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    run_context: RunContext | None,
    tool_call_id: str | None,
    title: str,
    status: str,
    rendered_output: str = "",
    mcp_server_prefix: str = WINK_MCP_SERVER_PREFIX,
) -> None:
    """Dispatch a ToolInvoked event for a completed ACP tool call.

    Skips WINK bridged tools (prefixed with *mcp_server_prefix*) since
    BridgedTool already emitted the event.
    """
    # Dedup: skip WINK bridged tools
    if title.startswith(mcp_server_prefix):
        return

    success = status == "completed"
    result: ToolResult[None]
    if success:
        result = ToolResult[None].ok(None, message=f"Tool {title} completed")
    else:
        result = ToolResult[None].error(f"Tool {title} failed")

    event = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=f"acp:{title}",
        params={},
        result=result,
        session_id=getattr(session, "session_id", None),
        created_at=datetime.now(UTC),
        usage=None,
        rendered_output=rendered_output,
        call_id=tool_call_id,
        run_context=run_context,
    )
    _ = session.dispatcher.dispatch(event)


def extract_token_usage(usage: object) -> TokenUsage | None:
    """Extract TokenUsage from an ACP Usage object.

    The ACP Usage object has: input_tokens, output_tokens,
    cached_read_tokens, thought_tokens.
    """
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cached_tokens = getattr(usage, "cached_read_tokens", None)

    if input_tokens is None and output_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )
