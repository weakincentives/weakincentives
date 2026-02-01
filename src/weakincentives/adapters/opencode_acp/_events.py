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

"""ACP session updates to WINK event mapping.

This module provides utilities for mapping ACP session update notifications
to WINK ToolInvoked events. OpenCode tools (not bridged WINK tools) are
reported via session/update notifications and need to be converted to WINK
telemetry events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ...runtime.events import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from ...runtime.run_context import RunContext

__all__ = [
    "WINK_MCP_SERVER_PREFIX",
    "is_wink_tool_call",
    "map_tool_call_to_event",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_events"})

# Prefix used to identify WINK bridged tools in MCP namespace
WINK_MCP_SERVER_PREFIX: str = "mcp__wink__"


def is_wink_tool_call(tool_name: str, mcp_server_name: str | None = None) -> bool:
    """Check if a tool call is a bridged WINK tool.

    WINK tools are identified either by explicit MCP server metadata or
    by the tool name prefix. Bridged tools already emit ToolInvoked events
    via BridgedTool, so we skip them to avoid duplicates.

    Args:
        tool_name: The tool name from the ACP update.
        mcp_server_name: The MCP server name if available from metadata.

    Returns:
        True if this is a bridged WINK tool (should be skipped).
    """
    # Check explicit MCP server metadata first
    if mcp_server_name in {"wink", "wink-tools"}:
        return True

    # Fall back to prefix check
    return tool_name.startswith(WINK_MCP_SERVER_PREFIX)


def map_tool_call_to_event(
    *,
    tool_name: str,
    tool_use_id: str,
    params: dict[str, Any] | None,
    result: str | None,
    success: bool,
    prompt_name: str,
    adapter_name: str,
    session_id: UUID | None,
    run_context: RunContext | None = None,
) -> ToolInvoked:
    """Map an ACP tool call update to a WINK ToolInvoked event.

    Args:
        tool_name: The tool name.
        tool_use_id: The tool call ID.
        params: The tool parameters, if available.
        result: The tool result text, if available.
        success: Whether the tool call succeeded.
        prompt_name: Name of the prompt for event context.
        adapter_name: Name of the adapter for event context.
        session_id: Session ID for event context.
        run_context: Optional run context for tracing.

    Returns:
        A ToolInvoked event ready for dispatch.
    """
    from ...prompt.tool import ToolResult

    # Create a ToolResult to represent the outcome
    if success:
        tool_result = ToolResult.ok(result, message=result or "")
    else:
        tool_result = ToolResult.error(result or "Tool call failed")

    event = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=tool_name,
        params=params or {},
        result=tool_result,
        session_id=session_id,
        created_at=datetime.now(UTC),
        usage=None,
        rendered_output=result[:1000] if result else "",
        call_id=tool_use_id,
        run_context=run_context,
    )

    logger.debug(
        "opencode_acp.events.tool_invoked_mapped",
        event="events.tool_invoked_mapped",
        context={
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "success": success,
        },
    )

    return event
