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

"""MCP tool bridge for exposing weakincentives tools to the Claude Agent SDK."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from ...budget import BudgetTracker
from ...contrib.tools.filesystem import Filesystem
from ...deadlines import Deadline
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ...runtime.events import ToolInvoked
from ...runtime.execution_state import CompositeSnapshot, ExecutionState
from ...runtime.logging import StructuredLogger, get_logger
from ...serde import parse, schema
from ..utilities import build_resources

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol, RenderedPromptProtocol
    from ...runtime.session.protocols import SessionProtocol
    from ..core import ProviderAdapter

__all__ = [
    "BridgedTool",
    "create_bridged_tools",
    "create_mcp_server",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "mcp_bridge"})


@dataclass(slots=True, frozen=True)
class _MCPToolCallFunction:
    """Adapter for MCP tool call function to ProviderToolCallFunction."""

    name: str
    arguments: str | None


@dataclass(slots=True, frozen=True)
class _MCPToolCall:
    """Adapter for MCP tool call to ProviderToolCall."""

    id: str
    function: _MCPToolCallFunction


class BridgedTool:
    """A weakincentives tool wrapped for MCP/SDK consumption.

    Attributes:
        name: Tool name for MCP registration.
        description: Tool description for MCP schema.
        input_schema: JSON schema for tool parameters.
        handler: Callable that executes the tool and returns MCP-format result.

    The execution_state provides transactional semantics: a snapshot is taken
    before execution and restored on failure, ensuring consistent state across
    failed or aborted tool calls. Session and filesystem are accessed from
    the execution_state.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        tool: Tool[Any, Any],
        execution_state: ExecutionState,
        adapter: ProviderAdapter[Any],
        prompt: PromptProtocol[Any],
        rendered_prompt: RenderedPromptProtocol[Any] | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        adapter_name: str = "claude_agent_sdk",
        prompt_name: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._tool = tool
        self._execution_state = execution_state
        self._adapter = adapter
        self._prompt = prompt
        self._rendered_prompt = rendered_prompt
        self._deadline = deadline
        self._budget_tracker = budget_tracker
        self._adapter_name = adapter_name
        self._prompt_name = prompt_name or f"{prompt.ns}:{prompt.key}"

    @property
    def _session(self) -> SessionProtocol:
        """Get session from execution state."""
        return self._execution_state.session

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return MCP-format result.

        Args:
            args: Tool arguments parsed from MCP request.

        Returns:
            MCP-format result dict with content and isError fields.

        Uses transactional semantics via execution_state: snapshot before
        execution, restore on failure.
        """
        handler = self._tool.handler
        if handler is None:
            return {
                "content": [{"type": "text", "text": "Tool has no handler"}],
                "isError": True,
            }

        with self._execution_state.tool_transaction(
            tag=f"tool:{self.name}"
        ) as snapshot:
            return self._execute_handler(handler, args, snapshot=snapshot)

    def _execute_handler(
        self,
        handler: ToolHandler[Any, Any],
        args: dict[str, Any],
        *,
        snapshot: CompositeSnapshot,
    ) -> dict[str, Any]:
        """Execute tool handler with transactional semantics.

        Args:
            handler: The tool handler to execute.
            args: Tool arguments.
            snapshot: Pre-execution snapshot for restore on failure.

        Returns:
            MCP-format result dict.
        """
        try:
            if self._tool.params_type is type(None):
                params = None
            else:
                params = parse(self._tool.params_type, args, extra="forbid")

            filesystem = self._execution_state.resources.get(Filesystem)
            resources = build_resources(
                filesystem=filesystem,
                budget_tracker=self._budget_tracker,
            )
            context = ToolContext(
                prompt=self._prompt,
                rendered_prompt=self._rendered_prompt,
                adapter=self._adapter,
                session=self._session,
                deadline=self._deadline,
                resources=resources,
            )

            result = handler(params, context=context)

            # Restore on tool failure (manual, since we're returning not raising)
            if not result.success:
                self._restore_snapshot(snapshot, reason="tool_failure")

            return self._format_success_result(args, result)

        except (TypeError, ValueError) as error:
            self._restore_snapshot(snapshot, reason="validation_error")
            logger.warning(
                "claude_agent_sdk.bridge.validation_error",
                event="bridge.validation_error",
                context={"tool_name": self.name, "error": str(error)},
            )
            return {
                "content": [{"type": "text", "text": f"Validation error: {error}"}],
                "isError": True,
            }

        except VisibilityExpansionRequired:
            # Context manager handles restore; just re-raise
            raise

        except Exception as error:
            # Context manager handles restore for propagating exceptions,
            # but we're catching and returning, so restore manually
            self._restore_snapshot(snapshot, reason="exception")
            logger.exception(
                "claude_agent_sdk.bridge.handler_error",
                event="bridge.handler_error",
                context={"tool_name": self.name, "error": str(error)},
            )
            return {
                "content": [{"type": "text", "text": f"Error: {error}"}],
                "isError": True,
            }

    def _restore_snapshot(self, snapshot: CompositeSnapshot, *, reason: str) -> None:
        """Restore from snapshot.

        Args:
            snapshot: CompositeSnapshot to restore.
            reason: Reason for restore (for logging).
        """
        self._execution_state.restore(snapshot)
        logger.debug(
            f"State restored after {reason}.",
            event=f"bridge.{reason}_restore",
            context={"tool_name": self.name},
        )

    def _format_success_result(
        self,
        args: dict[str, Any],
        result: ToolResult[Any],
    ) -> dict[str, Any]:
        """Format successful tool result as MCP response.

        Args:
            args: Original tool arguments.
            result: Tool execution result.

        Returns:
            MCP-format result dict.
        """
        # Respect exclude_value_from_context to avoid spilling large/sensitive
        # data into the model context
        if result.exclude_value_from_context:
            output_text = result.message
        else:
            # Use render() which calls render_tool_payload on the value,
            # falling back to message if render returns empty
            rendered = result.render()
            output_text = rendered if rendered else result.message

        # Publish ToolInvoked event with the actual tool result value
        # This enables session reducers to dispatch based on the value type
        self._publish_tool_invoked(args, result, output_text)

        return {
            "content": [{"type": "text", "text": output_text}],
            "isError": not result.success,
        }

    def _publish_tool_invoked(
        self,
        args: dict[str, Any],
        result: ToolResult[Any],
        rendered_output: str,
    ) -> None:
        """Publish a ToolInvoked event for session reducer dispatch.

        The session extracts the value from result.value for slice routing.
        """
        event = ToolInvoked(
            prompt_name=self._prompt_name,
            adapter=self._adapter_name,
            name=self.name,
            params=args,
            result=cast(ToolResult[object], result),
            session_id=None,
            created_at=datetime.now(UTC),
            usage=None,
            rendered_output=rendered_output[:1000] if rendered_output else "",
            call_id=None,
        )
        self._session.event_bus.publish(event)


def create_bridged_tools(
    tools: tuple[Tool[Any, Any], ...],
    *,
    execution_state: ExecutionState,
    adapter: ProviderAdapter[Any],
    prompt: PromptProtocol[Any],
    rendered_prompt: RenderedPromptProtocol[Any] | None,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    adapter_name: str = "claude_agent_sdk",
    prompt_name: str | None = None,
) -> tuple[BridgedTool, ...]:
    """Create MCP-compatible tool wrappers for weakincentives tools.

    Args:
        tools: Tuple of weakincentives Tool instances.
        execution_state: ExecutionState for transactional tool execution.
            Session and filesystem are accessed from this state container.
        adapter: Adapter for tool context.
        prompt: Prompt for tool context.
        rendered_prompt: Rendered prompt for tool context.
        deadline: Optional deadline for tool context.
        budget_tracker: Optional budget tracker for tool context.
        adapter_name: Name of the adapter for event publishing.
        prompt_name: Name of the prompt for event publishing.

    Returns:
        Tuple of BridgedTool instances ready for MCP registration.
    """
    bridged: list[BridgedTool] = []
    resolved_prompt_name = prompt_name or prompt.name or f"{prompt.ns}:{prompt.key}"

    for tool in tools:
        if tool.handler is None:
            continue

        input_schema: dict[str, Any]
        if tool.params_type is type(None):
            input_schema = {"type": "object", "properties": {}}
        else:
            input_schema = schema(tool.params_type)

        bridged_tool = BridgedTool(
            name=tool.name,
            description=tool.description,
            input_schema=input_schema,
            tool=tool,
            execution_state=execution_state,
            adapter=adapter,
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=adapter_name,
            prompt_name=resolved_prompt_name,
        )
        bridged.append(bridged_tool)

    return tuple(bridged)


def _make_async_handler(
    bt: BridgedTool,
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """Create an async handler wrapper for a bridged tool.

    The SDK expects async handlers, but our BridgedTool is synchronous.
    This wrapper creates an async function that calls the sync handler.
    The async def is required by the SDK even though we don't await anything.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:  # noqa: RUF029
        return bt(args)

    return handler


def create_mcp_server(
    bridged_tools: tuple[BridgedTool, ...],
    server_name: str = "wink-tools",
) -> object:
    """Create an MCP server config with the bridged tools registered.

    Uses the Claude Agent SDK's create_sdk_mcp_server function to create
    an in-process MCP server that can be passed to ClaudeAgentOptions.

    Args:
        bridged_tools: Tuple of BridgedTool instances to register.
        server_name: Name for the MCP server.

    Returns:
        McpSdkServerConfig instance ready for use with ClaudeAgentOptions.
    """
    try:
        from claude_agent_sdk import (
            SdkMcpTool,
            create_sdk_mcp_server,
            tool as sdk_tool,
        )
    except ImportError as error:
        raise ImportError(
            "claude-agent-sdk is required for custom tool bridging. "
            "Install it with: pip install claude-agent-sdk"
        ) from error

    sdk_tools: list[SdkMcpTool[Any]] = []

    for bridged_tool in bridged_tools:
        async_handler = _make_async_handler(bridged_tool)

        # Use the SDK's tool decorator to wrap the handler
        decorated_tool: SdkMcpTool[Any] = sdk_tool(
            bridged_tool.name,
            bridged_tool.description,
            bridged_tool.input_schema,
        )(async_handler)

        sdk_tools.append(decorated_tool)

        logger.debug(
            "claude_agent_sdk.bridge.tool_registered",
            event="bridge.tool_registered",
            context={"tool_name": bridged_tool.name},
        )

    # Create the SDK MCP server config
    mcp_server_config = create_sdk_mcp_server(
        name=server_name,
        version="1.0.0",
        tools=sdk_tools,
    )

    logger.info(
        "claude_agent_sdk.bridge.mcp_server_created",
        event="bridge.mcp_server_created",
        context={"server_name": server_name, "tool_count": len(bridged_tools)},
    )

    return mcp_server_config
