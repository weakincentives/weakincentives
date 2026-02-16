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

"""Shared tool bridge for exposing weakincentives tools to agentic harnesses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from ...budget import BudgetTracker
from ...clock import SYSTEM_CLOCK
from ...deadlines import Deadline
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.policy import PolicyDecision
from ...prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ...runtime.events import ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.transactions import (
    CompositeSnapshot,
    restore_snapshot,
    tool_transaction,
)
from ...runtime.watchdog import Heartbeat
from ...serde import parse, schema
from ._visibility_signal import VisibilityExpansionSignal

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol, RenderedPromptProtocol
    from ...runtime.session.protocols import SessionProtocol
    from ..core import ProviderAdapter

__all__ = [
    "BridgedTool",
    "MCPToolExecutionState",
    "create_bridged_tools",
    "create_mcp_server",
    "make_async_handler",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "mcp_bridge"})


MCP_TOOL_PREFIX = "mcp__wink__"


def _hash_params(params: dict[str, Any]) -> str:
    """Create a stable hash of tool parameters for keying.

    Uses JSON serialization with sorted keys for deterministic output,
    then MD5 hash for a compact key. MD5 is sufficient here since we're
    not using it for security, just for deduplication.
    """
    json_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode(), usedforsecurity=False).hexdigest()[:16]


def _normalize_tool_name(tool_name: str) -> str:
    """Normalize tool name by removing mcp__wink__ prefix if present."""
    if tool_name.startswith(MCP_TOOL_PREFIX):
        return tool_name[len(MCP_TOOL_PREFIX) :]
    return tool_name


@dataclass(slots=True)
class MCPToolExecutionState:
    """Thread-safe state for correlating tool_use_ids between hooks and MCP bridge.

    The MCP protocol doesn't pass tool_use_id to tool handlers, so we need shared
    state to correlate PreToolUse hook calls with BridgedTool executions.

    Thread Safety
    -------------
    Uses a composite key of (tool_name, params_hash) with FIFO queues per key.
    This handles all concurrency scenarios:

    - Different tools in parallel: Different keys, no conflict
    - Same tool with different params in parallel: Different hashes, no conflict
    - Same tool with same params in parallel: FIFO queue ensures correct ordering
      (assuming PreToolUse hooks fire in the same order as tool executions)

    The implementation uses:
    - threading.Lock for dict mutations (adding/removing queues)
    - collections.deque for thread-safe append/popleft within queues

    Usage Flow
    ----------
    1. PreToolUse hook: enqueue(tool_name, params, tool_use_id)
    2. BridgedTool: dequeue(tool_name, params) -> tool_use_id
    3. PostToolUse hook: (optional cleanup, queues auto-drain)
    """

    _queues: dict[str, deque[str]] = field(
        default_factory=lambda: dict[str, deque[str]]()
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _make_key(tool_name: str, params: dict[str, Any]) -> str:
        """Create composite key from tool name and params hash."""
        normalized = _normalize_tool_name(tool_name)
        params_hash = _hash_params(params)
        return f"{normalized}:{params_hash}"

    def enqueue(self, tool_name: str, params: dict[str, Any], tool_use_id: str) -> None:
        """Enqueue a tool_use_id for a specific tool+params combination.

        Called by PreToolUse hook before tool execution begins.

        Args:
            tool_name: The tool name (with or without mcp__wink__ prefix).
            params: The tool input parameters.
            tool_use_id: The SDK's tool_use_id for this invocation.
        """
        key = self._make_key(tool_name, params)
        with self._lock:
            if key not in self._queues:
                self._queues[key] = deque()
            self._queues[key].append(tool_use_id)

    def dequeue(self, tool_name: str, params: dict[str, Any]) -> str | None:
        """Dequeue the oldest tool_use_id for a specific tool+params combination.

        Called by BridgedTool when dispatching ToolInvoked events.

        Args:
            tool_name: The tool name (with or without mcp__wink__ prefix).
            params: The tool input parameters.

        Returns:
            The oldest tool_use_id for this tool+params, or None if queue is empty.
        """
        key = self._make_key(tool_name, params)
        with self._lock:
            queue = self._queues.get(key)
            if queue:
                try:
                    result = queue.popleft()
                    # Clean up empty queues to bound memory usage
                    if not queue:
                        del self._queues[key]
                    return result
                except (
                    IndexError
                ):  # pragma: no cover - defensive, can't happen with lock
                    return None
            return None


class BridgedTool:
    """A weakincentives tool wrapped for MCP/SDK consumption.

    Attributes:
        name: Tool name for MCP registration.
        description: Tool description for MCP schema.
        input_schema: JSON schema for tool parameters.
        handler: Callable that executes the tool and returns MCP-format result.

    Transactional semantics: a snapshot is taken before execution and restored
    on failure, ensuring consistent state across failed or aborted tool calls.
    Session and resources are accessed via the prompt.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        tool: Tool[Any, Any],
        session: SessionProtocol,
        adapter: ProviderAdapter[Any],
        prompt: PromptProtocol[Any],
        rendered_prompt: RenderedPromptProtocol[Any] | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        adapter_name: str = "claude_agent_sdk",
        prompt_name: str | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
        visibility_signal: VisibilityExpansionSignal | None = None,
        mcp_tool_state: MCPToolExecutionState | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._tool = tool
        self._session = session
        self._adapter = adapter
        self._prompt = prompt
        self._rendered_prompt = rendered_prompt
        self._deadline = deadline
        self._budget_tracker = budget_tracker
        self._adapter_name = adapter_name
        self._prompt_name = prompt_name or f"{prompt.ns}:{prompt.key}"
        self._heartbeat = heartbeat
        self._run_context = run_context
        self._visibility_signal = visibility_signal
        self._mcp_tool_state = mcp_tool_state

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return MCP-format result.

        Args:
            args: Tool arguments parsed from MCP request.

        Returns:
            MCP-format result dict with content and isError fields.

        Uses transactional semantics: snapshot before execution, restore on failure.
        """
        # Dequeue tool_use_id at the start to ensure it's always consumed,
        # even if tool execution fails early (validation error, exception, etc.)
        tool_use_id = (
            self._mcp_tool_state.dequeue(self.name, args)
            if self._mcp_tool_state
            else None
        )
        logger.debug(
            "claude_agent_sdk.bridge.tool_call.start",
            event="bridge.tool_call.start",
            context={
                "tool_name": self.name,
                "tool_use_id": tool_use_id,
                "prompt_name": self._prompt_name,
                "arguments": args,
            },
        )
        handler = self._tool.handler
        if handler is None:
            return {
                "content": [{"type": "text", "text": "Tool has no handler"}],
                "isError": True,
            }

        with tool_transaction(
            self._session,
            self._prompt.resources.context,
            tag=f"tool:{self.name}",
        ) as snapshot:
            return self._execute_handler(
                handler, args, snapshot=snapshot, tool_use_id=tool_use_id
            )

    def _execute_handler(
        self,
        handler: ToolHandler[Any, Any],
        args: dict[str, Any],
        *,
        snapshot: CompositeSnapshot,
        tool_use_id: str | None,
    ) -> dict[str, Any]:
        """Execute tool handler with transactional semantics.

        Args:
            handler: The tool handler to execute.
            args: Tool arguments.
            snapshot: Pre-execution snapshot for restore on failure.
            tool_use_id: The SDK's tool_use_id for this invocation.

        Returns:
            MCP-format result dict.
        """
        try:
            params = self._parse_params(args)
            context = self._make_tool_context()

            # Enforce tool policies before handler execution
            denial = self._check_policies(params, context=context)
            if denial is not None:
                self._restore_snapshot(snapshot, reason="policy_denial")
                return {
                    "content": [{"type": "text", "text": denial}],
                    "isError": True,
                }

            result = handler(params, context=context)

            # Notify policies of successful result
            if result.success:
                self._notify_policies(params, result, context=context)
            else:
                self._restore_snapshot(snapshot, reason="tool_failure")

            return self._format_success_result(args, result, tool_use_id)

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

        except VisibilityExpansionRequired as exc:
            # Store exception in signal for adapter to re-raise after SDK completes.
            # Manually restore snapshot since we're catching and returning, not re-raising.
            self._restore_snapshot(snapshot, reason="visibility_expansion")
            if self._visibility_signal is not None:
                self._visibility_signal.set(exc)
            logger.debug(
                "claude_agent_sdk.bridge.visibility_expansion_required",
                event="bridge.visibility_expansion_required",
                context={
                    "tool_name": self.name,
                    "section_keys": exc.section_keys,
                    "reason": exc.reason,
                },
            )
            # Return success (not error) - the tool worked correctly by identifying
            # sections that need expansion. The SDK will complete gracefully, then
            # the adapter checks the signal and re-raises the exception so the
            # caller can re-render the prompt with expanded sections.
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Sections expanded: {', '.join(exc.section_keys)}. "
                            "The prompt will be re-rendered with the full content "
                            "of the requested sections."
                        ),
                    }
                ],
                "isError": False,
            }

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

    def _parse_params(self, args: dict[str, Any]) -> object:
        """Parse tool arguments into the expected params type."""
        if self._tool.params_type is type(None):
            return None
        return parse(self._tool.params_type, args, extra="forbid", coerce=False)

    def _make_tool_context(self) -> ToolContext:
        """Create a ToolContext for handler execution."""
        return ToolContext(
            prompt=self._prompt,
            rendered_prompt=self._rendered_prompt,
            adapter=cast(Any, self._adapter),
            session=self._session,
            deadline=self._deadline,
            heartbeat=self._heartbeat,
            run_context=self._run_context,
        )

    def _restore_snapshot(self, snapshot: CompositeSnapshot, *, reason: str) -> None:
        """Restore from snapshot.

        Args:
            snapshot: CompositeSnapshot to restore.
            reason: Reason for restore (for logging).
        """
        restore_snapshot(self._session, self._prompt.resources.context, snapshot)
        logger.debug(
            f"State restored after {reason}.",
            event=f"bridge.{reason}_restore",
            context={"tool_name": self.name},
        )

    def _check_policies(
        self,
        params: object,
        *,
        context: ToolContext,
    ) -> str | None:
        """Check tool policies before handler execution.

        Returns denial message if any policy denies the call, None otherwise.
        """
        policies = self._prompt.policies_for_tool(self.name)
        for policy in policies:
            decision: PolicyDecision = policy.check(self._tool, params, context=context)
            if not decision.allowed:
                reason = decision.reason or "Denied by policy."
                logger.info(
                    "claude_agent_sdk.bridge.policy_denied",
                    event="bridge.policy_denied",
                    context={
                        "tool_name": self.name,
                        "policy": getattr(policy, "name", "unknown"),
                        "reason": reason,
                    },
                )
                return reason
        return None

    def _notify_policies(
        self,
        params: object,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Notify policies of successful tool execution."""
        policies = self._prompt.policies_for_tool(self.name)
        for policy in policies:
            policy.on_result(self._tool, params, result, context=context)

    def _format_success_result(
        self,
        args: dict[str, Any],
        result: ToolResult[Any],
        tool_use_id: str | None,
    ) -> dict[str, Any]:
        """Format successful tool result as MCP response.

        Args:
            args: Original tool arguments.
            result: Tool execution result.
            tool_use_id: The SDK's tool_use_id for this invocation.

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

        # Dispatch ToolInvoked event with the actual tool result value
        # This enables session reducers to dispatch based on the value type
        self._dispatch_tool_invoked(args, result, output_text, tool_use_id)

        logger.debug(
            "claude_agent_sdk.bridge.tool_call.complete",
            event="bridge.tool_call.complete",
            context={
                "tool_name": self.name,
                "success": result.success,
                "message": result.message,
                "value_type": (
                    type(result.value).__qualname__
                    if result.value is not None
                    else None
                ),
                "output_text": output_text,
            },
        )

        return {
            "content": [{"type": "text", "text": output_text}],
            "isError": not result.success,
        }

    def _dispatch_tool_invoked(
        self,
        args: dict[str, Any],
        result: ToolResult[Any],
        rendered_output: str,
        call_id: str | None,
    ) -> None:
        """Dispatch a ToolInvoked event for session reducer dispatch.

        The session extracts the value from result.value for slice routing.
        The call_id is passed through from __call__ which dequeues it at the
        start to ensure it's always consumed even on error paths.
        """
        event = ToolInvoked(
            prompt_name=self._prompt_name,
            adapter=self._adapter_name,
            name=self.name,
            params=args,
            result=cast(ToolResult[object], result),
            session_id=getattr(self._session, "session_id", None),
            created_at=SYSTEM_CLOCK.utcnow(),
            usage=None,
            rendered_output=rendered_output[:1000] if rendered_output else "",
            call_id=call_id,
            run_context=self._run_context,
        )
        _ = self._session.dispatcher.dispatch(event)


def create_bridged_tools(
    tools: tuple[Tool[Any, Any], ...],
    *,
    session: SessionProtocol,
    adapter: ProviderAdapter[Any],
    prompt: PromptProtocol[Any],
    rendered_prompt: RenderedPromptProtocol[Any] | None,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    adapter_name: str = "claude_agent_sdk",
    prompt_name: str | None = None,
    heartbeat: Heartbeat | None = None,
    run_context: RunContext | None = None,
    visibility_signal: VisibilityExpansionSignal | None = None,
    mcp_tool_state: MCPToolExecutionState | None = None,
) -> tuple[BridgedTool, ...]:
    """Create MCP-compatible tool wrappers for weakincentives tools.

    Args:
        tools: Tuple of weakincentives Tool instances.
        session: Session for tool execution context.
        adapter: Adapter for tool context.
        prompt: Prompt for tool context (must be in active context).
        rendered_prompt: Rendered prompt for tool context.
        deadline: Optional deadline for tool context.
        budget_tracker: Optional budget tracker for tool context.
        adapter_name: Name of the adapter for event dispatching.
        prompt_name: Name of the prompt for event dispatching.
        heartbeat: Optional heartbeat for tool context.
        run_context: Optional execution context with correlation identifiers.
        visibility_signal: Signal for propagating VisibilityExpansionRequired
            exceptions from tool handlers to the adapter.
        mcp_tool_state: Shared state for passing tool_use_id from hooks to bridge.

    Returns:
        Tuple of BridgedTool instances ready for MCP registration.
    """
    logger.debug(
        "claude_agent_sdk.bridge.create_bridged_tools",
        event="bridge.create_bridged_tools",
        context={
            "tool_count": len(tools),
            "tool_names": [t.name for t in tools],
            "prompt_name": prompt_name or prompt.name or f"{prompt.ns}:{prompt.key}",
        },
    )
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
            session=session,
            adapter=adapter,
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=adapter_name,
            prompt_name=resolved_prompt_name,
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
            mcp_tool_state=mcp_tool_state,
        )
        bridged.append(bridged_tool)

    return tuple(bridged)


def make_async_handler(
    bt: BridgedTool,
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """Create an async handler wrapper for a bridged tool.

    The SDK expects async handlers, but our BridgedTool is synchronous.
    This wrapper runs the sync handler in a thread via asyncio.to_thread()
    to avoid blocking the event loop during tool execution.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(bt, args)

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
        msg = "claude-agent-sdk is required. Install with: pip install claude-agent-sdk"
        raise ImportError(msg) from error

    sdk_tools: list[SdkMcpTool[Any]] = []

    for bridged_tool in bridged_tools:
        async_handler = make_async_handler(bridged_tool)

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
