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

"""MCP tool bridge for exposing weakincentives tools to Claude Agent SDK."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...prompt._types import SupportsDataclassOrNone, SupportsToolResult
from ...prompt.tool import Tool, ToolContext
from ...prompt.tool_result import ToolResult
from ...runtime.logging import StructuredLogger, get_logger
from ...serde import parse, schema

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol, RenderedPromptProtocol
    from ...runtime.session.protocols import SessionProtocol
    from ..core import ProviderAdapter

__all__ = [
    "WrappedToolSpec",
    "build_wrapped_tool_spec",
    "invoke_wrapped_tool",
]


logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk.bridge"}
)


@dataclass(slots=True)
class WrappedToolSpec:
    """Specification for a weakincentives tool wrapped for SDK use.

    Attributes:
        name: Tool name.
        description: Tool description.
        input_schema: JSON schema for tool parameters.
        tool: Original weakincentives tool.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    tool: Tool[Any, Any]


def build_wrapped_tool_spec(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
) -> WrappedToolSpec | None:
    """Build a wrapped tool specification from a weakincentives tool.

    Args:
        tool: The weakincentives tool to wrap.

    Returns:
        WrappedToolSpec if the tool has a handler, None otherwise.
    """
    if tool.handler is None:
        return None

    # Generate input schema from params type
    input_schema: dict[str, Any]
    if tool.params_type is type(None):
        input_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    else:
        input_schema = schema(tool.params_type)

    return WrappedToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema=input_schema,
        tool=tool,
    )


def build_tool_specs(
    tools: Sequence[Tool[Any, Any]],
) -> tuple[WrappedToolSpec, ...]:
    """Build wrapped tool specifications for all tools with handlers.

    Args:
        tools: The tools to wrap.

    Returns:
        Tuple of wrapped tool specifications.
    """
    specs: list[WrappedToolSpec] = []
    for tool in tools:
        spec = build_wrapped_tool_spec(tool)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


@dataclass(slots=True, frozen=True)
class _InvokeToolContext:
    """Context bundled for tool invocation."""

    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapter[Any]
    session: SessionProtocol
    deadline: Deadline | None
    budget_tracker: BudgetTracker | None


def invoke_wrapped_tool(
    spec: WrappedToolSpec,
    args: dict[str, Any],
    *,
    ctx: _InvokeToolContext,
) -> dict[str, Any]:
    """Invoke a wrapped tool with the given arguments.

    Args:
        spec: The wrapped tool specification.
        args: Arguments to pass to the tool.
        ctx: Bundled invocation context.

    Returns:
        MCP-style tool result with "content" and "isError" fields.
    """
    tool = spec.tool

    logger.debug(
        "invoke_wrapped_tool.start",
        event="sdk.bridge.invoke_tool.start",
        context={"tool_name": spec.name},
    )

    # Parse arguments to params type
    params: Any
    try:
        if tool.params_type is type(None):
            params = None
        else:
            params = parse(tool.params_type, args, extra="forbid")
    except Exception as exc:
        logger.warning(
            "invoke_wrapped_tool.parse_error",
            event="sdk.bridge.invoke_tool.parse_error",
            context={"tool_name": spec.name, "error": str(exc)},
        )
        return {
            "content": [{"type": "text", "text": f"Error parsing arguments: {exc}"}],
            "isError": True,
        }

    # Build context
    tool_context = ToolContext(
        prompt=ctx.prompt,
        rendered_prompt=ctx.rendered_prompt,
        adapter=ctx.adapter,
        session=ctx.session,
        deadline=ctx.deadline,
        budget_tracker=ctx.budget_tracker,
    )

    # Execute handler
    try:
        assert tool.handler is not None  # Already verified in build_wrapped_tool_spec
        result: ToolResult[Any] = tool.handler(params, context=tool_context)
    except Exception as exc:
        logger.exception(
            "invoke_wrapped_tool.error",
            event="sdk.bridge.invoke_tool.error",
            context={"tool_name": spec.name, "error": str(exc)},
        )
        return {
            "content": [{"type": "text", "text": f"Error: {exc}"}],
            "isError": True,
        }
    else:
        logger.debug(
            "invoke_wrapped_tool.success",
            event="sdk.bridge.invoke_tool.success",
            context={
                "tool_name": spec.name,
                "success": result.success,
            },
        )
        return {
            "content": [{"type": "text", "text": result.message}],
            "isError": not result.success,
        }
