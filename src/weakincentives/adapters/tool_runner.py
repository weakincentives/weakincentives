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

"""Unified tool execution with transaction semantics.

This module provides ToolRunner, which executes tool calls with automatic
snapshot and restore semantics. On tool failure, all state changes are
rolled back, ensuring consistent execution state.

Example usage::

    from weakincentives.adapters.tool_runner import ToolRunner
    from weakincentives.runtime.execution_state import ExecutionState

    runner = ToolRunner(
        execution_state=state,
        tool_registry=tools,
        prompt_name="my_prompt",
    )

    result = runner.execute(tool_call, context=context)
    if not result.success:
        # State was automatically restored
        pass

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from ..errors import DeadlineExceededError, ToolValidationError
from ..prompt.errors import VisibilityExpansionRequired
from ..prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ..runtime.execution_state import ExecutionState
from ..runtime.logging import StructuredLogger, get_logger
from ..serde import parse
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._provider_protocols import ProviderToolCall
from .core import PROMPT_EVALUATION_PHASE_TOOL, PromptEvaluationError

logger: StructuredLogger = get_logger(__name__, context={"component": "tool_runner"})


def _wrap_exception_as_tool_result(
    error: Exception,
    *,
    tool_name: str,
) -> ToolResult[Any]:
    """Wrap an exception as a failed ToolResult."""
    return ToolResult(
        message=f"Tool '{tool_name}' execution failed: {error}",
        value=None,
        success=False,
    )


@dataclass(slots=True)
class ToolRunner:
    """Unified tool execution with transaction semantics.

    ToolRunner provides identical transaction semantics for all adapters.
    Each tool call is a transaction - on failure, all state changes are
    rolled back via the ExecutionState snapshot/restore mechanism.

    Attributes:
        execution_state: The unified runtime state container.
        tool_registry: Mapping of tool names to tool definitions.
        prompt_name: Name of the prompt for error messages.

    Example usage::

        runner = ToolRunner(
            execution_state=state,
            tool_registry=tools,
            prompt_name="my_prompt",
        )

        # Execute with automatic rollback on failure
        result = runner.execute(tool_call, context=context)

    """

    execution_state: ExecutionState
    tool_registry: Mapping[str, Tool[Any, Any]]
    prompt_name: str

    def execute(
        self,
        tool_call: ProviderToolCall,
        *,
        context: ToolContext,
    ) -> ToolResult[Any]:
        """Execute a tool call with transaction semantics.

        Takes a snapshot before execution. On failure (exception or
        success=False result), restores the snapshot. On success, the
        changes are committed (no-op).

        Args:
            tool_call: The provider tool call to execute.
            context: Tool execution context.

        Returns:
            ToolResult from the tool handler.

        Raises:
            VisibilityExpansionRequired: If the tool requires visibility
                expansion. State is restored before re-raising.
            PromptEvaluationError: If deadline exceeded or other
                evaluation errors occur.
        """
        tool_name = tool_call.function.name
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return ToolResult(
                message=f"Unknown tool '{tool_name}'",
                value=None,
                success=False,
            )

        handler = tool.handler
        if handler is None:
            return ToolResult(
                message=f"Tool '{tool_name}' does not have a handler",
                value=None,
                success=False,
            )

        # Take snapshot before execution
        pre_snapshot = self.execution_state.snapshot(tag=f"pre:{tool_call.id}")

        try:
            result = self._invoke_handler(
                tool=tool,
                handler=cast(
                    ToolHandler[SupportsDataclassOrNone, SupportsToolResult], handler
                ),
                tool_call=tool_call,
                context=context,
            )
        except VisibilityExpansionRequired:
            # Restore state on visibility expansion
            self.execution_state.restore(pre_snapshot)
            logger.debug(
                "State restored after visibility expansion required.",
                event="tool_runner_visibility_restore",
                context={"tool": tool_name, "call_id": tool_call.id},
            )
            raise
        except DeadlineExceededError as error:
            # Restore state on deadline exceeded
            self.execution_state.restore(pre_snapshot)
            logger.debug(
                "State restored after deadline exceeded.",
                event="tool_runner_deadline_restore",
                context={"tool": tool_name, "call_id": tool_call.id},
            )
            raise PromptEvaluationError(
                str(error) or f"Tool '{tool_name}' exceeded the deadline.",
                prompt_name=self.prompt_name,
                phase=PROMPT_EVALUATION_PHASE_TOOL,
            ) from error
        except Exception as error:
            # Restore state on any exception
            self.execution_state.restore(pre_snapshot)
            logger.debug(
                "State restored after tool exception.",
                event="tool_runner_exception_restore",
                context={
                    "tool": tool_name,
                    "call_id": tool_call.id,
                    "error": str(error),
                },
            )
            return _wrap_exception_as_tool_result(error, tool_name=tool_name)

        # Restore if tool returned failure
        if not result.success:
            self.execution_state.restore(pre_snapshot)
            logger.debug(
                "State restored after tool failure.",
                event="tool_runner_failure_restore",
                context={"tool": tool_name, "call_id": tool_call.id},
            )

        return result

    def _invoke_handler(  # noqa: PLR6301
        self,
        *,
        tool: Tool[Any, Any],
        handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
        tool_call: ProviderToolCall,
        context: ToolContext,
    ) -> ToolResult[Any]:
        """Parse arguments and invoke the tool handler."""
        # Parse arguments
        import json

        arguments_json = tool_call.function.arguments
        arguments: dict[str, Any] = {}

        if arguments_json:
            try:
                parsed: object = json.loads(arguments_json)
            except json.JSONDecodeError as error:
                raise ToolValidationError(f"Invalid JSON arguments: {error}") from error

            if not isinstance(parsed, dict):
                raise ToolValidationError("Tool arguments must be a JSON object")

            arguments = cast(dict[str, Any], parsed)

        # Parse into typed params
        params: SupportsDataclass | None
        if tool.params_type is type(None):
            if arguments:
                raise ToolValidationError("Tool does not accept arguments")
            params = None
        else:
            try:
                params = parse(tool.params_type, arguments, extra="forbid")
            except (TypeError, ValueError) as error:
                raise ToolValidationError(str(error)) from error

        # Check deadline
        if context.deadline is not None and context.deadline.remaining() <= timedelta(
            0
        ):
            raise DeadlineExceededError(
                f"Deadline expired before executing '{tool.name}'"
            )

        # Invoke handler
        return handler(params, context=context)


__all__ = ["ToolRunner"]
