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

"""Tool execution and transaction management for provider adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from ..budget import BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import DeadlineExceededError, ToolValidationError
from ..prompt.errors import VisibilityExpansionRequired
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from ..prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ..runtime.events import HandlerFailure, ToolInvoked
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.transactions import (
    CompositeSnapshot,
    restore_snapshot,
    tool_transaction,
)
from ..serde import parse
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._names import AdapterName
from ._provider_protocols import ProviderToolCall
from .core import (
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    SessionProtocol,
)
from .utilities import (
    ToolArgumentsParser,
    ToolChoice,
    deadline_provider_payload,
    raise_tool_deadline_error,
    token_usage_from_payload,
)

if TYPE_CHECKING:
    from .core import ProviderAdapter


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.tool_executor"}
)


class ToolMessageSerializer(Protocol):
    def __call__(
        self,
        result: ToolResult[SupportsToolResult],
        *,
        payload: object | None = ...,
    ) -> object: ...


@FrozenDataclass()
class RejectedToolParams:
    """Dataclass used when provider arguments fail validation."""

    raw_arguments: dict[str, Any]
    error: str


@FrozenDataclass()
class ToolExecutionOutcome:
    """Result of executing a tool handler."""

    tool: Tool[SupportsDataclassOrNone, SupportsToolResult]
    params: SupportsDataclass | None
    result: ToolResult[SupportsToolResult]
    call_id: str | None
    log: StructuredLogger
    snapshot: CompositeSnapshot


@dataclass(slots=True)
class ToolExecutionContext:
    """Inputs and collaborators required to execute a provider tool call.

    Provides unified access to session and prompt resources for transactional
    tool execution.
    """

    adapter_name: AdapterName
    adapter: ProviderAdapter[Any]
    prompt: Prompt[Any]
    rendered_prompt: RenderedPrompt[Any] | None
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]
    session: SessionProtocol
    prompt_name: str
    parse_arguments: ToolArgumentsParser
    format_dispatch_failures: Callable[[Sequence[HandlerFailure]], str]
    deadline: Deadline | None
    provider_payload: dict[str, Any] | None = None
    logger_override: StructuredLogger | None = None
    budget_tracker: BudgetTracker | None = None

    def with_provider_payload(
        self, provider_payload: dict[str, Any] | None
    ) -> ToolExecutionContext:
        """Return a copy of the context with a new provider payload."""
        from dataclasses import replace

        return replace(self, provider_payload=provider_payload)


def _resolve_tool_and_handler(
    *,
    tool_call: ProviderToolCall,
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> tuple[
    Tool[SupportsDataclassOrNone, SupportsToolResult],
    ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
]:
    function = tool_call.function
    tool_name = function.name
    tool = tool_registry.get(tool_name)
    if tool is None:
        raise PromptEvaluationError(
            f"Unknown tool '{tool_name}' requested by provider.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )
    handler = tool.handler
    if handler is None:
        raise PromptEvaluationError(
            f"Tool '{tool_name}' does not have a registered handler.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )

    return tool, handler


def _parse_tool_call_arguments(
    *,
    tool_call: ProviderToolCall,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
    parse_arguments: ToolArgumentsParser,
) -> dict[str, Any]:
    return parse_arguments(
        tool_call.function.arguments,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
    )


def _build_tool_logger(
    *,
    context: ToolExecutionContext,
    tool_name: str,
    tool_call: ProviderToolCall,
) -> tuple[str | None, StructuredLogger]:
    call_id = tool_call.id
    bound_log = (context.logger_override or logger).bind(
        adapter=context.adapter_name,
        prompt=context.prompt_name,
        tool=tool_name,
        call_id=call_id,
    )
    return call_id, bound_log


def parse_tool_params(
    *,
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    arguments_mapping: Mapping[str, Any],
) -> SupportsDataclass | None:
    if tool.params_type is type(None):
        if arguments_mapping:
            raise ToolValidationError("Tool does not accept any arguments.")
        return None
    try:
        return parse(tool.params_type, arguments_mapping, extra="forbid")
    except (TypeError, ValueError) as error:
        raise ToolValidationError(str(error)) from error


def _rejected_params(
    *,
    arguments_mapping: Mapping[str, Any],
    error: Exception,
) -> RejectedToolParams:
    return RejectedToolParams(
        raw_arguments=dict(arguments_mapping),
        error=str(error),
    )


def _ensure_deadline_not_expired(
    *,
    deadline: Deadline | None,
    prompt_name: str,
    tool_name: str,
) -> None:
    if deadline is not None and deadline.remaining() <= timedelta(0):
        raise_tool_deadline_error(
            prompt_name=prompt_name,
            tool_name=tool_name,
            deadline=deadline,
        )


def _invoke_tool_handler(
    *,
    handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
    tool_params: SupportsDataclass | None,
    context: ToolContext,
) -> ToolResult[SupportsToolResult]:
    return handler(tool_params, context=context)


def _log_tool_completion(
    log: StructuredLogger, tool_result: ToolResult[SupportsToolResult]
) -> None:
    log.info(
        "Tool handler completed.",
        event="tool_handler_completed",
        context={
            "success": tool_result.success,
            "has_value": tool_result.value is not None,
        },
    )


def _handle_tool_validation_error(
    *,
    log: StructuredLogger,
    error: Exception,
) -> ToolResult[SupportsToolResult]:
    log.warning(
        "Tool validation failed.",
        event="tool_validation_failed",
        context={"reason": str(error)},
    )
    return ToolResult(
        message=f"Tool validation failed: {error}",
        value=None,
        success=False,
    )


def _handle_tool_deadline_error(
    *,
    error: DeadlineExceededError,
    prompt_name: str,
    tool_name: str,
    deadline: Deadline | None,
) -> PromptEvaluationError:
    return PromptEvaluationError(
        str(error) or f"Tool '{tool_name}' exceeded the deadline.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_TOOL,
        provider_payload=deadline_provider_payload(deadline),
    )


def _handle_type_error(
    *,
    log: StructuredLogger,
    tool_name: str,
    error: TypeError,
) -> ToolResult[SupportsToolResult]:
    """Handle TypeError during tool execution.

    TypeErrors may indicate handler signature mismatches (which pyright catches
    at development time) or other type-related issues within the handler logic.
    """
    log.error(
        "Tool raised TypeError.",
        event="tool_type_error",
        context={"error": str(error)},
    )
    return ToolResult(
        message=f"Tool '{tool_name}' raised TypeError: {error}",
        value=None,
        success=False,
    )


def _handle_unexpected_tool_error(
    *,
    log: StructuredLogger,
    tool_name: str,
    provider_payload: dict[str, Any] | None,
    error: Exception,
) -> ToolResult[SupportsToolResult]:
    log.exception(
        "Tool handler raised an unexpected exception.",
        event="tool_handler_exception",
        context={"provider_payload": provider_payload},
    )
    return ToolResult(
        message=f"Tool '{tool_name}' execution failed: {error}",
        value=None,
        success=False,
    )


def _restore_snapshot_if_needed(
    context: ToolExecutionContext,
    snapshot: CompositeSnapshot,
    log: StructuredLogger,
    *,
    reason: str,
) -> None:
    """Restore from snapshot."""
    restore_snapshot(context.session, context.prompt.resources, snapshot)
    log.debug(
        f"State restored after {reason}.",
        event=f"tool.{reason}_restore",
    )


def _execute_tool_handler(
    *,
    context: ToolExecutionContext,
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
    tool_name: str,
    tool_params: SupportsDataclass | None,
) -> ToolResult[SupportsToolResult]:
    """Execute the tool handler and build the tool context."""
    _ensure_deadline_not_expired(
        deadline=context.deadline,
        prompt_name=context.prompt_name,
        tool_name=tool_name,
    )
    # Resources are accessed via prompt.resources (through ToolContext.resources property)
    tool_context = ToolContext(
        prompt=cast(PromptProtocol[Any], context.prompt),
        rendered_prompt=context.rendered_prompt,
        adapter=cast(ProviderAdapterProtocol[Any], context.adapter),
        session=context.session,
        deadline=context.deadline,
    )
    return _invoke_tool_handler(
        handler=handler,
        tool_params=tool_params,
        context=tool_context,
    )


def _handle_tool_exception(  # noqa: PLR0913
    error: Exception,
    *,
    context: ToolExecutionContext,
    tool_name: str,
    log: StructuredLogger,
    snapshot: CompositeSnapshot,
    tool_params: SupportsDataclass | None,
    arguments_mapping: Mapping[str, Any],
) -> ToolResult[SupportsToolResult]:
    """Handle exceptions during tool execution, restoring snapshot as needed."""
    if isinstance(error, ToolValidationError):
        # Validation errors happen before tool invocation, no restore needed
        return _handle_tool_validation_error(log=log, error=error)
    if isinstance(error, DeadlineExceededError):
        # Context manager handles restore for re-raised exceptions
        raise _handle_tool_deadline_error(
            error=error,
            prompt_name=context.prompt_name,
            tool_name=tool_name,
            deadline=context.deadline,
        ) from error
    # Restore snapshot for all other exceptions since we're catching and returning
    _restore_snapshot_if_needed(context, snapshot, log, reason="exception")
    # TypeError may indicate signature mismatch or other type issues
    if isinstance(error, TypeError):
        return _handle_type_error(
            log=log,
            tool_name=tool_name,
            error=error,
        )
    return _handle_unexpected_tool_error(
        log=log,
        tool_name=tool_name,
        provider_payload=context.provider_payload,
        error=error,
    )


def _execute_tool_with_snapshot(  # noqa: PLR0913
    *,
    context: ToolExecutionContext,
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
    tool_name: str,
    arguments_mapping: Mapping[str, Any],
    call_id: str | None,
    log: StructuredLogger,
    snapshot: CompositeSnapshot,
) -> Iterator[ToolExecutionOutcome]:
    """Execute tool with transactional snapshot restore on failure."""
    tool_params: SupportsDataclass | None = None
    tool_result: ToolResult[SupportsToolResult]
    try:
        tool_params = parse_tool_params(tool=tool, arguments_mapping=arguments_mapping)
        tool_result = _execute_tool_handler(
            context=context,
            tool=tool,
            handler=handler,
            tool_name=tool_name,
            tool_params=tool_params,
        )
    except (VisibilityExpansionRequired, PromptEvaluationError):
        # Context manager handles restore; just re-raise
        raise
    except Exception as error:
        if tool_params is None:
            tool_params = cast(
                SupportsDataclass,
                _rejected_params(arguments_mapping=arguments_mapping, error=error),
            )
        tool_result = _handle_tool_exception(
            error,
            context=context,
            tool_name=tool_name,
            log=log,
            snapshot=snapshot,
            tool_params=tool_params,
            arguments_mapping=arguments_mapping,
        )
    else:
        # Manually restore if tool execution reported failure
        if not tool_result.success:
            _restore_snapshot_if_needed(context, snapshot, log, reason="tool_failure")
        _log_tool_completion(log, tool_result)

    # Defensive check: None is valid for parameterless tools (params_type is type(None))
    if tool_params is None and tool.params_type is not type(None):  # pragma: no cover
        raise RuntimeError("Tool parameters were not parsed.")

    yield ToolExecutionOutcome(
        tool=tool,
        params=tool_params,
        result=tool_result,
        call_id=call_id,
        log=log,
        snapshot=snapshot,
    )


@contextmanager
def tool_execution(
    *,
    context: ToolExecutionContext,
    tool_call: ProviderToolCall,
) -> Iterator[ToolExecutionOutcome]:
    """Context manager that executes a tool call and standardizes logging.

    Uses transactional semantics to ensure both session state and
    resources are rolled back on tool failure.
    """
    tool, handler = _resolve_tool_and_handler(
        tool_call=tool_call,
        tool_registry=context.tool_registry,
        prompt_name=context.prompt_name,
        provider_payload=context.provider_payload,
    )
    tool_name = tool.name
    arguments_mapping = _parse_tool_call_arguments(
        tool_call=tool_call,
        prompt_name=context.prompt_name,
        provider_payload=context.provider_payload,
        parse_arguments=context.parse_arguments,
    )

    call_id, log = _build_tool_logger(
        context=context,
        tool_name=tool_name,
        tool_call=tool_call,
    )

    # Use transactional execution
    with tool_transaction(
        context.session, context.prompt.resources, tag=f"tool:{tool_name}"
    ) as snapshot:
        yield from _execute_tool_with_snapshot(
            context=context,
            tool=tool,
            handler=handler,
            tool_name=tool_name,
            arguments_mapping=arguments_mapping,
            call_id=call_id,
            log=log,
            snapshot=snapshot,
        )


def dispatch_tool_invocation(
    *,
    context: ToolExecutionContext,
    outcome: ToolExecutionOutcome,
) -> ToolInvoked:
    """Send a tool invocation event to the session dispatcher."""
    session_id = getattr(context.session, "session_id", None)
    rendered_output = outcome.result.render()
    usage = token_usage_from_payload(context.provider_payload)

    invocation = ToolInvoked(
        prompt_name=context.prompt_name,
        adapter=context.adapter_name,
        name=outcome.tool.name,
        params=outcome.params,
        result=cast(ToolResult[object], outcome.result),
        session_id=session_id,
        created_at=datetime.now(UTC),
        usage=usage,
        rendered_output=rendered_output,
        call_id=outcome.call_id,
        event_id=uuid4(),
    )
    dispatch_result = context.session.dispatcher.dispatch(invocation)
    if not dispatch_result.ok:
        # Restore to pre-tool state if tool succeeded (not already restored)
        if outcome.result.success:
            _restore_snapshot_if_needed(
                context, outcome.snapshot, outcome.log, reason="dispatch_failure"
            )
        outcome.log.warning(
            "State rollback triggered after dispatch failure.",
            event="state_rollback_due_to_dispatch_failure",
        )
        failure_handlers = [
            getattr(failure.handler, "__qualname__", repr(failure.handler))
            for failure in dispatch_result.errors
        ]
        outcome.log.error(
            "Tool event dispatch failed.",
            event="tool_event_dispatch_failed",
            context={
                "failure_count": len(dispatch_result.errors),
                "failed_handlers": failure_handlers,
            },
        )
        outcome.result.message = context.format_dispatch_failures(
            dispatch_result.errors
        )
    else:
        outcome.log.debug(
            "Tool event dispatched.",
            event="tool_event_dispatched",
            context={"handler_count": dispatch_result.handled_count},
        )
    return invocation


def execute_tool_call(
    *,
    context: ToolExecutionContext,
    tool_call: ProviderToolCall,
) -> tuple[ToolInvoked, ToolResult[SupportsToolResult]]:
    """Execute a provider tool call and dispatch the resulting event."""

    with tool_execution(
        context=context,
        tool_call=tool_call,
    ) as outcome:
        invocation = dispatch_tool_invocation(
            context=context,
            outcome=outcome,
        )
    return invocation, outcome.result


@dataclass(slots=True)
class ToolExecutor:
    """Handles execution of tool calls and event dispatching.

    Provides unified access to session and prompt resources for transactional
    tool execution.
    """

    adapter_name: AdapterName
    adapter: ProviderAdapter[Any]
    prompt: Prompt[Any]
    prompt_name: str
    rendered: RenderedPrompt[Any]
    session: SessionProtocol
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]
    serialize_tool_message_fn: ToolMessageSerializer
    format_dispatch_failures: Callable[[Sequence[HandlerFailure]], str]
    parse_arguments: ToolArgumentsParser
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None
    _log: StructuredLogger = field(init=False)
    _context: ToolExecutionContext = field(init=False)
    _tool_message_records: list[
        tuple[ToolResult[SupportsToolResult], dict[str, Any]]
    ] = field(init=False)

    def __post_init__(self) -> None:
        self._log = (self.logger_override or logger).bind(
            adapter=self.adapter_name,
            prompt=self.prompt_name,
        )
        self._context = ToolExecutionContext(
            adapter_name=self.adapter_name,
            adapter=self.adapter,
            prompt=self.prompt,
            rendered_prompt=self.rendered,
            tool_registry=self.tool_registry,
            session=self.session,
            prompt_name=self.prompt_name,
            parse_arguments=self.parse_arguments,
            format_dispatch_failures=self.format_dispatch_failures,
            deadline=self.deadline,
            logger_override=self.logger_override,
            budget_tracker=self.budget_tracker,
        )
        self._tool_message_records = []

    def execute(
        self,
        tool_calls: Sequence[ProviderToolCall],
        provider_payload: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], ToolChoice]:
        """Execute tool calls and return resulting messages and next tool choice."""
        messages: list[dict[str, Any]] = []
        next_tool_choice: ToolChoice = "auto"

        self._log.debug(
            "Processing tool calls.",
            event="prompt_tool_calls_detected",
            context={"count": len(tool_calls)},
        )

        execution_context = self._context.with_provider_payload(provider_payload)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if self.deadline is not None and self.deadline.remaining() <= timedelta(0):
                raise_tool_deadline_error(
                    prompt_name=self.prompt_name,
                    tool_name=tool_name,
                    deadline=self.deadline,
                )
            with tool_execution(
                context=execution_context,
                tool_call=tool_call,
            ) as outcome:
                _ = dispatch_tool_invocation(
                    context=execution_context,
                    outcome=outcome,
                )

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": self.serialize_tool_message_fn(outcome.result),
            }
            messages.append(tool_message)
            self._tool_message_records.append((outcome.result, tool_message))

        return messages, next_tool_choice

    @property
    def tool_message_records(
        self,
    ) -> list[tuple[ToolResult[SupportsToolResult], dict[str, Any]]]:
        return self._tool_message_records


__all__ = [
    "ToolExecutionContext",
    "ToolExecutionOutcome",
    "ToolExecutor",
    "ToolMessageSerializer",
    "dispatch_tool_invocation",
    "execute_tool_call",
    "tool_execution",
]
