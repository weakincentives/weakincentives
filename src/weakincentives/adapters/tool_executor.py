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
from ..prompt.feedback import collect_feedback
from ..prompt.policy import PolicyDecision, ToolPolicy
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from ..prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ..runtime.events import HandlerFailure, ToolInvoked
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.run_context import RunContext
from ..runtime.transactions import (
    CompositeSnapshot,
    restore_snapshot,
    tool_transaction,
)
from ..serde import parse
from ..types import AdapterName
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
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
    from ..runtime.watchdog import Heartbeat
    from .core import ProviderAdapter


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.tool_executor"}
)


class ToolMessageSerializer(Protocol):
    """Protocol for serializing tool results into provider-specific message formats.

    Implementations convert a ToolResult into the format expected by a specific
    provider's API (e.g., OpenAI, Anthropic). The optional payload parameter
    allows passing additional provider-specific context.

    Args:
        result: The tool execution result to serialize.
        payload: Optional provider-specific payload for serialization context.

    Returns:
        A provider-specific message object (typically a dict or string).
    """

    def __call__(
        self,
        result: ToolResult[SupportsToolResult],
        *,
        payload: object | None = ...,
    ) -> object: ...


@FrozenDataclass()
class RejectedToolParams:
    """Represents failed tool parameter validation.

    Created when provider-supplied arguments cannot be parsed or validated
    against the tool's expected parameter schema. This preserves the raw
    arguments for debugging and error reporting while capturing the
    validation error message.

    Attributes:
        raw_arguments: The original unparsed arguments from the provider.
        error: Human-readable description of the validation failure.
    """

    raw_arguments: dict[str, Any]
    error: str


@FrozenDataclass()
class ToolExecutionOutcome:
    """Complete result of a tool handler execution.

    Captures all information needed for post-execution processing, including
    event dispatch, state restoration on failure, and logging correlation.
    Yielded by the ``tool_execution`` context manager.

    Attributes:
        tool: The tool definition that was executed.
        params: Parsed parameters passed to the handler, or RejectedToolParams
            on validation failure. None only for parameterless tools.
        result: The ToolResult returned by the handler (success or failure).
        call_id: Provider-assigned identifier for this tool call, used to
            correlate tool results with requests. May be None for some providers.
        log: Logger with bound context (adapter, prompt, tool, call_id) for
            correlated logging.
        snapshot: Pre-execution state snapshot for rollback on failure.
    """

    tool: Tool[SupportsDataclassOrNone, SupportsToolResult]
    params: SupportsDataclass | None
    result: ToolResult[SupportsToolResult]
    call_id: str | None
    log: StructuredLogger
    snapshot: CompositeSnapshot


@dataclass(slots=True)
class ToolExecutionContext:
    """Configuration and dependencies for executing provider tool calls.

    Groups all inputs needed to execute tools within a transactional context,
    including session state, prompt resources, and execution policies. Used
    by ``tool_execution`` and ``ToolExecutor`` to maintain consistent behavior.

    When ``heartbeat`` is provided, beats occur before and after each tool
    execution to prove liveness to external watchdogs.

    Attributes:
        adapter_name: Identifier for the provider adapter (e.g., "openai").
        adapter: The provider adapter instance handling LLM communication.
        prompt: The prompt template being evaluated.
        rendered_prompt: Pre-rendered prompt content, or None if not yet rendered.
        tool_registry: Mapping of tool names to Tool definitions.
        session: Session managing state and event dispatch.
        prompt_name: Human-readable prompt identifier for logging/errors.
        parse_arguments: Callable to parse raw argument strings from the provider.
        format_dispatch_failures: Callable to format handler failures into messages.
        deadline: Optional execution deadline for timeout enforcement.
        provider_payload: Raw response from the provider for the current turn.
        logger_override: Custom logger; if None, uses module-level logger.
        budget_tracker: Optional tracker for token/cost budgets.
        heartbeat: Optional watchdog heartbeat for liveness proofs.
        run_context: Optional context for distributed tracing/correlation.
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
    heartbeat: Heartbeat | None = None
    run_context: RunContext | None = None

    def with_provider_payload(
        self, provider_payload: dict[str, Any] | None
    ) -> ToolExecutionContext:
        """Create a new context with an updated provider payload.

        Returns a shallow copy of this context with the provider_payload
        replaced. All other fields remain unchanged.

        Args:
            provider_payload: New provider response payload, or None.

        Returns:
            A new ToolExecutionContext instance with the updated payload.
        """
        from dataclasses import replace

        return replace(self, provider_payload=provider_payload)

    def beat(self) -> None:
        """Record a heartbeat to prove liveness.

        Signals the watchdog that execution is progressing. Safe to call
        even when no heartbeat is configured (no-op in that case).
        """
        if self.heartbeat is not None:
            self.heartbeat.beat()


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
    """Parse and validate provider arguments into typed tool parameters.

    Converts raw argument mappings from the provider into the tool's
    expected parameter dataclass. Uses strict parsing that rejects
    unknown fields (extra="forbid").

    Args:
        tool: The tool definition containing the expected params_type.
        arguments_mapping: Raw arguments from the provider's tool call.

    Returns:
        Parsed parameter dataclass instance, or None for parameterless tools
        (where params_type is type(None)).

    Raises:
        ToolValidationError: If arguments are provided for a parameterless tool,
            or if parsing/validation fails against the params_type schema.

    Example:
        >>> tool = Tool(name="greet", params_type=GreetParams, ...)
        >>> params = parse_tool_params(tool=tool, arguments_mapping={"name": "World"})
        >>> assert isinstance(params, GreetParams)
    """
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


def _check_policies(
    *,
    policies: tuple[ToolPolicy, ...],
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    tool_params: SupportsDataclass | None,
    context: ToolContext,
    log: StructuredLogger,
) -> PolicyDecision:
    """Check all policies and return the first denial or allow."""
    for policy in policies:
        decision = policy.check(tool, tool_params, context=context)
        if not decision.allowed:
            log.info(
                "Policy denied tool execution.",
                event="policy_denied",
                context={
                    "policy": policy.name,
                    "reason": decision.reason,
                },
            )
            return decision
    return PolicyDecision.allow()


def _notify_policies_of_result(
    *,
    policies: tuple[ToolPolicy, ...],
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    tool_params: SupportsDataclass | None,
    result: ToolResult[SupportsToolResult],
    context: ToolContext,
) -> None:
    """Notify all policies of a successful tool execution."""
    if not result.success:
        return
    for policy in policies:
        policy.on_result(tool, tool_params, result, context=context)


def _invoke_tool_handler(
    *,
    handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
    tool_params: SupportsDataclass | None,
    context: ToolContext,
) -> ToolResult[SupportsToolResult]:
    return handler(tool_params, context=context)


def _log_tool_completion(
    log: StructuredLogger,
    tool_result: ToolResult[SupportsToolResult],
) -> None:
    log.info(
        "Tool handler completed.",
        event="tool_handler_completed",
        context={
            "success": tool_result.success,
            "has_value": tool_result.value is not None,
        },
    )
    # Log full result details at DEBUG level (using bound logger for correlation)
    log.debug(
        "tool.execution.complete",
        event="tool.execution.complete",
        context={
            "success": tool_result.success,
            "message": tool_result.message,
            "value": str(tool_result.value) if tool_result.value is not None else None,
            "value_type": (
                type(tool_result.value).__qualname__
                if tool_result.value is not None
                else None
            ),
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
    restore_snapshot(context.session, context.prompt.resources.context, snapshot)
    log.debug(
        f"State restored after {reason}.",
        event=f"tool.{reason}_restore",
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


def _build_policy_denied_result(
    decision: PolicyDecision,
) -> ToolResult[SupportsToolResult]:
    """Build a tool result for a policy-denied invocation."""
    return ToolResult(
        message=decision.reason or "Policy denied",
        value=None,
        success=False,
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

    # Build ToolContext once for policy checking and handler execution
    tool_context = ToolContext(
        prompt=cast(PromptProtocol[Any], context.prompt),
        rendered_prompt=context.rendered_prompt,
        adapter=cast(ProviderAdapterProtocol[Any], context.adapter),
        session=context.session,
        deadline=context.deadline,
        heartbeat=context.heartbeat,
        run_context=context.run_context,
    )

    # Get policies for this tool
    policies = context.prompt.policies_for_tool(tool_name)

    try:
        tool_params = parse_tool_params(tool=tool, arguments_mapping=arguments_mapping)

        # Check policies before executing handler
        decision = _check_policies(
            policies=policies,
            tool=tool,
            tool_params=tool_params,
            context=tool_context,
            log=log,
        )
        if (
            not decision.allowed
        ):  # pragma: no cover - integration path tested via helpers
            tool_result = _build_policy_denied_result(decision)
            _restore_snapshot_if_needed(context, snapshot, log, reason="policy_denied")
            yield ToolExecutionOutcome(
                tool=tool,
                params=tool_params,
                result=tool_result,
                call_id=call_id,
                log=log,
                snapshot=snapshot,
            )
            return

        # Execute handler
        _ensure_deadline_not_expired(
            deadline=context.deadline,
            prompt_name=context.prompt_name,
            tool_name=tool_name,
        )
        tool_result = _invoke_tool_handler(
            handler=handler,
            tool_params=tool_params,
            context=tool_context,
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
        else:
            # Notify policies of successful execution
            _notify_policies_of_result(
                policies=policies,
                tool=tool,
                tool_params=tool_params,
                result=tool_result,
                context=tool_context,
            )
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
    """Execute a tool call with transactional state management.

    Context manager that resolves the tool from the registry, parses arguments,
    executes the handler, and yields the outcome. Provides automatic rollback
    of session state and prompt resources if the tool fails or raises an
    exception.

    Args:
        context: Execution context with session, prompt, and tool registry.
        tool_call: Provider-formatted tool call with function name and arguments.

    Yields:
        ToolExecutionOutcome containing the tool result and execution metadata.
        The outcome is yielded once before the context manager exits.

    Raises:
        PromptEvaluationError: If the tool is not found in the registry, has
            no handler, or a deadline is exceeded.
        VisibilityExpansionRequired: If the tool requires expanding prompt
            visibility (re-raised without rollback).

    Note:
        - Heartbeat beats occur before and after tool execution when available.
        - Tool policies are checked before handler execution; denied tools
          return a failure result without invoking the handler.
        - On handler failure, the pre-execution snapshot is restored before
          the outcome is yielded.

    Example:
        >>> with tool_execution(context=ctx, tool_call=call) as outcome:
        ...     if outcome.result.success:
        ...         dispatch_tool_invocation(context=ctx, outcome=outcome)
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

    # Log tool execution start with full arguments (using bound logger for correlation)
    log.debug(
        "tool.execution.start",
        event="tool.execution.start",
        context={
            "arguments": dict(arguments_mapping),
        },
    )

    # Beat before tool execution
    context.beat()

    # Use transactional execution
    with tool_transaction(
        context.session, context.prompt.resources.context, tag=f"tool:{tool_name}"
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

    # Beat after tool execution
    context.beat()


def dispatch_tool_invocation(
    *,
    context: ToolExecutionContext,
    outcome: ToolExecutionOutcome,
) -> ToolInvoked:
    """Dispatch a ToolInvoked event to session handlers.

    Creates a ToolInvoked event from the execution outcome and dispatches it
    through the session's event system. If any handler fails, the session
    state is rolled back to the pre-tool snapshot and the result message is
    updated with failure information.

    Args:
        context: Execution context containing the session dispatcher.
        outcome: Completed tool execution outcome from ``tool_execution``.

    Returns:
        The ToolInvoked event that was dispatched (useful for logging or
        correlation, regardless of handler success).

    Note:
        - If dispatch fails and the tool succeeded, state is rolled back.
        - The outcome's result.message may be mutated to include dispatch
          failure information.
        - Token usage is extracted from the provider payload if available.
    """
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
        run_context=context.run_context,
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


def _append_feedback_to_result(
    result: ToolResult[SupportsToolResult],
    feedback_text: str | None,
) -> ToolResult[SupportsToolResult]:
    """Append feedback text to tool result message.

    Feedback is always delivered when present, even if the tool message is empty.
    This ensures feedback providers work for tools that return meaningful output
    in their value/rendered payload rather than the message field.

    Args:
        result: The original tool result.
        feedback_text: Feedback text to append, or None.

    Returns:
        Updated tool result with feedback appended, or original if no feedback.
    """
    if not feedback_text:
        return result

    from dataclasses import replace

    if result.message:
        return replace(result, message=f"{result.message}\n\n{feedback_text}")
    return replace(result, message=feedback_text)


def execute_tool_call(
    *,
    context: ToolExecutionContext,
    tool_call: ProviderToolCall,
) -> tuple[ToolInvoked, ToolResult[SupportsToolResult]]:
    """Execute a tool call, dispatch the event, and collect feedback.

    Combines tool execution with event dispatch and feedback collection into
    a single high-level operation. This is the primary entry point for
    executing individual tool calls when not using the batched ToolExecutor.

    Args:
        context: Execution context with session, prompt, and tool registry.
        tool_call: Provider-formatted tool call to execute.

    Returns:
        A tuple of:
        - ToolInvoked: The dispatched event for logging/correlation.
        - ToolResult: The tool result with any feedback appended to the message.

    Raises:
        PromptEvaluationError: If the tool is not found, has no handler, or
            a deadline is exceeded.
        VisibilityExpansionRequired: If the tool requires visibility expansion.

    Note:
        Feedback providers registered on the prompt are run after tool
        completion, and their output is appended to the result message.
    """
    with tool_execution(
        context=context,
        tool_call=tool_call,
    ) as outcome:
        invocation = dispatch_tool_invocation(
            context=context,
            outcome=outcome,
        )

    # Run feedback providers after tool completion
    feedback_text = collect_feedback(
        prompt=cast(PromptProtocol[Any], context.prompt),
        session=context.session,
        deadline=context.deadline,
    )

    tool_result = _append_feedback_to_result(outcome.result, feedback_text)
    return invocation, tool_result


@dataclass(slots=True)
class ToolExecutor:
    """Batch executor for provider tool calls with event dispatch.

    Manages execution of multiple tool calls from a single LLM response turn,
    handling transactional state management, event dispatch, feedback collection,
    and message serialization. Tracks execution records for debugging and audit.

    When ``heartbeat`` is provided, beats occur before and after each tool
    execution to prove liveness. Tool handlers receive the heartbeat via
    ``ToolContext.beat()`` for additional beats during long-running operations.

    Attributes:
        adapter_name: Provider adapter identifier for logging.
        adapter: The provider adapter handling LLM communication.
        prompt: The prompt template being evaluated.
        prompt_name: Human-readable prompt identifier.
        rendered: Pre-rendered prompt content.
        session: Session managing state and event dispatch.
        tool_registry: Mapping of tool names to Tool definitions.
        serialize_tool_message_fn: Serializer for converting results to messages.
        format_dispatch_failures: Formatter for handler failure messages.
        parse_arguments: Parser for raw provider arguments.
        logger_override: Custom logger; if None, uses module-level logger.
        deadline: Optional execution deadline.
        budget_tracker: Optional token/cost budget tracker.
        heartbeat: Optional watchdog heartbeat.
        run_context: Optional distributed tracing context.

    Example:
        >>> executor = ToolExecutor(
        ...     adapter_name="openai",
        ...     adapter=adapter,
        ...     prompt=prompt,
        ...     # ... other required fields
        ... )
        >>> messages, choice = executor.execute(tool_calls, payload)
        >>> # messages ready to send back to provider
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
    heartbeat: Heartbeat | None = None
    run_context: RunContext | None = None
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
            heartbeat=self.heartbeat,
            run_context=self.run_context,
        )
        self._tool_message_records = []

    def execute(
        self,
        tool_calls: Sequence[ProviderToolCall],
        provider_payload: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], ToolChoice]:
        """Execute a batch of tool calls and return serialized result messages.

        Processes each tool call sequentially: executes the handler, dispatches
        the ToolInvoked event, collects feedback, and serializes the result
        into a provider message format. Results are recorded in
        ``tool_message_records`` for inspection.

        Args:
            tool_calls: Sequence of provider-formatted tool calls to execute.
            provider_payload: Raw provider response for the current turn,
                used for token usage extraction and error context.

        Returns:
            A tuple of:
            - List of serialized tool result messages (role="tool" format).
            - ToolChoice for the next turn (currently always "auto").

        Raises:
            PromptEvaluationError: If a tool is not found, has no handler, or
                a deadline is exceeded before/during execution.

        Note:
            - Tool calls are executed sequentially in order.
            - Each result message includes the tool_call_id for correlation.
            - Feedback from providers is appended to result messages.
            - Records are available via ``tool_message_records`` property.
        """
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

            # Run feedback providers after tool completion
            feedback_text = collect_feedback(
                prompt=cast(PromptProtocol[Any], self.prompt),
                session=self.session,
                deadline=self.deadline,
            )
            tool_result = _append_feedback_to_result(outcome.result, feedback_text)

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": self.serialize_tool_message_fn(tool_result),
            }
            messages.append(tool_message)
            self._tool_message_records.append((tool_result, tool_message))

        return messages, next_tool_choice

    @property
    def tool_message_records(
        self,
    ) -> list[tuple[ToolResult[SupportsToolResult], dict[str, Any]]]:
        """Access records of executed tool calls and their serialized messages.

        Returns a list of (result, message) tuples for each tool call processed
        by ``execute()``. Useful for debugging, testing, or audit logging.

        Returns:
            List of tuples where each tuple contains:
            - ToolResult: The original tool result (with feedback appended).
            - dict: The serialized message sent to the provider.
        """
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
