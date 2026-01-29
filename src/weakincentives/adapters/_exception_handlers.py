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

"""Chain of Responsibility pattern for tool exception handling.

This module provides a pluggable exception handling mechanism for tool execution.
Each handler decides whether it can process a given exception type and either
returns a ToolResult or re-raises the exception.

Example usage::

    from weakincentives.adapters._exception_handlers import (
        DEFAULT_EXCEPTION_CHAIN,
        ExceptionContext,
    )

    ctx = ExceptionContext(
        error=some_error,
        tool_name="my_tool",
        prompt_name="my_prompt",
        deadline=deadline,
        provider_payload={},
        log=logger,
        snapshot=snapshot,
        tool_params=params,
        arguments_mapping=args,
    )
    result = DEFAULT_EXCEPTION_CHAIN.handle(ctx, restore_fn=my_restore_fn)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import DeadlineExceededError, ToolValidationError
from ..prompt.tool import ToolResult
from ..types.dataclass import SupportsDataclass, SupportsToolResult
from .core import PROMPT_EVALUATION_PHASE_TOOL, PromptEvaluationError
from .utilities import deadline_provider_payload

if TYPE_CHECKING:
    from ..runtime.logging import StructuredLogger
    from ..runtime.transactions import CompositeSnapshot


@FrozenDataclass()
class ExceptionContext:
    """Context for exception handling.

    Contains all information needed by exception handlers to process an error
    and decide on the appropriate response strategy.

    Attributes:
        error: The exception that occurred during tool execution.
        tool_name: Name of the tool that raised the exception.
        prompt_name: Name of the prompt being evaluated.
        deadline: Optional deadline for the tool execution.
        provider_payload: Provider-specific metadata from the LLM response.
        log: Structured logger bound to the current tool context.
        snapshot: Composite snapshot for potential rollback.
        tool_params: Parsed tool parameters, or None if parsing failed.
        arguments_mapping: Raw arguments as provided by the LLM.
    """

    error: Exception = field(compare=False)
    tool_name: str
    prompt_name: str
    deadline: Deadline | None = field(compare=False)
    provider_payload: Mapping[str, Any] = field(compare=False)
    log: StructuredLogger = field(compare=False, repr=False)
    snapshot: CompositeSnapshot = field(compare=False, repr=False)
    tool_params: SupportsDataclass | None = field(compare=False)
    arguments_mapping: Mapping[str, Any] = field(compare=False)


class ExceptionHandler(Protocol):
    """Handler for a specific exception type.

    Implementations decide whether they can process a given exception and
    either return a ToolResult or an Exception to be re-raised.
    """

    def can_handle(self, error: Exception) -> bool:
        """Return True if this handler can process the error."""
        ...

    def handle(
        self, ctx: ExceptionContext
    ) -> ToolResult[SupportsToolResult] | Exception:
        """Handle the error.

        Returns:
            ToolResult to return to caller, or Exception to propagate.
        """
        ...


@FrozenDataclass()
class ValidationErrorHandler:
    """Handles ToolValidationError - no snapshot restore needed.

    Validation errors occur before tool invocation, so no state changes
    need to be rolled back.
    """

    def can_handle(self, error: Exception) -> bool:  # noqa: PLR6301
        """Return True for ToolValidationError instances."""
        return isinstance(error, ToolValidationError)

    def handle(  # noqa: PLR6301
        self, ctx: ExceptionContext
    ) -> ToolResult[SupportsToolResult]:
        """Log the validation failure and return an error result."""
        ctx.log.warning(
            "Tool validation failed.",
            event="tool_validation_failed",
            context={"reason": str(ctx.error)},
        )
        return ToolResult(
            message=f"Tool validation failed: {ctx.error}",
            value=None,
            success=False,
        )


@FrozenDataclass()
class DeadlineErrorHandler:
    """Handles DeadlineExceededError - re-raises for context manager.

    Deadline errors propagate up so the transaction context manager can
    perform the rollback. Returns an exception to signal re-raising.
    """

    def can_handle(self, error: Exception) -> bool:  # noqa: PLR6301
        """Return True for DeadlineExceededError instances."""
        return isinstance(error, DeadlineExceededError)

    def handle(self, ctx: ExceptionContext) -> Exception:  # noqa: PLR6301
        """Convert to PromptEvaluationError for the caller to raise."""
        error = cast(DeadlineExceededError, ctx.error)
        return PromptEvaluationError(
            str(error) or f"Tool '{ctx.tool_name}' exceeded the deadline.",
            prompt_name=ctx.prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=deadline_provider_payload(ctx.deadline),
        )


@FrozenDataclass()
class TypeErrorHandler:
    """Handles TypeError - indicates signature mismatch or type issues.

    TypeErrors may indicate handler signature mismatches (which pyright catches
    at development time) or other type-related issues within the handler logic.
    Requires snapshot restoration since tool execution may have partially completed.
    """

    requires_restore: bool = True
    """Whether this handler requires snapshot restoration before returning."""

    def can_handle(self, error: Exception) -> bool:  # noqa: PLR6301
        """Return True for TypeError instances."""
        return isinstance(error, TypeError)

    def handle(  # noqa: PLR6301
        self, ctx: ExceptionContext
    ) -> ToolResult[SupportsToolResult]:
        """Log the type error and return an error result."""
        ctx.log.error(
            "Tool raised TypeError.",
            event="tool_type_error",
            context={"error": str(ctx.error)},
        )
        return ToolResult(
            message=f"Tool '{ctx.tool_name}' raised TypeError: {ctx.error}",
            value=None,
            success=False,
        )


@FrozenDataclass()
class UnexpectedErrorHandler:
    """Handles any unexpected exception as a fallback.

    This handler catches all exceptions not handled by more specific handlers
    and returns a generic error result. Always requires snapshot restoration.
    """

    requires_restore: bool = True
    """Whether this handler requires snapshot restoration before returning."""

    def can_handle(self, error: Exception) -> bool:  # noqa: PLR6301
        """Return True for any exception (fallback handler)."""
        return True

    def handle(  # noqa: PLR6301
        self, ctx: ExceptionContext
    ) -> ToolResult[SupportsToolResult]:
        """Log the unexpected exception and return an error result."""
        ctx.log.exception(
            "Tool handler raised an unexpected exception.",
            event="tool_handler_exception",
            context={"provider_payload": dict(ctx.provider_payload)},
        )
        return ToolResult(
            message=f"Tool '{ctx.tool_name}' execution failed: {ctx.error}",
            value=None,
            success=False,
        )


type RestoreFunction = Callable[[str], None]
"""Signature for snapshot restoration callback: restore_fn(reason: str) -> None."""


class ExceptionHandlerChain:
    """Chain of exception handlers with ordered dispatch.

    Handlers are checked in order and the first matching handler processes
    the exception. If a handler returns an Exception, it is re-raised.
    If a handler returns a ToolResult, snapshot restoration is performed
    if the handler requires it.

    Example::

        chain = ExceptionHandlerChain([
            ValidationErrorHandler(),
            DeadlineErrorHandler(),
            TypeErrorHandler(),
            UnexpectedErrorHandler(),  # Fallback
        ])
        result = chain.handle(ctx, restore_fn=my_restore_fn)
    """

    __slots__ = ("_handlers",)

    def __init__(self, handlers: Sequence[ExceptionHandler]) -> None:
        """Initialize the chain with an ordered sequence of handlers.

        Args:
            handlers: Sequence of handlers to check in order. Should include
                a fallback handler that accepts all exceptions.
        """
        super().__init__()
        self._handlers = tuple(handlers)

    @property
    def handlers(self) -> tuple[ExceptionHandler, ...]:
        """Return the ordered tuple of handlers in this chain."""
        return self._handlers

    def handle(
        self,
        ctx: ExceptionContext,
        restore_fn: RestoreFunction,
    ) -> ToolResult[SupportsToolResult]:
        """Process an exception through the handler chain.

        Args:
            ctx: Exception context containing the error and related state.
            restore_fn: Callback to restore from snapshot. Called with a reason
                string if the handler requires restoration.

        Returns:
            ToolResult from the matching handler.

        Raises:
            Exception: If a handler returns an Exception to propagate.
        """
        for handler in self._handlers:
            if handler.can_handle(ctx.error):
                # Check if handler requires snapshot restoration
                requires_restore = getattr(handler, "requires_restore", False)
                if requires_restore:
                    restore_fn("exception")

                result = handler.handle(ctx)
                if isinstance(result, Exception):
                    raise result from ctx.error
                return result

        # Should not reach here if chain includes a fallback handler
        restore_fn("exception")  # pragma: no cover
        return ToolResult(  # pragma: no cover
            message=f"Unhandled exception: {ctx.error}",
            value=None,
            success=False,
        )


# Default chain used by tool executor
DEFAULT_EXCEPTION_CHAIN = ExceptionHandlerChain(
    [
        ValidationErrorHandler(),
        DeadlineErrorHandler(),
        TypeErrorHandler(),
        UnexpectedErrorHandler(),
    ]
)


__all__ = [
    "DEFAULT_EXCEPTION_CHAIN",
    "DeadlineErrorHandler",
    "ExceptionContext",
    "ExceptionHandler",
    "ExceptionHandlerChain",
    "RestoreFunction",
    "TypeErrorHandler",
    "UnexpectedErrorHandler",
    "ValidationErrorHandler",
]
