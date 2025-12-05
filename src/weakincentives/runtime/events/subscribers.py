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

"""Standard event subscribers for structured logging and observability."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

from ...serde import dump
from ...types import JSONValue
from ..logging import StructuredLogger, get_logger
from ._types import EventBus, EventHandler, TokenUsage

if TYPE_CHECKING:
    from . import PromptExecuted, PromptRendered, ToolInvoked


_DEFAULT_TRUNCATE_LIMIT: Final[int] = 256
_MODULE_LOGGER_NAME: Final[str] = "weakincentives.runtime.events.subscribers"


def _truncate(text: str, *, limit: int) -> str:
    """Truncate text to the specified limit, appending ellipsis if needed."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _coerce_for_json(payload: object) -> JSONValue:
    """Recursively coerce a payload to JSON-serializable types."""
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Mapping):
        return {str(key): _coerce_for_json(value) for key, value in payload.items()}
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        return [_coerce_for_json(item) for item in payload]
    if hasattr(payload, "__dataclass_fields__"):
        return cast(JSONValue, dump(payload, exclude_none=True))
    if isinstance(payload, set):
        return [_coerce_for_json(item) for item in sorted(payload)]
    return str(payload)


def _build_token_usage_context(usage: TokenUsage | None) -> dict[str, JSONValue]:
    """Build a context dict for token usage."""
    if usage is None:
        return {}

    result: dict[str, JSONValue] = {}
    if usage.input_tokens is not None:
        result["input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        result["output_tokens"] = usage.output_tokens
    if usage.cached_tokens is not None:
        result["cached_tokens"] = usage.cached_tokens
    total = usage.total_tokens
    if total is not None:
        result["total_tokens"] = total
    return result


@dataclass(slots=True)
class StandardLoggingSubscribers:
    """Attach standard structured logging for prompt and tool events.

    This helper provides ready-to-use event handlers that log prompt renders,
    tool invocations, and execution summaries using structured logging.

    Example usage::

        from weakincentives.runtime import Session, StandardLoggingSubscribers

        session = Session()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(session.event_bus)

        # ... run your agent ...

        # Optionally detach when done
        subscribers.detach(session.event_bus)

    The class can also be used as a context manager::

        with StandardLoggingSubscribers().attached(session.event_bus):
            # ... run your agent ...

    """

    truncate_limit: int = field(default=_DEFAULT_TRUNCATE_LIMIT)
    """Maximum length for serialized payloads before truncation."""

    logger: StructuredLogger | logging.Logger | None = field(default=None)
    """Logger instance to use. Defaults to module-level StructuredLogger when None."""

    show_rendered_prompt: bool = field(default=True)
    """Whether to log prompt renders."""

    show_tool_invocations: bool = field(default=True)
    """Whether to log tool invocation details."""

    show_execution_summary: bool = field(default=True)
    """Whether to log execution completion summaries."""

    _handlers: dict[type[object], EventHandler] = field(
        default_factory=dict, init=False, repr=False
    )
    _resolved_logger: StructuredLogger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the logger instance after initialization."""
        if self.logger is None:
            self._resolved_logger = get_logger(
                _MODULE_LOGGER_NAME,
                context={"component": "event_subscribers"},
            )
        elif isinstance(self.logger, StructuredLogger):
            self._resolved_logger = self.logger
        else:
            self._resolved_logger = get_logger(
                _MODULE_LOGGER_NAME,
                logger_override=self.logger,
                context={"component": "event_subscribers"},
            )

    def _handle_prompt_rendered(self, event: object) -> None:
        """Handle PromptRendered events."""
        from . import PromptRendered

        prompt_event = cast(PromptRendered, event)
        prompt_label = prompt_event.prompt_name or (
            f"{prompt_event.prompt_ns}:{prompt_event.prompt_key}"
        )
        self._resolved_logger.info(
            "Prompt rendered.",
            event="prompt_rendered",
            context={
                "prompt_name": prompt_label,
                "prompt_ns": prompt_event.prompt_ns,
                "prompt_key": prompt_event.prompt_key,
                "adapter": prompt_event.adapter,
                "rendered_prompt": prompt_event.rendered_prompt,
            },
        )

    def _handle_tool_invoked(self, event: object) -> None:
        """Handle ToolInvoked events."""
        from . import ToolInvoked

        tool_event = cast(ToolInvoked, event)

        # Coerce params for JSON serialization
        params_data = _coerce_for_json(dump(tool_event.params, exclude_none=True))

        # Build result context
        result_message = _truncate(
            tool_event.result.message or "", limit=self.truncate_limit
        )

        context: dict[str, JSONValue] = {
            "tool_name": tool_event.name,
            "prompt_name": tool_event.prompt_name,
            "params": params_data,
            "result_message": result_message,
            "result_success": tool_event.result.success,
        }

        # Add token usage if available
        usage_context = _build_token_usage_context(tool_event.usage)
        if usage_context:
            context["token_usage"] = usage_context

        # Add payload if present
        payload = tool_event.result.value
        if payload is not None:
            # Use dump for dataclasses, coerce directly for other types
            if hasattr(payload, "__dataclass_fields__"):
                payload_data = _coerce_for_json(dump(payload, exclude_none=True))
            else:
                payload_data = _coerce_for_json(payload)
            context["result_payload"] = payload_data

        self._resolved_logger.info(
            f"Tool invoked: {tool_event.name}",
            event="tool_invoked",
            context=context,
        )

    def _handle_prompt_executed(self, event: object) -> None:
        """Handle PromptExecuted events."""
        from . import PromptExecuted

        prompt_event = cast(PromptExecuted, event)

        context: dict[str, JSONValue] = {
            "prompt_name": prompt_event.prompt_name,
            "adapter": prompt_event.adapter,
        }

        # Add token usage if available
        usage_context = _build_token_usage_context(prompt_event.usage)
        if usage_context:
            context["token_usage"] = usage_context

        self._resolved_logger.info(
            "Prompt execution complete.",
            event="prompt_executed",
            context=context,
        )

    def attach(self, bus: EventBus) -> None:
        """Subscribe logging handlers to the event bus.

        Args:
            bus: The event bus to subscribe handlers to.
        """
        from . import PromptExecuted, PromptRendered, ToolInvoked

        if self.show_rendered_prompt:
            handler: EventHandler = self._handle_prompt_rendered
            bus.subscribe(PromptRendered, handler)
            self._handlers[PromptRendered] = handler

        if self.show_tool_invocations:
            handler = self._handle_tool_invoked
            bus.subscribe(ToolInvoked, handler)
            self._handlers[ToolInvoked] = handler

        if self.show_execution_summary:
            handler = self._handle_prompt_executed
            bus.subscribe(PromptExecuted, handler)
            self._handlers[PromptExecuted] = handler

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe logging handlers from the event bus.

        Note: This relies on the bus supporting handler removal by identity.
        The InProcessEventBus does not currently support unsubscription,
        so this method clears the internal handler registry but may not
        remove handlers from all bus implementations.

        Args:
            bus: The event bus to unsubscribe handlers from.
        """
        # Clear internal registry; actual unsubscription depends on bus impl
        self._handlers.clear()

    def attached(self, bus: EventBus) -> _AttachedContext:
        """Return a context manager that attaches/detaches handlers.

        Example::

            with subscribers.attached(bus):
                # handlers are active
                ...
            # handlers detached

        Args:
            bus: The event bus to manage subscription for.

        Returns:
            A context manager for scoped subscription.
        """
        return _AttachedContext(subscribers=self, bus=bus)


class _AttachedContext:
    """Context manager for scoped event subscription."""

    def __init__(self, *, subscribers: StandardLoggingSubscribers, bus: EventBus):
        self._subscribers = subscribers
        self._bus = bus

    def __enter__(self) -> StandardLoggingSubscribers:
        self._subscribers.attach(self._bus)
        return self._subscribers

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._subscribers.detach(self._bus)


def attach_standard_logging(
    bus: EventBus,
    *,
    truncate_limit: int = _DEFAULT_TRUNCATE_LIMIT,
    logger: StructuredLogger | logging.Logger | None = None,
    show_rendered_prompt: bool = True,
    show_tool_invocations: bool = True,
    show_execution_summary: bool = True,
) -> StandardLoggingSubscribers:
    """Convenience function to attach standard logging subscribers.

    This is a shorthand for creating a StandardLoggingSubscribers instance
    and calling attach() on it.

    Example::

        from weakincentives.runtime import Session, attach_standard_logging

        session = Session()
        attach_standard_logging(session.event_bus)

    Args:
        bus: The event bus to subscribe handlers to.
        truncate_limit: Maximum length for serialized payloads.
        logger: Logger instance to use (defaults to module-level StructuredLogger).
        show_rendered_prompt: Whether to log rendered prompts.
        show_tool_invocations: Whether to log tool invocations.
        show_execution_summary: Whether to log execution summaries.

    Returns:
        The StandardLoggingSubscribers instance for later detachment if needed.
    """
    subscribers = StandardLoggingSubscribers(
        truncate_limit=truncate_limit,
        logger=logger,
        show_rendered_prompt=show_rendered_prompt,
        show_tool_invocations=show_tool_invocations,
        show_execution_summary=show_execution_summary,
    )
    subscribers.attach(bus)
    return subscribers


__all__ = [
    "StandardLoggingSubscribers",
    "attach_standard_logging",
]
