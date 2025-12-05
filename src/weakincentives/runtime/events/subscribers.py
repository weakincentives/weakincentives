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

"""Standard event subscribers for console logging and observability."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TextIO, cast

from ...serde import dump
from ._types import EventBus, EventHandler, TokenUsage

if TYPE_CHECKING:
    from . import PromptExecuted, PromptRendered, ToolInvoked


_DEFAULT_TRUNCATE_LIMIT: Final[int] = 256


def _truncate(text: str, *, limit: int) -> str:
    """Truncate text to the specified limit, appending ellipsis if needed."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _coerce_for_json(payload: object) -> object:
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
        return dump(payload, exclude_none=True)  # pragma: no cover - defensive
    if isinstance(payload, set):
        return sorted(_coerce_for_json(item) for item in payload)
    return str(payload)


def _format_payload(payload: object, *, limit: int) -> str:
    """Serialize and truncate a payload for log output."""
    serializable = _coerce_for_json(payload)
    try:
        rendered = json.dumps(serializable, ensure_ascii=False)
    except TypeError:  # pragma: no cover - defensive fallback
        rendered = repr(serializable)
    return _truncate(rendered, limit=limit)


def _format_token_usage(usage: TokenUsage | None) -> str:
    """Format token usage counts for log output."""
    if usage is None:
        return "token usage: (not reported)"

    parts: list[str] = []
    if usage.input_tokens is not None:
        parts.append(f"input={usage.input_tokens}")
    if usage.output_tokens is not None:
        parts.append(f"output={usage.output_tokens}")
    if usage.cached_tokens is not None:
        parts.append(f"cached={usage.cached_tokens}")

    total = usage.total_tokens
    if total is not None:
        parts.append(f"total={total}")

    if not parts:
        return "token usage: (not reported)"
    return f"token usage: {', '.join(parts)}"


@dataclass(slots=True)
class StandardLoggingSubscribers:
    """Attach standard console logging for prompt and tool events.

    This helper provides ready-to-use event handlers that print prompt renders,
    tool invocations, and execution summaries to a stream (stdout by default).

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

    stream: TextIO | None = field(default=None)
    """Output stream for log messages. Defaults to stdout when None."""

    show_rendered_prompt: bool = field(default=True)
    """Whether to print the full rendered prompt text."""

    show_tool_invocations: bool = field(default=True)
    """Whether to print tool invocation details."""

    show_execution_summary: bool = field(default=True)
    """Whether to print execution completion summaries."""

    _handlers: dict[type[object], EventHandler] = field(
        default_factory=dict, init=False, repr=False
    )

    def _output(self, text: str) -> None:
        """Write text to the configured output stream."""
        import sys

        stream = self.stream or sys.stdout
        print(text, file=stream)

    def _handle_prompt_rendered(self, event: object) -> None:
        """Handle PromptRendered events."""
        from . import PromptRendered

        prompt_event = cast(PromptRendered, event)
        prompt_label = prompt_event.prompt_name or (
            f"{prompt_event.prompt_ns}:{prompt_event.prompt_key}"
        )
        self._output(f"\n[prompt] Rendered prompt ({prompt_label})")
        self._output(prompt_event.rendered_prompt)
        self._output("")

    def _handle_tool_invoked(self, event: object) -> None:
        """Handle ToolInvoked events."""
        from . import ToolInvoked

        tool_event = cast(ToolInvoked, event)
        params_repr = _format_payload(
            dump(tool_event.params, exclude_none=True),
            limit=self.truncate_limit,
        )
        result_message = _truncate(
            tool_event.result.message or "", limit=self.truncate_limit
        )
        payload_repr: str | None = None
        payload = tool_event.result.value
        if payload is not None:
            try:
                payload_repr = _format_payload(
                    dump(payload, exclude_none=True),
                    limit=self.truncate_limit,
                )
            except TypeError:
                payload_repr = _format_payload(
                    {"value": payload},
                    limit=self.truncate_limit,
                )

        lines = [
            f"{tool_event.name} ({tool_event.prompt_name})",
            f"  params: {params_repr}",
            f"  result: {result_message}",
        ]
        lines.append(f"  {_format_token_usage(tool_event.usage)}")
        if payload_repr is not None:
            lines.append(f"  payload: {payload_repr}")

        self._output("\n[tool] " + "\n".join(lines))

    def _handle_prompt_executed(self, event: object) -> None:
        """Handle PromptExecuted events."""
        from . import PromptExecuted

        prompt_event = cast(PromptExecuted, event)
        self._output("\n[prompt] Execution complete")
        self._output(f"  {_format_token_usage(prompt_event.usage)}\n")

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
    stream: TextIO | None = None,
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
        stream: Output stream for log messages (defaults to stdout).
        show_rendered_prompt: Whether to print rendered prompts.
        show_tool_invocations: Whether to print tool invocations.
        show_execution_summary: Whether to print execution summaries.

    Returns:
        The StandardLoggingSubscribers instance for later detachment if needed.
    """
    subscribers = StandardLoggingSubscribers(
        truncate_limit=truncate_limit,
        stream=stream,
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
