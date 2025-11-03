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

"""In-process event primitives for adapter telemetry."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from .prompt._types import SupportsDataclass

if TYPE_CHECKING:
    from .adapters.core import PromptResponse
    from .prompt.tool import ToolResult

EventHandler = Callable[[object], None]

logger = logging.getLogger(__name__)


class EventBus(Protocol):
    """Minimal synchronous publish/subscribe abstraction."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """Register a handler for the given event type."""
        ...

    def publish(self, event: object) -> PublishResult:
        """Publish an event instance to subscribers."""
        ...


class NullEventBus:
    """Event bus implementation that discards all events."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """No-op subscription hook."""

    def publish(self, event: object) -> PublishResult:
        """Drop the provided event instance."""

        return PublishResult(
            event=event,
            handlers_invoked=(),
            errors=(),
        )


class InProcessEventBus:
    """Process-local event bus that delivers events synchronously."""

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[EventHandler]] = {}

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        handlers = self._handlers.setdefault(event_type, [])
        handlers.append(handler)

    def publish(self, event: object) -> PublishResult:
        handlers = tuple(self._handlers.get(type(event), ()))
        invoked: list[EventHandler] = []
        failures: list[HandlerFailure] = []
        for handler in handlers:
            invoked.append(handler)
            try:
                handler(event)
            except Exception as error:
                logger.exception(
                    "Error delivering event %s to handler %r",
                    type(event).__name__,
                    handler,
                )
                failures.append(HandlerFailure(handler=handler, error=error))

        return PublishResult(
            event=event,
            handlers_invoked=tuple(invoked),
            errors=tuple(failures),
        )


@dataclass(slots=True, frozen=True)
class HandlerFailure:
    """Container describing a handler error captured during publish."""

    handler: EventHandler
    error: BaseException

    def __str__(self) -> str:
        return f"{self.handler!r} -> {self.error!r}"


@dataclass(slots=True, frozen=True)
class PublishResult:
    """Summary of an event publish invocation."""

    event: object
    handlers_invoked: tuple[EventHandler, ...]
    errors: tuple[HandlerFailure, ...]
    handled_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "handled_count", len(self.handlers_invoked))

    @property
    def ok(self) -> bool:
        """Return ``True`` when no handler failures were recorded."""

        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise an ``ExceptionGroup`` if any handlers failed."""

        if not self.errors:
            return

        failures = ", ".join(str(failure) for failure in self.errors)
        message = f"Errors while publishing {type(self.event).__name__}: {failures}"
        raise ExceptionGroup(
            message,
            tuple(cast(Exception, failure.error) for failure in self.errors),
        )


@dataclass(slots=True, frozen=True)
class PromptExecuted:
    """Event emitted after an adapter finishes evaluating a prompt."""

    prompt_name: str
    adapter: str
    result: PromptResponse[object]


@dataclass(slots=True, frozen=True)
class ToolInvoked:
    """Event emitted after an adapter executes a tool handler."""

    prompt_name: str
    adapter: str
    name: str
    params: SupportsDataclass
    result: ToolResult[object]
    call_id: str | None = None


__all__ = [
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "NullEventBus",
    "PromptExecuted",
    "PublishResult",
    "ToolInvoked",
]
