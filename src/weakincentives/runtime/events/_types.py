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

"""Shared typing primitives for event integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast, override
from uuid import UUID, uuid4

from ...adapters._names import AdapterName
from ...dataclasses import FrozenDataclass

EventHandler = Callable[[object], None]


class Dispatcher(Protocol):
    """Minimal synchronous event dispatcher with delivery tracking."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """Register a handler for the given event type."""
        ...

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        """Remove a handler for the given event type.

        Returns ``True`` if the handler was found and removed, ``False`` otherwise.
        """
        ...

    def dispatch(self, event: object) -> DispatchResult:
        """Dispatch an event to registered handlers and return the result."""
        ...


# Type aliases to clarify dispatcher usage patterns.
#
# ControlDispatcher: Used by MainLoop for request/response orchestration
# (MainLoopRequest, MainLoopCompleted, MainLoopFailed events).
#
# TelemetryDispatcher: Used by adapters and sessions for observability
# (PromptRendered, ToolInvoked, PromptExecuted events).
#
# Both aliases resolve to Dispatcher at runtime; the distinction is semantic.

type ControlDispatcher = Dispatcher
"""Dispatcher used for MainLoop request/response control flow."""

type TelemetryDispatcher = Dispatcher
"""Dispatcher used for session telemetry and adapter observability events."""


@FrozenDataclass()
class HandlerFailure:
    """Container describing a handler error captured during publish."""

    handler: EventHandler
    error: BaseException

    @override
    def __str__(self) -> str:
        return f"{self.handler!r} -> {self.error!r}"


@dataclass(slots=True, frozen=True)
class DispatchResult:
    """Summary of an event dispatch invocation."""

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
        message = f"Errors while dispatching {type(self.event).__name__}: {failures}"
        raise ExceptionGroup(
            message,
            tuple(cast(Exception, failure.error) for failure in self.errors),
        )


@FrozenDataclass()
class TokenUsage:
    """Token accounting captured from provider responses."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Return a best-effort total when counts are available."""

        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@FrozenDataclass()
class ToolInvoked:
    """Event emitted after an adapter executes a tool handler."""

    prompt_name: str
    adapter: AdapterName
    name: str
    params: Any
    result: Any
    session_id: UUID | None
    created_at: datetime
    usage: TokenUsage | None = None
    rendered_output: str = ""
    call_id: str | None = None
    event_id: UUID = field(default_factory=uuid4)


__all__ = [
    "ControlDispatcher",
    "DispatchResult",
    "Dispatcher",
    "EventHandler",
    "HandlerFailure",
    "TelemetryDispatcher",
    "TokenUsage",
    "ToolInvoked",
]
