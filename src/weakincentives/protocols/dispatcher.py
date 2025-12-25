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

"""Dispatcher protocol definitions.

This module defines the minimal event dispatcher contract used throughout
weakincentives. It has ZERO dependencies on other weakincentives modules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

EventHandler = Callable[[object], None]
"""Callback type for event subscribers."""


class HandlerFailureProtocol(Protocol):
    """Protocol for handler failure information."""

    handler: EventHandler
    error: BaseException


class DispatchResultProtocol(Protocol):
    """Protocol for dispatch operation results."""

    event: object
    handlers_invoked: Sequence[EventHandler]
    errors: Sequence[HandlerFailureProtocol]
    handled_count: int

    @property
    def ok(self) -> bool:
        """Return True when no handler failures were recorded."""
        ...

    def raise_if_errors(self) -> None:
        """Raise an ExceptionGroup if any handlers failed."""
        ...


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

    def dispatch(self, event: object) -> DispatchResultProtocol:
        """Dispatch an event to registered handlers and return the result."""
        ...


# Semantic type aliases for dispatcher usage patterns.
#
# ControlDispatcher: Used by MainLoop for request/response orchestration
# (MainLoopRequest, MainLoopCompleted, MainLoopFailed events).
#
# TelemetryDispatcher: Used by adapters and sessions for observability
# (PromptRendered, ToolInvoked, PromptExecuted events).

type ControlDispatcher = Dispatcher
"""Dispatcher used for MainLoop request/response control flow."""

type TelemetryDispatcher = Dispatcher
"""Dispatcher used for session telemetry and adapter observability events."""


__all__ = [
    "ControlDispatcher",
    "DispatchResultProtocol",
    "Dispatcher",
    "EventHandler",
    "HandlerFailureProtocol",
    "TelemetryDispatcher",
]
