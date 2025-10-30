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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .prompts._types import SupportsDataclass

if TYPE_CHECKING:
    from .adapters.core import PromptResponse
    from .prompts.tool import ToolResult

EventHandler = Callable[[object], None]

logger = logging.getLogger(__name__)


class EventBus(Protocol):
    """Minimal synchronous publish/subscribe abstraction."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """Register a handler for the given event type."""

    def publish(self, event: object) -> None:
        """Publish an event instance to subscribers."""


class NullEventBus:
    """Event bus implementation that discards all events."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:  # noqa: D401
        """No-op subscription hook."""

    def publish(self, event: object) -> None:  # noqa: D401
        """Drop the provided event instance."""


class InProcessEventBus:
    """Process-local event bus that delivers events synchronously."""

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[EventHandler]] = {}

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        handlers = self._handlers.setdefault(event_type, [])
        handlers.append(handler)

    def publish(self, event: object) -> None:
        handlers = tuple(self._handlers.get(type(event), ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error delivering event %s to handler %r",
                    type(event).__name__,
                    handler,
                )


@dataclass(slots=True, frozen=True)
class PromptExecuted:
    """Event emitted after an adapter finishes evaluating a prompt."""

    prompt_name: str
    adapter: str
    response: PromptResponse[object]


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
    "InProcessEventBus",
    "NullEventBus",
    "PromptExecuted",
    "ToolInvoked",
]
