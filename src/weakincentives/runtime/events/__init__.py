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

from dataclasses import field
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ...adapters._names import AdapterName
from ...dataclasses import FrozenDataclass
from ..logging import StructuredLogger, get_logger
from ._types import (
    ControlBus,
    EventBus,
    EventHandler,
    HandlerFailure,
    PublishResult,
    TelemetryBus,
    TokenUsage,
    ToolInvoked,
)

if TYPE_CHECKING:
    from ...prompt.overrides import PromptDescriptor
else:  # pragma: no cover - runtime alias avoids import cycles during module init
    PromptDescriptor = Any


def _describe_handler(handler: EventHandler) -> str:
    module_name = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None)
    if isinstance(qualname, str):
        prefix = f"{module_name}." if isinstance(module_name, str) else ""
        return f"{prefix}{qualname}"
    return repr(handler)  # pragma: no cover - defensive fallback


logger: StructuredLogger = get_logger(__name__, context={"component": "event_bus"})


class InProcessEventBus:
    """Process-local event bus that delivers events synchronously."""

    def __init__(self) -> None:
        super().__init__()
        self._handlers: dict[type[object], list[EventHandler]] = {}
        self._lock = RLock()

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.setdefault(event_type, [])
            handlers.append(handler)

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers is None:
                return False
            try:
                handlers.remove(handler)
            except ValueError:
                return False
            else:
                return True

    def publish(self, event: object) -> PublishResult:
        with self._lock:
            handlers = tuple(self._handlers.get(type(event), ()))
        invoked: list[EventHandler] = []
        failures: list[HandlerFailure] = []
        for handler in handlers:
            invoked.append(handler)
            try:
                handler(event)
            except Exception as error:
                logger.exception(
                    "Error delivering event.",
                    event="event_delivery_failed",
                    context={
                        "handler": _describe_handler(handler),
                        "event_type": type(event).__name__,
                    },
                )
                failures.append(HandlerFailure(handler=handler, error=error))

        return PublishResult(
            event=event,
            handlers_invoked=tuple(invoked),
            errors=tuple(failures),
        )


@FrozenDataclass()
class PromptExecuted:
    """Event emitted after an adapter finishes evaluating a prompt."""

    prompt_name: str
    adapter: AdapterName
    result: Any
    session_id: UUID | None
    created_at: datetime
    usage: TokenUsage | None = None
    event_id: UUID = field(default_factory=uuid4)


@FrozenDataclass()
class PromptRendered:
    """Event emitted immediately before dispatching a rendered prompt."""

    prompt_ns: str
    prompt_key: str
    prompt_name: str | None
    adapter: AdapterName
    session_id: UUID | None
    render_inputs: tuple[Any, ...]
    rendered_prompt: str
    created_at: datetime
    descriptor: PromptDescriptor | None = None
    event_id: UUID = field(default_factory=uuid4)


__all__ = [
    "ControlBus",
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "PromptExecuted",
    "PromptRendered",
    "PublishResult",
    "TelemetryBus",
    "TokenUsage",
    "ToolInvoked",
]
