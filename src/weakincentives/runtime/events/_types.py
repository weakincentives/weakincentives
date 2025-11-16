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
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast, override
from uuid import UUID, uuid4

from ...prompt._types import SupportsDataclass, SupportsToolResult
from ...prompt.tool_result import ToolResult

if TYPE_CHECKING:
    from ...adapters._names import AdapterName


class EventProtocol(Protocol):
    """Structural type describing non-dataclass event payloads."""

    event_id: UUID
    created_at: datetime


type EventPayload = SupportsDataclass | EventProtocol


EventT = TypeVar("EventT", bound=EventPayload)

EventHandler = Callable[[EventT], None]


class EventBus(Protocol[EventT]):
    """Minimal synchronous publish/subscribe abstraction."""

    def subscribe(self, event_type: type[EventT], handler: EventHandler[EventT]) -> None:
        """Register a handler for the given event type."""
        ...

    def publish(self, event: EventT) -> PublishResult[EventT]:
        """Publish an event instance to subscribers."""
        ...


@dataclass(slots=True, frozen=True)
class HandlerFailure(Generic[EventT]):
    """Container describing a handler error captured during publish."""

    handler: EventHandler[EventT]
    error: BaseException

    @override
    def __str__(self) -> str:
        return f"{self.handler!r} -> {self.error!r}"


@dataclass(slots=True, frozen=True)
class PublishResult(Generic[EventT]):
    """Summary of an event publish invocation."""

    event: EventT
    handlers_invoked: tuple[EventHandler[EventT], ...]
    errors: tuple[HandlerFailure[EventT], ...]
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
class ToolInvoked:
    """Event emitted after an adapter executes a tool handler."""

    prompt_name: str
    adapter: AdapterName
    name: str
    params: SupportsDataclass
    result: ToolResult[SupportsToolResult]
    session_id: UUID | None
    created_at: datetime
    value: SupportsDataclass | None = None
    call_id: str | None = None
    event_id: UUID = field(default_factory=uuid4)


__all__ = [
    "EventBus",
    "EventHandler",
    "EventPayload",
    "EventProtocol",
    "HandlerFailure",
    "PublishResult",
    "ToolInvoked",
]
