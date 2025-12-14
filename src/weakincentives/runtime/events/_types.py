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
from ..annotations import SliceMeta

EventHandler = Callable[[object], None]


class EventBus(Protocol):
    """Minimal synchronous publish/subscribe abstraction."""

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """Register a handler for the given event type."""
        ...

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        """Remove a handler for the given event type.

        Returns ``True`` if the handler was found and removed, ``False`` otherwise.
        """
        ...

    def publish(self, event: object) -> PublishResult:
        """Publish an event instance to subscribers."""
        ...


# Type aliases to clarify bus usage patterns.
#
# ControlBus: Used by MainLoop for request/response orchestration
# (MainLoopRequest, MainLoopCompleted, MainLoopFailed events).
#
# TelemetryBus: Used by adapters and sessions for observability
# (PromptRendered, ToolInvoked, PromptExecuted events).
#
# Both aliases resolve to EventBus at runtime; the distinction is semantic.

type ControlBus = EventBus
"""EventBus used for MainLoop request/response control flow."""

type TelemetryBus = EventBus
"""EventBus used for session telemetry and adapter observability events."""


@FrozenDataclass()
class HandlerFailure:
    """Container describing a handler error captured during publish."""

    handler: EventHandler
    error: BaseException

    @override
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

    __slice_meta__ = SliceMeta(
        label="Tool Invoked",
        description="Event emitted after an adapter executes a tool handler.",
        icon="wrench",
        sort_key="created_at",
        sort_order="desc",
    )

    prompt_name: str = field(
        metadata={
            "display": "secondary",
            "description": "Name of the prompt containing the tool.",
        }
    )
    adapter: AdapterName = field(
        metadata={
            "display": "hidden",
            "description": "Adapter used for execution.",
        }
    )
    name: str = field(
        metadata={
            "display": "primary",
            "label": "Tool",
            "description": "Name of the tool that was invoked.",
        }
    )
    params: Any = field(
        metadata={
            "display": "primary",
            "format": "json",
            "description": "Parameters passed to the tool handler.",
        }
    )
    result: Any = field(
        metadata={
            "display": "primary",
            "format": "json",
            "description": "Result returned by the tool handler.",
        }
    )
    session_id: UUID | None = field(
        metadata={
            "display": "hidden",
            "description": "Session identifier for the invocation.",
        }
    )
    created_at: datetime = field(
        metadata={
            "display": "secondary",
            "description": "Timestamp of tool invocation.",
        }
    )
    usage: TokenUsage | None = field(
        default=None,
        metadata={
            "display": "secondary",
            "format": "json",
            "description": "Token usage statistics if applicable.",
        },
    )
    rendered_output: str = field(
        default="",
        metadata={
            "display": "secondary",
            "format": "text",
            "description": "Rendered output from the tool result.",
        },
    )
    call_id: str | None = field(
        default=None,
        metadata={
            "display": "hidden",
            "description": "Provider-assigned call identifier.",
        },
    )
    event_id: UUID = field(
        default_factory=uuid4,
        metadata={
            "display": "hidden",
            "description": "Unique identifier for this event.",
        },
    )


__all__ = [
    "ControlBus",
    "EventBus",
    "EventHandler",
    "HandlerFailure",
    "PublishResult",
    "TelemetryBus",
    "TokenUsage",
    "ToolInvoked",
]
