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

"""In-process event primitives for adapter telemetry and observability.

This package provides synchronous event dispatching for telemetry and
observability events emitted by adapters during prompt evaluation and
tool execution. Unlike the mailbox system (which provides durable
point-to-point delivery), events are broadcast to all subscribers
in-process with fire-and-forget semantics.

When to Use Events vs Mailbox
-----------------------------

**Use Events (this package) for:**

- Telemetry and observability data
- In-process event notifications
- Fire-and-forget broadcasts to multiple subscribers
- Metrics collection and monitoring

**Use Mailbox for:**

- Durable request processing that survives restarts
- Work distribution across multiple consumers
- Cross-process communication
- Tasks requiring acknowledgment and retry on failure

Dispatcher Protocol
-------------------

The :class:`Dispatcher` protocol defines a minimal interface for event
delivery with subscription management and delivery tracking::

    from weakincentives.runtime.events import InProcessDispatcher

    dispatcher = InProcessDispatcher()

    # Subscribe to events
    def on_tool_invoked(event: object) -> None:
        tool_event = cast(ToolInvoked, event)
        print(f"Tool {tool_event.name} executed")

    dispatcher.subscribe(ToolInvoked, on_tool_invoked)

    # Dispatch events
    result = dispatcher.dispatch(ToolInvoked(...))
    if not result.ok:
        result.raise_if_errors()  # Raises ExceptionGroup

    # Unsubscribe when done
    dispatcher.unsubscribe(ToolInvoked, on_tool_invoked)

Telemetry Events
----------------

The package defines three primary telemetry events emitted by adapters:

**PromptRendered**
    Emitted immediately before dispatching a rendered prompt to the
    provider. Contains the rendered prompt text, adapter name, and
    optional prompt descriptor for debugging.

**PromptExecuted**
    Emitted after an adapter finishes evaluating a prompt. Contains
    the result, token usage statistics, and timing information.

**ToolInvoked**
    Emitted after an adapter executes a tool handler. Contains the
    tool name, parameters, result, and optional rendered output.

Example: Token Usage Tracking
-----------------------------

Subscribe to events to track token usage across prompts::

    from weakincentives.runtime.events import InProcessDispatcher, PromptExecuted

    dispatcher = InProcessDispatcher()
    total_tokens = 0

    def track_tokens(event: object) -> None:
        global total_tokens
        prompt_event = cast(PromptExecuted, event)
        if prompt_event.usage is not None:
            total_tokens += prompt_event.usage.total_tokens or 0

    dispatcher.subscribe(PromptExecuted, track_tokens)

Type Aliases
------------

The package provides semantic type aliases to clarify dispatcher usage:

- :data:`ControlDispatcher` - Used by AgentLoop for request/response orchestration
- :data:`TelemetryDispatcher` - Used by adapters and sessions for observability

Both resolve to :class:`Dispatcher` at runtime; the distinction is purely semantic.

Exports
-------

**Protocols:**
    - :class:`Dispatcher` - Minimal event dispatch protocol
    - :data:`ControlDispatcher` - Alias for AgentLoop control flow
    - :data:`TelemetryDispatcher` - Alias for telemetry events

**Implementations:**
    - :class:`InProcessDispatcher` - Synchronous in-process delivery

**Events:**
    - :class:`PromptRendered` - Pre-dispatch prompt event
    - :class:`PromptExecuted` - Post-evaluation completion event
    - :class:`ToolInvoked` - Tool execution event

**Results:**
    - :class:`DispatchResult` - Delivery tracking with handler errors
    - :class:`HandlerFailure` - Individual handler error container
    - :class:`TokenUsage` - Token accounting from providers
"""

from __future__ import annotations

from dataclasses import field
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass
from ...types import AdapterName
from ..logging import StructuredLogger, get_logger
from ..run_context import RunContext
from .types import (
    ControlDispatcher,
    Dispatcher,
    DispatchResult,
    EventHandler,
    HandlerFailure,
    TelemetryDispatcher,
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


logger: StructuredLogger = get_logger(__name__, context={"component": "dispatcher"})


class InProcessDispatcher:
    """Process-local dispatcher that delivers events synchronously."""

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

    def dispatch(self, event: object) -> DispatchResult:
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

        return DispatchResult(
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
    run_context: RunContext | None = None
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
    run_context: RunContext | None = None
    event_id: UUID = field(default_factory=uuid4)


__all__ = [
    "ControlDispatcher",
    "DispatchResult",
    "Dispatcher",
    "HandlerFailure",
    "InProcessDispatcher",
    "PromptExecuted",
    "PromptRendered",
    "TelemetryDispatcher",
    "TokenUsage",
    "ToolInvoked",
]
