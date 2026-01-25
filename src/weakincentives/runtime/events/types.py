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

"""Shared typing primitives for event integrations.

This module defines the core event dispatch infrastructure used throughout
the runtime, including:

- :class:`Dispatcher` protocol for event routing
- :class:`DispatchResult` for tracking handler invocations
- :class:`TokenUsage` for LLM token accounting
- :class:`ToolInvoked` telemetry event for tool execution
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast, override
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass
from ...types import AdapterName
from ..run_context import RunContext

EventHandler = Callable[[object], None]
"""Type alias for event handler callables.

An event handler is any callable that accepts an event object and returns
nothing. Handlers are invoked synchronously when events are dispatched.

Example::

    def my_handler(event: object) -> None:
        if isinstance(event, MyEvent):
            print(f"Received: {event}")

    dispatcher.subscribe(MyEvent, my_handler)
"""


class Dispatcher(Protocol):
    """Minimal synchronous event dispatcher with delivery tracking.

    A dispatcher routes events to registered handlers based on event type.
    All handlers are invoked synchronously, and any errors are captured
    in the returned :class:`DispatchResult` rather than propagating.

    Example::

        def on_completed(event: object) -> None:
            print(f"Task completed: {event}")

        dispatcher.subscribe(TaskCompleted, on_completed)
        result = dispatcher.dispatch(TaskCompleted(task_id="abc"))
        if not result.ok:
            result.raise_if_errors()
    """

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        """Register a handler for the given event type.

        Args:
            event_type: The event class to subscribe to. Handlers receive
                events of this exact type (not subclasses).
            handler: Callable to invoke when events of this type are dispatched.
        """
        ...

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        """Remove a handler for the given event type.

        Args:
            event_type: The event class the handler was subscribed to.
            handler: The handler callable to remove.

        Returns:
            ``True`` if the handler was found and removed, ``False`` if
            the handler was not registered for this event type.
        """
        ...

    def dispatch(self, event: object) -> DispatchResult:
        """Dispatch an event to all registered handlers for its type.

        Handlers are invoked synchronously in registration order. If any
        handler raises an exception, the error is captured in the result
        and remaining handlers continue to execute.

        Args:
            event: The event object to dispatch.

        Returns:
            A :class:`DispatchResult` containing the list of invoked handlers
            and any errors that occurred during dispatch.
        """
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
"""Dispatcher used for MainLoop request/response control flow.

Control dispatchers route orchestration events like ``MainLoopRequest``,
``MainLoopCompleted``, and ``MainLoopFailed`` between the main loop and
its callers. Use this type alias to indicate a dispatcher handles control
flow rather than observability.
"""

type TelemetryDispatcher = Dispatcher
"""Dispatcher used for session telemetry and adapter observability events.

Telemetry dispatchers route events like ``PromptRendered``, ``ToolInvoked``,
and ``PromptExecuted`` for monitoring, logging, and metrics collection.
Use this type alias to indicate a dispatcher handles observability events
rather than control flow.
"""


@FrozenDataclass()
class HandlerFailure:
    """Container describing a handler error captured during dispatch.

    When an event handler raises an exception, the dispatcher captures it
    in a ``HandlerFailure`` rather than propagating immediately. This allows
    remaining handlers to execute and provides a complete error report.

    Attributes:
        handler: The event handler callable that raised the exception.
        error: The exception raised by the handler.
    """

    handler: EventHandler
    error: BaseException

    @override
    def __str__(self) -> str:
        return f"{self.handler!r} -> {self.error!r}"


@dataclass(slots=True, frozen=True)
class DispatchResult:
    """Summary of an event dispatch invocation.

    After dispatching an event, callers receive a ``DispatchResult`` that
    reports which handlers ran and whether any failed. Use :attr:`ok` to
    check for success, or :meth:`raise_if_errors` to convert failures into
    an exception group.

    Attributes:
        event: The event object that was dispatched.
        handlers_invoked: Tuple of handlers that were called, in invocation order.
        errors: Tuple of :class:`HandlerFailure` instances for handlers that raised.
        handled_count: Number of handlers invoked (computed automatically).

    Example::

        result = dispatcher.dispatch(MyEvent())
        if result.ok:
            print(f"Dispatched to {result.handled_count} handlers")
        else:
            for failure in result.errors:
                print(f"Handler {failure.handler} failed: {failure.error}")
    """

    event: object
    handlers_invoked: tuple[EventHandler, ...]
    errors: tuple[HandlerFailure, ...]
    handled_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "handled_count", len(self.handlers_invoked))

    @property
    def ok(self) -> bool:
        """Check whether all handlers completed successfully.

        Returns:
            ``True`` if no handler failures were recorded, ``False`` otherwise.
        """
        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise an ``ExceptionGroup`` if any handlers failed.

        This method provides a convenient way to propagate handler errors
        after dispatch. If all handlers succeeded, this method returns
        normally. If any handler raised an exception, an ``ExceptionGroup``
        is raised containing all captured exceptions.

        Raises:
            ExceptionGroup: If one or more handlers raised exceptions during
                dispatch. The group message includes the event type name and
                a summary of failures.
        """

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
    """Token accounting captured from LLM provider responses.

    Tracks token consumption for cost monitoring and rate limit management.
    All fields are optional since not all providers report all metrics.

    Attributes:
        input_tokens: Number of tokens in the prompt/input. ``None`` if not
            reported by the provider.
        output_tokens: Number of tokens generated in the response. ``None``
            if not reported.
        cached_tokens: Number of input tokens served from cache (e.g., prompt
            caching). ``None`` if not supported or not reported.

    Example::

        usage = TokenUsage(input_tokens=150, output_tokens=50)
        print(f"Total: {usage.total_tokens}")  # 200
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Compute total tokens consumed (input + output).

        Returns:
            Sum of input and output tokens, treating ``None`` as zero.
            Returns ``None`` only if both input and output are ``None``.
        """

        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@FrozenDataclass()
class ToolInvoked:
    """Telemetry event emitted after an adapter executes a tool handler.

    This event provides observability into tool execution during prompt
    processing. Subscribe to this event type on a :class:`TelemetryDispatcher`
    to log tool calls, collect metrics, or implement debugging features.

    Attributes:
        prompt_name: Name of the prompt template that declared the tool.
        adapter: Identifier of the adapter that executed the tool (e.g.,
            ``"openai"``, ``"litellm"``, ``"claude_agent_sdk"``).
        name: The tool name as declared in the prompt's ``tools`` tuple.
        params: The parameters passed to the tool handler (type varies by tool).
        result: The return value from the tool handler (type varies by tool).
        session_id: UUID of the session context, or ``None`` if no session.
        created_at: Timestamp when the tool invocation completed.
        usage: Token usage associated with the tool call, if reported.
        rendered_output: String representation of the tool result for display.
        call_id: Provider-assigned identifier for the tool call, if available.
        run_context: Contextual metadata about the current run, if available.
        event_id: Unique identifier for this event instance (auto-generated).
    """

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
    run_context: RunContext | None = None
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
