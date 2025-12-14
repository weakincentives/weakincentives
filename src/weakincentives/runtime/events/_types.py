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

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, cast, override
from uuid import UUID, uuid4

from ...adapters._names import AdapterName
from ...dataclasses import FrozenDataclass


class ToolCallStatus(Enum):
    """Lifecycle states for tool calls."""

    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


ToolCallStatusLiteral = Literal["requested", "running", "completed", "failed"]


def compute_correlation_key(tool_name: str, args: dict[str, Any]) -> str:
    """Compute a stable correlation key from tool name and arguments.

    This enables correlation between SDK tool_use_id events and MCP tool
    invocations that don't have direct access to the tool_use_id.

    Args:
        tool_name: Name of the tool being invoked.
        args: Arguments passed to the tool.

    Returns:
        A hex string derived from tool name and canonicalized arguments.
    """
    # Canonicalize args by sorting keys for deterministic hashing
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@FrozenDataclass()
class ToolCall:
    """Canonical state slice representing a tool call through its lifecycle.

    This slice tracks tool calls from request to completion, enabling
    correlation between PreToolUse/PostToolUse hooks and actual tool execution.

    Attributes:
        call_id: Provider's tool_use_id (None for MCP tools before correlation).
        tool_name: Name of the tool being invoked.
        params: Arguments passed to the tool.
        status: Current lifecycle state.
        correlation_key: Hash-based key for MCP tool correlation.
        prompt_name: Name of the prompt that invoked the tool.
        adapter: Name of the adapter executing the tool.
        requested_at: When the tool call was first requested.
        started_at: When tool execution began.
        completed_at: When tool execution finished.
        result: Tool result (set on completion).
        error: Error message (set on failure).
        metadata: Additional tool metadata.
        event_id: Unique identifier for this tool call.
    """

    tool_name: str
    params: dict[str, Any]
    status: ToolCallStatus
    correlation_key: str
    prompt_name: str
    adapter: AdapterName
    requested_at: datetime
    call_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    event_id: UUID = field(default_factory=uuid4)

    @property
    def duration_ms(self) -> float | None:
        """Return execution duration in milliseconds if timing is available."""
        if self.started_at is None or self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    @classmethod
    def create_requested(  # noqa: PLR0913
        cls,
        *,
        tool_name: str,
        params: dict[str, Any],
        prompt_name: str,
        adapter: AdapterName,
        requested_at: datetime,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCall:
        """Create a new tool call in REQUESTED state."""
        return cls(
            tool_name=tool_name,
            params=params,
            status=ToolCallStatus.REQUESTED,
            correlation_key=compute_correlation_key(tool_name, params),
            prompt_name=prompt_name,
            adapter=adapter,
            requested_at=requested_at,
            call_id=call_id,
            metadata=metadata,
        )

    def with_status(  # noqa: PLR0913
        self,
        status: ToolCallStatus,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        result: object = None,
        error: str | None = None,
        call_id: str | None = None,
    ) -> ToolCall:
        """Return a copy with updated status and optional timing/result."""
        from dataclasses import replace

        return replace(
            self,
            status=status,
            started_at=started_at if started_at is not None else self.started_at,
            completed_at=completed_at
            if completed_at is not None
            else self.completed_at,
            result=result if result is not None else self.result,
            error=error if error is not None else self.error,
            call_id=call_id if call_id is not None else self.call_id,
        )


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
    """Event emitted after an adapter executes a tool handler.

    Attributes:
        prompt_name: Name of the prompt that invoked the tool.
        adapter: Name of the adapter that executed the tool.
        name: Name of the tool that was invoked.
        params: Parameters passed to the tool handler.
        result: Result returned by the tool handler.
        session_id: Session ID for correlation.
        created_at: When the event was created.
        usage: Token usage from the provider response.
        rendered_output: Rendered output text (truncated).
        call_id: Provider's tool call ID for correlation.
        started_at: When tool execution started (if available).
        completed_at: When tool execution completed (if available).
        metadata: Additional tool metadata (e.g., tool version, source).
        correlation_key: Key for correlating MCP tools with SDK tool_use_id.
        event_id: Unique identifier for this event.
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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    correlation_key: str | None = None
    event_id: UUID = field(default_factory=uuid4)

    @property
    def duration_ms(self) -> float | None:
        """Return execution duration in milliseconds if timing is available."""
        if self.started_at is None or self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000


__all__ = [
    "ControlBus",
    "EventBus",
    "EventHandler",
    "HandlerFailure",
    "PublishResult",
    "TelemetryBus",
    "TokenUsage",
    "ToolCall",
    "ToolCallStatus",
    "ToolCallStatusLiteral",
    "ToolInvoked",
    "compute_correlation_key",
]
