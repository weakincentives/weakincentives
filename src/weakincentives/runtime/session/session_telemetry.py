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

# pyright: reportPrivateUsage=false

"""Session telemetry event handlers.

This module provides handlers for telemetry events (ToolInvoked, PromptExecuted,
PromptRendered) that dispatch payloads to session reducers.

These handlers are called by Session's telemetry subscription callbacks and
route events to the session's internal dispatch mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from ...types.dataclass import SupportsDataclass
from ..events import PromptExecuted, PromptRendered, ToolInvoked
from ._types import ReducerEvent
from .dataclasses import is_dataclass_instance
from .rendered_tools import RenderedTools
from .session_dispatch import dispatch_data_event

if TYPE_CHECKING:
    from .session import Session

# Type casts for telemetry event types
_PROMPT_RENDERED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptRendered
)
_TOOL_INVOKED_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolInvoked)
_PROMPT_EXECUTED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecuted
)
_RENDERED_TOOLS_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], RenderedTools
)


def handle_tool_invoked(session: Session, event: ToolInvoked) -> None:
    """Handle ToolInvoked telemetry event.

    Dispatches the ToolInvoked event to any registered reducers, then extracts
    the payload from the ToolResult and dispatches it to slice reducers.

    Args:
        session: The session to dispatch to.
        event: The ToolInvoked telemetry event.
    """
    # Dispatch ToolInvoked to any reducers registered for ToolInvoked events
    dispatch_data_event(
        session,
        _TOOL_INVOKED_TYPE,
        cast(ReducerEvent, event),
    )

    # Extract payload from ToolResult for slice dispatch
    result = event.result
    if hasattr(result, "value"):
        # ToolResult dataclass
        payload = result.value
    elif isinstance(result, dict):
        # Raw dict from SDK native tools - no typed value
        payload = None
    else:
        payload = None

    # Dispatch payload directly to slice reducers
    if is_dataclass_instance(payload):
        # Narrow for ty: payload is SupportsDataclass after TypeGuard
        narrowed = cast(SupportsDataclass, payload)  # pyright: ignore[reportUnnecessaryCast]
        dispatch_data_event(session, type(narrowed), narrowed)


def handle_prompt_executed(session: Session, event: PromptExecuted) -> None:
    """Handle PromptExecuted telemetry event.

    Dispatches the PromptExecuted event to any registered reducers, then extracts
    the output and dispatches it to slice reducers. Handles both single outputs
    and iterable outputs.

    Args:
        session: The session to dispatch to.
        event: The PromptExecuted telemetry event.
    """
    # Dispatch PromptExecuted to any reducers registered for PromptExecuted events
    dispatch_data_event(
        session,
        _PROMPT_EXECUTED_TYPE,
        cast(ReducerEvent, event),
    )

    # Dispatch output directly to slice reducers
    output = event.result.output
    if is_dataclass_instance(output):
        dispatch_data_event(session, type(output), output)
        return

    # Handle iterable outputs (dispatch each item directly)
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
        for item in cast(Iterable[object], output):
            if is_dataclass_instance(item):
                # Narrow for ty: item is SupportsDataclass after TypeGuard
                narrowed_item = cast(SupportsDataclass, item)  # pyright: ignore[reportUnnecessaryCast]
                dispatch_data_event(session, type(narrowed_item), narrowed_item)


def handle_prompt_rendered(session: Session, event: PromptRendered) -> None:
    """Handle PromptRendered telemetry event.

    Dispatches the PromptRendered event to any registered reducers.

    Args:
        session: The session to dispatch to.
        event: The PromptRendered telemetry event.
    """
    dispatch_data_event(
        session,
        _PROMPT_RENDERED_TYPE,
        cast(ReducerEvent, event),
    )


def handle_rendered_tools(session: Session, event: RenderedTools) -> None:
    """Handle RenderedTools telemetry event.

    Dispatches the RenderedTools event to any registered reducers.
    Stores the complete list of tools available at render time with
    their JSON Schema definitions.

    Args:
        session: The session to dispatch to.
        event: The RenderedTools telemetry event.
    """
    dispatch_data_event(
        session,
        _RENDERED_TOOLS_TYPE,
        cast(ReducerEvent, event),
    )


__all__ = [
    "handle_prompt_executed",
    "handle_prompt_rendered",
    "handle_rendered_tools",
    "handle_tool_invoked",
]
