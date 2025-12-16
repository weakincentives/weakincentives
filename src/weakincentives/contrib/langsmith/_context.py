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

"""Trace context management for LangSmith integration.

This module handles parent-child relationship tracking and context propagation
for LangSmith runs. It uses ``contextvars`` to ensure thread-safe trace context
propagation that composes with LangSmith's native integrations.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from uuid import UUID

if TYPE_CHECKING:
    from langsmith.run_trees import RunTree


@dataclass(slots=True)
class TraceContext:
    """Tracks parent-child relationships for a single evaluation.

    This context is stored per-session and links all runs within a single
    prompt evaluation into a unified trace.
    """

    trace_id: UUID
    """Unique identifier for the trace (shared by all runs in the trace)."""

    root_run_id: UUID
    """ID of the root chain run created from PromptRendered."""

    current_run_id: UUID
    """ID of the currently active run (updated as new runs are created)."""

    session_id: UUID | None
    """WINK session ID for correlation."""

    run_count: int = field(default=0)
    """Count of runs created in this trace."""

    total_tokens: int = field(default=0)
    """Accumulated token count across all runs."""


# Thread-local storage for active trace contexts, keyed by session_id
# Note: We use None as default and initialize on first access to avoid mutable default
_active_contexts: ContextVar[dict[UUID, TraceContext] | None] = cast(
    "ContextVar[dict[UUID, TraceContext] | None]",
    ContextVar("langsmith_contexts", default=None),
)

# Current run tree for manual control (set by telemetry handler)
_current_run_tree: ContextVar[object | None] = cast(
    "ContextVar[object | None]",
    ContextVar("langsmith_run_tree", default=None),
)


def _get_contexts() -> dict[UUID, TraceContext]:
    """Get or initialize the contexts dict."""
    contexts = _active_contexts.get()
    if contexts is None:
        contexts = {}
        _ = _active_contexts.set(contexts)
    return contexts


def get_context(session_id: UUID | None) -> TraceContext | None:
    """Retrieve the active trace context for a session.

    Args:
        session_id: The WINK session ID. If ``None``, returns ``None``.

    Returns:
        The active ``TraceContext`` or ``None`` if no trace is active.
    """
    if session_id is None:
        return None
    return _get_contexts().get(session_id)


def set_context(session_id: UUID, context: TraceContext) -> None:
    """Set the active trace context for a session.

    Args:
        session_id: The WINK session ID.
        context: The trace context to associate with the session.
    """
    contexts = _get_contexts()
    # Create new dict to avoid mutating shared state
    new_contexts = dict(contexts)
    new_contexts[session_id] = context
    _ = _active_contexts.set(new_contexts)


def clear_context(session_id: UUID | None) -> TraceContext | None:
    """Remove and return the trace context for a session.

    Args:
        session_id: The WINK session ID. If ``None``, returns ``None``.

    Returns:
        The removed ``TraceContext`` or ``None`` if no context was active.
    """
    if session_id is None:
        return None
    contexts = _get_contexts()
    if session_id not in contexts:
        return None
    # Create new dict to avoid mutating shared state
    new_contexts = dict(contexts)
    context = new_contexts.pop(session_id)
    _ = _active_contexts.set(new_contexts)
    return context


def get_current_run_tree() -> RunTree | None:
    """Get the current LangSmith RunTree for manual control.

    This allows adding custom child runs, metadata, or tags within
    tool handlers or other traced contexts.

    Returns:
        The current ``RunTree`` or ``None`` if no trace is active.

    Example::

        from weakincentives.contrib.langsmith import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree:
            with run_tree.create_child(
                name="custom_step",
                run_type="chain",
                inputs={"key": "value"},
            ) as child:
                result = do_custom_step()
                child.end(outputs={"result": result})
    """
    value = _current_run_tree.get()
    return cast("RunTree | None", value)


def set_current_run_tree(run_tree: object) -> None:
    """Set the current LangSmith RunTree.

    Called by the telemetry handler when processing events.

    Args:
        run_tree: The LangSmith RunTree to set as current.
    """
    _ = _current_run_tree.set(run_tree)


def clear_current_run_tree() -> None:
    """Clear the current LangSmith RunTree."""
    _ = _current_run_tree.set(None)


__all__ = [
    "TraceContext",
    "clear_context",
    "clear_current_run_tree",
    "get_context",
    "get_current_run_tree",
    "set_context",
    "set_current_run_tree",
]
