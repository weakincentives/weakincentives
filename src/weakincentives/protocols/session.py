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

"""Session protocol definitions.

This module defines the session contract used by prompt, adapters, and tools.
It has ZERO dependencies on runtime modules to prevent circular imports.

Dependency rule: Only imports from:
- Standard library
- weakincentives.types
- Other weakincentives.protocols modules
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, override
from uuid import UUID

from ..types.dataclass import SupportsDataclass
from .dispatcher import DispatchResultProtocol, TelemetryDispatcher
from .slice_policy import SlicePolicy

type SliceObserver[T: SupportsDataclass] = Callable[
    [tuple[T, ...], tuple[T, ...]],
    None,
]
"""Observer callback invoked when a slice changes.

Receives ``(old_values, new_values)`` after each state update.
"""


class Subscription:
    """Handle for unsubscribing an observer."""

    __slots__ = ("_unsubscribe_fn", "subscription_id")

    subscription_id: UUID
    """Unique identifier for this subscription."""

    def __init__(
        self,
        *,
        unsubscribe_fn: Callable[[], None],
        subscription_id: UUID,
    ) -> None:
        super().__init__()
        self._unsubscribe_fn: Callable[[], None] | None = unsubscribe_fn
        self.subscription_id = subscription_id

    def unsubscribe(self) -> bool:
        """Remove the observer. Returns True if successfully unsubscribed."""
        if self._unsubscribe_fn is not None:
            self._unsubscribe_fn()
            self._unsubscribe_fn = None
            return True
        return False

    @override
    def __repr__(self) -> str:
        return f"Subscription(subscription_id={self.subscription_id!r})"


class SnapshotProtocol(Protocol):
    """Protocol for session state snapshots."""

    @property
    def slices(self) -> Mapping[type[SupportsDataclass], tuple[SupportsDataclass, ...]]:
        """Mapping of slice types to their values."""
        ...

    @property
    def policies(self) -> Mapping[type[SupportsDataclass], SlicePolicy]:
        """Mapping of slice types to their policies."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Session metadata tags."""
        ...

    def to_json(self) -> str:
        """Serialize the snapshot to a JSON string."""
        ...


class SliceAccessorProtocol[T: SupportsDataclass](Protocol):
    """Protocol for accessing and mutating a session slice."""

    def latest(self) -> T | None:
        """Return the most recent value, or None if empty."""
        ...

    def all(self) -> tuple[T, ...]:
        """Return all values in the slice."""
        ...

    def where(self, predicate: Callable[[T], bool]) -> Iterable[T]:
        """Return values matching the predicate."""
        ...

    def seed(self, initial: T) -> None:
        """Initialize the slice with a value if empty."""
        ...

    def append(self, value: T) -> None:
        """Append a value to the slice."""
        ...

    def clear(self) -> None:
        """Remove all values from the slice."""
        ...


class SessionProtocol(Protocol):
    """Structural protocol implemented by session state containers.

    Session provides a Redux-style state container with typed slices, reducers,
    and event-driven dispatch. Access slices via indexing::

        # Query operations
        session[Plan].latest()
        session[Plan].all()
        session[Plan].where(lambda p: p.active)

        # Direct mutations
        session[Plan].seed(initial_plan)
        session[Plan].clear()
        session[Plan].register(AddStep, reducer)

    For dispatch (routes to all reducers for the event type)::

        session.dispatch(AddStep(step="x"))

    Global operations are available directly on the session::

        session.reset()                  # Clear all slices
        session.restore(snapshot)        # Restore from snapshot

    """

    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] | None = None,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture the current session state."""
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """The telemetry dispatcher for this session."""
        ...

    def __getitem__[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> SliceAccessorProtocol[S]:
        """Access a slice for querying and mutation operations."""
        ...

    def install[S: SupportsDataclass](
        self,
        slice_type: type[S],
        *,
        initial: Callable[[], S] | None = None,
    ) -> None:
        """Install a declarative state slice."""
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResultProtocol:
        """Dispatch an event to all reducers registered for its type."""
        ...

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations."""
        ...

    def restore(
        self, snapshot: SnapshotProtocol, *, preserve_logs: bool = True
    ) -> None:
        """Restore session slices from the provided snapshot."""
        ...

    @property
    def parent(self) -> SessionProtocol | None:
        """Parent session in a hierarchy, or None if root."""
        ...

    @property
    def children(self) -> tuple[SessionProtocol, ...]:
        """Child sessions in a hierarchy."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Metadata tags for this session."""
        ...


class SessionViewProtocol(Protocol):
    """Read-only protocol for session access.

    SessionView provides read-only access to session state, suitable for
    contexts where mutation is not allowed (e.g., reducers).
    """

    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] | None = None,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture the current session state."""
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """The telemetry dispatcher for this session."""
        ...

    def __getitem__[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> SliceAccessorProtocol[S]:
        """Access a slice for querying operations (read-only)."""
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResultProtocol:
        """Dispatch an event to all reducers registered for its type."""
        ...

    @property
    def parent(self) -> SessionViewProtocol | None:
        """Parent session in a hierarchy, or None if root."""
        ...

    @property
    def children(self) -> tuple[SessionViewProtocol, ...]:
        """Child sessions in a hierarchy."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Metadata tags for this session."""
        ...


__all__ = [
    "SessionProtocol",
    "SessionViewProtocol",
    "SliceAccessorProtocol",
    "SliceObserver",
    "SnapshotProtocol",
    "Subscription",
]
