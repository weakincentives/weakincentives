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

"""Session protocols - import this for type hints, not implementations.

This module provides protocol definitions that break circular import
dependencies between Session, SliceAccessor, TypedReducer, and related
classes. All modules in the session package should import protocols
from here for type hints.

The protocols serve as:
1. Type hints that can be imported without causing cycles
2. Documentation of interfaces
3. Easier mocking for tests
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, Self

from ...types.dataclass import SupportsDataclass
from ..events.types import DispatchResult, TelemetryDispatcher
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .slices import SliceOp, SliceView

# ──────────────────────────────────────────────────────────────────────
# Slice Accessor Protocols
# ──────────────────────────────────────────────────────────────────────


class ReadOnlySliceAccessorProtocol[S: SupportsDataclass](Protocol):
    """Protocol for read-only slice access operations.

    Provides query-only operations for accessing session state slices.
    This is the interface exposed by SessionView for reducers.
    """

    def latest(self) -> S | None:
        """Return the most recent item in the slice, if any."""
        ...

    def all(self) -> tuple[S, ...]:
        """Return the entire slice as an immutable tuple."""
        ...

    def where(self, predicate: Callable[[S], bool]) -> tuple[S, ...]:
        """Return items that satisfy the predicate."""
        ...


class SliceAccessorProtocol[S: SupportsDataclass](Protocol):
    """Protocol for slice access operations with mutations.

    Provides both query and mutation operations for session state slices.
    This is the interface exposed by Session for full slice access.
    """

    def latest(self) -> S | None:
        """Return the most recent item in the slice, if any."""
        ...

    def all(self) -> tuple[S, ...]:
        """Return the entire slice as an immutable tuple."""
        ...

    def where(self, predicate: Callable[[S], bool]) -> tuple[S, ...]:
        """Return items that satisfy the predicate."""
        ...

    def seed(self, values: S | tuple[S, ...]) -> None:
        """Initialize or replace the slice contents."""
        ...

    def clear(self, predicate: Callable[[S], bool] | None = None) -> None:
        """Remove items from the slice, optionally filtered by predicate."""
        ...

    def append(self, value: S) -> None:
        """Append a value to the slice."""
        ...

    def register(
        self,
        event_type: type[SupportsDataclass],
        reducer: TypedReducerProtocol[S],
        *,
        policy: SlicePolicy | None = None,
    ) -> None:
        """Register a reducer for events of the given type."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Reducer Protocols
# ──────────────────────────────────────────────────────────────────────


class ReducerContextProtocol(Protocol):
    """Protocol implemented by reducer context objects.

    Provides read-only access to session state during reducer execution.
    """

    session: SessionViewProtocol
    """Read-only session view for accessing other slices."""


class TypedReducerProtocol[S: SupportsDataclass](Protocol):
    """Protocol for reducer callables maintained by Session.

    Reducers receive a lazy SliceView and return a SliceOp describing
    the mutation to apply.
    """

    def __call__(
        self,
        view: SliceView[S],
        event: SupportsDataclass,
        *,
        context: ReducerContextProtocol,
    ) -> SliceOp[S]:
        """Execute the reducer to produce a slice operation.

        Args:
            view: Lazy read-only view of the current slice state.
            event: The event being dispatched.
            context: Context providing session access.

        Returns:
            A SliceOp describing how to mutate the slice.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# Session Protocols
# ──────────────────────────────────────────────────────────────────────


class SnapshotProtocol(Protocol):
    """Protocol for session snapshots.

    Snapshots capture immutable state that can be used to restore sessions.
    """

    @property
    def slices(self) -> Mapping[type[SupportsDataclass], tuple[SupportsDataclass, ...]]:
        """Return mapping of slice types to their contents."""
        ...

    @property
    def policies(self) -> Mapping[type[SupportsDataclass], SlicePolicy]:
        """Return mapping of slice types to their policies."""
        ...

    def to_json(self) -> str:
        """Serialize the snapshot to a JSON string."""
        ...


class SessionViewProtocol(Protocol):
    """Read-only protocol for session access.

    SessionView provides read-only access to session state, suitable for
    contexts where mutation is not allowed (e.g., reducers). Access slices
    via indexing::

        # Query operations (read-only)
        view[Plan].latest()
        view[Plan].all()
        view[Plan].where(lambda p: p.active)

    Unlike full SessionProtocol, SessionView does not expose:
    - ``install()`` for registering state slices
    - ``reset()`` for clearing state
    - ``restore()`` for loading snapshots
    - Mutation methods on slice accessors (``seed``, ``clear``, ``register``)

    Dispatch is still available for broadcasting events to reducers.
    """

    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state."""
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """Return the dispatcher backing this session."""
        ...

    def __getitem__[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> ReadOnlySliceAccessorProtocol[T]:
        """Access a slice for read-only query operations."""
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type."""
        ...

    @property
    def parent(self) -> Self | None:
        """Return the parent session if one was provided."""
        ...

    @property
    def children(self) -> tuple[Self, ...]:
        """Return direct child sessions in registration order."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Return immutable tags associated with this session."""
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
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state."""
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """Return the dispatcher backing this session."""
        ...

    def __getitem__[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> SliceAccessorProtocol[T]:
        """Access a slice for querying and mutation operations."""
        ...

    def install[T: SupportsDataclass](
        self,
        slice_type: type[T],
        *,
        initial: Callable[[], T] | None = None,
    ) -> None:
        """Install a declarative state slice with @reducer methods."""
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
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
    def parent(self) -> Self | None:
        """Return the parent session if one was provided."""
        ...

    @property
    def children(self) -> tuple[Self, ...]:
        """Return direct child sessions in registration order."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Return immutable tags associated with this session."""
        ...


__all__ = [
    "ReadOnlySliceAccessorProtocol",
    "ReducerContextProtocol",
    "SessionProtocol",
    "SessionViewProtocol",
    "SliceAccessorProtocol",
    "SnapshotProtocol",
    "TypedReducerProtocol",
]
