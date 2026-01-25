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

# pyright: reportImportCycles=false

"""Protocols describing Session behavior exposed to other modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, Self

from ...types.dataclass import SupportsDataclass
from ..events.types import DispatchResult, TelemetryDispatcher
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .snapshots import Snapshot

if TYPE_CHECKING:
    from .slice_accessor import ReadOnlySliceAccessor, SliceAccessor

type SnapshotProtocol = Snapshot
"""Type alias for the Snapshot class used in protocol definitions.

Snapshots capture session state at a point in time for persistence or rollback.
See :class:`~weakincentives.runtime.session.snapshots.Snapshot` for details.
"""


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
        """Capture the current session state as an immutable snapshot.

        Snapshots are used for state persistence and rollback operations.
        By default, only slices with STATE policy are included; LOG slices
        are excluded to preserve append-only semantics.

        Args:
            tag: Optional label for identifying the snapshot purpose.
            policies: Which slice policies to include (default: STATE only).
            include_all: If True, include all slices regardless of policy.

        Returns:
            Frozen snapshot containing serialized slice state.

        Example::

            # Capture state before risky operation
            checkpoint = session.snapshot(tag="pre-tool")

            # On failure, restore to checkpoint
            session.restore(checkpoint)
        """
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """Telemetry dispatcher for session observability events.

        Use this dispatcher to subscribe to or emit telemetry events such as
        PromptRendered, ToolInvoked, and PromptExecuted. These events enable
        logging, metrics collection, and debugging.

        Returns:
            The session's telemetry dispatcher instance.
        """
        ...

    def __getitem__[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> SliceAccessor[T]:
        """Access a typed slice by its dataclass type.

        Returns a :class:`SliceAccessor` providing query and mutation operations
        for the specified slice type. The slice is automatically installed if
        it does not already exist.

        Args:
            slice_type: The dataclass type identifying the slice.

        Returns:
            SliceAccessor for querying and mutating the slice.

        Example::

            # Query the Plan slice
            current_plan = session[Plan].latest()

            # Seed initial data
            session[Plan].seed(Plan(steps=()))
        """
        ...

    def install[T: SupportsDataclass](
        self,
        slice_type: type[T],
        *,
        initial: Callable[[], T] | None = None,
    ) -> None:
        """Explicitly install a slice type with optional initial value factory.

        Most users do not need to call this directly; slices are auto-installed
        on first access via ``session[SliceType]``. Use this method when you
        need to pre-register slices or provide a default value factory.

        Args:
            slice_type: The dataclass type to register as a slice.
            initial: Optional factory function returning the initial slice value.
                Called lazily on first access if the slice is empty.

        Example::

            session.install(Config, initial=lambda: Config(debug=False))
        """
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type.

        Routes the event to every reducer that was registered for this event
        type across all slices. Reducers are invoked synchronously in
        registration order.

        Args:
            event: A dataclass instance representing the event to dispatch.

        Returns:
            DispatchResult summarizing which handlers were invoked and any errors.

        Example::

            result = session.dispatch(AddStep(step="validate inputs"))
            if not result.ok:
                result.raise_if_errors()
        """
        ...

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations.

        Removes all data from every slice but keeps the slice types and their
        associated reducers intact. Use this to reinitialize session state
        without losing configuration.

        Example::

            session.reset()
            assert session[Plan].latest() is None
        """
        ...

    def restore(
        self, snapshot: SnapshotProtocol, *, preserve_logs: bool = True
    ) -> None:
        """Restore session slices from the provided snapshot.

        Replaces current slice state with the values captured in the snapshot.
        By default, LOG-policy slices are preserved (not overwritten) to
        maintain audit history.

        Args:
            snapshot: The snapshot to restore from.
            preserve_logs: If True (default), do not overwrite LOG-policy slices.
                Set to False to fully replace all state.

        Example::

            checkpoint = session.snapshot()
            # ... perform operations that might fail ...
            session.restore(checkpoint)  # Rollback on failure
        """
        ...

    @property
    def parent(self) -> Self | None:
        """Parent session in the session hierarchy, if any.

        Sessions can form a tree structure for hierarchical state management.
        Returns None for root sessions.
        """
        ...

    @property
    def children(self) -> tuple[Self, ...]:
        """Child sessions spawned from this session.

        Returns a tuple of all direct child sessions. Children inherit slice
        registrations from their parent but maintain independent state.
        """
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Metadata tags associated with this session.

        Tags are string key-value pairs for labeling and filtering sessions.
        Common uses include environment identifiers, user context, and
        debugging markers.
        """
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
        """Capture the current session state as an immutable snapshot.

        Snapshots are used for state persistence and rollback operations.
        By default, only slices with STATE policy are included; LOG slices
        are excluded to preserve append-only semantics.

        Args:
            tag: Optional label for identifying the snapshot purpose.
            policies: Which slice policies to include (default: STATE only).
            include_all: If True, include all slices regardless of policy.

        Returns:
            Frozen snapshot containing serialized slice state.
        """
        ...

    @property
    def dispatcher(self) -> TelemetryDispatcher:
        """Telemetry dispatcher for session observability events.

        Use this dispatcher to subscribe to or emit telemetry events such as
        PromptRendered, ToolInvoked, and PromptExecuted. These events enable
        logging, metrics collection, and debugging.

        Returns:
            The session's telemetry dispatcher instance.
        """
        ...

    def __getitem__[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> ReadOnlySliceAccessor[T]:
        """Access a typed slice by its dataclass type (read-only).

        Returns a :class:`ReadOnlySliceAccessor` providing query-only operations
        for the specified slice type. Unlike the full SliceAccessor, this does
        not expose mutation methods.

        Args:
            slice_type: The dataclass type identifying the slice.

        Returns:
            ReadOnlySliceAccessor for querying the slice.

        Example::

            # Query the Plan slice (read-only)
            current_plan = view[Plan].latest()
            all_plans = view[Plan].all()
        """
        ...

    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type.

        Routes the event to every reducer that was registered for this event
        type across all slices. Reducers are invoked synchronously in
        registration order.

        Args:
            event: A dataclass instance representing the event to dispatch.

        Returns:
            DispatchResult summarizing which handlers were invoked and any errors.
        """
        ...

    @property
    def parent(self) -> Self | None:
        """Parent session in the session hierarchy, if any.

        Sessions can form a tree structure for hierarchical state management.
        Returns None for root sessions.
        """
        ...

    @property
    def children(self) -> tuple[Self, ...]:
        """Child sessions spawned from this session.

        Returns a tuple of all direct child sessions. Children inherit slice
        registrations from their parent but maintain independent state.
        """
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Metadata tags associated with this session.

        Tags are string key-value pairs for labeling and filtering sessions.
        Common uses include environment identifiers, user context, and
        debugging markers.
        """
        ...


__all__ = ["SessionProtocol", "SessionViewProtocol", "SnapshotProtocol"]
