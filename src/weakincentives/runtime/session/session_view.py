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

"""Read-only session view for reducer contexts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from ...types.dataclass import SupportsDataclass
from ..events.types import DispatchResult, TelemetryDispatcher
from .protocols import SessionProtocol, SessionViewProtocol, SnapshotProtocol
from .slice_accessor import ReadOnlySliceAccessor
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy


class SessionView(SessionViewProtocol):
    """Read-only view of a session for use in reducers.

    SessionView wraps a :class:`Session` and exposes only read operations.
    It prevents accidental mutation of session state from within reducers.

    All query operations delegate to the underlying session::

        view[Plan].latest()     # Query current plan
        view[Plan].all()        # Get all plans
        view.snapshot()         # Capture current state

    Dispatch is available for broadcasting events (handled by reducers)::

        view.dispatch(MyEvent())

    Mutation methods are intentionally not exposed::

        view.reset()            # Not available
        view.restore(snapshot)  # Not available
        view.install(Type)      # Not available
        view[Plan].seed(...)    # Not available (ReadOnlySliceAccessor)
        view[Plan].clear()      # Not available (ReadOnlySliceAccessor)

    """

    __slots__ = ("_session",)

    _session: Any  # SessionProtocol with private method access

    def __init__(self, session: SessionProtocol) -> None:
        """Create a read-only view of the provided session.

        Args:
            session: The session to wrap with read-only access.

        """
        super().__init__()
        self._session = session

    # ──────────────────────────────────────────────────────────────────────
    # Read-Only Slice Access
    # ──────────────────────────────────────────────────────────────────────

    def _select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Delegate to underlying session's slice selection."""
        return self._session._select_all(slice_type)

    @override
    def __getitem__[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> ReadOnlySliceAccessor[S]:
        """Access a slice for read-only query operations.

        Returns a :class:`ReadOnlySliceAccessor` that provides only query
        methods (``all()``, ``latest()``, ``where()``). Mutation methods
        are not available through this accessor.

        Usage::

            view[Plan].latest()
            view[Plan].all()
            view[Plan].where(lambda p: p.active)

        """
        return ReadOnlySliceAccessor(self, slice_type)

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch (Event Broadcasting)
    # ──────────────────────────────────────────────────────────────────────

    @override
    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type.

        Delegates to the underlying session's dispatch mechanism.

        Args:
            event: The event to dispatch.

        Returns:
            DispatchResult containing dispatch outcome.

        """
        return self._session.dispatch(event)

    # ──────────────────────────────────────────────────────────────────────
    # Snapshot
    # ──────────────────────────────────────────────────────────────────────

    @override
    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state.

        Delegates to the underlying session's snapshot mechanism.

        Args:
            tag: Optional label for the snapshot.
            policies: Slice policies to include when include_all is False.
            include_all: If True, snapshot all slices regardless of policy.

        Returns:
            An immutable snapshot of the session state.

        """
        return self._session.snapshot(
            tag=tag, policies=policies, include_all=include_all
        )

    # ──────────────────────────────────────────────────────────────────────
    # Properties (Read-Only)
    # ──────────────────────────────────────────────────────────────────────

    @property
    @override
    def dispatcher(self) -> TelemetryDispatcher:
        """Return the dispatcher backing the underlying session."""
        return self._session.dispatcher

    @property
    @override
    def parent(self) -> SessionView | None:
        """Return a view of the parent session, if one exists."""
        parent = self._session.parent
        if parent is None:
            return None
        return SessionView(parent)

    @property
    @override
    def children(self) -> tuple[SessionView, ...]:
        """Return views of direct child sessions in registration order."""
        return tuple(SessionView(child) for child in self._session.children)

    @property
    @override
    def tags(self) -> Mapping[str, str]:
        """Return immutable tags associated with the underlying session."""
        return self._session.tags


def as_view(session: SessionProtocol) -> SessionView:
    """Create a read-only view of the provided session.

    This is a convenience function for creating SessionView instances.

    Args:
        session: The session to wrap with read-only access.

    Returns:
        A SessionView providing read-only access to the session.

    Example::

        view = as_view(session)
        plan = view[Plan].latest()

    """
    return SessionView(session)


__all__ = ["SessionView", "as_view"]
