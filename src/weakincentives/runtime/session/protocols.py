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
from ..events._types import TelemetryBus
from ._observer_types import SliceObserver, Subscription
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .snapshots import Snapshot

if TYPE_CHECKING:
    from .slice_accessor import SliceAccessor

type SnapshotProtocol = Snapshot


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

    For broadcast dispatch (routes to all reducers for the event type)::

        session.broadcast(AddStep(step="x"))

    Global operations are available directly on the session::

        session.reset()                  # Clear all slices
        session.rollback(snapshot)       # Restore from snapshot

    """

    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol: ...

    @property
    def event_bus(self) -> TelemetryBus: ...

    def observe[T: SupportsDataclass](
        self,
        slice_type: type[T],
        observer: SliceObserver[T],
    ) -> Subscription: ...

    def __getitem__[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> SliceAccessor[T]: ...

    def install[T: SupportsDataclass](
        self,
        slice_type: type[T],
        *,
        initial: Callable[[], T] | None = None,
    ) -> None: ...

    def broadcast(self, event: SupportsDataclass) -> None:
        """Broadcast an event to all reducers registered for its type."""
        ...

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations."""
        ...

    def rollback(
        self, snapshot: SnapshotProtocol, *, preserve_logs: bool = True
    ) -> None:
        """Restore session slices from the provided snapshot."""
        ...

    @property
    def parent(self) -> Self | None: ...

    @property
    def children(self) -> tuple[Self, ...]: ...

    @property
    def tags(self) -> Mapping[str, str]: ...


__all__ = ["SessionProtocol", "SnapshotProtocol"]
