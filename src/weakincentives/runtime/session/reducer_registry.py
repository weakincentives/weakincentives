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

"""Reducer registration and lookup for session event routing.

ReducerRegistry manages the mapping from event types to reducers,
supporting multiple reducers per event type with different target slices.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ...types.dataclass import SupportsDataclass
from ._slice_types import SessionSliceType
from ._types import TypedReducer


@dataclass(slots=True)
class ReducerRegistration:
    """A single reducer registration.

    Attributes:
        reducer: The reducer function to invoke.
        slice_type: The target slice type for the reducer's output.
    """

    reducer: TypedReducer[Any]
    slice_type: SessionSliceType


class ReducerRegistry:
    """Thread-safe registry for event-to-reducer mappings.

    ReducerRegistry tracks which reducers should be invoked for each event
    type, along with their target slice types. Multiple reducers can be
    registered for the same event type, targeting different slices.

    Example::

        registry = ReducerRegistry()

        # Register reducers for an event type
        registry.register(AddStep, step_reducer, target_slice=AgentPlan)

        # Get all reducers for an event type
        for registration in registry.get_registrations(AddStep):
            op = registration.reducer(slice_view, event, context=ctx)

    """

    __slots__ = ("_lock", "_reducers")

    def __init__(self) -> None:
        """Initialize an empty reducer registry."""
        super().__init__()
        self._reducers: dict[SessionSliceType, list[ReducerRegistration]] = {}
        self._lock = RLock()

    def register[S: SupportsDataclass](
        self,
        event_type: SessionSliceType,
        reducer: TypedReducer[S],
        *,
        target_slice: type[S],
    ) -> None:
        """Register a reducer for an event type.

        Args:
            event_type: The event type that triggers this reducer.
            reducer: The reducer function to invoke.
            target_slice: The slice type that the reducer operates on.
        """
        with self._lock:
            registration = ReducerRegistration(
                reducer=reducer,
                slice_type=target_slice,
            )
            bucket = self._reducers.setdefault(event_type, [])
            bucket.append(registration)

    def get_registrations(
        self, event_type: SessionSliceType
    ) -> tuple[ReducerRegistration, ...]:
        """Get all reducer registrations for an event type.

        Args:
            event_type: The event type to look up.

        Returns:
            Tuple of ReducerRegistration objects, empty if none registered.
        """
        with self._lock:
            return tuple(self._reducers.get(event_type, ()))

    def has_registrations(self, event_type: SessionSliceType) -> bool:
        """Check if any reducers are registered for an event type.

        Args:
            event_type: The event type to check.

        Returns:
            True if at least one reducer is registered.
        """
        with self._lock:
            return event_type in self._reducers and len(self._reducers[event_type]) > 0

    def all_event_types(self) -> set[SessionSliceType]:
        """Return all event types with registered reducers.

        Returns:
            Set of event types that have at least one reducer.
        """
        with self._lock:
            return set(self._reducers)

    def all_target_slice_types(self) -> set[SessionSliceType]:
        """Return all target slice types from all registrations.

        Returns:
            Set of slice types that are targets of registered reducers.
        """
        with self._lock:
            types: set[SessionSliceType] = set()
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def iter_registrations(
        self,
    ) -> Iterator[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]]:
        """Iterate over all event types and their registrations.

        Thread-safe snapshot iteration.

        Yields:
            Tuples of (event_type, registrations).
        """
        with self._lock:
            items = [
                (event_type, tuple(registrations))
                for event_type, registrations in self._reducers.items()
            ]
        yield from items

    def snapshot(
        self,
    ) -> list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]]:
        """Create a snapshot of all registrations for cloning.

        Returns:
            List of (event_type, registrations) tuples.
        """
        with self._lock:
            return [
                (event_type, tuple(registrations))
                for event_type, registrations in self._reducers.items()
            ]

    def copy_from(
        self,
        snapshot: list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]],
        *,
        skip_existing: bool = True,
    ) -> None:
        """Copy registrations from a snapshot.

        Args:
            snapshot: List of (event_type, registrations) tuples.
            skip_existing: If True, skip event types already registered.
        """
        with self._lock:
            for event_type, registrations in snapshot:
                if skip_existing and event_type in self._reducers:
                    continue
                for registration in registrations:
                    bucket = self._reducers.setdefault(event_type, [])
                    bucket.append(
                        ReducerRegistration(
                            reducer=registration.reducer,
                            slice_type=registration.slice_type,
                        )
                    )


__all__ = ["ReducerRegistration", "ReducerRegistry"]
