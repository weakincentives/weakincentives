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
# pyright: reportPrivateUsage=false

"""Fluent slice accessor combining query, dispatch, and mutation operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast

from ...dbc import pure
from ...types import SupportsDataclass

if TYPE_CHECKING:
    from ._types import TypedReducer


class _SliceAccessorProvider(Protocol):
    """Protocol for objects that support slice access with targeted dispatch."""

    def select_all[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> tuple[S, ...]: ...

    def _apply_to_slice(
        self,
        slice_type: type[SupportsDataclass],
        event: SupportsDataclass,
    ) -> None: ...

    def _mutation_seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None: ...

    def _mutation_clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None: ...

    def _mutation_register_reducer[S: SupportsDataclass](
        self,
        data_type: type[SupportsDataclass],
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
    ) -> None: ...

    def _mutation_dispatch_event(
        self, slice_type: type[SupportsDataclass], event: SupportsDataclass
    ) -> None: ...


class SliceAccessor[T: SupportsDataclass]:
    """Fluent interface for querying, dispatching, and mutating a specific slice.

    Accessed via ``session[SliceType]``, providing:

    - **Query operations**: Retrieve state from the slice
    - **Dispatch operations**: Send events to slice-specific reducers
    - **Mutation operations**: Directly modify slice state (bypassing reducers)

    Usage::

        # Query operations
        session[Plan].all()
        session[Plan].latest()
        session[Plan].where(lambda p: p.active)

        # Targeted dispatch (only runs reducers for Plan slice)
        session[Plan].apply(AddStep(step="x"))

        # Direct mutations (bypass reducers)
        session[Plan].seed(initial_plan)
        session[Plan].clear()
        session[Plan].append(new_plan)

        # Reducer registration
        session[Plan].register(AddStep, add_step_reducer)

    For broadcast dispatch that runs all reducers for an event type::

        session.apply(AddStep(step="x"))

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(self, provider: _SliceAccessorProvider, slice_type: type[T]) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

    # ──────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────

    @pure
    def all(self) -> tuple[T, ...]:
        """Return the entire slice for the provided type."""
        return self._provider.select_all(self._slice_type)

    @pure
    def latest(self) -> T | None:
        """Return the most recent item in the slice, if any."""
        values = self._provider.select_all(self._slice_type)
        if not values:
            return None
        return values[-1]

    @pure
    def where(self, predicate: Callable[[T], bool]) -> tuple[T, ...]:
        """Return items that satisfy the predicate."""
        return tuple(
            value
            for value in self._provider.select_all(self._slice_type)
            if predicate(value)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch Operations
    # ──────────────────────────────────────────────────────────────────────

    def apply(self, event: SupportsDataclass) -> None:
        """Dispatch an event targeting only this slice's reducers.

        Unlike ``session.apply(event)`` which broadcasts to all reducers
        registered for the event type, this method filters to only run
        reducers that target this specific slice type.

        Args:
            event: The event to dispatch. Only reducers registered for
                ``(type(event), self._slice_type)`` will be executed.

        Example::

            # Only runs reducers for AddStep that target the Plan slice
            session[Plan].apply(AddStep(step="implement feature"))

        """
        self._provider._apply_to_slice(self._slice_type, event)

    # ──────────────────────────────────────────────────────────────────────
    # Mutation Operations (bypass reducers)
    # ──────────────────────────────────────────────────────────────────────

    def seed(self, values: T | Iterable[T]) -> None:
        """Initialize or replace the stored tuple for the slice type.

        Bypasses reducers; useful for initial state setup or restoration.

        Args:
            values: A single value or iterable of values to seed the slice.

        Example::

            session[Plan].seed(Plan(steps=()))
            session[Plan].seed([Plan(steps=("a",)), Plan(steps=("b",))])

        """
        if isinstance(values, self._slice_type):
            self._provider._mutation_seed_slice(self._slice_type, (values,))
        else:
            self._provider._mutation_seed_slice(
                self._slice_type, cast(Iterable[T], values)
            )

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        Bypasses reducers; useful for cache invalidation or cleanup.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.

        Example::

            session[Plan].clear()  # Remove all
            session[Plan].clear(lambda p: not p.active)  # Remove inactive

        """
        self._provider._mutation_clear_slice(self._slice_type, predicate)

    def append(self, value: T) -> None:
        """Append a value to the slice using the default append reducer.

        Shorthand for dispatching an event where the event type equals
        the slice type.

        Args:
            value: The value to append to the slice.

        Example::

            session[Plan].append(Plan(steps=("new step",)))

        """
        self._provider._mutation_dispatch_event(self._slice_type, value)

    def register(
        self,
        event_type: type[SupportsDataclass],
        reducer: Any,  # TypedReducer[T] - avoiding import cycle  # noqa: ANN401
    ) -> None:
        """Register a reducer for events of the given type.

        When events of ``event_type`` are dispatched, the reducer will be
        called to transform the slice.

        Args:
            event_type: The event type that triggers this reducer.
            reducer: Pure function transforming (slice_values, event) -> slice_values.

        Example::

            session[Plan].register(AddStep, add_step_reducer)

        """
        self._provider._mutation_register_reducer(
            event_type, reducer, slice_type=self._slice_type
        )


__all__ = ["SliceAccessor"]
