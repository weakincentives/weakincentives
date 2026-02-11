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

from ...types.dataclass import SupportsDataclass
from .slice_mutations import ClearSlice, InitializeSlice
from .slice_policy import SlicePolicy

if TYPE_CHECKING:
    from ._types import TypedReducer


class _ReadOnlySliceAccessorProvider(Protocol):
    """Protocol for objects that support read-only slice access."""

    def _select_all[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> tuple[S, ...]: ...


class _SliceAccessorProvider(Protocol):
    """Protocol for objects that support slice access."""

    def _select_all[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> tuple[S, ...]: ...

    def _mutation_register_reducer[S: SupportsDataclass](
        self,
        data_type: type[SupportsDataclass],
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
        policy: SlicePolicy | None = None,
    ) -> None: ...

    def _mutation_dispatch_event(
        self, slice_type: type[SupportsDataclass], event: SupportsDataclass
    ) -> None: ...


class SliceAccessor[T: SupportsDataclass]:
    """Fluent interface for querying and mutating a specific slice.

    Accessed via ``session[SliceType]``, providing:

    - **Query operations**: Retrieve state from the slice
    - **Mutation operations**: Modify slice state (all via dispatch)

    Usage::

        # Query operations
        session[Plan].all()
        session[Plan].latest()
        session[Plan].where(lambda p: p.active)

        # Mutations (all go through dispatch)
        session[Plan].seed(initial_plan)   # Dispatches InitializeSlice
        session[Plan].clear()              # Dispatches ClearSlice
        session[Plan].append(new_plan)     # Dispatches to reducers

        # Reducer registration
        session[Plan].register(AddStep, add_step_reducer)

    For dispatch that runs all reducers for an event type::

        session.dispatch(AddStep(step="x"))

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(self, provider: _SliceAccessorProvider, slice_type: type[T]) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

    # ──────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────

    def all(self) -> tuple[T, ...]:
        """Return the entire slice for the provided type."""
        return self._provider._select_all(self._slice_type)

    def latest(self) -> T | None:
        """Return the most recent item in the slice, if any."""
        values = self._provider._select_all(self._slice_type)
        if not values:
            return None
        return values[-1]

    def where(self, predicate: Callable[[T], bool]) -> tuple[T, ...]:
        """Return items that satisfy the predicate."""
        return tuple(
            value
            for value in self._provider._select_all(self._slice_type)
            if predicate(value)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Mutation Operations (all via dispatch)
    # ──────────────────────────────────────────────────────────────────────

    def seed(self, values: T | Iterable[T]) -> None:
        """Initialize or replace the stored tuple for the slice type.

        Dispatches an ``InitializeSlice`` event for auditability.

        Args:
            values: A single value or iterable of values to seed the slice.

        Example::

            session[Plan].seed(Plan(steps=()))
            session[Plan].seed([Plan(steps=("a",)), Plan(steps=("b",))])

        """
        if isinstance(values, self._slice_type):
            values_tuple: tuple[T, ...] = (values,)
        else:
            values_tuple = tuple(cast(Iterable[T], values))
        event: InitializeSlice[T] = InitializeSlice(
            slice_type=self._slice_type, values=values_tuple
        )
        self._provider._mutation_dispatch_event(InitializeSlice, event)

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        Dispatches a ``ClearSlice`` event for auditability.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.

        Example::

            session[Plan].clear()  # Remove all
            session[Plan].clear(lambda p: not p.active)  # Remove inactive

        """
        event: ClearSlice[T] = ClearSlice(
            slice_type=self._slice_type, predicate=predicate
        )
        self._provider._mutation_dispatch_event(ClearSlice, event)

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
        *,
        policy: SlicePolicy | None = None,
    ) -> None:
        """Register a reducer for events of the given type.

        When events of ``event_type`` are dispatched, the reducer will be
        called to transform the slice.

        Args:
            event_type: The event type that triggers this reducer.
            reducer: Pure function transforming (slice_values, event) -> slice_values.
            policy: Optional slice policy for rollback behavior.

        Example::

            session[Plan].register(AddStep, add_step_reducer)

        """
        self._provider._mutation_register_reducer(
            event_type,
            reducer,
            slice_type=self._slice_type,
            policy=policy,
        )


class ReadOnlySliceAccessor[T: SupportsDataclass]:
    """Read-only accessor for querying a specific slice.

    Accessed via ``session_view[SliceType]``, providing only query operations:

    - **Query operations**: Retrieve state from the slice

    Unlike :class:`SliceAccessor`, this class does not expose mutation methods
    (``seed``, ``clear``, ``append``, ``register``). Use this in contexts where
    read-only access is required, such as reducers.

    Usage::

        # Query operations
        view[Plan].all()
        view[Plan].latest()
        view[Plan].where(lambda p: p.active)

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(
        self, provider: _ReadOnlySliceAccessorProvider, slice_type: type[T]
    ) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

    # ──────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────

    def all(self) -> tuple[T, ...]:
        """Return the entire slice for the provided type."""
        return self._provider._select_all(self._slice_type)

    def latest(self) -> T | None:
        """Return the most recent item in the slice, if any."""
        values = self._provider._select_all(self._slice_type)
        if not values:
            return None
        return values[-1]

    def where(self, predicate: Callable[[T], bool]) -> tuple[T, ...]:
        """Return items that satisfy the predicate."""
        return tuple(
            value
            for value in self._provider._select_all(self._slice_type)
            if predicate(value)
        )


__all__ = ["ReadOnlySliceAccessor", "SliceAccessor"]
