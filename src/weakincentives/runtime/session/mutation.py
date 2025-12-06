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

"""Fluent mutation builders for Session slices."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

from ...prompt._types import SupportsDataclass
from ._mutation_types import MutationProvider

if TYPE_CHECKING:
    from .snapshots import Snapshot


class MutationBuilder[T: SupportsDataclass]:
    """Fluent interface for mutating session slices.

    Usage::

        # Direct mutations (bypass reducers)
        session.mutate(Plan).seed(initial_plan)
        session.mutate(Plan).clear()
        session.mutate(Plan).clear(lambda p: not p.active)

        # Event-driven mutations (through reducers)
        session.mutate(Plan).dispatch(SetupPlan(objective="Build feature"))
        session.mutate(Plan).append(new_plan)

        # Reducer registration
        session.mutate(Plan).register(SetupPlan, setup_plan_reducer)

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(self, provider: MutationProvider, slice_type: type[T]) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

    def seed(self, values: T | Iterable[T]) -> None:
        """Initialize or replace the stored tuple for the slice type.

        Bypasses reducers; useful for initial state setup or restoration.

        Args:
            values: A single value or iterable of values to seed the slice.

        """
        if isinstance(values, self._slice_type):
            self._provider.mutation_seed_slice(self._slice_type, (values,))
        else:
            # values is Iterable[T] at this point
            self._provider.mutation_seed_slice(
                self._slice_type, cast(Iterable[T], values)
            )

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        Bypasses reducers; useful for cache invalidation or cleanup.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.

        """
        self._provider.mutation_clear_slice(self._slice_type, predicate)

    def dispatch(self, event: SupportsDataclass) -> None:
        """Dispatch an event to be processed by registered reducers.

        This is the preferred mutation path as it:
        - Flows through registered reducers
        - Maintains traceability of state changes

        Args:
            event: The event to dispatch. Must have a reducer registered
                for its type that targets this slice type.

        """
        self._provider.mutation_dispatch_event(type(event), event)

    def append(self, value: T) -> None:
        """Append a value to the slice using the default append reducer.

        Shorthand for dispatching an event where the event type equals
        the slice type.

        Args:
            value: The value to append to the slice.

        """
        self._provider.mutation_dispatch_event(self._slice_type, value)

    def register(
        self,
        data_type: type[SupportsDataclass],
        reducer: Any,  # TypedReducer[T] - avoiding import cycle  # noqa: ANN401
    ) -> None:
        """Register a reducer for events of the given type.

        When events of ``data_type`` are dispatched, the reducer will be
        called to transform the slice.

        Args:
            data_type: The event type that triggers this reducer.
            reducer: Pure function transforming (slice_values, event) -> slice_values.

        """
        self._provider.mutation_register_reducer(
            data_type, reducer, slice_type=self._slice_type
        )


class GlobalMutationBuilder:
    """Fluent interface for session-wide mutations.

    Usage::

        session.mutate().reset()
        session.mutate().rollback(snapshot)

    """

    __slots__ = ("_provider",)

    def __init__(self, provider: MutationProvider) -> None:
        super().__init__()
        self._provider = provider

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations."""
        self._provider.mutation_reset()

    def rollback(self, snapshot: Snapshot) -> None:
        """Restore session slices from the provided snapshot."""
        self._provider.mutation_rollback(snapshot)


__all__ = ["GlobalMutationBuilder", "MutationBuilder", "MutationProvider"]
