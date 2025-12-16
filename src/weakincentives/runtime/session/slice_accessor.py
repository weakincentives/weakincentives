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

"""Fluent slice accessor combining query and targeted dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ...dbc import pure
from ...prompt._types import SupportsDataclass


class _SliceAccessorProvider(Protocol):
    """Protocol for objects that support slice access with targeted dispatch."""

    def select_all[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> tuple[S, ...]: ...

    def apply_to_slice(
        self,
        slice_type: type[SupportsDataclass],
        event: SupportsDataclass,
    ) -> None: ...


class SliceAccessor[T: SupportsDataclass]:
    """Fluent interface for querying and mutating a specific slice.

    Accessed via ``session[SliceType]``, providing both query operations
    and slice-targeted dispatch.

    Usage::

        # Query operations
        session[Plan].all()
        session[Plan].latest()
        session[Plan].where(lambda p: p.active)

        # Targeted dispatch (only runs reducers for Plan slice)
        session[Plan].apply(AddStep(step="x"))

    For broadcast dispatch that runs all reducers for an event type::

        session.apply(AddStep(step="x"))

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(self, provider: _SliceAccessorProvider, slice_type: type[T]) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

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
        self._provider.apply_to_slice(self._slice_type, event)


__all__ = ["SliceAccessor"]
