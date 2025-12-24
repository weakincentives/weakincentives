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

"""Slice storage protocols for session state backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, runtime_checkable

from ....types.dataclass import SupportsDataclass


@runtime_checkable
class SliceView[T: SupportsDataclass](Protocol):
    """Readonly lazy view of a slice for reducer input.

    SliceView provides lazy access to slice contents, enabling reducers
    that only append to avoid loading existing data entirely. For
    file-backed slices, methods like `is_empty` and `latest()` can be
    optimized to avoid full file reads.

    This is the input type for reducers, replacing the eager tuple.
    """

    @property
    def is_empty(self) -> bool:
        """Check if slice is empty without loading all items.

        For file-backed slices, this can check file existence/size
        without parsing contents.
        """
        ...

    def __len__(self) -> int:
        """Return item count.

        May require loading for some backends, but can be optimized
        (e.g., counting newlines in JSONL without parsing).
        """
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate items lazily.

        Enables streaming for large slices without loading all into memory.
        """
        ...

    def all(self) -> tuple[T, ...]:
        """Load and return all items as a tuple.

        This is the expensive operation - avoid if possible.
        Results may be cached by the implementation.
        """
        ...

    def latest(self) -> T | None:
        """Return most recent item, or None if empty.

        Can be optimized for file-backed slices (e.g., read last line).
        """
        ...

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        """Yield items matching predicate.

        Enables filtered iteration without materializing full tuple.
        """
        ...


@runtime_checkable
class Slice[T: SupportsDataclass](Protocol):
    """Protocol for slice storage backends.

    Slices store immutable tuples of dataclass instances. All read operations
    return tuples; mutations replace the underlying storage atomically.
    """

    def all(self) -> tuple[T, ...]:
        """Return all items in the slice as an immutable tuple."""
        ...

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        ...

    def append(self, item: T) -> None:
        """Append a single item to the slice."""
        ...

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items to the slice."""
        ...

    def replace(self, items: tuple[T, ...]) -> None:
        """Replace all items atomically.

        Used by reducers after transforming state. The tuple is the
        complete new state for the slice.
        """
        ...

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.
        """
        ...

    def __len__(self) -> int:
        """Return the number of items in the slice."""
        ...

    def snapshot(self) -> tuple[T, ...]:
        """Create a snapshot of current state for serialization.

        Returns the same as all() but signals intent for snapshot use.
        Backends may optimize for this (e.g., flush pending writes).
        """
        ...

    def view(self) -> SliceView[T]:
        """Return a lazy readonly view for reducer input.

        The view provides lazy access to slice contents, enabling
        efficient append-only reducers that never load existing data.
        """
        ...


class SliceFactory(Protocol):
    """Factory for creating slice storage backends."""

    def create[T: SupportsDataclass](self, slice_type: type[T]) -> Slice[T]:
        """Create a new slice for the given dataclass type.

        Args:
            slice_type: The dataclass type this slice will store.

        Returns:
            A new empty Slice instance.
        """
        ...


__all__ = [
    "Slice",
    "SliceFactory",
    "SliceView",
]
