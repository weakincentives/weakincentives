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

"""In-memory slice implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ....types.dataclass import SupportsDataclass


@dataclass(slots=True, frozen=True)
class MemorySliceView[T: SupportsDataclass]:
    """Immutable readonly view of a MemorySlice.

    Provides a snapshot of slice data at view creation time.
    Used for reducer contract compliance.
    """

    _data: tuple[T, ...]

    @property
    def is_empty(self) -> bool:
        """Check if slice is empty."""
        return len(self._data) == 0

    def __len__(self) -> int:
        """Return the number of items in the slice."""
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        """Iterate items lazily."""
        return iter(self._data)

    def all(self) -> tuple[T, ...]:
        """Return all items as an immutable tuple."""
        return self._data

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        return self._data[-1] if self._data else None

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        """Yield items matching predicate."""
        return (item for item in self._data if predicate(item))


@dataclass(slots=True)
class MemorySlice[T: SupportsDataclass]:
    """In-memory tuple-backed slice.

    Provides O(1) reads and O(n) appends. All operations are performed
    on immutable tuples, matching current Session semantics.
    """

    _data: tuple[T, ...] = ()

    # --- SliceView protocol (readonly) ---

    @property
    def is_empty(self) -> bool:
        """Check if slice is empty."""
        return len(self._data) == 0

    def __len__(self) -> int:
        """Return the number of items in the slice."""
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        """Iterate items lazily."""
        return iter(self._data)

    def all(self) -> tuple[T, ...]:
        """Return all items as an immutable tuple."""
        return self._data

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        return self._data[-1] if self._data else None

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        """Yield items matching predicate."""
        return (item for item in self._data if predicate(item))

    # --- Slice protocol (mutable) ---

    def append(self, item: T) -> None:
        """Append a single item to the slice."""
        self._data = (*self._data, item)

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items to the slice."""
        self._data = (*self._data, *items)

    def replace(self, items: tuple[T, ...]) -> None:
        """Replace all items atomically."""
        self._data = items

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.
        """
        if predicate is None:
            self._data = ()
        else:
            self._data = tuple(v for v in self._data if not predicate(v))

    def snapshot(self) -> tuple[T, ...]:
        """Create a snapshot of current state for serialization."""
        return self._data

    def view(self) -> MemorySliceView[T]:
        """Return an immutable readonly view for reducer input.

        Returns a snapshot that won't be affected by concurrent mutations.
        """
        return MemorySliceView(self._data)


@dataclass(slots=True, frozen=True)
class MemorySliceFactory:
    """Factory that creates in-memory slices."""

    def create[T: SupportsDataclass](self, slice_type: type[T]) -> MemorySlice[T]:
        """Create a new empty in-memory slice.

        Args:
            slice_type: The dataclass type this slice will store (unused for memory).

        Returns:
            A new empty MemorySlice instance.
        """
        del slice_type  # Unused for memory slices
        return MemorySlice()


__all__ = [
    "MemorySlice",
    "MemorySliceFactory",
    "MemorySliceView",
]
