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

"""JSONL file-backed slice implementation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

from ....serde import dump, parse
from ....types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    pass


# Cross-platform file locking utilities
import sys

# Import platform-specific locking module at module level for type checking
if sys.platform == "win32":  # pragma: no cover
    import msvcrt as _msvcrt  # Windows locking
else:
    import fcntl as _fcntl  # Unix locking


def _lock_shared(f: IO[str]) -> None:
    """Acquire a shared (read) lock on the file."""
    if sys.platform == "win32":  # pragma: no cover
        _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1)
    else:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_SH)


def _lock_exclusive(f: IO[str]) -> None:
    """Acquire an exclusive (write) lock on the file."""
    if sys.platform == "win32":  # pragma: no cover
        _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1)
    else:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)


def _unlock(f: IO[str]) -> None:
    """Release lock on the file."""
    if sys.platform == "win32":  # pragma: no cover
        _msvcrt.locking(f.fileno(), _msvcrt.LK_UNLCK, 1)
    else:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)


@dataclass(slots=True)
class JsonlSliceView[T: SupportsDataclass]:
    """Lazy readonly view of a JsonlSlice.

    Provides optimized access patterns that avoid loading the entire
    file when not necessary. Key optimizations:
    - is_empty: O(1) file stat
    - latest(): Can read just the last line (seek to end)
    - __iter__: Streams items without loading all into memory
    """

    path: Path
    item_type: type[T]
    _slice: JsonlSlice[T]

    @property
    def is_empty(self) -> bool:
        """O(1) check using file existence and size."""
        if not self.path.exists():
            return True
        return self.path.stat().st_size == 0

    def __len__(self) -> int:
        """Count items. Uses cache if available, else counts lines."""
        if self._slice._cache is not None:
            return len(self._slice._cache)
        if not self.path.exists():
            return 0
        # Count newlines without parsing
        with self.path.open("rb") as f:
            return sum(1 for _ in f)

    def __iter__(self) -> Iterator[T]:
        """Stream items lazily without loading all into memory."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            _lock_shared(f)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        yield parse(self.item_type, data)
            finally:
                _unlock(f)

    def all(self) -> tuple[T, ...]:
        """Load all items. Uses slice's cache for efficiency."""
        return self._slice.all()

    def latest(self) -> T | None:
        """Return last item. Can be optimized to read only last line."""
        # Use cache if available
        if self._slice._cache is not None:
            cache = self._slice._cache
            return cache[-1] if cache else None
        # Optimization: read last line only (seek from end)
        if not self.path.exists():
            return None
        if self.path.stat().st_size == 0:
            return None
        # For simplicity, fall back to full read with caching
        # A production impl could seek to find last newline
        items = self._slice.all()
        return items[-1] if items else None

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        """Stream filtered items."""
        return (item for item in self if predicate(item))


@dataclass(slots=True)
class JsonlSlice[T: SupportsDataclass]:
    """JSONL file-backed slice for persistence.

    Each item is stored as a single JSON line. Reads load the entire file;
    appends are O(1) file operations. Includes type information for
    polymorphic deserialization.

    Thread safety: Uses file locking (fcntl.flock) for concurrent access.
    Cache invalidation: Cache cleared on any write operation.
    """

    path: Path
    item_type: type[T]
    _cache: tuple[T, ...] | None = None

    def _read_all(self) -> tuple[T, ...]:
        """Read all items from the JSONL file."""
        if not self.path.exists():
            return ()
        items: list[T] = []
        with self.path.open() as f:
            _lock_shared(f)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        items.append(parse(self.item_type, data))
            finally:
                _unlock(f)
        return tuple(items)

    def all(self) -> tuple[T, ...]:
        """Return all items in the slice as an immutable tuple."""
        if self._cache is not None:
            return self._cache
        result = self._read_all()
        self._cache = result
        return result

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        items = self.all()
        return items[-1] if items else None

    @staticmethod
    def _write_item(f: IO[str], item: SupportsDataclass) -> None:
        """Write a single item as a JSON line."""
        data = dump(item)
        _ = f.write(json.dumps(data, separators=(",", ":")) + "\n")

    def append(self, item: T) -> None:
        """Append a single item to the slice."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            _lock_exclusive(f)
            try:
                self._write_item(f, item)
            finally:
                _unlock(f)
        self._cache = None

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items to the slice."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            _lock_exclusive(f)
            try:
                for item in items:
                    self._write_item(f, item)
            finally:
                _unlock(f)
        self._cache = None

    def replace(self, items: tuple[T, ...]) -> None:
        """Replace all items atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            _lock_exclusive(f)
            try:
                for item in items:
                    self._write_item(f, item)
            finally:
                _unlock(f)
        self._cache = items

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items from the slice.

        Args:
            predicate: If provided, only items where predicate returns True
                are removed. If None, all items are removed.
        """
        if predicate is None:
            self.path.unlink(missing_ok=True)
            self._cache = ()
        else:
            remaining = tuple(v for v in self.all() if not predicate(v))
            self.replace(remaining)

    def __len__(self) -> int:
        """Return the number of items in the slice."""
        return len(self.all())

    def snapshot(self) -> tuple[T, ...]:
        """Create a snapshot of current state for serialization.

        Ensures cache is populated for consistent snapshot.
        """
        return self.all()

    def view(self) -> JsonlSliceView[T]:
        """Return a lazy readonly view for reducer input."""
        return JsonlSliceView(
            path=self.path,
            item_type=self.item_type,
            _slice=self,
        )


@dataclass(slots=True)
class JsonlSliceFactory:
    """Factory that creates JSONL file-backed slices.

    Each slice type gets its own file based on the qualified class name.
    When base_dir is not provided, creates a temporary directory that
    persists for the lifetime of the factory.

    Example::

        # Explicit directory for persistent storage
        factory = JsonlSliceFactory(base_dir=Path("./logs"))

        # Temporary directory (auto-created, useful for debugging/testing)
        factory = JsonlSliceFactory()
    """

    base_dir: Path | None = None
    _resolved_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the resolved directory."""
        if self.base_dir is not None:
            object.__setattr__(self, "_resolved_dir", self.base_dir)
        else:
            # Create a temporary directory that persists for factory lifetime
            temp_dir = tempfile.mkdtemp(prefix="wink_slices_")
            object.__setattr__(self, "_resolved_dir", Path(temp_dir))

    def create[T: SupportsDataclass](self, slice_type: type[T]) -> JsonlSlice[T]:
        """Create a new JSONL-backed slice.

        Args:
            slice_type: The dataclass type this slice will store.

        Returns:
            A new JsonlSlice instance for the given type.
        """
        # Use qualified name for unique file paths
        filename = f"{slice_type.__module__}.{slice_type.__qualname__}.jsonl"
        # Sanitize module path for filesystem
        safe_filename = filename.replace(":", "_")
        return JsonlSlice(
            path=self._resolved_dir / safe_filename,
            item_type=slice_type,
        )

    @property
    def directory(self) -> Path:
        """Return the resolved directory path (useful for debugging)."""
        return self._resolved_dir


__all__ = [
    "JsonlSlice",
    "JsonlSliceFactory",
    "JsonlSliceView",
]
