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

"""Slice operation types for reducer return values."""

from __future__ import annotations

from collections.abc import Callable

from ....dataclasses import FrozenDataclass
from ....types.dataclass import SupportsDataclass


@FrozenDataclass()
class Append[T: SupportsDataclass]:
    """Append a single item to the slice.

    Most efficient operation for file-backed slices - just appends
    to the file without reading existing contents.
    """

    item: T


@FrozenDataclass()
class Extend[T: SupportsDataclass]:
    """Append multiple items to the slice."""

    items: tuple[T, ...]


@FrozenDataclass()
class Replace[T: SupportsDataclass]:
    """Replace entire slice contents.

    Required when reducer transforms existing state. For file-backed
    slices, this rewrites the entire file.
    """

    items: tuple[T, ...]


@FrozenDataclass()
class Clear[T: SupportsDataclass]:
    """Clear items from the slice.

    Args:
        predicate: If provided, only items where predicate returns True
            are removed. If None, all items are removed.
    """

    predicate: Callable[[T], bool] | None = None


# Union of all slice operations - reducers must return one of these
type SliceOp[T: SupportsDataclass] = Append[T] | Extend[T] | Replace[T] | Clear[T]


__all__ = [
    "Append",
    "Clear",
    "Extend",
    "Replace",
    "SliceOp",
]
