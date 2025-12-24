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

"""System events for unified slice mutation dispatch.

This module provides system events that unify all slice mutations through
the dispatch mechanism. Instead of having some mutations bypass reducers
(seed, clear) while others go through dispatch (append), all mutations
now use dispatch for consistency and auditability.

Usage::

    from weakincentives.runtime.session import InitializeSlice, ClearSlice

    # Initialize a slice (replaces all values)
    session.dispatch(InitializeSlice(Plan, (initial_plan,)))

    # Clear all items from a slice
    session.dispatch(ClearSlice(Plan))

    # Clear items matching a predicate
    session.dispatch(ClearSlice(Plan, predicate=lambda p: not p.active))

These events are handled specially by Session before normal reducer dispatch,
ensuring they always succeed regardless of registered reducers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field

from ...dataclasses import FrozenDataclass
from ...types.dataclass import SupportsDataclass


@FrozenDataclass()
class InitializeSlice[T: SupportsDataclass]:
    """System event to initialize or replace a slice.

    This event replaces all values in a slice with the provided values.
    Equivalent to the previous ``session[T].seed()`` behavior, but now
    goes through dispatch for auditability.

    Attributes:
        slice_type: The type of slice to initialize.
        values: The values to set in the slice.

    Example::

        # Initialize with a single value
        session.dispatch(InitializeSlice(Plan, (Plan(steps=()),)))

        # Initialize with multiple values
        session.dispatch(InitializeSlice(Plan, (plan1, plan2)))

    """

    slice_type: type[T] = field(metadata={"description": "Slice type to initialize."})
    values: tuple[T, ...] = field(
        metadata={"description": "Values to set in the slice."}
    )


@FrozenDataclass()
class ClearSlice[T: SupportsDataclass]:
    """System event to clear items from a slice.

    This event removes items from a slice, optionally filtering by predicate.
    Equivalent to the previous ``session[T].clear()`` behavior, but now
    goes through dispatch for auditability.

    Attributes:
        slice_type: The type of slice to clear.
        predicate: Optional predicate; items where predicate returns True
            are removed. If None, all items are removed.

    Example::

        # Clear all items
        session.dispatch(ClearSlice(Plan))

        # Clear items matching predicate
        session.dispatch(ClearSlice(Plan, predicate=lambda p: not p.active))

    """

    slice_type: type[T] = field(metadata={"description": "Slice type to clear."})
    predicate: Callable[[T], bool] | None = field(
        default=None,
        metadata={"description": "Predicate for selective removal. None clears all."},
    )


__all__ = [
    "ClearSlice",
    "InitializeSlice",
]
