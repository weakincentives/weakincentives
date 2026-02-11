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

"""Built-in reducers for Session state slices."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ...types.dataclass import SupportsDataclass
from ._types import (
    ReducerContextProtocol,
    ReducerEvent,
    TypedReducer,
)
from .slices import Append, Replace, SliceView


def append_all[T: SupportsDataclass](
    view: SliceView[T],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> Append[T]:
    """Append event to slice (ledger semantics).

    Never accesses view - O(1) for file-backed slices.
    Use for event streams and ledgers where every event should be recorded,
    even if it equals a previous entry.
    """
    del context, view
    return Append(cast(T, event))


def replace_latest[T: SupportsDataclass](
    view: SliceView[T],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> Replace[T]:
    """Keep only the most recent value.

    Never accesses view - always replaces with singleton.
    """
    del context, view
    return Replace((cast(T, event),))


def upsert_by[T: SupportsDataclass, K](
    key_fn: Callable[[T], K],
) -> TypedReducer[T]:
    """Create reducer that upserts by derived key.

    Must access view to find existing item - O(n) for any backend.
    """

    def reducer(
        view: SliceView[T],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[T]:
        del context
        value = cast(T, event)
        event_key = key_fn(value)
        items = [item for item in view if key_fn(item) != event_key]
        items.append(value)
        return Replace(tuple(items))

    return reducer


def replace_latest_by[T: SupportsDataclass, K](
    key_fn: Callable[[T], K],
) -> TypedReducer[T]:
    """Return a reducer that keeps only the most recent item for each key.

    Must access view to filter by key - O(n) for any backend.
    """

    def reducer(
        view: SliceView[T],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[T]:
        del context
        value = cast(T, event)
        key = key_fn(value)
        filtered = tuple(item for item in view if key_fn(item) != key)
        return Replace((*filtered, value))

    return reducer


__all__ = [
    "append_all",
    "replace_latest",
    "replace_latest_by",
    "upsert_by",
]
