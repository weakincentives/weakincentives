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

from ...dbc import pure
from ...prompt._types import SupportsDataclass
from ._types import (
    ReducerContextProtocol,
    ReducerEvent,
    ReducerEventWithValue,
    SimpleReducer,
    TypedReducer,
)


def as_typed_reducer[T: SupportsDataclass](fn: SimpleReducer[T]) -> TypedReducer[T]:
    """Wrap a simple reducer to match the :class:`TypedReducer` signature.

    Use this when you have a stateless reducer that doesn't need access to the
    :class:`ReducerContext`. The wrapper discards the context argument
    automatically.

    Example::

        def my_reducer(
            slice_values: tuple[MyItem, ...],
            event: ReducerEvent,
        ) -> tuple[MyItem, ...]:
            value = cast(MyItem, event)
            return (*slice_values, value)

        session.mutate(MyItem).register(MyItem, as_typed_reducer(my_reducer))
    """

    def wrapper(
        slice_values: tuple[T, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[T, ...]:
        del context
        return fn(slice_values, event)

    return pure(wrapper)


def _resolve_event_value(event: ReducerEvent) -> SupportsDataclass:
    if isinstance(event, ReducerEventWithValue):
        value = event.value
        if value is not None:
            return value
    return cast(SupportsDataclass, event)


def _append_simple[T: SupportsDataclass](
    slice_values: tuple[T, ...],
    event: ReducerEvent,
) -> tuple[T, ...]:
    value = cast(T, _resolve_event_value(event))
    if value in slice_values:
        return slice_values
    return (*slice_values, value)


@pure
def append[T: SupportsDataclass](
    slice_values: tuple[T, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[T, ...]:
    """Append the event value if it is not already present."""
    del context
    return _append_simple(slice_values, event)


def upsert_by[T: SupportsDataclass, K](key_fn: Callable[[T], K]) -> TypedReducer[T]:
    """Return a reducer that upserts items sharing the same derived key."""

    def reducer(
        slice_values: tuple[T, ...],
        event: ReducerEvent,
    ) -> tuple[T, ...]:
        value = cast(T, _resolve_event_value(event))
        key = key_fn(value)
        updated: list[T] = []
        replaced = False
        for existing in slice_values:
            if key_fn(existing) == key:
                if not replaced:
                    updated.append(value)
                    replaced = True
                continue
            updated.append(existing)
        if not replaced:
            updated.append(value)
        return tuple(updated)

    return as_typed_reducer(reducer)


def _replace_latest_simple[T: SupportsDataclass](
    slice_values: tuple[T, ...],
    event: ReducerEvent,
) -> tuple[T, ...]:
    del slice_values
    return (cast(T, _resolve_event_value(event)),)


@pure
def replace_latest[T: SupportsDataclass](
    slice_values: tuple[T, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[T, ...]:
    """Keep only the most recent event value."""
    del context
    return _replace_latest_simple(slice_values, event)


def replace_latest_by[T: SupportsDataclass, K](
    key_fn: Callable[[T], K],
) -> TypedReducer[T]:
    """Return a reducer that keeps only the most recent item for each key."""

    def reducer(
        slice_values: tuple[T, ...],
        event: ReducerEvent,
    ) -> tuple[T, ...]:
        value = cast(T, _resolve_event_value(event))
        key = key_fn(value)
        filtered = tuple(item for item in slice_values if key_fn(item) != key)
        return (*filtered, value)

    return as_typed_reducer(reducer)


__all__ = [
    "append",
    "as_typed_reducer",
    "replace_latest",
    "replace_latest_by",
    "upsert_by",
]
