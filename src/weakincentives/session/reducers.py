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

from .session import DataEvent, TypedReducer


def append[T](slice_values: tuple[T, ...], event: DataEvent) -> tuple[T, ...]:
    """Append the event value if it is not already present."""

    value = cast(T, event.value)
    if value in slice_values:
        return slice_values
    return slice_values + (value,)


def upsert_by[T, K](key_fn: Callable[[T], K]) -> TypedReducer[T]:
    """Return a reducer that upserts items sharing the same derived key."""

    def reducer(slice_values: tuple[T, ...], event: DataEvent) -> tuple[T, ...]:
        value = cast(T, event.value)
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

    return reducer


def replace_latest[T](slice_values: tuple[T, ...], event: DataEvent) -> tuple[T, ...]:
    """Keep only the most recent event value."""

    return (cast(T, event.value),)


__all__ = ["append", "upsert_by", "replace_latest"]
