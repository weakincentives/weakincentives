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

"""Type narrowing helpers for session modules.

These helpers reduce ``cast()`` noise by providing runtime-validated type
narrowing for common patterns in reducers and session handling.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import TypeGuard

from ...prompt._types import SupportsDataclass
from ._slice_types import SessionSliceType
from ._types import ReducerEvent, ReducerEventWithValue


def is_dataclass_type(cls: type[object]) -> TypeGuard[type[SupportsDataclass]]:
    """Return ``True`` when ``cls`` is a dataclass type (not an instance)."""

    return is_dataclass(cls)


def as_slice_type(cls: type[object]) -> SessionSliceType:
    """Narrow a type to SessionSliceType with runtime validation.

    Raises:
        TypeError: If ``cls`` is not a dataclass type.
    """
    if not is_dataclass_type(cls):
        msg = f"{cls} is not a dataclass type"
        raise TypeError(msg)
    return cls


def as_dataclass_type(cls: type[object]) -> type[SupportsDataclass]:
    """Narrow a type to type[SupportsDataclass] with runtime validation.

    Raises:
        TypeError: If ``cls`` is not a dataclass type.
    """
    if not is_dataclass_type(cls):
        msg = f"{cls} is not a dataclass type"
        raise TypeError(msg)
    return cls


def extract_event_value[T: SupportsDataclass](
    event: ReducerEvent, expected_type: type[T]
) -> T:
    """Extract and narrow event value to the expected type.

    This helper reduces cast() noise in reducers by combining the isinstance
    check with type narrowing in a single call.

    Raises:
        TypeError: If the event does not have a value or the value is not of
            the expected type.
    """
    if not isinstance(event, ReducerEventWithValue):
        msg = f"Event {type(event).__name__} does not have a value attribute"
        raise TypeError(msg)
    value = event.value
    if value is None:
        msg = f"Event {type(event).__name__} has a None value"
        raise TypeError(msg)
    if not isinstance(value, expected_type):
        msg = f"Expected {expected_type.__name__}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


__all__ = [
    "as_dataclass_type",
    "as_slice_type",
    "extract_event_value",
    "is_dataclass_type",
]
