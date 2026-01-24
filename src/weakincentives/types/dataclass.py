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

"""Dataclass typing helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Protocol, TypeGuard, runtime_checkable

type DataclassFieldMapping = dict[str, Field[Any]]


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances.

    This protocol uses structural typing to detect any class decorated with
    :func:`dataclasses.dataclass` or equivalent. It's marked ``@runtime_checkable``
    to allow ``isinstance()`` checks for duck-typing scenarios.
    """

    __dataclass_fields__: ClassVar[DataclassFieldMapping]


# NOTE: These must remain old-style assignments (not PEP 695 type statements)
# because they are used at runtime in Tool[SupportsDataclassOrNone, SupportsToolResult]
# subscriptions. PEP 695 creates TypeAliasType objects that don't work with
# __class_getitem__ validation which expects actual types.
SupportsDataclassOrNone = SupportsDataclass | None
SupportsToolResult = SupportsDataclass | Sequence[SupportsDataclass] | None


def is_dataclass_instance(value: object) -> TypeGuard[SupportsDataclass]:
    """Return ``True`` when ``value`` is a dataclass instance (not a type).

    This is the canonical helper for checking whether an object is an instance
    of a dataclass rather than the dataclass type itself. Use this function
    throughout the codebase for consistent behavior.

    Args:
        value: The object to check.

    Returns:
        True if ``value`` is a dataclass instance, False otherwise.
    """
    return is_dataclass(value) and not isinstance(value, type)


__all__ = [
    "DataclassFieldMapping",
    "SupportsDataclass",
    "SupportsDataclassOrNone",
    "SupportsToolResult",
    "is_dataclass_instance",
]
