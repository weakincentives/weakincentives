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

"""Dataclass typing helpers for structural typing and runtime checks.

This module provides protocols and type aliases for working with dataclasses
in a type-safe manner, enabling generic serialization, tool results, and
duck-typing patterns.

Type Aliases:
    DataclassFieldMapping: Mapping of field names to Field descriptors.
    SupportsDataclassOrNone: Optional dataclass for nullable returns.
    SupportsToolResult: Valid tool handler return types.

Protocols:
    SupportsDataclass: Structural protocol matching any dataclass.

Functions:
    is_dataclass_instance: Type guard for dataclass instances.

Example:
    >>> from weakincentives.types import SupportsDataclass, is_dataclass_instance
    >>> @dataclass
    ... class User:
    ...     name: str
    >>> is_dataclass_instance(User("alice"))
    True
    >>> isinstance(User("alice"), SupportsDataclass)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Protocol, TypeGuard, runtime_checkable

type DataclassFieldMapping = dict[str, Field[Any]]
"""Mapping from field names to their :class:`dataclasses.Field` descriptors.

This is the type of the ``__dataclass_fields__`` class attribute present on
all dataclasses. Use it when introspecting dataclass field metadata.
"""


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances.

    This protocol uses structural typing to detect any class decorated with
    :func:`dataclasses.dataclass` or equivalent. It's marked ``@runtime_checkable``
    to allow ``isinstance()`` checks for duck-typing scenarios.

    The protocol matches any object with a ``__dataclass_fields__`` class variable,
    which is the canonical marker for dataclass types.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        >>> isinstance(Point(1.0, 2.0), SupportsDataclass)
        True
        >>> isinstance(Point, SupportsDataclass)  # The class itself
        True

    Note:
        For checking instances specifically (not types), use
        :func:`is_dataclass_instance` instead.
    """

    __dataclass_fields__: ClassVar[DataclassFieldMapping]


SupportsDataclassOrNone = SupportsDataclass | None
"""Optional dataclass type for functions that may return None.

Use this for return type annotations where a dataclass or None are valid results,
such as lookup functions that return None when an item is not found.
"""

SupportsToolResult = SupportsDataclass | Sequence[SupportsDataclass] | None
"""Valid return types for tool handler result values.

Tool handlers can return:
    - A single dataclass instance (most common case)
    - A sequence of dataclass instances (for batch operations)
    - None (for side-effect-only tools)

The actual return should be wrapped in ``ToolResult.ok()`` or ``ToolResult.error()``.
"""


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
