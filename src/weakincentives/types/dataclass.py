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

"""Dataclass typing helpers.

This module is the canonical location for dataclass-related type protocols.
Import from :mod:`weakincentives.types` for public use::

    from weakincentives.types import SupportsDataclass, SupportsToolResult
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import Field
from typing import Any, ClassVar, Protocol, runtime_checkable

type DataclassFieldMapping = dict[str, Field[Any]]


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances.

    This protocol uses structural typing to detect any class decorated with
    :func:`dataclasses.dataclass` or equivalent. It's marked ``@runtime_checkable``
    to allow ``isinstance()`` checks for duck-typing scenarios.
    """

    __dataclass_fields__: ClassVar[DataclassFieldMapping]


SupportsDataclassOrNone = SupportsDataclass | None
SupportsToolResult = SupportsDataclass | Sequence[SupportsDataclass] | None


__all__ = [
    "DataclassFieldMapping",
    "SupportsDataclass",
    "SupportsDataclassOrNone",
    "SupportsToolResult",
]
