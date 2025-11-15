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

"""Internal typing helpers for the :mod:`weakincentives.prompt` package."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import Field
from typing import Any, ClassVar, Protocol, runtime_checkable

type DataclassFieldMapping = dict[str, Field[Any]]


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances."""

    __dataclass_fields__: ClassVar[DataclassFieldMapping]


SupportsToolResult = SupportsDataclass | Sequence[SupportsDataclass]


__all__ = ["DataclassFieldMapping", "SupportsDataclass", "SupportsToolResult"]
