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

"""JSON typing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

_JSONPrimitive = str | int | float | bool | None


type JSONArray = Sequence["JSONValue"]
type JSONObject = Mapping[str, "JSONValue"]
type JSONValue = _JSONPrimitive | JSONObject | JSONArray

type ContractResult = bool | tuple[bool, *tuple[object, ...]] | None

if TYPE_CHECKING:  # pragma: no cover - import guard for typing
    from .dataclass import SupportsDataclass

ParseableDataclassT = TypeVar("ParseableDataclassT", bound="SupportsDataclass")
JSONObjectT = TypeVar("JSONObjectT", bound=JSONObject)
JSONArrayT = TypeVar("JSONArrayT", bound=JSONArray)


def as_json_object(value: object) -> JSONObject:
    """Narrow *value* to :data:`JSONObject` or raise :exc:`TypeError`.

    Centralizes the ``cast()`` needed because ``isinstance(v, Mapping)``
    narrows to ``Mapping[Unknown, object]``, not ``Mapping[str, JSONValue]``.
    """
    if not isinstance(value, Mapping):
        msg = f"Expected JSON object (Mapping), got {type(value).__name__}"
        raise TypeError(msg)
    return cast(JSONObject, value)


def as_json_array(value: object) -> JSONArray:
    """Narrow *value* to :data:`JSONArray` or raise :exc:`TypeError`.

    Centralizes the ``cast()`` needed because ``isinstance(v, Sequence)``
    narrows to ``Sequence[Unknown]``, not ``Sequence[JSONValue]``.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        msg = f"Expected JSON array (Sequence), got {type(value).__name__}"
        raise TypeError(msg)
    return cast(JSONArray, value)


__all__ = [
    "ContractResult",
    "JSONArray",
    "JSONArrayT",
    "JSONObject",
    "JSONObjectT",
    "JSONValue",
    "ParseableDataclassT",
    "as_json_array",
    "as_json_object",
]
