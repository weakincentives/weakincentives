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
from typing import TYPE_CHECKING, TypeVar

_JSONPrimitive = str | int | float | bool | None


type JSONArray = Sequence["JSONValue"]
type JSONObject = Mapping[str, "JSONValue"]
type JSONValue = _JSONPrimitive | JSONObject | JSONArray

type ContractResult = bool | tuple[bool, *tuple[object, ...]] | None

if TYPE_CHECKING:  # pragma: no cover - import guard for typing
    from ..prompt._types import SupportsDataclass

ParseableDataclassT = TypeVar("ParseableDataclassT", bound="SupportsDataclass")
JSONObjectT = TypeVar("JSONObjectT", bound=JSONObject)
JSONArrayT = TypeVar("JSONArrayT", bound=JSONArray)


__all__ = [
    "ContractResult",
    "JSONArray",
    "JSONArrayT",
    "JSONObject",
    "JSONObjectT",
    "JSONValue",
    "ParseableDataclassT",
]
