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

"""JSON typing helpers for serialization and contract validation.

This module provides type aliases for JSON-compatible data structures and
contract result types used throughout the weakincentives codebase.

Type Aliases:
    JSONValue: Any valid JSON value (primitives, objects, or arrays).
    JSONObject: A JSON object as a string-keyed mapping.
    JSONArray: A JSON array as a sequence of JSON values.
    ContractResult: Return type for design-by-contract predicates.

Type Variables:
    ParseableDataclassT: Bound to SupportsDataclass for generic parsing.
    JSONObjectT: Bound to JSONObject for generic object handling.
    JSONArrayT: Bound to JSONArray for generic array handling.

Example:
    >>> from weakincentives.types import JSONValue, JSONObject
    >>> data: JSONValue = {"key": [1, 2, 3], "nested": {"a": True}}
    >>> obj: JSONObject = {"name": "example", "count": 42}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

_JSONPrimitive = str | int | float | bool | None


type JSONArray = Sequence["JSONValue"]
"""A JSON array represented as a sequence of JSON values.

This includes Python lists and other sequence types containing valid JSON values.
"""

type JSONObject = Mapping[str, "JSONValue"]
"""A JSON object represented as a string-keyed mapping.

This includes Python dicts and other mappings with string keys and JSON values.
"""

type JSONValue = _JSONPrimitive | JSONObject | JSONArray
"""Any valid JSON value: primitives (str, int, float, bool, None), objects, or arrays.

Use this type to annotate parameters or return values that accept arbitrary JSON data.
"""

type ContractResult = bool | tuple[bool, *tuple[object, ...]] | None
"""Return type for design-by-contract predicates (@require, @ensure, @invariant).

Valid return values:
    - ``True`` or ``None``: Contract satisfied.
    - ``False``: Contract violated (raises ContractViolation).
    - ``(False, message, ...)``: Contract violated with custom diagnostic message(s).
    - ``(True, ...)``: Contract satisfied (extra values ignored).

Example:
    >>> @require(lambda x: (x > 0, f"x must be positive, got {x}"))
    ... def sqrt(x: float) -> float: ...
"""

if TYPE_CHECKING:  # pragma: no cover - import guard for typing
    from .dataclass import SupportsDataclass

ParseableDataclassT = TypeVar("ParseableDataclassT", bound="SupportsDataclass")
"""Type variable for dataclass types that can be parsed from JSON.

Bound to :class:`SupportsDataclass` to ensure the target type has dataclass fields
for serialization/deserialization via ``serde.parse()`` and ``serde.dump()``.
"""

JSONObjectT = TypeVar("JSONObjectT", bound=JSONObject)
"""Type variable for JSON object subtypes.

Use in generic functions that need to preserve the specific mapping type.
"""

JSONArrayT = TypeVar("JSONArrayT", bound=JSONArray)
"""Type variable for JSON array subtypes.

Use in generic functions that need to preserve the specific sequence type.
"""


__all__ = [
    "ContractResult",
    "JSONArray",
    "JSONArrayT",
    "JSONObject",
    "JSONObjectT",
    "JSONValue",
    "ParseableDataclassT",
]
