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

"""Dataclass serialization helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import cast, no_type_check
from uuid import UUID

from ..types import JSONValue
from ..types.dataclass import SupportsDataclass
from ._utils import MISSING_SENTINEL


def _serialize(
    value: object,
    *,
    by_alias: bool,
    exclude_none: bool,
) -> JSONValue | object:
    primitive = _serialize_primitive(value, exclude_none)
    if primitive is not None:
        return primitive

    if dataclasses.is_dataclass(value):
        return _serialize_dataclass(value, by_alias, exclude_none)
    if isinstance(value, Mapping):
        return _serialize_mapping(
            cast(Mapping[object, object], value),
            by_alias,
            exclude_none,
        )
    if isinstance(value, (set, frozenset)):
        return _serialize_set(cast(set[object], value), by_alias, exclude_none)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _serialize_sequence(
            cast(Sequence[object], value), by_alias, exclude_none
        )
    return value


def _serialize_primitive(
    value: object, exclude_none: bool
) -> JSONValue | object | None:
    if value is None:
        return MISSING_SENTINEL if exclude_none else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal, Path)):
        return str(value)
    return None


def _serialize_dataclass(
    value: object,
    by_alias: bool,
    exclude_none: bool,
) -> JSONValue | object:
    return dump(
        value,
        by_alias=by_alias,
        exclude_none=exclude_none,
        computed=False,
    )


def _serialize_mapping(
    mapping: Mapping[object, object],
    by_alias: bool,
    exclude_none: bool,
) -> dict[object, JSONValue]:
    serialized: dict[object, JSONValue] = {}
    for key, item in mapping.items():
        item_value = _serialize(
            item,
            by_alias=by_alias,
            exclude_none=exclude_none,
        )
        if item_value is MISSING_SENTINEL:
            continue
        serialized[key] = cast(JSONValue, item_value)
    return serialized


def _serialize_set(
    items: set[object],
    by_alias: bool,
    exclude_none: bool,
) -> list[JSONValue]:
    serialized_items = _serialize_iterable(items, by_alias, exclude_none)
    try:
        return sorted(serialized_items, key=repr)
    except TypeError:
        return serialized_items


def _serialize_sequence(
    sequence: Sequence[object],
    by_alias: bool,
    exclude_none: bool,
) -> list[JSONValue]:
    return _serialize_iterable(sequence, by_alias, exclude_none)


def _serialize_iterable(
    items: Iterable[object],
    by_alias: bool,
    exclude_none: bool,
) -> list[JSONValue]:
    serialized_items: list[JSONValue] = []
    for item in items:
        item_value = _serialize(
            item,
            by_alias=by_alias,
            exclude_none=exclude_none,
        )
        if item_value is MISSING_SENTINEL:
            continue
        serialized_items.append(cast(JSONValue, item_value))
    return serialized_items


def dump(
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
) -> dict[str, JSONValue]:
    """Serialize a dataclass instance to a JSON-compatible dictionary."""

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("dump() requires a dataclass instance")

    result: dict[str, JSONValue] = {}
    dataclass_obj = cast(SupportsDataclass, obj)
    _serialize_fields(dataclass_obj, result, by_alias, exclude_none)
    if computed and hasattr(obj.__class__, "__computed__"):
        _serialize_computed_fields(dataclass_obj, result, by_alias, exclude_none)

    return result


@no_type_check
def _serialize_fields(
    obj: SupportsDataclass,
    result: dict[str, JSONValue],
    by_alias: bool,
    exclude_none: bool,
) -> None:
    for field in dataclasses.fields(obj):
        key = _field_key(field, by_alias)
        serialized = _serialize(
            getattr(obj, field.name), by_alias=by_alias, exclude_none=exclude_none
        )
        if serialized is MISSING_SENTINEL:
            continue
        result[key] = cast(JSONValue, serialized)


def _serialize_computed_fields(
    obj: SupportsDataclass,
    result: dict[str, JSONValue],
    by_alias: bool,
    exclude_none: bool,
) -> None:
    computed_fields = cast(Sequence[str], getattr(obj.__class__, "__computed__", ()))
    for name in computed_fields:
        serialized = _serialize(
            getattr(obj, name), by_alias=by_alias, exclude_none=exclude_none
        )
        if serialized is MISSING_SENTINEL:
            continue
        result[name] = cast(JSONValue, serialized)


def _field_key(
    field: dataclasses.Field[object],
    by_alias: bool,
) -> str:
    if not by_alias:
        return field.name
    alias = dict(field.metadata).get("alias")
    return alias or field.name


def clone[T](obj: T, **overrides: object) -> T:
    """Create a deep copy of a dataclass by serializing and deserializing.

    This is useful for creating independent copies of frozen dataclasses
    with optional field overrides.

    Args:
        obj: A dataclass instance to clone.
        **overrides: Optional field values to override in the cloned instance.

    Returns:
        A new instance of the same dataclass type with field values from
        the original, optionally overridden by the provided kwargs.

    Raises:
        TypeError: If obj is not a dataclass instance.
    """
    from .parse import parse

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("clone() requires a dataclass instance")

    serialized = dump(obj)
    serialized.update(cast(dict[str, JSONValue], overrides))

    return parse(type(obj), serialized)


__all__ = ["clone", "dump"]
