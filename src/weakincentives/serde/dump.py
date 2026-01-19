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

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import cast, no_type_check
from uuid import UUID

from ..types import JSONValue
from ..types.dataclass import SupportsDataclass
from ._utils import MISSING_SENTINEL, TYPE_REF_KEY, _set_extras, _type_identifier


def _serialize(
    value: object,
    *,
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> JSONValue | object:
    primitive = _serialize_primitive(value, exclude_none)
    if primitive is not None:
        return primitive

    if dataclasses.is_dataclass(value):
        return _serialize_dataclass(
            value,
            by_alias,
            exclude_none,
            alias_generator,
        )
    if isinstance(value, Mapping):
        return _serialize_mapping(
            cast(Mapping[object, object], value),
            by_alias,
            exclude_none,
            alias_generator,
        )
    if isinstance(value, (set, frozenset)):
        return _serialize_set(
            cast(set[object], value),
            by_alias,
            exclude_none,
            alias_generator,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _serialize_sequence(value, by_alias, exclude_none, alias_generator)
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
    alias_generator: Callable[[str], str] | None,
) -> JSONValue | object:
    return dump(
        value,
        by_alias=by_alias,
        exclude_none=exclude_none,
        computed=False,
        alias_generator=alias_generator,
    )


def _serialize_mapping(
    mapping: Mapping[object, object],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> dict[object, JSONValue]:
    serialized: dict[object, JSONValue] = {}
    for key, item in mapping.items():
        item_value = _serialize(
            item,
            by_alias=by_alias,
            exclude_none=exclude_none,
            alias_generator=alias_generator,
        )
        if item_value is MISSING_SENTINEL:
            continue
        serialized[key] = cast(JSONValue, item_value)
    return serialized


def _serialize_set(
    items: set[object],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> list[JSONValue]:
    serialized_items = _serialize_iterable(
        items, by_alias, exclude_none, alias_generator
    )
    try:
        return sorted(serialized_items, key=repr)
    except TypeError:
        return serialized_items


def _serialize_sequence(
    sequence: Sequence[object],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> list[JSONValue]:
    return _serialize_iterable(sequence, by_alias, exclude_none, alias_generator)


def _serialize_iterable(
    items: Iterable[object],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> list[JSONValue]:
    serialized_items: list[JSONValue] = []
    for item in items:
        item_value = _serialize(
            item,
            by_alias=by_alias,
            exclude_none=exclude_none,
            alias_generator=alias_generator,
        )
        if item_value is MISSING_SENTINEL:
            continue
        serialized_items.append(cast(JSONValue, item_value))
    return serialized_items


def dump(  # noqa: PLR0913
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    include_dataclass_type: bool = False,
    type_key: str = TYPE_REF_KEY,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, JSONValue]:
    """Serialize a dataclass instance to a JSON-compatible dictionary."""

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("dump() requires a dataclass instance")

    result: dict[str, JSONValue] = {}
    if include_dataclass_type:
        result[type_key] = _type_identifier(type(obj))
    dataclass_obj = cast(SupportsDataclass, obj)
    _serialize_fields(dataclass_obj, result, by_alias, exclude_none, alias_generator)
    if computed and hasattr(obj.__class__, "__computed__"):
        _serialize_computed_fields(
            dataclass_obj, result, by_alias, exclude_none, alias_generator
        )

    return result


@no_type_check
def _serialize_fields(
    obj: SupportsDataclass,
    result: dict[str, JSONValue],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> None:
    for field in dataclasses.fields(obj):
        key = _field_key(field, by_alias, alias_generator)
        serialized = _serialize_field_value(
            getattr(obj, field.name),
            by_alias,
            exclude_none,
            alias_generator,
        )
        if serialized is MISSING_SENTINEL:
            continue
        result[key] = cast(JSONValue, serialized)


def _serialize_computed_fields(
    obj: SupportsDataclass,
    result: dict[str, JSONValue],
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> None:
    computed_fields = cast(Sequence[str], getattr(obj.__class__, "__computed__", ()))
    for name in computed_fields:
        serialized = _serialize_field_value(
            getattr(obj, name),
            by_alias,
            exclude_none,
            alias_generator,
        )
        if serialized is MISSING_SENTINEL:
            continue
        key = name if not (by_alias and alias_generator) else alias_generator(name)
        result[key] = cast(JSONValue, serialized)


def _field_key(
    field: dataclasses.Field[object],
    by_alias: bool,
    alias_generator: Callable[[str], str] | None,
) -> str:
    if not by_alias:
        return field.name

    field_meta = dict(field.metadata)
    alias = field_meta.get("alias")
    if alias is None and alias_generator is not None:
        alias = alias_generator(field.name)
    return alias or field.name


def _serialize_field_value(
    value: object,
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> JSONValue | object:
    return _serialize(
        value,
        by_alias=by_alias,
        exclude_none=exclude_none,
        alias_generator=alias_generator,
    )


def _extract_extras(obj: object, field_names: set[str]) -> dict[str, object]:
    """Extract extra attributes from a dataclass instance."""
    extras_attr = getattr(obj, "__extras__", None)
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        return {key: value for key, value in obj_dict.items() if key not in field_names}
    if isinstance(extras_attr, Mapping):
        return dict(extras_attr)
    return {}  # pragma: no cover - defensive fallback for unusual dataclasses


def _apply_extras(cloned: object, extras: dict[str, object]) -> None:
    """Apply extra attributes to a cloned instance."""
    if hasattr(cloned, "__dict__"):
        for key, value in extras.items():
            object.__setattr__(cloned, key, value)
    else:
        _set_extras(cloned, extras)


def clone[T](obj: T, **updates: object) -> T:
    """Clone a dataclass instance and re-run model-level validation hooks."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("clone() requires a dataclass instance")

    field_names = {field.name for field in dataclasses.fields(obj)}
    extras = _extract_extras(obj, field_names)
    cloned = dataclasses.replace(obj, **updates)

    if extras:
        _apply_extras(cloned, extras)

    validator = getattr(cloned, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(cloned, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()
    return cloned


__all__ = ["clone", "dump"]
