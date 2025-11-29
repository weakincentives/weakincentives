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
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast, overload
from uuid import UUID

from ..prompt._types import SupportsDataclass
from ..types import JSONValue
from ._utils import MISSING_SENTINEL, set_extras

DataclassInstance = SupportsDataclass
TDataclass = TypeVar("TDataclass", bound=SupportsDataclass)


def _require_dataclass_instance(obj: object) -> SupportsDataclass:
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("dump() requires a dataclass instance")
    return cast(SupportsDataclass, obj)


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

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialize_dataclass(
            cast(DataclassInstance, value),
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
    if isinstance(value, set):
        return _serialize_set(
            cast(set[object], value),
            by_alias,
            exclude_none,
            alias_generator,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _serialize_sequence(
            cast(Sequence[object], value),
            by_alias,
            exclude_none,
            alias_generator,
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
    value: DataclassInstance,
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


@overload
def dump(
    obj: SupportsDataclass,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, JSONValue]: ...


@overload
def dump(
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, JSONValue]: ...


def dump(
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, JSONValue]:
    """Serialize a dataclass instance to a JSON-compatible dictionary."""

    dataclass_obj = _require_dataclass_instance(obj)

    result: dict[str, JSONValue] = {}
    _serialize_fields(dataclass_obj, result, by_alias, exclude_none, alias_generator)
    if computed and hasattr(obj.__class__, "__computed__"):
        _serialize_computed_fields(
            dataclass_obj, result, by_alias, exclude_none, alias_generator
        )

    return result


def _serialize_fields(
    obj: DataclassInstance,
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
    obj: DataclassInstance,
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


@overload
def clone[TDataclass: DataclassInstance](
    obj: TDataclass, **updates: object
) -> TDataclass: ...


@overload
def clone[TDataclass: DataclassInstance](
    obj: TDataclass | None, **updates: object
) -> TDataclass: ...


@overload
def clone(obj: object, **updates: object) -> SupportsDataclass: ...


def clone[TDataclass: DataclassInstance](
    obj: TDataclass | None, **updates: object
) -> TDataclass:
    """Clone a dataclass instance and re-run model-level validation hooks."""

    dataclass_obj = _require_dataclass_instance(obj)
    typed_obj = cast(TDataclass, dataclass_obj)
    field_names = {field.name for field in dataclasses.fields(dataclass_obj)}
    extras: dict[str, object] = {}
    extras_attr = getattr(dataclass_obj, "__extras__", None)
    if hasattr(dataclass_obj, "__dict__"):
        extras = {
            key: value
            for key, value in dataclass_obj.__dict__.items()
            if key not in field_names
        }
    elif isinstance(extras_attr, Mapping):
        extras = dict(cast(Mapping[str, object], extras_attr))

    cloned: TDataclass = dataclasses.replace(typed_obj, **updates)

    if extras:
        if hasattr(cloned, "__dict__"):
            for key, value in extras.items():
                object.__setattr__(cloned, key, value)
        else:
            set_extras(cloned, extras)

    validator = getattr(cloned, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(cloned, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()
    return cloned


__all__ = ["clone", "dump"]
