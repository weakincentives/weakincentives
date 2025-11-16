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
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import cast
from uuid import UUID

from ..types import JSONValue
from ._utils import MISSING_SENTINEL, _set_extras


def _serialize(
    value: object,
    *,
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> JSONValue | object:
    if value is None:
        return MISSING_SENTINEL if exclude_none else None
    if dataclasses.is_dataclass(value):
        return dump(
            value,
            by_alias=by_alias,
            exclude_none=exclude_none,
            computed=False,
            alias_generator=alias_generator,
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal, Path)):
        return str(value)
    if isinstance(value, Mapping):
        serialized: dict[object, JSONValue] = {}
        for key, item in value.items():
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
    if isinstance(value, set):
        items: list[JSONValue] = []
        for member in value:
            item_value = _serialize(
                member,
                by_alias=by_alias,
                exclude_none=exclude_none,
                alias_generator=alias_generator,
            )
            if item_value is MISSING_SENTINEL:
                continue
            items.append(cast(JSONValue, item_value))
        try:
            return sorted(items, key=repr)
        except TypeError:
            return items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[JSONValue] = []
        for item in value:
            item_value = _serialize(
                item,
                by_alias=by_alias,
                exclude_none=exclude_none,
                alias_generator=alias_generator,
            )
            if item_value is MISSING_SENTINEL:
                continue
            items.append(cast(JSONValue, item_value))
        return items
    return value


def dump(
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, JSONValue]:
    """Serialize a dataclass instance to a JSON-compatible dictionary."""

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("dump() requires a dataclass instance")

    result: dict[str, JSONValue] = {}
    for field in dataclasses.fields(obj):
        field_meta = dict(field.metadata)
        key = field.name
        if by_alias:
            alias = field_meta.get("alias")
            if alias is None and alias_generator is not None:
                alias = alias_generator(field.name)
            if alias:
                key = alias
        value = getattr(obj, field.name)
        serialized = _serialize(
            value,
            by_alias=by_alias,
            exclude_none=exclude_none,
            alias_generator=alias_generator,
        )
        if serialized is MISSING_SENTINEL:
            continue
        result[key] = cast(JSONValue, serialized)

    if computed and hasattr(obj.__class__, "__computed__"):
        computed_fields = cast(
            Sequence[str], getattr(obj.__class__, "__computed__", ())
        )
        for name in computed_fields:
            value = getattr(obj, name)
            serialized = _serialize(
                value,
                by_alias=by_alias,
                exclude_none=exclude_none,
                alias_generator=alias_generator,
            )
            if serialized is MISSING_SENTINEL:
                continue
            key = name
            if by_alias and alias_generator is not None:
                key = alias_generator(name)
            result[key] = cast(JSONValue, serialized)

    return result


def clone[T](obj: T, **updates: object) -> T:
    """Clone a dataclass instance and re-run model-level validation hooks."""

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("clone() requires a dataclass instance")
    field_names = {field.name for field in dataclasses.fields(obj)}
    extras: dict[str, object] = {}
    extras_attr = getattr(obj, "__extras__", None)
    if hasattr(obj, "__dict__"):
        extras = {
            key: value for key, value in obj.__dict__.items() if key not in field_names
        }
    elif isinstance(extras_attr, Mapping):
        extras = dict(extras_attr)

    cloned = dataclasses.replace(obj, **updates)

    if extras:
        if hasattr(cloned, "__dict__"):
            for key, value in extras.items():
                object.__setattr__(cloned, key, value)
        else:
            _set_extras(cloned, extras)

    validator = getattr(cloned, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(cloned, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()
    return cloned


__all__ = ["clone", "dump"]
