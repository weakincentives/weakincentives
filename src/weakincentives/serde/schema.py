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

"""Dataclass schema generation helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal, cast, get_args, get_origin, get_type_hints
from uuid import UUID

from ..prompt._types import SupportsDataclass
from ..types import JSONValue
from ._utils import AnyType, UnionType, merge_annotated_meta, ordered_values

NULL_TYPE = type(None)
NULL_JSON_TYPE = "null"
ELLIPSIS_SENTINEL = Ellipsis
IGNORE_EXTRA = "ignore"
FORBID_EXTRA = "forbid"
ALLOW_EXTRA = "allow"
EXTRA_MODES = {IGNORE_EXTRA, FORBID_EXTRA, ALLOW_EXTRA}
NULL_TYPE_SCHEMA: dict[str, JSONValue] = {"type": NULL_JSON_TYPE}

PRIMITIVE_FORMATS: dict[type[object], dict[str, JSONValue]] = {
    bool: {"type": "boolean"},
    int: {"type": "integer"},
    float: {"type": "number"},
    Decimal: {"type": "number"},
    str: {"type": "string"},
    datetime: {"type": "string", "format": "date-time"},
    date: {"type": "string", "format": "date"},
    time: {"type": "string", "format": "time"},
    UUID: {"type": "string", "format": "uuid"},
    Path: {"type": "string"},
}


def _schema_constraints(meta: Mapping[str, object]) -> dict[str, JSONValue]:
    schema_meta: dict[str, JSONValue] = {}
    mapping = {
        "ge": "minimum",
        "minimum": "minimum",
        "gt": "exclusiveMinimum",
        "exclusiveMinimum": "exclusiveMinimum",
        "le": "maximum",
        "maximum": "maximum",
        "lt": "exclusiveMaximum",
        "exclusiveMaximum": "exclusiveMaximum",
        "min_length": "minLength",
        "minLength": "minLength",
        "max_length": "maxLength",
        "maxLength": "maxLength",
        "regex": "pattern",
        "pattern": "pattern",
    }
    for key, target in mapping.items():
        if key in meta and target not in schema_meta:
            schema_meta[target] = cast(JSONValue, meta[key])
    members = meta.get("enum") or meta.get("in")
    if isinstance(members, Iterable) and not isinstance(members, (str, bytes)):
        enum_values = ordered_values(cast(Iterable[JSONValue], members))
        _ = schema_meta.setdefault("enum", enum_values)
    not_members = meta.get("not_in")
    if (
        isinstance(not_members, Iterable)
        and not isinstance(not_members, (str, bytes))
        and "not" not in schema_meta
    ):
        schema_meta["not"] = {
            "enum": ordered_values(cast(Iterable[JSONValue], not_members))
        }
    return schema_meta


def _resolve_schema(
    base_type: object,
    origin: object,
    alias_generator: Callable[[str], str] | None,
) -> dict[str, JSONValue] | None:
    for builder in (
        _schema_for_dataclass,
        _schema_for_literal,
        _schema_for_union,
        _schema_for_collection,
        _schema_for_enum,
        _schema_for_primitive,
    ):
        schema_data = builder(base_type, alias_generator, origin)
        if schema_data is not None:
            return schema_data
    return None


def _schema_for_type(
    typ: object,
    meta: Mapping[str, object] | None,
    alias_generator: Callable[[str], str] | None,
) -> dict[str, JSONValue]:
    base_type, merged_meta = merge_annotated_meta(typ, meta)
    origin = get_origin(base_type)

    schema_data = _resolve_schema(base_type, origin, alias_generator) or {}

    schema_data.update(_schema_constraints(merged_meta))
    return schema_data


def _schema_for_dataclass(
    base_type: object, alias_generator: Callable[[str], str] | None, origin: object
) -> dict[str, JSONValue] | None:
    if base_type is object or base_type is AnyType:
        return {}
    if not dataclasses.is_dataclass(base_type):
        return None
    dataclass_type = base_type if isinstance(base_type, type) else type(base_type)
    return schema(dataclass_type, alias_generator=alias_generator)


def _schema_for_literal(
    base_type: object, alias_generator: Callable[[str], str] | None, origin: object
) -> dict[str, JSONValue] | None:
    if base_type is NULL_TYPE:
        return dict(NULL_TYPE_SCHEMA)
    if origin is not Literal:
        return None

    literal_values = list(get_args(base_type))
    schema_data: dict[str, JSONValue] = {"enum": literal_values}
    if not literal_values:
        return schema_data  # pragma: no cover - defensive

    if all(isinstance(value, bool) for value in literal_values):
        schema_data["type"] = "boolean"
    elif all(isinstance(value, str) for value in literal_values):
        schema_data["type"] = "string"
    elif all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in literal_values
    ):
        schema_data["type"] = "integer"
    elif all(isinstance(value, (float, Decimal)) for value in literal_values):
        schema_data["type"] = "number"
    return schema_data


def _is_object_schema(schema_data: Mapping[str, JSONValue]) -> bool:
    return schema_data.get("type") == "object"


def _collect_union_subschemas(
    base_type: object, alias_generator: Callable[[str], str] | None
) -> tuple[bool, list[dict[str, JSONValue]], Mapping[str, JSONValue] | None]:
    includes_null = False
    subschemas: list[dict[str, JSONValue]] = []
    base_schema_ref: Mapping[str, JSONValue] | None = None
    for arg in get_args(base_type):
        if arg is NULL_TYPE:
            includes_null = True
            continue
        subschema = _schema_for_type(arg, None, alias_generator)
        subschemas.append(subschema)
        if base_schema_ref is None and _is_object_schema(subschema):
            base_schema_ref = subschema
    return includes_null, subschemas, base_schema_ref


def _merge_union_schema(
    subschemas: Sequence[dict[str, JSONValue]],
    base_schema_ref: Mapping[str, JSONValue] | None,
    includes_null: bool,
) -> dict[str, JSONValue]:
    any_of = list(subschemas)
    if includes_null:
        any_of.append(dict(NULL_TYPE_SCHEMA))
    schema_data: dict[str, JSONValue] = (
        dict(base_schema_ref)
        if base_schema_ref is not None and len(subschemas) == 1
        else {}
    )
    schema_data["anyOf"] = any_of
    return schema_data


def _apply_union_metadata(
    schema_data: dict[str, JSONValue],
    subschemas: Sequence[dict[str, JSONValue]],
    base_schema_ref: Mapping[str, JSONValue] | None,
) -> None:
    non_null_types = [
        subschema.get("type")
        for subschema in subschemas
        if isinstance(subschema.get("type"), str)
        and subschema.get("type") != NULL_JSON_TYPE
    ]
    if non_null_types and len(set(non_null_types)) == 1:
        schema_data["type"] = non_null_types[0]
    if len(subschemas) == 1 and base_schema_ref is None:
        title = subschemas[0].get("title")
        if isinstance(title, str):  # pragma: no cover - not triggered in tests
            _ = schema_data.setdefault("title", title)
        required = subschemas[0].get("required")
        if isinstance(required, (list, tuple)):  # pragma: no cover - defensive
            _ = schema_data.setdefault("required", list(required))


def _schema_for_union(
    base_type: object, alias_generator: Callable[[str], str] | None, origin: object
) -> dict[str, JSONValue] | None:
    if origin is not UnionType:
        return None

    includes_null, subschemas, base_schema_ref = _collect_union_subschemas(
        base_type, alias_generator
    )
    schema_data = _merge_union_schema(subschemas, base_schema_ref, includes_null)
    _apply_union_metadata(schema_data, subschemas, base_schema_ref)
    return schema_data


def _collection_item_type(base_type: object, index: int = 0) -> object:
    args = get_args(base_type)
    return args[index] if len(args) > index else object


def _list_schema(
    base_type: object, alias_generator: Callable[[str], str] | None
) -> dict[str, JSONValue]:
    item_type = _collection_item_type(base_type)
    return {
        "type": "array",
        "items": _schema_for_type(item_type, None, alias_generator),
    }


def _set_schema(
    base_type: object, alias_generator: Callable[[str], str] | None
) -> dict[str, JSONValue]:
    schema_data = _list_schema(base_type, alias_generator)
    schema_data["uniqueItems"] = True
    return schema_data


def _tuple_schema(
    base_type: object, alias_generator: Callable[[str], str] | None
) -> dict[str, JSONValue]:
    args = get_args(base_type)
    if args and args[-1] is ELLIPSIS_SENTINEL:
        return {
            "type": "array",
            "items": _schema_for_type(args[0], None, alias_generator),
        }
    return {
        "type": "array",
        "prefixItems": [_schema_for_type(arg, None, alias_generator) for arg in args],
        "minItems": len(args),
        "maxItems": len(args),
    }


def _mapping_schema(
    base_type: object, alias_generator: Callable[[str], str] | None
) -> dict[str, JSONValue]:
    value_type = _collection_item_type(base_type, index=1)
    return {
        "type": "object",
        "additionalProperties": _schema_for_type(value_type, None, alias_generator),
    }


_COLLECTION_BUILDERS: Mapping[
    object, Callable[[object, Callable[[str], str] | None], dict[str, JSONValue]]
] = {
    list: _list_schema,
    Sequence: _list_schema,
    set: _set_schema,
    tuple: _tuple_schema,
    dict: _mapping_schema,
    Mapping: _mapping_schema,
}


def _schema_for_collection(
    base_type: object,
    alias_generator: Callable[[str], str] | None,
    origin: object,
) -> dict[str, JSONValue] | None:
    builder = _COLLECTION_BUILDERS.get(origin)
    if builder is None:
        return None
    return builder(base_type, alias_generator)


def _schema_for_enum(
    base_type: object, alias_generator: Callable[[str], str] | None, origin: object
) -> dict[str, JSONValue] | None:
    if not isinstance(base_type, type) or not issubclass(base_type, Enum):
        return None

    enum_values = [member.value for member in base_type]
    schema_data: dict[str, JSONValue] = {"enum": enum_values}
    if not enum_values:
        return schema_data  # pragma: no cover - defensive

    if all(isinstance(value, str) for value in enum_values):
        schema_data["type"] = "string"
    elif all(isinstance(value, bool) for value in enum_values):
        schema_data["type"] = "boolean"
    elif all(
        isinstance(value, int) and not isinstance(value, bool) for value in enum_values
    ):
        schema_data["type"] = "integer"
    elif all(isinstance(value, (float, Decimal)) for value in enum_values):
        schema_data["type"] = "number"
    return schema_data


def _schema_for_primitive(
    base_type: object, alias_generator: Callable[[str], str] | None, origin: object
) -> dict[str, JSONValue] | None:
    for primitive, schema_data in PRIMITIVE_FORMATS.items():
        if base_type is primitive:
            return dict(schema_data)
    return None


def schema[SchemaDataclass: SupportsDataclass](
    cls: type[SchemaDataclass],
    *,
    alias_generator: Callable[[str], str] | None = None,
    extra: Literal["ignore", "forbid", "allow"] = IGNORE_EXTRA,
) -> dict[str, JSONValue]:
    """Produce a minimal JSON Schema description for a dataclass."""

    if not dataclasses.is_dataclass(cls):
        raise TypeError("schema() requires a dataclass type")
    if extra not in EXTRA_MODES:
        raise ValueError("extra must be one of 'ignore', 'forbid', or 'allow'")

    properties: dict[str, dict[str, JSONValue]] = {}
    required: list[str] = []
    type_hints = get_type_hints(cls, include_extras=True)

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_meta = dict(field.metadata)
        alias = field_meta.get("alias")
        if alias_generator is not None and not alias:
            alias = alias_generator(field.name)
        property_name = alias or field.name
        field_type = type_hints.get(field.name, field.type)
        properties[property_name] = _schema_for_type(
            field_type, field_meta, alias_generator
        )
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(property_name)

    schema_dict: dict[str, JSONValue] = {
        "title": cls.__name__,
        "type": "object",
        "properties": properties,
        "additionalProperties": extra != FORBID_EXTRA,
    }
    if required:
        schema_dict["required"] = required
    if not required:
        _ = schema_dict.pop("required", None)
    return schema_dict


__all__ = ["schema"]
