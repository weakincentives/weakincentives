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

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnnecessaryIsInstance=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints
from uuid import UUID

from ._utils import _UNION_TYPE, _AnyType, _merge_annotated_meta, _ordered_values


def _schema_constraints(meta: Mapping[str, object]) -> dict[str, object]:
    schema_meta: dict[str, object] = {}
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
            schema_meta[target] = meta[key]
    members = meta.get("enum") or meta.get("in")
    if isinstance(members, Iterable) and not isinstance(members, (str, bytes)):
        _ = schema_meta.setdefault("enum", _ordered_values(members))
    not_members = meta.get("not_in")
    if (
        isinstance(not_members, Iterable)
        and not isinstance(not_members, (str, bytes))
        and "not" not in schema_meta
    ):
        schema_meta["not"] = {"enum": _ordered_values(not_members)}
    return schema_meta


def _schema_for_type(
    typ: object,
    meta: Mapping[str, object] | None,
    alias_generator: Callable[[str], str] | None,
) -> dict[str, object]:
    base_type, merged_meta = _merge_annotated_meta(typ, meta)
    origin = get_origin(base_type)

    if base_type is object or base_type is _AnyType:
        schema_data: dict[str, object] = {}
    elif dataclasses.is_dataclass(base_type):
        dataclass_type = base_type if isinstance(base_type, type) else type(base_type)
        schema_data = schema(dataclass_type, alias_generator=alias_generator)
    elif base_type is type(None):
        schema_data = {"type": "null"}
    elif isinstance(base_type, type) and issubclass(base_type, Enum):
        enum_values = [member.value for member in base_type]
        schema_data = {"enum": enum_values}
        if enum_values:
            if all(isinstance(value, str) for value in enum_values):
                schema_data["type"] = "string"
            elif all(isinstance(value, bool) for value in enum_values):
                schema_data["type"] = "boolean"
            elif all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in enum_values
            ):
                schema_data["type"] = "integer"
            elif all(isinstance(value, (float, Decimal)) for value in enum_values):
                schema_data["type"] = "number"
    elif base_type is bool:
        schema_data = {"type": "boolean"}
    elif base_type is int:
        schema_data = {"type": "integer"}
    elif base_type in {float, Decimal}:
        schema_data = {"type": "number"}
    elif base_type is str:
        schema_data = {"type": "string"}
    elif base_type is datetime:
        schema_data = {"type": "string", "format": "date-time"}
    elif base_type is date:
        schema_data = {"type": "string", "format": "date"}
    elif base_type is time:
        schema_data = {"type": "string", "format": "time"}
    elif base_type is UUID:
        schema_data = {"type": "string", "format": "uuid"}
    elif base_type is Path:
        schema_data = {"type": "string"}
    elif origin is Literal:
        literal_values = list(get_args(base_type))
        schema_data = {"enum": literal_values}
        if literal_values:
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
    elif origin in {list, Sequence}:
        item_type = get_args(base_type)[0] if get_args(base_type) else object
        schema_data = {
            "type": "array",
            "items": _schema_for_type(item_type, None, alias_generator),
        }
    elif origin is set:
        item_type = get_args(base_type)[0] if get_args(base_type) else object
        schema_data = {
            "type": "array",
            "items": _schema_for_type(item_type, None, alias_generator),
            "uniqueItems": True,
        }
    elif origin is tuple:
        args = get_args(base_type)
        if args and args[-1] is Ellipsis:
            schema_data = {
                "type": "array",
                "items": _schema_for_type(args[0], None, alias_generator),
            }
        else:
            schema_data = {
                "type": "array",
                "prefixItems": [
                    _schema_for_type(arg, None, alias_generator) for arg in args
                ],
                "minItems": len(args),
                "maxItems": len(args),
            }
    elif origin in {dict, Mapping}:
        args = get_args(base_type)
        value_type = args[1] if len(args) == 2 else object
        schema_data = {
            "type": "object",
            "additionalProperties": _schema_for_type(value_type, None, alias_generator),
        }
    elif origin is _UNION_TYPE:
        subschemas = []
        includes_null = False
        base_schema_ref: Mapping[str, object] | None = None
        for arg in get_args(base_type):
            if arg is type(None):
                includes_null = True
                continue
            subschema = _schema_for_type(arg, None, alias_generator)
            subschemas.append(subschema)
            if (
                base_schema_ref is None
                and isinstance(subschema, Mapping)
                and subschema.get("type") == "object"
            ):
                base_schema_ref = subschema
        any_of = list(subschemas)
        if includes_null:
            any_of.append({"type": "null"})
        if base_schema_ref is not None and len(subschemas) == 1:
            schema_data = dict(base_schema_ref)
        else:
            schema_data = {}
        schema_data["anyOf"] = any_of
        non_null_types = [
            subschema.get("type")
            for subschema in subschemas
            if isinstance(subschema.get("type"), str)
            and subschema.get("type") != "null"
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
    else:
        schema_data = {}

    schema_data.update(_schema_constraints(merged_meta))
    return schema_data


def schema(
    cls: type[object],
    *,
    alias_generator: Callable[[str], str] | None = None,
    extra: Literal["ignore", "forbid", "allow"] = "ignore",
) -> dict[str, object]:
    """Produce a minimal JSON Schema description for a dataclass."""

    if not dataclasses.is_dataclass(cls) or not isinstance(cls, type):
        raise TypeError("schema() requires a dataclass type")
    if extra not in {"ignore", "forbid", "allow"}:
        raise ValueError("extra must be one of 'ignore', 'forbid', or 'allow'")

    properties: dict[str, object] = {}
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

    schema_dict: dict[str, object] = {
        "title": cls.__name__,
        "type": "object",
        "properties": properties,
        "additionalProperties": extra != "forbid",
    }
    if required:
        schema_dict["required"] = required
    if not required:
        _ = schema_dict.pop("required", None)
    return schema_dict


__all__ = ["schema"]
