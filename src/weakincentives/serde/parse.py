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

"""Dataclass parsing helpers."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnnecessaryIsInstance=false, reportCallIssue=false, reportArgumentType=false, reportPossiblyUnboundVariable=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    cast,
    get_args as typing_get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from ..types import JSONValue
from ._utils import (
    _UNION_TYPE,
    _AnyType,
    _apply_constraints,
    _merge_annotated_meta,
    _ParseConfig,
    _set_extras,
)

get_args = typing_get_args


_UNHANDLED = object()


def _bool_from_str(value: str) -> bool:
    lowered = value.strip().lower()
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise TypeError(f"Cannot interpret '{value}' as boolean")


def _coerce_union(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if origin is not _UNION_TYPE:
        return _UNHANDLED
    if (
        config.coerce
        and isinstance(value, str)
        and value.strip() == ""
        and any(arg is type(None) for arg in get_args(base_type))
    ):
        return _apply_constraints(None, merged_meta, path)
    last_error: Exception | None = None
    for arg in get_args(base_type):
        if arg is type(None):
            if value is None:
                return _apply_constraints(None, merged_meta, path)
            continue
        try:
            coerced = _coerce_to_type(value, arg, None, path, config)
        except (TypeError, ValueError) as error:
            last_error = error
            continue
        return _apply_constraints(coerced, merged_meta, path)
    if last_error is not None:
        message = str(last_error)
        if message.startswith(f"{path}:") or message.startswith(f"{path}."):
            raise last_error
        if isinstance(last_error, TypeError):
            raise TypeError(f"{path}: {message}") from last_error
        raise ValueError(f"{path}: {message}") from last_error
    raise TypeError(f"{path}: no matching type in Union")


def _coerce_none_value(
    value: object, base_type: object, path: str, _merged_meta: Mapping[str, object]
) -> object:
    if base_type is type(None):
        if value is not None:
            raise TypeError(f"{path}: expected None")
        return None
    if value is None:
        raise TypeError(f"{path}: value cannot be None")
    return _UNHANDLED


def _coerce_literal(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if get_origin(base_type) is not Literal:
        return _UNHANDLED

    literals = get_args(base_type)
    matched_literal, last_literal_error = _match_literal_value(value, literals, config)
    if matched_literal is not None:
        return _apply_constraints(matched_literal, merged_meta, path)
    if last_literal_error is not None:
        if isinstance(last_literal_error, TypeError):
            message = str(last_literal_error)
            if not message.startswith(f"{path}:"):
                message = f"{path}: {message}"
            raise TypeError(message) from last_literal_error
        raise last_literal_error
    raise ValueError(f"{path}: expected one of {list(literals)}")


def _convert_literal_value(literal: object, value: object) -> object:
    literal_type = cast(type[object], type(literal))
    if isinstance(literal, bool) and isinstance(value, str):
        return _bool_from_str(value)
    return literal_type(value)


def _match_literal_value(
    value: object, literals: tuple[object, ...], config: _ParseConfig
) -> tuple[object | None, Exception | None]:
    last_literal_error: Exception | None = None
    for literal in literals:
        if value == literal:
            return literal, None
        if not config.coerce:
            continue
        try:
            coerced_literal = _convert_literal_value(literal, value)
        except Exception as error:  # pragma: no cover - defensive guard
            last_literal_error = error
            continue
        if coerced_literal == literal:
            return literal, None
    return None, last_literal_error


def _coerce_bool(
    value: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if isinstance(value, bool):
        return _apply_constraints(value, merged_meta, path)
    if config.coerce and isinstance(value, str):
        try:
            coerced_bool = _bool_from_str(value)
        except TypeError as error:
            raise TypeError(f"{path}: {error}") from error
        return _apply_constraints(coerced_bool, merged_meta, path)
    if config.coerce and isinstance(value, (int, float)):
        return _apply_constraints(bool(value), merged_meta, path)
    raise TypeError(f"{path}: expected bool")


def _coerce_primitives(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if base_type is bool:
        return _coerce_bool(value, merged_meta, path, config)

    constructors: Mapping[type[object], Callable[[object], object]] = {
        int: int,
        float: float,
        str: str,
        Decimal: lambda raw: Decimal(str(raw)),
        UUID: lambda raw: UUID(str(raw)),
        Path: lambda raw: Path(str(raw)),
        datetime: lambda raw: datetime.fromisoformat(str(raw)),
        date: lambda raw: date.fromisoformat(str(raw)),
        time: lambda raw: time.fromisoformat(str(raw)),
    }

    literal_type = cast(type[object], base_type)
    constructor = constructors.get(literal_type)
    if constructor is None:
        return _UNHANDLED

    if isinstance(value, literal_type):
        return _apply_constraints(value, merged_meta, path)
    if not config.coerce:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: expected {type_name}")
    try:
        coerced_value = constructor(value)
    except Exception as error:
        raise TypeError(
            f"{path}: unable to coerce {value!r} to {literal_type.__name__}"
        ) from error
    return _apply_constraints(coerced_value, merged_meta, path)


def _coerce_dataclass_value(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if not dataclasses.is_dataclass(base_type):
        return _UNHANDLED

    dataclass_type = base_type if isinstance(base_type, type) else type(base_type)
    if isinstance(value, dataclass_type):
        return _apply_constraints(value, merged_meta, path)
    if not isinstance(value, Mapping):
        type_name = getattr(dataclass_type, "__name__", type(dataclass_type).__name__)
        raise TypeError(f"{path}: expected mapping for dataclass {type_name}")
    try:
        parsed = parse(
            cast(type[object], dataclass_type),
            cast(Mapping[str, object], value),
            extra=config.extra,
            coerce=config.coerce,
            case_insensitive=config.case_insensitive,
            alias_generator=config.alias_generator,
            aliases=config.aliases,
        )
    except (TypeError, ValueError) as error:
        message = str(error)
        if ":" in message:
            prefix, suffix = message.split(":", 1)
            if " " not in prefix:
                message = f"{path}.{prefix}:{suffix}"
            else:
                message = f"{path}: {message}"
        else:
            message = f"{path}: {message}"
        raise type(error)(message) from error
    return _apply_constraints(parsed, merged_meta, path)


def _normalize_sequence_input(
    value: object, origin: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    is_sequence_like = isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )
    if origin in {list, Sequence} and not is_sequence_like:
        if config.coerce and isinstance(value, str):
            return [value]
        raise TypeError(f"{path}: expected sequence")
    if origin in {set, tuple} and not is_sequence_like:
        if config.coerce and isinstance(value, str):
            return [value]
        if origin is set and config.coerce and isinstance(value, Iterable):
            return list(cast(Iterable[JSONValue], value))
        expected = "set" if origin is set else "tuple"
        raise TypeError(f"{path}: expected {expected}")

    if isinstance(value, str):  # pragma: no cover - handled by earlier coercion
        return [value]
    if isinstance(value, Iterable):
        return list(cast(Iterable[JSONValue], value))
    raise TypeError(f"{path}: expected iterable")


def _coerce_sequence_items(
    items: list[JSONValue],
    args: tuple[object, ...],
    origin: object,
    path: str,
    config: _ParseConfig,
) -> list[object]:
    if (
        origin is tuple
        and args
        and args[-1] is not Ellipsis
        and len(args) != len(items)
    ):
        raise ValueError(f"{path}: expected {len(args)} items")
    coerced_items: list[object] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if origin is tuple and args:
            item_type = args[0] if args[-1] is Ellipsis else args[index]
        else:
            item_type = args[0] if args else object
        coerced_items.append(_coerce_to_type(item, item_type, None, item_path, config))
    return coerced_items


def _coerce_sequence_value(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
    origin: object,
) -> object:
    if origin not in {list, Sequence, tuple, set}:
        return _UNHANDLED

    items = _normalize_sequence_input(value, origin, path, config)
    args = get_args(base_type)
    coerced_items = _coerce_sequence_items(items, args, origin, path, config)
    if origin is set:
        value_out: object = set(coerced_items)
    elif origin is tuple:
        value_out = tuple(coerced_items)
    else:
        value_out = list(coerced_items)
    return _apply_constraints(value_out, merged_meta, path)


def _coerce_mapping_value(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
    origin: object,
) -> object:
    if origin is not dict and origin is not Mapping:
        return _UNHANDLED

    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping")
    key_type, value_type = (
        get_args(base_type) if get_args(base_type) else (object, object)
    )
    mapping_value = cast(Mapping[JSONValue, JSONValue], value)
    result_dict: dict[object, object] = {}
    for key, item in mapping_value.items():
        coerced_key = _coerce_to_type(key, key_type, None, f"{path} keys", config)
        coerced_value = _coerce_to_type(
            item, value_type, None, f"{path}[{coerced_key}]", config
        )
        result_dict[coerced_key] = coerced_value
    return _apply_constraints(result_dict, merged_meta, path)


def _coerce_enum_value(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
    type_name: str,
) -> object:
    if not isinstance(base_type, type) or not issubclass(base_type, Enum):
        return _UNHANDLED

    if isinstance(value, base_type):
        enum_value = value
    elif config.coerce:
        try:
            enum_value = base_type[value]
        except KeyError:
            try:
                enum_value = base_type(value)
            except ValueError as error:
                raise ValueError(f"{path}: invalid enum value {value!r}") from error
        except TypeError:
            try:
                enum_value = base_type(value)
            except ValueError as error:
                raise ValueError(f"{path}: invalid enum value {value!r}") from error
    else:
        raise TypeError(f"{path}: expected {type_name}")
    return _apply_constraints(enum_value, merged_meta, path)


def _coerce_to_type(
    value: object,
    typ: object,
    meta: Mapping[str, object] | None,
    path: str,
    config: _ParseConfig,
) -> object:
    base_type, merged_meta = _merge_annotated_meta(typ, meta)
    origin = get_origin(base_type)
    type_name = getattr(base_type, "__name__", type(base_type).__name__)

    if base_type is object or base_type is _AnyType:
        return _apply_constraints(value, merged_meta, path)

    for result in (
        _coerce_union(value, base_type, merged_meta, path, config),
        _coerce_none_value(value, base_type, path, merged_meta),
        _coerce_literal(value, base_type, merged_meta, path, config),
        _coerce_dataclass_value(value, base_type, merged_meta, path, config),
        _coerce_sequence_value(value, base_type, merged_meta, path, config, origin),
        _coerce_mapping_value(value, base_type, merged_meta, path, config, origin),
        _coerce_enum_value(value, base_type, merged_meta, path, config, type_name),
        _coerce_primitives(value, base_type, merged_meta, path, config),
    ):
        if result is not _UNHANDLED:
            return result

    try:
        coerced = base_type(value)
    except Exception as error:
        raise type(error)(str(error)) from error
    return _apply_constraints(coerced, merged_meta, path)


def _find_key(
    data: Mapping[str, object], name: str, alias: str | None, case_insensitive: bool
) -> str | None:
    candidates = [alias, name]
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in data:
            return candidate
    if not case_insensitive:
        return None
    lowered_map: dict[str, str] = {}
    for key in data:
        if isinstance(key, str):
            _ = lowered_map.setdefault(key.lower(), key)
    for candidate in candidates:
        if candidate is None or not isinstance(candidate, str):
            continue
        lowered = candidate.lower()
        if lowered in lowered_map:
            return lowered_map[lowered]
    return None


def parse[T](  # noqa: C901
    cls: type[T],
    data: Mapping[str, object] | object,
    *,
    extra: Literal["ignore", "forbid", "allow"] = "ignore",
    coerce: bool = True,
    case_insensitive: bool = False,
    alias_generator: Callable[[str], str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> T:
    """Parse a mapping into a dataclass instance."""

    if not dataclasses.is_dataclass(cls) or not isinstance(cls, type):
        raise TypeError("parse() requires a dataclass type")
    if not isinstance(data, Mapping):
        raise TypeError("parse() requires a mapping input")
    if extra not in {"ignore", "forbid", "allow"}:
        raise ValueError("extra must be one of 'ignore', 'forbid', or 'allow'")

    config = _ParseConfig(
        extra=extra,
        coerce=coerce,
        case_insensitive=case_insensitive,
        alias_generator=alias_generator,
        aliases=aliases,
    )

    mapping_data = cast(Mapping[str, object], data)
    type_hints = get_type_hints(cls, include_extras=True)
    kwargs: dict[str, object] = {}
    used_keys: set[str] = set()

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_meta = dict(field.metadata)
        field_alias = None
        if aliases and field.name in aliases:
            field_alias = aliases[field.name]
        elif (alias := field_meta.get("alias")) is not None:
            field_alias = alias
        elif alias_generator is not None:
            field_alias = alias_generator(field.name)

        key = _find_key(mapping_data, field.name, field_alias, case_insensitive)
        if key is None:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"Missing required field: '{field.name}'")
            continue
        used_keys.add(key)
        raw_value = mapping_data[key]
        field_type = type_hints.get(field.name, field.type)
        try:
            value = _coerce_to_type(
                raw_value, field_type, field_meta, field.name, config
            )
        except (TypeError, ValueError) as error:
            raise type(error)(str(error)) from error
        kwargs[field.name] = value

    instance = cls(**kwargs)

    extras = {key: mapping_data[key] for key in mapping_data if key not in used_keys}
    if extras:
        if extra == "forbid":
            raise ValueError(f"Extra keys not permitted: {list(extras.keys())}")
        if extra == "allow":
            if hasattr(instance, "__dict__"):
                for key, value in extras.items():
                    object.__setattr__(instance, key, value)
            else:
                _set_extras(instance, extras)

    validator = getattr(instance, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(instance, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()

    return instance


__all__ = ["parse"]
