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

"""Type coercion functions for serde parsing."""

# pyright: reportImportCycles=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnnecessaryIsInstance=false, reportCallIssue=false, reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    Union as _TypingUnion,  # pyright: ignore[reportDeprecated]
    cast,
    get_args,
    get_origin,
)
from uuid import UUID

from ..types import JSONValue
from ._generics import _build_typevar_map, _is_typevar
from ._utils import (
    _UNION_TYPE,
    _AnyType,
    _apply_constraints,
    _build_item_meta,
    _merge_annotated_meta,
    _ParseConfig,
)

# typing.Union origin (for Optional[X] and Union[X, Y] constructs from type aliases)
_TYPING_UNION = _TypingUnion


def _bool_from_str(value: str) -> bool:
    lowered = value.strip().lower()
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise TypeError(f"Cannot interpret '{value}' as boolean")


_NOT_HANDLED = object()


def _decimal_from_any(value: object) -> object:
    return Decimal(str(value))


def _uuid_from_any(value: object) -> object:
    return UUID(str(value))


def _path_from_any(value: object) -> object:
    return Path(str(value))


def _datetime_from_any(value: object) -> object:
    return datetime.fromisoformat(str(value))


def _date_from_any(value: object) -> object:
    return date.fromisoformat(str(value))


def _time_from_any(value: object) -> object:
    return time.fromisoformat(str(value))


_PRIMITIVE_COERCERS: dict[type[object], object] = cast(
    dict[type[object], object],
    {
        int: int,
        float: float,
        str: str,
        Decimal: _decimal_from_any,
        UUID: _uuid_from_any,
        Path: _path_from_any,
        datetime: _datetime_from_any,
        date: _date_from_any,
        time: _time_from_any,
    },
)


def _is_union_type(origin: object) -> bool:
    """Check if origin is a union type (either types.UnionType or typing.Union)."""
    return origin is _UNION_TYPE or origin is _TYPING_UNION


def _is_empty_string_coercible_to_none(
    value: object, base_type: object, config: _ParseConfig
) -> bool:
    """Check if empty string can be coerced to None in an optional union."""
    return (
        config.coerce
        and isinstance(value, str)
        and value.strip() == ""
        and any(arg is type(None) for arg in get_args(base_type))
    )


def _raise_union_error(last_error: Exception, path: str) -> None:
    """Raise an appropriately prefixed error from union coercion."""
    message = str(last_error)
    if message.startswith(f"{path}:") or message.startswith(f"{path}."):
        raise last_error
    if isinstance(last_error, TypeError):
        raise TypeError(f"{path}: {message}") from last_error
    raise ValueError(f"{path}: {message}") from last_error


def _coerce_union(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if not _is_union_type(origin):
        return _NOT_HANDLED
    if _is_empty_string_coercible_to_none(value, base_type, config):
        return _apply_constraints(None, merged_meta, path)
    last_error: Exception | None = None
    for arg in get_args(base_type):
        if arg is type(None):
            if value is None:
                return _apply_constraints(None, merged_meta, path)
            continue
        try:
            coerced = _coerce_to_type(
                value, arg, _build_item_meta(merged_meta, arg), path, config
            )
        except (TypeError, ValueError) as error:
            last_error = error
            continue
        return _apply_constraints(coerced, merged_meta, path)
    if last_error is not None:
        _raise_union_error(last_error, path)
    raise TypeError(f"{path}: no matching type in Union")


def _try_coerce_literal(
    value: object, literal: object
) -> tuple[object | None, Exception | None]:
    """Attempt to coerce value to match a literal. Returns (coerced, error)."""
    literal_type = type(literal)
    try:
        if isinstance(literal, bool) and isinstance(value, str):
            coerced = _bool_from_str(value)
        else:
            coerced = literal_type(value)
    except (TypeError, ValueError) as error:
        return None, error
    else:
        return coerced, None


def _coerce_literal(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if origin is not Literal:
        return _NOT_HANDLED
    literals = get_args(base_type)
    last_literal_error: Exception | None = None
    for literal in literals:
        if value == literal:
            return _apply_constraints(literal, merged_meta, path)
        if not config.coerce:
            continue  # pragma: no cover - coerce=False rarely used with Literal
        coerced_literal, error = _try_coerce_literal(value, literal)
        if error is not None:
            last_literal_error = error
            continue
        if coerced_literal == literal:
            return _apply_constraints(literal, merged_meta, path)
    if last_literal_error is not None:
        raise type(last_literal_error)(
            f"{path}: {last_literal_error}"
        ) from last_literal_error
    raise ValueError(f"{path}: expected one of {list(literals)}")


def _coerce_none(value: object, base_type: object, path: str) -> object:
    if base_type is type(None):
        if value is not None:
            raise TypeError(f"{path}: expected None")
        return None
    if value is None:
        raise TypeError(f"{path}: value cannot be None")
    return _NOT_HANDLED


def _coerce_primitive(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    coercer = _PRIMITIVE_COERCERS.get(cast(type[object], base_type))
    if coercer is None:
        return _NOT_HANDLED

    literal_type = cast(type[object], base_type)
    if isinstance(value, literal_type):
        return _apply_constraints(value, merged_meta, path)
    if not config.coerce:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: expected {type_name}")
    try:
        coerced_value = coercer(value)
    except Exception as error:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: unable to coerce {value!r} to {type_name}") from error
    return _apply_constraints(coerced_value, merged_meta, path)


def _build_nested_config(base_type: object, config: _ParseConfig) -> _ParseConfig:
    """Build config with typevar_map for nested generic type parsing."""
    origin = get_origin(base_type)
    if origin is None:
        return config
    nested_typevar_map = _build_typevar_map(
        cast(type[object], base_type), parent_typevar_map=config.typevar_map
    )
    if not nested_typevar_map:  # pragma: no cover - always true for PEP 695 generics
        return config
    return _ParseConfig(
        extra=config.extra,
        coerce=config.coerce,
        typevar_map=nested_typevar_map,
    )


def _format_nested_error(error: Exception, path: str) -> str:
    """Format error message with proper path prefix."""
    message = str(error)
    if ":" in message:
        prefix, suffix = message.split(":", 1)
        if " " not in prefix:
            return f"{path}.{prefix}:{suffix}"
    return f"{path}: {message}"


def _coerce_dataclass(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    from .parse import _parse_dataclass

    # Handle generic aliases like MyClass[int] - extract the origin class
    origin = get_origin(base_type)
    target_type = origin if origin is not None else base_type

    if not dataclasses.is_dataclass(target_type):
        return _NOT_HANDLED

    dataclass_type = target_type if isinstance(target_type, type) else type(target_type)
    if isinstance(value, dataclass_type):
        return _apply_constraints(value, merged_meta, path)
    if not isinstance(value, Mapping):
        type_name = getattr(dataclass_type, "__name__", type(dataclass_type).__name__)
        raise TypeError(f"{path}: expected mapping for dataclass {type_name}")

    nested_data = cast(Mapping[str, object], value)
    nested_config = _build_nested_config(base_type, config)

    try:
        parsed = _parse_dataclass(dataclass_type, nested_data, config=nested_config)
    except (TypeError, ValueError) as error:
        raise type(error)(_format_nested_error(error, path)) from error
    return _apply_constraints(parsed, merged_meta, path)


def _normalize_list_like(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce and isinstance(value, str):
        return [value]
    raise TypeError(f"{path}: expected sequence")


def _normalize_set_value(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, (set, list, tuple)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return list(cast(Iterable[JSONValue], value))
    raise TypeError(f"{path}: expected set")


def _normalize_tuple_value(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce and isinstance(value, str):
        return [value]
    raise TypeError(f"{path}: expected tuple")


def _normalize_sequence_value(
    value: object, origin: type[object] | None, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if origin in {list, Sequence}:
        return _normalize_list_like(value, path, config)
    if origin is set:
        return _normalize_set_value(value, path, config)
    return _normalize_tuple_value(value, path, config)


def _resolve_sequence_origin(
    base_type: object,
) -> tuple[type[object] | None, bool]:
    """Return (effective_origin, is_bare) for a potential sequence type."""
    origin = cast(type[object] | None, get_origin(base_type))
    if origin is list or origin is Sequence or origin is tuple or origin is set:
        return origin, False
    # Also handle bare types (list, tuple, set) which have no origin.
    # Use identity checks to avoid hashing unhashable generic aliases.
    if base_type is list:
        return list, True
    if base_type is tuple:
        return tuple, True
    if base_type is set:
        return set, True
    return None, False


def _resolve_item_types(
    args: tuple[object, ...],
    origin: type[object] | None,
    count: int,
) -> list[object]:
    """Resolve the expected type for each item in a sequence."""
    if origin is tuple and args and args[-1] is not Ellipsis:
        if len(args) != count:
            raise ValueError(f"expected {len(args)} items")
        return list(args)
    if origin is tuple and args:
        return [args[0]] * count
    return [args[0] if args else object] * count


def _coerce_sequence(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    effective_origin, is_bare = _resolve_sequence_origin(base_type)
    if effective_origin is None:
        return _NOT_HANDLED
    args = get_args(base_type)
    # Reject unparameterized collections (bare list/tuple/set)
    if not args and is_bare:
        type_name = getattr(effective_origin, "__name__", "sequence")
        raise TypeError(
            f"{path}: bare {type_name} is not allowed; use {type_name}[<element_type>] instead"
        )
    items = _normalize_sequence_value(value, effective_origin, path, config)
    try:
        item_types = _resolve_item_types(args, effective_origin, len(items))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    coerced_items: list[object] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        item_type = item_types[index]
        item_meta = _build_item_meta(merged_meta, item_type)
        coerced_items.append(
            _coerce_to_type(item, item_type, item_meta, item_path, config)
        )
    if effective_origin is set:
        value_out: object = set(coerced_items)
    elif effective_origin is tuple:
        value_out = tuple(coerced_items)
    else:
        value_out = list(coerced_items)
    return _apply_constraints(value_out, merged_meta, path)


def _coerce_mapping(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    # Also handle bare dict type which has no origin
    if origin is not dict and origin is not Mapping and base_type is not dict:
        return _NOT_HANDLED
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping")
    args = get_args(base_type)
    if args:
        key_type, value_type = args
    elif base_type is dict and origin is None:
        # Bare dict without type parameters
        raise TypeError(
            f"{path}: bare dict is not allowed; use dict[<key_type>, <value_type>] instead"
        )
    else:
        key_type, value_type = (object, object)  # pragma: no cover - defensive
    mapping_value = cast(Mapping[JSONValue, JSONValue], value)
    result_dict: dict[object, object] = {}
    key_meta = _build_item_meta(merged_meta, key_type)
    val_meta = _build_item_meta(merged_meta, value_type)
    for key, item in mapping_value.items():
        coerced_key = _coerce_to_type(key, key_type, key_meta, f"{path} keys", config)
        coerced_value = _coerce_to_type(
            item, value_type, val_meta, f"{path}[{coerced_key}]", config
        )
        result_dict[coerced_key] = coerced_value
    return _apply_constraints(result_dict, merged_meta, path)


def _coerce_enum(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if not (isinstance(base_type, type) and issubclass(base_type, Enum)):
        return _NOT_HANDLED
    if isinstance(value, base_type):
        enum_value = value
    elif config.coerce:
        if isinstance(value, str):
            try:
                enum_value = base_type[value]
            except KeyError:
                try:
                    enum_value = base_type(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: invalid enum value {value!r}") from error
        else:
            try:
                enum_value = base_type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid enum value {value!r}") from error
    else:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: expected {type_name}")
    return _apply_constraints(enum_value, merged_meta, path)


def _coerce_bool(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if base_type is not bool:
        return _NOT_HANDLED
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


def _coerce_to_type(
    value: object,
    typ: object,
    meta: Mapping[str, object] | None,
    path: str,
    config: _ParseConfig,
) -> object:
    base_type, merged_meta = _merge_annotated_meta(typ, meta)

    # Resolve TypeVar from typevar_map (populated from generic alias type args)
    if _is_typevar(base_type) and base_type in config.typevar_map:
        base_type = config.typevar_map[base_type]

    if base_type is object or base_type is _AnyType:
        if merged_meta.get("untyped", False) is not True:
            raise TypeError(
                f'{path}: unbound type (Any/object) requires {{"untyped": True}} in Annotated metadata'
            )
        return _apply_constraints(value, merged_meta, path)

    # Check for unresolved TypeVar - this means the caller didn't provide
    # a fully specialized generic alias
    if _is_typevar(base_type):
        msg = (
            f"{path}: cannot parse TypeVar field - use a fully specialized "
            f"generic type like MyClass[ConcreteType] instead of MyClass"
        )
        raise TypeError(msg)

    coercers = (
        lambda: _coerce_union(value, base_type, merged_meta, path, config),
        lambda: _coerce_none(value, base_type, path),
        lambda: _coerce_literal(value, base_type, merged_meta, path, config),
        lambda: _coerce_dataclass(value, base_type, merged_meta, path, config),
        lambda: _coerce_sequence(value, base_type, merged_meta, path, config),
        lambda: _coerce_mapping(value, base_type, merged_meta, path, config),
        lambda: _coerce_enum(value, base_type, merged_meta, path, config),
        lambda: _coerce_bool(value, base_type, merged_meta, path, config),
        lambda: _coerce_primitive(value, base_type, merged_meta, path, config),
    )
    for coercer in coercers:
        result = coercer()
        if result is not _NOT_HANDLED:
            return result

    try:
        coerced = base_type(value)
    except Exception as error:
        raise type(error)(str(error)) from error
    return _apply_constraints(coerced, merged_meta, path)
