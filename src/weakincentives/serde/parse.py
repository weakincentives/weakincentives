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
    TypeVar,
    Union as _TypingUnion,  # pyright: ignore[reportDeprecated]
    cast,
    get_args as typing_get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from ..types import JSONValue
from ._utils import (
    _UNION_TYPE,
    TYPE_REF_KEY,
    _AnyType,
    _apply_constraints,
    _merge_annotated_meta,
    _ParseConfig,
    _resolve_type_identifier,
    _set_extras,
    _type_identifier,
)

# typing.Union origin (for Optional[X] and Union[X, Y] constructs from type aliases)
_TYPING_UNION = _TypingUnion

get_args = typing_get_args


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


_PRIMITIVE_COERCERS: dict[type[object], Callable[[object], object]] = cast(
    dict[type[object], Callable[[object], object]],
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
            coerced = _coerce_to_type(value, arg, None, path, config)
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


def _coerce_dataclass(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if not dataclasses.is_dataclass(base_type):
        return _NOT_HANDLED
    dataclass_type = base_type if isinstance(base_type, type) else type(base_type)
    if isinstance(value, dataclass_type):
        return _apply_constraints(value, merged_meta, path)
    if not isinstance(value, Mapping):
        type_name = getattr(dataclass_type, "__name__", type(dataclass_type).__name__)
        raise TypeError(f"{path}: expected mapping for dataclass {type_name}")

    # Strip type_key from nested data if present - we already know the target type,
    # and leaving it would cause extra="forbid" to fail on recursively embedded types
    nested_data = cast(Mapping[str, object], value)
    if config.allow_dataclass_type and config.type_key in nested_data:
        nested_data = {k: v for k, v in nested_data.items() if k != config.type_key}

    try:
        parsed = parse(
            dataclass_type,
            nested_data,
            extra=config.extra,
            coerce=config.coerce,
            case_insensitive=config.case_insensitive,
            alias_generator=config.alias_generator,
            aliases=config.aliases,
            allow_dataclass_type=config.allow_dataclass_type,
            type_key=config.type_key,
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


def _coerce_sequence_items(
    items: list[JSONValue],
    args: tuple[object, ...],
    origin: type[object] | None,
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


def _coerce_sequence(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = cast(type[object] | None, get_origin(base_type))
    if origin not in {list, Sequence, tuple, set}:
        return _NOT_HANDLED
    items = _normalize_sequence_value(value, origin, path, config)
    args = get_args(base_type)
    coerced_items = _coerce_sequence_items(items, args, origin, path, config)
    if origin is set:
        value_out: object = set(coerced_items)
    elif origin is tuple:
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
    if origin is not dict and origin is not Mapping:
        return _NOT_HANDLED
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


def _is_typevar(typ: object) -> bool:
    """Check if a type is a TypeVar (unresolved generic parameter)."""
    return isinstance(typ, TypeVar)


def _build_annotation_namespace(
    cls: type[object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build globalns and localns for evaluating annotations."""
    import sys

    # Get module globals
    module = sys.modules.get(cls.__module__, None)
    globalns: dict[str, object] = vars(module).copy() if module else {}

    # Add class type parameters (Python 3.12+ generic syntax)
    localns: dict[str, object] = {}
    type_params = getattr(cls, "__type_params__", ())
    for tp in type_params:
        localns[tp.__name__] = tp

    return globalns, localns


def _evaluate_annotations_individually(cls: type[object]) -> dict[str, object]:
    """Evaluate each annotation individually, preserving TypeVars."""
    globalns, localns = _build_annotation_namespace(cls)

    # Collect annotations from the class hierarchy
    raw_annotations: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        if hasattr(base, "__annotations__"):
            raw_annotations.update(base.__annotations__)

    # Evaluate each annotation
    result: dict[str, object] = {}
    for name, annotation in raw_annotations.items():
        if isinstance(annotation, str):
            try:
                # Evaluate type annotation string from class definition (safe)
                result[name] = eval(annotation, globalns, localns)  # nosec B307
            except NameError:
                # Unresolvable annotation (e.g., TYPE_CHECKING import)
                # Use object as fallback - coercion will fail if field is required
                result[name] = object
        else:
            result[name] = annotation

    return result


def _safe_get_type_hints(cls: type[object]) -> dict[str, object]:
    """Get type hints, handling generic classes with unbound type parameters.

    Generic classes with unbound type parameters cause NameError in get_type_hints().
    This function catches that error and evaluates annotations individually,
    preserving TypeVar objects for the TypeVar coercer to handle.
    """
    try:
        return get_type_hints(cls, include_extras=True)
    except NameError:
        # Generic class with unresolved type parameters or TYPE_CHECKING imports
        # Evaluate annotations individually, preserving TypeVars
        return _evaluate_annotations_individually(cls)


def _coerce_typevar(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    """Handle TypeVar types by looking for __type__ in the value."""
    if not _is_typevar(base_type):
        return _NOT_HANDLED

    # TypeVar fields require __type__ in the value to determine concrete type
    if not config.allow_dataclass_type:
        raise TypeError(
            f"{path}: cannot parse TypeVar field without allow_dataclass_type=True"
        )

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{path}: expected mapping with {config.type_key} for TypeVar field"
        )

    mapping_value = cast(Mapping[str, object], value)
    type_key = config.type_key
    if type_key not in mapping_value:
        raise TypeError(
            f"{path}: TypeVar field requires {type_key} in value to determine type"
        )

    # Resolve the concrete type from __type__
    type_identifier = mapping_value[type_key]
    if not isinstance(type_identifier, str):
        raise TypeError(f"{path}.{type_key}: must be a string type reference")

    try:
        resolved_cls = _resolve_type_identifier(type_identifier)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{path}.{type_key}: {error}") from error

    if not dataclasses.is_dataclass(resolved_cls):
        raise TypeError(f"{path}.{type_key}: resolved type is not a dataclass")

    # Parse with the resolved type, excluding the type key from the payload
    payload = {k: v for k, v in mapping_value.items() if k != type_key}
    return _coerce_dataclass(payload, resolved_cls, merged_meta, path, config)


def _coerce_to_type(
    value: object,
    typ: object,
    meta: Mapping[str, object] | None,
    path: str,
    config: _ParseConfig,
) -> object:
    base_type, merged_meta = _merge_annotated_meta(typ, meta)

    if base_type is object or base_type is _AnyType:
        return _apply_constraints(value, merged_meta, path)

    coercers = (
        lambda: _coerce_union(value, base_type, merged_meta, path, config),
        lambda: _coerce_none(value, base_type, path),
        lambda: _coerce_literal(value, base_type, merged_meta, path, config),
        lambda: _coerce_typevar(value, base_type, merged_meta, path, config),
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


def _find_key_exact(
    data: Mapping[str, object], candidates: list[str | None]
) -> str | None:
    """Find exact match for any candidate key."""
    for candidate in candidates:
        if candidate is not None and candidate in data:
            return candidate
    return None


def _build_lowered_key_map(data: Mapping[str, object]) -> dict[str, str]:
    """Build a case-insensitive key lookup map."""
    lowered_map: dict[str, str] = {}
    for key in data:
        if isinstance(key, str):
            _ = lowered_map.setdefault(key.lower(), key)
    return lowered_map


def _find_key(
    data: Mapping[str, object], name: str, alias: str | None, case_insensitive: bool
) -> str | None:
    candidates = [alias, name]
    exact = _find_key_exact(data, candidates)
    if exact is not None or not case_insensitive:
        return exact
    lowered_map = _build_lowered_key_map(data)
    for candidate in candidates:
        if candidate is None:
            continue
        lowered = candidate.lower()
        if lowered in lowered_map:
            return lowered_map[lowered]
    return None


def _resolve_field_alias(
    field: dataclasses.Field[object],
    aliases: Mapping[str, str] | None,
    alias_generator: Callable[[str], str] | None,
    field_meta: dict[str, object],
) -> str | None:
    if aliases and field.name in aliases:
        return aliases[field.name]
    alias_value = field_meta.get("alias")
    if alias_value is not None:
        return cast(str, alias_value)
    if alias_generator is not None:
        return alias_generator(field.name)
    return None


def _coerce_field_value(
    field: dataclasses.Field[object],
    raw_value: object,
    field_meta: Mapping[str, object],
    field_type: object,
    config: _ParseConfig,
) -> object:
    try:
        return _coerce_to_type(raw_value, field_type, field_meta, field.name, config)
    except (TypeError, ValueError) as error:
        raise type(error)(str(error)) from error


def _collect_field_kwargs(
    cls: type[object],
    mapping_data: Mapping[str, object],
    type_hints: Mapping[str, object],
    config: _ParseConfig,
    *,
    aliases: Mapping[str, str] | None,
    alias_generator: Callable[[str], str] | None,
) -> tuple[dict[str, object], set[str]]:
    kwargs: dict[str, object] = {}
    used_keys: set[str] = set()

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_meta = dict(field.metadata)
        field_alias = _resolve_field_alias(field, aliases, alias_generator, field_meta)

        key = _find_key(mapping_data, field.name, field_alias, config.case_insensitive)
        if key is None:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"Missing required field: '{field.name}'")
            continue
        used_keys.add(key)
        raw_value = mapping_data[key]
        field_type = type_hints.get(field.name, field.type)
        kwargs[field.name] = _coerce_field_value(
            field, raw_value, field_meta, field_type, config
        )

    return kwargs, used_keys


def _apply_extra_fields(
    instance: object,
    mapping_data: Mapping[str, object],
    used_keys: set[str],
    extra: Literal["ignore", "forbid", "allow"],
) -> None:
    extras = {key: mapping_data[key] for key in mapping_data if key not in used_keys}
    if not extras:
        return
    if extra == "forbid":
        raise ValueError(f"Extra keys not permitted: {list(extras.keys())}")
    if extra == "allow":
        if hasattr(instance, "__dict__"):
            for key, value in extras.items():
                object.__setattr__(instance, key, value)
        else:
            _set_extras(instance, extras)


def _run_validation_hooks(instance: object) -> None:
    validator = getattr(instance, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(instance, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()


def _resolve_type_from_payload(
    mapping_data: Mapping[str, object], type_key: str
) -> tuple[type[object], Mapping[str, object]]:
    """Resolve dataclass type from payload type_key reference."""
    type_identifier = mapping_data[type_key]
    if not isinstance(type_identifier, str):
        raise TypeError(f"{type_key} must be a string type reference")
    try:
        resolved_cls = _resolve_type_identifier(type_identifier)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{type_key}: {error}") from error
    if not dataclasses.is_dataclass(resolved_cls):
        raise TypeError(f"{type_key}: resolved type is not a dataclass")
    payload = cast(
        Mapping[str, object],
        {key: value for key, value in mapping_data.items() if key != type_key},
    )
    return resolved_cls, payload


def _resolve_target_dataclass[T](
    cls: type[T] | None,
    mapping_data: Mapping[str, object],
    *,
    allow_dataclass_type: bool,
    type_key: str,
) -> tuple[type[T], Mapping[str, object]]:
    target_cls: type[T] | None = cls
    referenced_cls: type[object] | None = None
    payload = mapping_data

    if allow_dataclass_type and type_key in mapping_data:
        referenced_cls, payload = _resolve_type_from_payload(mapping_data, type_key)

    if target_cls is None:
        if referenced_cls is None:
            raise TypeError("parse() requires a dataclass type")
        target_cls = cast(type[T], referenced_cls)

    if not dataclasses.is_dataclass(target_cls) or not isinstance(target_cls, type):
        raise TypeError("parse() requires a dataclass type")

    if referenced_cls is not None and referenced_cls is not target_cls:
        expected = _type_identifier(target_cls)
        found = _type_identifier(referenced_cls)
        raise TypeError(
            f"{type_key} does not match target dataclass {expected}; found {found}"
        )

    return target_cls, payload


def parse[T](
    cls: type[T] | None,
    data: Mapping[str, object] | object,
    *,
    extra: Literal["ignore", "forbid", "allow"] = "ignore",
    coerce: bool = True,
    case_insensitive: bool = False,
    alias_generator: Callable[[str], str] | None = None,
    aliases: Mapping[str, str] | None = None,
    allow_dataclass_type: bool = False,
    type_key: str = TYPE_REF_KEY,
) -> T:
    """Parse a mapping into a dataclass instance."""

    if not isinstance(data, Mapping):
        raise TypeError("parse() requires a mapping input")
    if extra not in {"ignore", "forbid", "allow"}:
        raise ValueError("extra must be one of 'ignore', 'forbid', or 'allow'")

    mapping_data = cast(Mapping[str, object], data)
    target_cls, mapping_data = _resolve_target_dataclass(
        cls,
        mapping_data,
        allow_dataclass_type=allow_dataclass_type,
        type_key=type_key,
    )

    config = _ParseConfig(
        extra=extra,
        coerce=coerce,
        case_insensitive=case_insensitive,
        alias_generator=alias_generator,
        aliases=aliases,
        allow_dataclass_type=allow_dataclass_type,
        type_key=type_key,
    )

    type_hints = _safe_get_type_hints(target_cls)
    kwargs, used_keys = _collect_field_kwargs(
        target_cls,
        mapping_data,
        type_hints,
        config,
        aliases=aliases,
        alias_generator=alias_generator,
    )

    instance = target_cls(**kwargs)

    _apply_extra_fields(instance, mapping_data, used_keys, extra)
    _run_validation_hooks(instance)

    return instance


__all__ = ["parse"]
