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


def _bool_from_str(value: str) -> bool:
    lowered = value.strip().lower()
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise TypeError(f"Cannot interpret '{value}' as boolean")


def _coerce_to_type(  # noqa: C901
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

    if origin is _UNION_TYPE:
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

    if base_type is type(None):
        if value is not None:
            raise TypeError(f"{path}: expected None")
        return None

    if value is None:
        raise TypeError(f"{path}: value cannot be None")

    if origin is Literal:
        literals = get_args(base_type)
        last_literal_error: Exception | None = None
        for literal in literals:
            if value == literal:
                return _apply_constraints(literal, merged_meta, path)
            if config.coerce:
                literal_type = cast(type[object], type(literal))
                try:
                    if isinstance(literal, bool) and isinstance(value, str):
                        coerced_literal = _bool_from_str(value)
                    else:
                        coerced_literal = literal_type(value)
                except (TypeError, ValueError) as error:
                    last_literal_error = error
                    continue
                if coerced_literal == literal:
                    return _apply_constraints(literal, merged_meta, path)
        if last_literal_error is not None:
            raise type(last_literal_error)(
                f"{path}: {last_literal_error}"
            ) from last_literal_error
        raise ValueError(f"{path}: expected one of {list(literals)}")

    if dataclasses.is_dataclass(base_type):
        dataclass_type = base_type if isinstance(base_type, type) else type(base_type)
        if isinstance(value, dataclass_type):
            return _apply_constraints(value, merged_meta, path)
        if not isinstance(value, Mapping):
            type_name = getattr(
                dataclass_type, "__name__", type(dataclass_type).__name__
            )
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

    if origin in {list, Sequence, tuple, set}:
        is_sequence_like = isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )
        if origin in {list, Sequence} and not is_sequence_like:
            if config.coerce and isinstance(value, str):
                value = [value]
            else:
                raise TypeError(f"{path}: expected sequence")
        if origin is set and not isinstance(value, (set, list, tuple)):
            if config.coerce:
                if isinstance(value, str):
                    value = [value]
                elif isinstance(value, Iterable):
                    value = list(cast(Iterable[JSONValue], value))
                else:
                    raise TypeError(f"{path}: expected set")
            else:
                raise TypeError(f"{path}: expected set")
        if origin is tuple and not is_sequence_like:
            if config.coerce and isinstance(value, str):
                value = [value]
            else:
                raise TypeError(f"{path}: expected tuple")

        if isinstance(value, str):  # pragma: no cover - handled by earlier coercion
            items = [value]
        elif isinstance(value, Iterable):
            items = list(cast(Iterable[JSONValue], value))
        else:  # pragma: no cover - defensive guard
            raise TypeError(f"{path}: expected iterable")
        args = get_args(base_type)
        coerced_items: list[object] = []
        if (
            origin is tuple
            and args
            and args[-1] is not Ellipsis
            and len(args) != len(items)
        ):
            raise ValueError(f"{path}: expected {len(args)} items")
        for index, item in enumerate(items):
            item_path = f"{path}[{index}]"
            if origin is tuple and args:
                item_type = args[0] if args[-1] is Ellipsis else args[index]
            else:
                item_type = args[0] if args else object
            coerced_items.append(
                _coerce_to_type(item, item_type, None, item_path, config)
            )
        if origin is set:
            value_out: object = set(coerced_items)
        elif origin is tuple:
            value_out = tuple(coerced_items)
        else:
            value_out = list(coerced_items)
        return _apply_constraints(value_out, merged_meta, path)

    if origin is dict or origin is Mapping:
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

    if isinstance(base_type, type) and issubclass(base_type, Enum):
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

    if base_type is bool:
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

    if base_type in {int, float, str, Decimal, UUID, Path, datetime, date, time}:
        literal_type = cast(type[object], base_type)
        if isinstance(value, literal_type):
            return _apply_constraints(value, merged_meta, path)
        if not config.coerce:
            raise TypeError(f"{path}: expected {type_name}")
        coerced_value: object | None = None
        try:
            if literal_type is int:
                coerced_value = int(value)
            elif literal_type is float:
                coerced_value = float(value)
            elif literal_type is str:
                coerced_value = str(value)
            elif literal_type is Decimal:
                coerced_value = Decimal(str(value))
            elif literal_type is UUID:
                coerced_value = UUID(str(value))
            elif literal_type is Path:
                coerced_value = Path(str(value))
            elif literal_type is datetime:
                coerced_value = datetime.fromisoformat(str(value))
            elif literal_type is date:
                coerced_value = date.fromisoformat(str(value))
            elif literal_type is time:
                coerced_value = time.fromisoformat(str(value))
        except Exception as error:
            raise TypeError(
                f"{path}: unable to coerce {value!r} to {type_name}"
            ) from error
        if (
            coerced_value is None
        ):  # pragma: no cover - impossible when branches exhaust types
            raise AssertionError("Unhandled literal type coercion")
        return _apply_constraints(coerced_value, merged_meta, path)

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


def parse[T](
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
    kwargs, used_keys = _collect_field_kwargs(
        cls, mapping_data, type_hints, config, aliases, alias_generator
    )

    instance = cls(**kwargs)

    _apply_extra_fields(instance, mapping_data, used_keys, extra)
    _run_validation_hooks(instance)

    return instance


__all__ = ["parse"]
