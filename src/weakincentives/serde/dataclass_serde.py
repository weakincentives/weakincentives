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

# Copyright 2025 weak incentives
#
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

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import MISSING
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any as _AnyType
from typing import Final, Literal, Union, cast, get_args, get_origin, get_type_hints
from uuid import UUID

MISSING_SENTINEL: Final[object] = object()


class _ExtrasDescriptor:
    """Descriptor storing extras for slotted dataclasses."""

    def __init__(self) -> None:
        self._store: dict[int, dict[str, object]] = {}

    def __get__(
        self, instance: object | None, owner: type[object]
    ) -> dict[str, object] | None:
        if instance is None:
            return None
        return self._store.get(id(instance))

    def __set__(self, instance: object, value: dict[str, object] | None) -> None:
        key = id(instance)
        if value is None:
            self._store.pop(key, None)
        else:
            self._store[key] = dict(value)


_SLOTTED_EXTRAS: Final[dict[type[object], _ExtrasDescriptor]] = {}


def _ordered_values(values: Iterable[object]) -> list[object]:
    """Return a deterministic list of metadata values."""

    items = list(values)
    if isinstance(values, (set, frozenset)):
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=repr)
    return items


def _set_extras(instance: object, extras: Mapping[str, object]) -> None:
    """Attach extras to an instance, handling slotted dataclasses."""

    extras_dict = dict(extras)
    try:
        object.__setattr__(instance, "__extras__", extras_dict)
    except AttributeError:
        cls = instance.__class__
        descriptor = _SLOTTED_EXTRAS.get(cls)
        if descriptor is None:
            descriptor = _ExtrasDescriptor()
            _SLOTTED_EXTRAS[cls] = descriptor
            cls.__extras__ = descriptor  # type: ignore[attr-defined]
        descriptor.__set__(instance, extras_dict)


@dataclasses.dataclass(frozen=True)
class _ParseConfig:
    extra: Literal["ignore", "forbid", "allow"]
    coerce: bool
    case_insensitive: bool
    alias_generator: Callable[[str], str] | None
    aliases: Mapping[str, str] | None


def _merge_annotated_meta(
    typ: object, meta: Mapping[str, object] | None
) -> tuple[object, dict[str, object]]:
    merged: dict[str, object] = dict(meta or {})
    base = typ
    while getattr(base, "__metadata__", None) is not None:
        args = get_args(base)
        if not args:
            break
        base = args[0]
        for extra in args[1:]:
            if isinstance(extra, Mapping):
                merged.update(extra)
    return base, merged


def _bool_from_str(value: str) -> bool:
    lowered = value.strip().lower()
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise TypeError(f"Cannot interpret '{value}' as boolean")


def _apply_constraints(value: object, meta: Mapping[str, object], path: str) -> object:
    if not meta:
        return value

    result = value
    if isinstance(result, str):
        if meta.get("strip"):
            result = result.strip()
        if meta.get("lower") or meta.get("lowercase"):
            result = result.lower()
        if meta.get("upper") or meta.get("uppercase"):
            result = result.upper()

    def _normalize_option(option: object) -> object:
        if isinstance(result, str) and isinstance(option, str):
            candidate: str = option
            if meta.get("strip"):
                candidate = candidate.strip()
            if meta.get("lower") or meta.get("lowercase"):
                candidate = candidate.lower()
            if meta.get("upper") or meta.get("uppercase"):
                candidate = candidate.upper()
            return candidate
        return option

    def _fail(message: str) -> None:
        raise ValueError(f"{path}: {message}")

    numeric_value = result
    if isinstance(numeric_value, (int, float, Decimal)):
        numeric = numeric_value
        minimum_candidate = meta.get("ge", meta.get("minimum"))
        if (
            isinstance(minimum_candidate, (int, float, Decimal))
            and numeric < minimum_candidate
        ):
            _fail(f"must be >= {minimum_candidate}")
        exclusive_min_candidate = meta.get("gt", meta.get("exclusiveMinimum"))
        if (
            isinstance(exclusive_min_candidate, (int, float, Decimal))
            and numeric <= exclusive_min_candidate
        ):
            _fail(f"must be > {exclusive_min_candidate}")
        maximum_candidate = meta.get("le", meta.get("maximum"))
        if (
            isinstance(maximum_candidate, (int, float, Decimal))
            and numeric > maximum_candidate
        ):
            _fail(f"must be <= {maximum_candidate}")
        exclusive_max_candidate = meta.get("lt", meta.get("exclusiveMaximum"))
        if (
            isinstance(exclusive_max_candidate, (int, float, Decimal))
            and numeric >= exclusive_max_candidate
        ):
            _fail(f"must be < {exclusive_max_candidate}")

    if isinstance(result, Sized):
        min_length_candidate = meta.get("min_length", meta.get("minLength"))
        if isinstance(min_length_candidate, int) and len(result) < min_length_candidate:
            _fail(f"length must be >= {min_length_candidate}")
        max_length_candidate = meta.get("max_length", meta.get("maxLength"))
        if isinstance(max_length_candidate, int) and len(result) > max_length_candidate:
            _fail(f"length must be <= {max_length_candidate}")

    pattern = meta.get("regex", meta.get("pattern"))
    if isinstance(pattern, str) and isinstance(result, str):
        if not re.search(pattern, result):
            _fail(f"does not match pattern {pattern}")
    elif isinstance(pattern, Pattern) and isinstance(result, str):
        compiled_pattern = cast(Pattern[str], pattern)
        if not compiled_pattern.search(result):
            _fail(f"does not match pattern {pattern}")

    members = meta.get("in") or meta.get("enum")
    if isinstance(members, Iterable) and not isinstance(members, (str, bytes)):
        options = _ordered_values(members)
        normalized_options = [_normalize_option(option) for option in options]
        if result not in normalized_options:
            _fail(f"must be one of {normalized_options}")

    not_members = meta.get("not_in")
    if isinstance(not_members, Iterable) and not isinstance(not_members, (str, bytes)):
        forbidden = _ordered_values(not_members)
        normalized_forbidden = [_normalize_option(option) for option in forbidden]
        if result in normalized_forbidden:
            _fail(f"may not be one of {normalized_forbidden}")

    validators = meta.get("validators", meta.get("validate"))
    if validators:
        callables: Iterable[Callable[[object], object]]
        if isinstance(validators, Iterable) and not isinstance(
            validators, (str, bytes)
        ):
            callables = cast(Iterable[Callable[[object], object]], validators)
        else:
            callables = (cast(Callable[[object], object], validators),)
        for validator in callables:
            try:
                result = validator(result)
            except (TypeError, ValueError) as error:
                raise type(error)(f"{path}: {error}") from error
            except Exception as error:  # pragma: no cover - defensive
                raise ValueError(f"{path}: validator raised {error!r}") from error

    converter = meta.get("convert", meta.get("transform"))
    if converter:
        try:
            result = converter(result)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{path}: {error}") from error
        except Exception as error:  # pragma: no cover - defensive
            raise ValueError(f"{path}: converter raised {error!r}") from error

    return result


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

    if base_type in {object, _AnyType}:
        return _apply_constraints(value, merged_meta, path)

    if origin is Union:
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
                literal_type = type(literal)
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
                dataclass_type, "__name__", dataclass_type.__class__.__name__
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
                    value = list(value)
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
            items = list(value)
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
        result_dict: dict[object, object] = {}
        for key, item in value.items():
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
                enum_value = base_type[value]  # type: ignore[index]
            except KeyError:
                try:
                    enum_value = base_type(value)  # type: ignore[call-arg]
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
        if isinstance(value, base_type):
            return _apply_constraints(value, merged_meta, path)
        if not config.coerce:
            raise TypeError(f"{path}: expected {type_name}")
        try:
            if base_type is int:
                coerced_value = int(value)
            elif base_type is float:
                coerced_value = float(value)
            elif base_type is str:
                coerced_value = str(value)
            elif base_type is Decimal:
                coerced_value = Decimal(str(value))
            elif base_type is UUID:
                coerced_value = UUID(str(value))
            elif base_type is Path:
                coerced_value = Path(str(value))
            elif base_type is datetime:
                coerced_value = datetime.fromisoformat(str(value))
            elif base_type is date:
                coerced_value = date.fromisoformat(str(value))
            elif base_type is time:
                coerced_value = time.fromisoformat(str(value))
        except Exception as error:
            raise TypeError(
                f"{path}: unable to coerce {value!r} to {type_name}"
            ) from error
        return _apply_constraints(coerced_value, merged_meta, path)

    try:
        coerced = base_type(value)  # type: ignore[call-arg]
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
            lowered_map.setdefault(key.lower(), key)
    for candidate in candidates:
        if candidate is None or not isinstance(candidate, str):
            continue
        lowered = candidate.lower()
        if lowered in lowered_map:
            return lowered_map[lowered]
    return None


def _serialize(
    value: object,
    *,
    by_alias: bool,
    exclude_none: bool,
    alias_generator: Callable[[str], str] | None,
) -> object:
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
        serialized: dict[object, object] = {}
        for key, item in value.items():
            item_value = _serialize(
                item,
                by_alias=by_alias,
                exclude_none=exclude_none,
                alias_generator=alias_generator,
            )
            if item_value is MISSING_SENTINEL:
                continue
            serialized[key] = item_value
        return serialized
    if isinstance(value, set):
        items = [
            item
            for item in (
                _serialize(
                    member,
                    by_alias=by_alias,
                    exclude_none=exclude_none,
                    alias_generator=alias_generator,
                )
                for member in value
            )
            if item is not MISSING_SENTINEL
        ]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = []
        for item in value:
            item_value = _serialize(
                item,
                by_alias=by_alias,
                exclude_none=exclude_none,
                alias_generator=alias_generator,
            )
            if item_value is MISSING_SENTINEL:
                continue
            items.append(item_value)
        return items
    return value


def parse[T](
    cls: type[T],
    data: Mapping[str, object],
    *,
    extra: Literal["ignore", "forbid", "allow"] = "ignore",
    coerce: bool = True,
    case_insensitive: bool = False,
    alias_generator: Callable[[str], str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> T:
    """Parse a mapping into a dataclass instance.

    Parameters
    ----------
    cls:
        Dataclass type to instantiate.
    data:
        Mapping payload describing the instance.

    Returns
    -------
    T
        Parsed dataclass instance after type coercion and validation.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Example:
    ...     name: str
    >>> parse(Example, {"name": "Ada"})
    Example(name='Ada')
    """
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

    type_hints = get_type_hints(cls, include_extras=True)
    kwargs: dict[str, object] = {}
    used_keys: set[str] = set()

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_meta = dict(field.metadata) if field.metadata is not None else {}
        field_alias = None
        if aliases and field.name in aliases:
            field_alias = aliases[field.name]
        elif (alias := field_meta.get("alias")) is not None:
            field_alias = alias
        elif alias_generator is not None:
            field_alias = alias_generator(field.name)

        key = _find_key(data, field.name, field_alias, case_insensitive)
        if key is None:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"Missing required field: '{field.name}'")
            continue
        used_keys.add(key)
        raw_value = data[key]
        field_type = type_hints.get(field.name, field.type)
        try:
            value = _coerce_to_type(
                raw_value, field_type, field_meta, field.name, config
            )
        except (TypeError, ValueError) as error:
            raise type(error)(str(error)) from error
        kwargs[field.name] = value

    instance = cls(**kwargs)

    extras = {key: data[key] for key in data if key not in used_keys}
    if extras:
        if extra == "forbid":
            raise ValueError(f"Extra keys not permitted: {list(extras.keys())}")
        if extra == "allow":
            if hasattr(instance, "__dict__"):
                for key, value in extras.items():
                    object.__setattr__(instance, key, value)
            else:
                _set_extras(instance, extras)

    if extra == "allow" and not extras:
        pass

    validator = getattr(instance, "__validate__", None)
    if callable(validator):
        validator()
    post_validator = getattr(instance, "__post_validate__", None)
    if callable(post_validator):
        post_validator()

    return instance


def dump(
    obj: object,
    *,
    by_alias: bool = True,
    exclude_none: bool = False,
    computed: bool = False,
    alias_generator: Callable[[str], str] | None = None,
) -> dict[str, object]:
    """Serialize a dataclass instance to a JSON-compatible dictionary.

    Parameters
    ----------
    obj:
        Dataclass instance to serialize.

    Returns
    -------
    dict[str, object]
        Serialized representation with nested dataclasses expanded.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("dump() requires a dataclass instance")

    result: dict[str, object] = {}
    for field in dataclasses.fields(obj):
        field_meta = dict(field.metadata) if field.metadata is not None else {}
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
        result[key] = serialized

    if computed and hasattr(obj.__class__, "__computed__"):
        for name in getattr(obj.__class__, "__computed__", ()):  # type: ignore[attr-defined]
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
            result[key] = serialized

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
        validator()
    post_validator = getattr(cloned, "__post_validate__", None)
    if callable(post_validator):
        post_validator()
    return cloned


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
        schema_meta.setdefault("enum", _ordered_values(members))
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

    if base_type in {object, _AnyType}:
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
    elif origin is Union:
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
                schema_data.setdefault("title", title)
            required = subschemas[0].get("required")
            if isinstance(required, (list, tuple)):  # pragma: no cover - defensive
                schema_data.setdefault("required", list(required))
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
        field_meta = dict(field.metadata) if field.metadata is not None else {}
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

    schema_dict = {
        "title": cls.__name__,
        "type": "object",
        "properties": properties,
        "additionalProperties": extra != "forbid",
    }
    if required:
        schema_dict["required"] = required
    if not required:
        schema_dict.pop("required", None)
    return schema_dict


__all__: Final[list[str]] = ["parse", "dump", "clone", "schema"]
