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

"""Shared helpers for dataclass serde operations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from re import Pattern
from typing import Any as _AnyType, Final, Literal, cast, get_args
from uuid import UUID

from ..dataclasses import FrozenDataclass
from ..types import JSONValue

MISSING_SENTINEL: Final[object] = object()
_UNION_TYPE = type(int | str)
TYPE_REF_KEY: Final[str] = "__type__"


@dataclass(slots=True, frozen=True)
class TypeCoercer:
    """Bidirectional type coercion specification.

    Defines how to parse values from JSON-like inputs and serialize values back
    to JSON-compatible outputs for a specific type.

    Attributes:
        parse: Callable to convert JSON-like value to the target Python type.
        dump: Optional callable to convert Python type to JSON-like value.
              If None, the value is assumed to be JSON-compatible as-is.
    """

    parse: Callable[[object], object]
    dump: Callable[[object], object] | None = None


def _parse_decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _parse_uuid(value: object) -> UUID:
    return UUID(str(value))


def _parse_path(value: object) -> Path:
    return Path(str(value))


def _parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def _parse_date(value: object) -> date:
    return date.fromisoformat(str(value))


def _parse_time(value: object) -> time:
    return time.fromisoformat(str(value))


def _dump_isoformat(value: object) -> str:
    return cast(datetime | date | time, value).isoformat()


def _parse_int(value: object) -> int:
    return int(cast(int | float | str, value))


def _parse_float(value: object) -> float:
    return float(cast(int | float | str, value))


def _parse_str(value: object) -> str:
    return str(value)


TYPE_COERCERS: Final[Mapping[type[object], TypeCoercer]] = {
    int: TypeCoercer(parse=_parse_int),
    float: TypeCoercer(parse=_parse_float),
    str: TypeCoercer(parse=_parse_str),
    Decimal: TypeCoercer(parse=_parse_decimal, dump=str),
    UUID: TypeCoercer(parse=_parse_uuid, dump=str),
    Path: TypeCoercer(parse=_parse_path, dump=str),
    datetime: TypeCoercer(parse=_parse_datetime, dump=_dump_isoformat),
    date: TypeCoercer(parse=_parse_date, dump=_dump_isoformat),
    time: TypeCoercer(parse=_parse_time, dump=_dump_isoformat),
}


class _ExtrasDescriptor:
    """Descriptor storing extras for slotted dataclasses."""

    def __init__(self) -> None:
        super().__init__()
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
            _ = self._store.pop(key, None)
        else:
            self._store[key] = dict(value)


_SLOTTED_EXTRAS: Final[dict[type[object], _ExtrasDescriptor]] = {}


def _ordered_values(values: Iterable[JSONValue]) -> list[JSONValue]:
    """Return a deterministic list of metadata values."""

    items = list(values)
    if isinstance(values, (set, frozenset)):
        return sorted(items, key=repr)
    return items


def _get_or_create_extras_descriptor(cls: type[object]) -> _ExtrasDescriptor:
    """Get or create an extras descriptor for a slotted class."""
    descriptor = _SLOTTED_EXTRAS.get(cls)
    if descriptor is not None:
        return descriptor
    descriptor = _ExtrasDescriptor()
    _SLOTTED_EXTRAS[cls] = descriptor
    cls.__extras__ = descriptor  # type: ignore[attr-defined]
    return descriptor


def _set_extras(instance: object, extras: Mapping[str, object]) -> None:
    """Attach extras to an instance, handling slotted dataclasses."""

    extras_dict = dict(extras)
    try:
        object.__setattr__(instance, "__extras__", extras_dict)
    except AttributeError:
        descriptor = _get_or_create_extras_descriptor(instance.__class__)
        descriptor.__set__(instance, extras_dict)


@FrozenDataclass()
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
                merged.update(cast(Mapping[str, object], extra))
    return base, merged


def _apply_constraints[ConstrainedT](
    value: ConstrainedT, meta: Mapping[str, object], path: str
) -> ConstrainedT:
    if not meta:
        return value

    normalized_value = _normalize_string(value, meta)
    _validate_numeric_constraints(normalized_value, meta, path)
    _validate_length(normalized_value, meta, path)
    _validate_pattern(normalized_value, meta, path)
    _validate_membership(normalized_value, meta, path)
    validated = _run_validators(normalized_value, meta, path)
    converted = _apply_converter(validated, meta, path)
    return cast(ConstrainedT, converted)


def _normalize_string(value: object, meta: Mapping[str, object]) -> object:
    if not isinstance(value, str):
        return value

    result = value
    if meta.get("strip"):
        result = result.strip()
    if meta.get("lower") or meta.get("lowercase"):
        result = result.lower()
    if meta.get("upper") or meta.get("uppercase"):
        result = result.upper()
    return result


def _validate_numeric_constraints(
    candidate: object, meta: Mapping[str, object], path: str
) -> None:
    if not isinstance(candidate, (int, float, Decimal)):
        return

    numeric = candidate
    _enforce_bound(numeric, meta.get("ge", meta.get("minimum")), path, "ge")
    _enforce_bound(numeric, meta.get("gt", meta.get("exclusiveMinimum")), path, "gt")
    _enforce_bound(numeric, meta.get("le", meta.get("maximum")), path, "le")
    _enforce_bound(numeric, meta.get("lt", meta.get("exclusiveMaximum")), path, "lt")


def _enforce_bound(
    numeric: Decimal | float | int, bound: object, path: str, kind: str
) -> None:
    if not isinstance(bound, (int, float, Decimal)):
        return

    match kind:
        case "ge" if numeric < bound:
            _fail(path, f"must be >= {bound}")
        case "gt" if numeric <= bound:
            _fail(path, f"must be > {bound}")
        case "le" if numeric > bound:
            _fail(path, f"must be <= {bound}")
        case "lt" if numeric >= bound:
            _fail(path, f"must be < {bound}")
        case _:
            pass


def _validate_length(candidate: object, meta: Mapping[str, object], path: str) -> None:
    if not isinstance(candidate, Sized):
        return

    min_length_candidate = meta.get("min_length", meta.get("minLength"))
    if isinstance(min_length_candidate, int) and len(candidate) < min_length_candidate:
        _fail(path, f"length must be >= {min_length_candidate}")

    max_length_candidate = meta.get("max_length", meta.get("maxLength"))
    if isinstance(max_length_candidate, int) and len(candidate) > max_length_candidate:
        _fail(path, f"length must be <= {max_length_candidate}")


def _validate_pattern(candidate: object, meta: Mapping[str, object], path: str) -> None:
    if not isinstance(candidate, str):
        return

    pattern = meta.get("regex", meta.get("pattern"))
    if isinstance(pattern, str):
        if not re.search(pattern, candidate):
            _fail(path, f"does not match pattern {pattern}")
    elif isinstance(pattern, Pattern):
        compiled_pattern = cast(Pattern[str], pattern)
        if not compiled_pattern.search(candidate):
            _fail(path, f"does not match pattern {pattern}")


def _validate_membership(
    candidate: object, meta: Mapping[str, object], path: str
) -> None:
    _validate_inclusion(candidate, meta, path)
    _validate_exclusion(candidate, meta, path)


def _validate_inclusion(
    candidate: object, meta: Mapping[str, object], path: str
) -> None:
    members = meta.get("in") or meta.get("enum")
    if not isinstance(members, Iterable) or isinstance(members, (str, bytes)):
        return

    options_iter = cast(Iterable[JSONValue], members)
    options = _ordered_values(options_iter)
    normalized_options = [
        _normalize_option(option, candidate, meta) for option in options
    ]
    if candidate not in normalized_options:
        _fail(path, f"must be one of {normalized_options}")


def _validate_exclusion(
    candidate: object, meta: Mapping[str, object], path: str
) -> None:
    not_members = meta.get("not_in")
    if not isinstance(not_members, Iterable) or isinstance(not_members, (str, bytes)):
        return

    forbidden_iter = cast(Iterable[JSONValue], not_members)
    forbidden = _ordered_values(forbidden_iter)
    normalized_forbidden = [
        _normalize_option(option, candidate, meta) for option in forbidden
    ]
    if candidate in normalized_forbidden:
        _fail(path, f"may not be one of {normalized_forbidden}")


def _normalize_option(
    option: JSONValue, candidate: object, meta: Mapping[str, object]
) -> JSONValue:
    if not isinstance(candidate, str) or not isinstance(option, str):
        return option

    normalized_option = option
    if meta.get("strip"):
        normalized_option = normalized_option.strip()
    if meta.get("lower") or meta.get("lowercase"):
        normalized_option = normalized_option.lower()
    if meta.get("upper") or meta.get("uppercase"):
        normalized_option = normalized_option.upper()
    return normalized_option


def _run_validators(candidate: object, meta: Mapping[str, object], path: str) -> object:
    validators = meta.get("validators", meta.get("validate"))
    if not validators:
        return candidate

    callables: Iterable[Callable[[object], object]]
    if isinstance(validators, Iterable) and not isinstance(validators, (str, bytes)):
        callables = cast(Iterable[Callable[[object], object]], validators)
    else:
        callables = (cast(Callable[[object], object], validators),)

    current = candidate
    for validator in callables:
        current = _run_validator(validator, current, path)
    return current


def _run_validator(
    validator: Callable[[object], object], candidate: object, path: str
) -> object:
    try:
        return validator(candidate)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from error
    except Exception as error:  # pragma: no cover - defensive
        raise ValueError(f"{path}: validator raised {error!r}") from error


def _apply_converter(
    candidate: object, meta: Mapping[str, object], path: str
) -> object:
    converter = meta.get("convert", meta.get("transform"))
    if not converter:
        return candidate

    converter_fn = cast(Callable[[object], object], converter)
    try:
        return converter_fn(candidate)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from error
    except Exception as error:  # pragma: no cover - defensive
        raise ValueError(f"{path}: converter raised {error!r}") from error


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _type_identifier(cls: type[object]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_type_identifier(identifier: str) -> type[object]:
    module_name, _, qualname = identifier.partition(":")
    if not module_name or not qualname:
        raise ValueError(f"Invalid type identifier: {identifier!r}")

    module = import_module(module_name)
    target: object = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError(f"Type {identifier!r} could not be resolved")
    if not isinstance(target, type):
        raise TypeError(f"Resolved object for {identifier!r} is not a type")
    return target


def resolve_type_identifier(identifier: str) -> type[object]:
    return _resolve_type_identifier(identifier)


def type_identifier(cls: type[object]) -> str:
    return _type_identifier(cls)


__all__ = [  # noqa: RUF022
    "MISSING_SENTINEL",
    "TYPE_COERCERS",
    "TYPE_REF_KEY",
    "TypeCoercer",
    "_AnyType",
    "_ExtrasDescriptor",
    "_ParseConfig",
    "_SLOTTED_EXTRAS",
    "_UNION_TYPE",
    "_apply_constraints",
    "_merge_annotated_meta",
    "_ordered_values",
    "_resolve_type_identifier",
    "_set_extras",
    "_type_identifier",
    "resolve_type_identifier",
    "type_identifier",
]
