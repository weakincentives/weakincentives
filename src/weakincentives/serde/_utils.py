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

# pyright: reportPrivateUsage=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import field
from decimal import Decimal
from importlib import import_module
from re import Pattern
from typing import Any as _AnyType, Final, Literal, cast, get_args

from ..dataclasses import FrozenDataclass
from ..types import JSONValue

# Import SerdeScope lazily to avoid circular imports
# The actual import happens in _ParseConfig usage

MISSING_SENTINEL: Final[object] = object()
_UNION_TYPE = type(int | str)
TYPE_REF_KEY: Final[str] = "__type__"


def _ordered_values(values: Iterable[JSONValue]) -> list[JSONValue]:
    """Return a deterministic list of metadata values."""

    items = list(values)
    if isinstance(values, (set, frozenset)):
        return sorted(items, key=repr)
    return items


def _is_unbound_type(typ: object) -> bool:
    """Check if a type is unbound (Any, object, or TypeVar).

    Unbound types require the {"untyped": True} marker to be valid for
    parsing and schema generation.
    """
    from ._generics import _is_typevar

    return typ is _AnyType or typ is object or _is_typevar(typ)


def _build_item_meta(
    merged_meta: Mapping[str, object], item_type: object
) -> dict[str, object] | None:
    """Build metadata to propagate to a collection item or union branch.

    Only propagates the untyped marker if the item_type is actually unbound
    (Any, object, or TypeVar). This ensures that ``dict[Any, str]`` with
    ``{"untyped": True}`` allows untyped keys but still validates string values.
    """
    if merged_meta.get("untyped", False) is True and _is_unbound_type(item_type):
        return {"untyped": True}
    return None


@FrozenDataclass()
class _ParseConfig:
    extra: Literal["ignore", "forbid"]
    coerce: bool
    typevar_map: Mapping[object, type] = field(
        default_factory=lambda: dict[object, type]()
    )
    """Mapping from TypeVar objects to their concrete types for generic alias support."""
    scope: object = None  # SerdeScope | None, using object to avoid circular import
    """Serialization scope for field visibility. Use SerdeScope enum values."""


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

    callables: Iterable[object]
    if isinstance(validators, Iterable) and not isinstance(validators, (str, bytes)):
        callables = cast("Iterable[object]", validators)
    else:
        callables = (validators,)

    current = candidate
    for validator in callables:
        current = _run_validator(validator, current, path)
    return current


def _run_validator(validator: object, candidate: object, path: str) -> object:
    if not callable(validator):
        return candidate  # pragma: no cover - defensive
    fn = cast("Callable[[object], object]", validator)
    try:
        return fn(candidate)
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

    if not callable(converter):
        return candidate  # pragma: no cover - defensive
    try:
        return converter(candidate)
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
    "TYPE_REF_KEY",
    "_AnyType",
    "_ParseConfig",
    "_UNION_TYPE",
    "_apply_constraints",
    "_merge_annotated_meta",
    "_ordered_values",
    "_resolve_type_identifier",
    "_type_identifier",
    "resolve_type_identifier",
    "type_identifier",
]
