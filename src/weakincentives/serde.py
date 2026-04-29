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

"""Dataclass-aware JSON serialization with constraints and unions.

The spine ships a minimal encoder for snapshots; ``serde`` covers the
richer cases that show up at IO boundaries: typed unions discriminated by
``__type__``, validation via ``Annotated`` constraint dicts, primitive
coercions, and partial-update semantics for tuples / lists / dicts.
"""

from __future__ import annotations

import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import (
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

_UNION_ORIGINS = (types.UnionType,)
_VARIADIC_TUPLE = 2

__all__ = [
    "ParseError",
    "SerdeError",
    "dump",
    "parse",
    "type_identifier",
]


class SerdeError(Exception):
    """Base class for serde failures."""


class ParseError(SerdeError):
    """Raised when input data cannot be parsed into the requested type."""


def type_identifier(cls: type[object]) -> str:
    """Return a stable ``module.qualname`` identifier for ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


# =============================================================================
# Public API
# =============================================================================


def parse[T](cls: type[T], data: object) -> T:
    """Build an instance of ``cls`` from JSON-shaped ``data``."""
    return cast("T", _parse_typed(cls, data, path="$"))


def dump(value: object) -> object:
    """Convert a value into JSON-compatible primitives."""
    return _dump(value)


# =============================================================================
# Parsing
# =============================================================================


def _parse_typed(annotation: object, data: object, *, path: str) -> object:
    annotation, constraints = _split_annotated(annotation)
    origin = get_origin(annotation)

    if annotation is None or annotation is type(None):
        if data is not None:
            raise ParseError(f"{path}: expected null, got {type(data).__name__}")
        return None

    if origin in _UNION_ORIGINS:
        return _parse_union(get_args(annotation), data, path=path)

    if origin in (list, tuple, Sequence):
        return _parse_sequence(annotation, origin, data, path=path)

    if origin in (dict, Mapping):
        return _parse_mapping(annotation, data, path=path)

    if isinstance(annotation, type):
        result = _parse_concrete(annotation, data, path=path)
    else:
        raise ParseError(f"{path}: unsupported annotation {annotation!r}")

    for constraint, expected in constraints.items():
        _check_constraint(result, constraint, expected, path=path)
    return result


def _parse_bool(data: object, path: str) -> bool:
    if not isinstance(data, bool):
        raise ParseError(f"{path}: expected bool")
    return data


def _parse_int(data: object, path: str) -> int:
    if isinstance(data, bool) or not isinstance(data, int):
        raise ParseError(f"{path}: expected int")
    return data


def _parse_float(data: object, path: str) -> float:
    if isinstance(data, bool):
        raise ParseError(f"{path}: expected number")
    if isinstance(data, int):
        return float(data)
    if not isinstance(data, float):
        raise ParseError(f"{path}: expected number")
    return data


def _parse_str(data: object, path: str) -> str:
    if not isinstance(data, str):
        raise ParseError(f"{path}: expected string")
    return data


_PRIMITIVE_PARSERS: dict[type[object], Callable[[object, str], object]] = {
    bool: _parse_bool,
    int: _parse_int,
    float: _parse_float,
    str: _parse_str,
}


def _parse_concrete(cls: type[object], data: object, *, path: str) -> object:
    parser = _PRIMITIVE_PARSERS.get(cls)
    if parser is not None:
        return parser(data, path)
    if is_dataclass(cls):
        return _parse_dataclass(cls, data, path=path)
    raise ParseError(f"{path}: unsupported concrete type {cls!r}")


def _parse_dataclass(cls: type[object], data: object, *, path: str) -> object:
    if not isinstance(data, dict):
        raise ParseError(f"{path}: expected object for {cls.__qualname__}")
    payload = cast("dict[str, Any]", data)
    hints = get_type_hints(cls, include_extras=True)
    kwargs: dict[str, object] = {}
    seen: set[str] = set()
    dataclass_cls = cast("type[Any]", cls)
    for f in fields(dataclass_cls):
        seen.add(f.name)
        annotation: object = hints.get(f.name, f.type)
        if f.name in payload:
            value: object = payload[f.name]
            kwargs[f.name] = _parse_typed(annotation, value, path=f"{path}.{f.name}")
        elif _has_default(f):
            continue
        else:
            raise ParseError(f"{path}.{f.name}: missing required field")
    extra = set(payload.keys()) - seen - {"__type__"}
    if extra:
        raise ParseError(f"{path}: unexpected fields {sorted(extra)!r}")
    return cls(**kwargs)


def _has_default(f: object) -> bool:
    from dataclasses import MISSING

    default = getattr(f, "default", MISSING)
    factory = getattr(f, "default_factory", MISSING)
    return default is not MISSING or factory is not MISSING


def _parse_sequence(
    annotation: object, origin: object, data: object, *, path: str
) -> object:
    if not isinstance(data, list):
        raise ParseError(f"{path}: expected array")
    args = get_args(annotation)
    elements = cast("list[Any]", data)
    if origin is tuple and len(args) == _VARIADIC_TUPLE and args[-1] is Ellipsis:
        item_type = args[0]
        return tuple(
            _parse_typed(item_type, item, path=f"{path}[{i}]")
            for i, item in enumerate(elements)
        )
    if origin is tuple and args and Ellipsis not in args:
        if len(args) != len(elements):
            raise ParseError(
                f"{path}: tuple expected {len(args)} items, got {len(elements)}"
            )
        return tuple(
            _parse_typed(arg, item, path=f"{path}[{i}]")
            for i, (arg, item) in enumerate(zip(args, elements, strict=True))
        )
    item_type = args[0] if args else object
    parsed = [
        _parse_typed(item_type, item, path=f"{path}[{i}]")
        for i, item in enumerate(elements)
    ]
    return parsed if origin is list else tuple(parsed)


def _parse_mapping(annotation: object, data: object, *, path: str) -> object:
    if not isinstance(data, dict):
        raise ParseError(f"{path}: expected object")
    args = get_args(annotation)
    key_type = args[0] if args else object
    value_type = args[1] if len(args) > 1 else object
    payload = cast("dict[Any, Any]", data)
    return {
        _parse_typed(key_type, k, path=f"{path}.<key>"): _parse_typed(
            value_type, v, path=f"{path}.{k!r}"
        )
        for k, v in payload.items()
    }


def _parse_union(args: tuple[object, ...], data: object, *, path: str) -> object:
    if data is None and type(None) in args:
        return None
    candidates = [a for a in args if a is not type(None)]
    if len(candidates) == 1:
        return _parse_typed(candidates[0], data, path=path)
    tagged = _try_tagged_union(candidates, data, path=path)
    if tagged is not _UNRESOLVED:
        return tagged
    for candidate in candidates:
        try:
            return _parse_typed(candidate, cast("object", data), path=path)
        except ParseError:
            continue
    raise ParseError(f"{path}: no union member matched")


_UNRESOLVED = object()


def _try_tagged_union(candidates: list[object], data: object, *, path: str) -> object:
    if not isinstance(data, dict):
        return _UNRESOLVED
    mapping = cast("dict[str, Any]", data)
    if "__type__" not in mapping:
        return _UNRESOLVED
    type_id = mapping["__type__"]
    for candidate in candidates:
        if isinstance(candidate, type) and type_identifier(candidate) == type_id:
            return _parse_typed(candidate, mapping, path=path)
    raise ParseError(f"{path}: no union member matches __type__={type_id!r}")


# =============================================================================
# Constraints
# =============================================================================


def _split_annotated(annotation: object) -> tuple[object, dict[str, object]]:
    if get_origin(annotation) is not Annotated:
        return annotation, {}
    args = get_args(annotation)
    base = args[0]
    constraints: dict[str, object] = {}
    for extra in args[1:]:
        if isinstance(extra, dict):
            constraints.update(cast("dict[str, object]", extra))
    return base, constraints


_NUMERIC_RULES: dict[str, Callable[[float, float], tuple[bool, str]]] = {
    "ge": lambda v, e: (v >= e, f"{v} < ge={e}"),
    "le": lambda v, e: (v <= e, f"{v} > le={e}"),
    "gt": lambda v, e: (v > e, f"{v} not > {e}"),
    "lt": lambda v, e: (v < e, f"{v} not < {e}"),
}


def _check_constraint(value: object, name: str, expected: object, *, path: str) -> None:
    rule = _NUMERIC_RULES.get(name)
    if rule is not None:
        if not isinstance(value, int | float):
            return
        ok, message = rule(float(value), float(cast("int | float", expected)))
        if not ok:
            raise ParseError(f"{path}: {message}")
        return
    if name in {"min_length", "max_length"} and isinstance(value, str):
        threshold = cast("int", expected)
        ok = (
            len(value) >= threshold if name == "min_length" else len(value) <= threshold
        )
        if not ok:
            raise ParseError(f"{path}: length {len(value)} bound {name}={threshold}")
        return
    if name == "pattern" and isinstance(value, str):
        if not re.match(cast("str", expected), value):
            raise ParseError(f"{path}: does not match pattern {expected!r}")
        return
    raise SerdeError(f"unknown constraint {name!r}")


# =============================================================================
# Dumping
# =============================================================================


def _dump(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_dump(item) for item in cast("Sequence[object]", value)]
    if isinstance(value, dict):
        return {
            str(k): _dump(v) for k, v in cast("dict[object, object]", value).items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, object] = {"__type__": type_identifier(type(value))}
        for f in fields(value):
            encoded[f.name] = _dump(getattr(value, f.name))
        return encoded
    raise SerdeError(f"cannot dump value of type {type(value).__name__}")
