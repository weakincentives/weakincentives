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
from decimal import Decimal
from re import Pattern
from typing import Any as _AnyType
from typing import Final, Literal, cast, get_args

MISSING_SENTINEL: Final[object] = object()
_UNION_TYPE = type(int | str)


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


def _ordered_values(values: Iterable[object]) -> list[object]:
    """Return a deterministic list of metadata values."""

    items = list(values)
    if isinstance(values, (set, frozenset)):
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


@dataclass(frozen=True)
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
        options_iter = cast(Iterable[object], members)
        options = _ordered_values(options_iter)
        normalized_options = [_normalize_option(option) for option in options]
        if result not in normalized_options:
            _fail(f"must be one of {normalized_options}")

    not_members = meta.get("not_in")
    if isinstance(not_members, Iterable) and not isinstance(not_members, (str, bytes)):
        forbidden_iter = cast(Iterable[object], not_members)
        forbidden = _ordered_values(forbidden_iter)
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
        converter_fn = cast(Callable[[object], object], converter)
        try:
            result = converter_fn(result)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{path}: {error}") from error
        except Exception as error:  # pragma: no cover - defensive
            raise ValueError(f"{path}: converter raised {error!r}") from error

    return result


__all__ = [
    "MISSING_SENTINEL",
    "_SLOTTED_EXTRAS",
    "_UNION_TYPE",
    "_AnyType",
    "_ExtrasDescriptor",
    "_ParseConfig",
    "_apply_constraints",
    "_merge_annotated_meta",
    "_ordered_values",
    "_set_extras",
]
