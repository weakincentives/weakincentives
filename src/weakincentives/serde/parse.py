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

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnnecessaryIsInstance=false, reportCallIssue=false, reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import MISSING
from typing import (
    Literal,
    cast,
    get_origin,
)

from ._coercers import _coerce_to_type
from ._generics import _build_typevar_map, _get_field_types
from ._scope import SerdeScope, is_hidden_in_scope
from ._utils import _ParseConfig


def _find_key(data: Mapping[str, object], name: str, alias: str | None) -> str | None:
    """Find a matching key in data by alias or field name (exact match only)."""
    if alias is not None and alias in data:
        return alias
    if name in data:
        return name
    return None


def _resolve_field_alias(
    field: dataclasses.Field[object],
    field_meta: dict[str, object],
) -> str | None:
    """Resolve field alias from field metadata only."""
    alias_value = field_meta.get("alias")
    if alias_value is not None:
        return cast(str, alias_value)
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
) -> tuple[dict[str, object], set[str]]:
    kwargs: dict[str, object] = {}
    used_keys: set[str] = set()

    # Determine scope for hidden field checks
    scope = (
        cast(SerdeScope, config.scope)
        if config.scope is not None
        else SerdeScope.DEFAULT
    )

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_type = type_hints.get(field.name, field.type)

        # Skip hidden fields in the current scope - they use defaults
        if is_hidden_in_scope(field_type, scope):
            continue

        field_meta = dict(field.metadata)
        field_alias = _resolve_field_alias(field, field_meta)

        key = _find_key(mapping_data, field.name, field_alias)
        if key is None:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"Missing required field: '{field.name}'")
            continue
        used_keys.add(key)
        raw_value = mapping_data[key]
        kwargs[field.name] = _coerce_field_value(
            field, raw_value, field_meta, field_type, config
        )

    return kwargs, used_keys


def _apply_extra_fields(
    mapping_data: Mapping[str, object],
    used_keys: set[str],
    extra: Literal["ignore", "forbid"],
) -> None:
    if extra == "ignore":
        return
    extras = {key: mapping_data[key] for key in mapping_data if key not in used_keys}
    if extras:
        raise ValueError(f"Extra keys not permitted: {list(extras.keys())}")


def _run_validation_hooks(instance: object) -> None:
    validator = getattr(instance, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(instance, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()


def _parse_dataclass[T](
    cls: type[T],
    mapping_data: Mapping[str, object],
    *,
    config: _ParseConfig,
) -> T:
    """Internal parse implementation that takes pre-built config."""
    # Resolve generic alias to concrete class
    origin = get_origin(cls)
    target_cls = cast(type[T], origin if origin is not None else cls)

    type_hints = _get_field_types(target_cls)
    kwargs, used_keys = _collect_field_kwargs(
        target_cls,
        mapping_data,
        type_hints,
        config,
    )

    instance = target_cls(**kwargs)

    _apply_extra_fields(mapping_data, used_keys, config.extra)
    _run_validation_hooks(instance)

    return instance


def parse[T](
    cls: type[T],
    data: Mapping[str, object] | object,
    *,
    extra: Literal["ignore", "forbid"] = "ignore",
    coerce: bool = True,
    scope: SerdeScope = SerdeScope.DEFAULT,
) -> T:
    """Parse a mapping into a dataclass instance.

    Supports generic dataclasses via generic aliases. For example:
        parse(MyGenericClass[str], data)

    The type arguments are used to resolve TypeVar fields during parsing.

    Args:
        cls: The dataclass type to parse into.
        data: A mapping (dict) containing the field values.
        extra: How to handle extra fields: "ignore" or "forbid".
        coerce: Whether to coerce values to match declared types.
        scope: The serialization scope. Fields marked with
            ``HiddenInStructuredOutput`` are skipped (use defaults) when
            ``scope=SerdeScope.STRUCTURED_OUTPUT``.

    Returns:
        An instance of the dataclass with parsed values.
    """
    if not isinstance(data, Mapping):
        raise TypeError("parse() requires a mapping input")
    if extra not in {"ignore", "forbid"}:
        raise ValueError("extra must be one of 'ignore' or 'forbid'")

    # Resolve generic alias to concrete class
    origin = get_origin(cls)
    target_cls = cast(type[T], origin if origin is not None else cls)

    if not dataclasses.is_dataclass(target_cls) or not isinstance(target_cls, type):
        raise TypeError("parse() requires a dataclass type")

    # Build TypeVar mapping from generic alias type arguments
    typevar_map = _build_typevar_map(cls)

    mapping_data = cast(Mapping[str, object], data)
    config = _ParseConfig(
        extra=extra,
        coerce=coerce,
        typevar_map=typevar_map,
        scope=scope,
    )

    return _parse_dataclass(target_cls, mapping_data, config=config)


__all__ = ["parse"]
