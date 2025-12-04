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

"""Frozen dataclass helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, Field, dataclass, fields
from typing import (
    Any,
    Protocol,
    Self,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    dataclass_transform,
)

__all__ = ["FrozenDataclass"]

T = TypeVar("T")


class _SupportsUpdate(Protocol):
    def update(self, **changes: object) -> Self:
        """Return a modified copy of the dataclass."""
        ...

    def merge(self, mapping_or_obj: object) -> Self:
        """Merge fields into a new instance."""
        ...

    def map(
        self, transform: Callable[[dict[str, object]], Mapping[str, object]]
    ) -> Self:
        """Apply a mapping-based transform."""
        ...


class DataclassOptions(TypedDict, total=False):
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    frozen: bool
    match_args: bool
    kw_only: bool
    slots: bool


@dataclass_transform()
def FrozenDataclass(
    **dataclass_kwargs: Unpack[DataclassOptions],
) -> Callable[[type[T]], type[T]]:
    """Dataclass decorator with frozen, slotted defaults plus helpers.

    The decorator mirrors :func:`dataclasses.dataclass` while defaulting to
    ``frozen=True`` and ``slots=True``. Classes may optionally define a
    ``__pre_init__`` classmethod to normalise inputs before construction; it is
    invoked with keyword arguments derived from the initialiser inputs.

    Copy helpers are injected on the decorated class:

    - ``update(**changes)``: return a modified copy via
      :func:`dataclasses.replace`.
    - ``merge(mapping_or_obj)``: merge fields from a mapping or attribute-bearing
      object into a copy.
    - ``map(transform)``: provide a mapping of current fields to ``transform``
      and apply the returned replacements.
    """

    options: DataclassOptions = {
        "init": True,
        "repr": True,
        "eq": True,
        "order": False,
        "unsafe_hash": False,
        "frozen": True,
        "match_args": True,
        "kw_only": False,
        "slots": True,
        **dataclass_kwargs,
    }

    def decorator(cls: type[T]) -> type[T]:
        dataclass_cls = cast(Callable[[type[T]], type[T]], dataclass(**options))(cls)
        _attach_helpers(dataclass_cls)
        return dataclass_cls

    return decorator


def _attach_helpers(cls: type[Any]) -> None:
    original_init = cls.__init__
    pre_init = getattr(cls, "__pre_init__", None)

    if pre_init is not None:
        cls.__init__ = _build_pre_init_wrapper(cls, pre_init, original_init)

    cls.update = _build_update_helper(cls)
    cls.merge = _build_merge_helper(cls)
    cls.map = _build_map_helper(cls)


def _build_pre_init_wrapper(
    cls: type[Any],
    pre_init: Callable[..., Mapping[str, object]],
    original_init: Callable[..., None],
) -> Callable[..., None]:
    field_defs = [field for field in fields(cls) if field.init]
    field_names = [field.name for field in field_defs]

    def wrapper(self: object, *args: object, **kwargs: object) -> None:
        if len(args) > len(field_defs):
            raise TypeError(
                f"{cls.__name__}() takes {len(field_defs)} positional arguments but {len(args)} were given"
            )

        provided = dict(kwargs)
        bound = _bind_fields(cls, field_defs, field_names, args, provided)
        if provided:
            unexpected = ", ".join(sorted(provided))
            raise TypeError(
                f"{cls.__name__}() got unexpected keyword arguments: {unexpected}"
            )

        normalized = pre_init(**bound)
        _ensure_mapping(cls, normalized)
        _validate_normalized_fields(cls, field_names, normalized)

        original_init(self, **normalized)

    return wrapper


def _bind_fields(
    cls: type[Any],
    field_defs: Iterable[Field[Any]],
    field_names: list[str],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    bound: dict[str, object] = {}
    for index, (field_def, name) in enumerate(
        zip(field_defs, field_names, strict=False)
    ):
        if index < len(args):
            bound[name] = args[index]
            continue
        if name in kwargs:
            bound[name] = kwargs.pop(name)
            continue
        if field_def.default is not MISSING:
            bound[name] = field_def.default
            continue
        if field_def.default_factory is not MISSING:
            bound[name] = field_def.default_factory()
            continue
        bound[name] = MISSING

    return bound


def _ensure_mapping(cls: type[Any], value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__}.__pre_init__() must return a mapping")


def _validate_normalized_fields(
    cls: type[Any], field_names: list[str], normalized: Mapping[str, object]
) -> None:
    unexpected = normalized.keys() - set(field_names)
    missing = {
        name
        for name in field_names
        if name not in normalized or normalized[name] is MISSING
    }

    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise TypeError(
            f"{cls.__name__}.__pre_init__() returned unexpected fields: {joined}"
        )
    if missing:
        joined = ", ".join(sorted(missing))
        raise TypeError(
            f"{cls.__name__}.__pre_init__() is missing required fields: {joined}"
        )


def _build_update_helper(cls: type[Any]) -> Callable[..., _SupportsUpdate]:
    def update(self: _SupportsUpdate, **changes: object) -> _SupportsUpdate:
        return _apply_changes(self, changes)

    return update


def _build_merge_helper(cls: type[Any]) -> Callable[..., _SupportsUpdate]:
    field_names = [field.name for field in fields(cls) if field.init]

    def merge(self: _SupportsUpdate, mapping_or_obj: object) -> _SupportsUpdate:
        updates = _extract_updates(cls, field_names, mapping_or_obj)
        return self.update(**updates)

    return merge


def _build_map_helper(cls: type[Any]) -> Callable[..., _SupportsUpdate]:
    def map_(
        self: _SupportsUpdate,
        transform: Callable[[dict[str, object]], Mapping[str, object]],
    ) -> _SupportsUpdate:
        current = {
            field.name: getattr(self, field.name)
            for field in fields(self)  # type: ignore[arg-type]
        }
        updates = transform(current)
        _ensure_mapping(cls, updates)
        return self.update(**dict(updates))

    map_.__name__ = "map"
    return map_


def _extract_updates(
    cls: type[Any], field_names: list[str], source: object
) -> dict[str, object]:
    if isinstance(source, Mapping):
        typed_source = cast(Mapping[str, object], source)
        extra_keys = typed_source.keys() - set(field_names)
        if extra_keys:
            joined = ", ".join(sorted(extra_keys))
            raise TypeError(
                f"{cls.__name__}.merge() received unexpected fields: {joined}"
            )
        return {key: typed_source[key] for key in field_names if key in typed_source}

    updates: dict[str, object] = {}
    for name in field_names:
        try:
            updates[name] = getattr(source, name)
        except AttributeError as error:
            raise TypeError(
                f"{cls.__name__}.merge() source missing attribute: {name}"
            ) from error
    return updates


def _apply_changes(
    instance: _SupportsUpdate, changes: Mapping[str, object]
) -> _SupportsUpdate:
    cls = type(instance)
    field_defs = fields(instance)  # type: ignore[arg-type]
    values = {field.name: getattr(instance, field.name) for field in field_defs}

    unknown = set(changes) - values.keys()
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise TypeError(f"{cls.__name__}() got unexpected field(s): {joined}")

    values.update(changes)
    new_instance = cls.__new__(cls)
    for name, value in values.items():
        object.__setattr__(new_instance, name, value)

    post_init = getattr(new_instance, "__post_init__", None)
    if callable(post_init):
        _ = post_init()

    return new_instance
