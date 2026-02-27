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

"""Enhanced dataclass utilities with immutability-first defaults.

This module provides :class:`FrozenDataclassMixin` and :func:`pre_init` for
building immutable, memory-efficient data structures with convenient
copy-and-modify helper methods.

Usage Pattern
-------------
Combine ``@dataclass(slots=True, frozen=True)`` with ``FrozenDataclassMixin``::

    from dataclasses import dataclass
    from weakincentives.dataclasses import FrozenDataclassMixin

    @dataclass(slots=True, frozen=True)
    class Point(FrozenDataclassMixin):
        x: float
        y: float

    p1 = Point(1.0, 2.0)
    p2 = p1.update(x=3.0)  # Point(x=3.0, y=2.0)

    # Attempting mutation raises FrozenInstanceError
    p1.x = 5.0  # Raises dataclasses.FrozenInstanceError

Copy Helper Methods
-------------------
All subclasses of ``FrozenDataclassMixin`` receive three helper methods:

**update(**changes)**
    Return a new instance with specified fields replaced::

        @dataclass(slots=True, frozen=True)
        class User(FrozenDataclassMixin):
            name: str
            email: str
            active: bool = True

        user = User("Alice", "alice@example.com")
        updated = user.update(email="alice@newdomain.com", active=False)

**merge(mapping_or_obj)**
    Merge fields from a dictionary or another object::

        changes = {"email": "new@example.com"}
        merged = user.merge(changes)

**map(transform)**
    Apply a transformation function to current field values::

        transformed = user.map(lambda f: {"name": str(f["name"]).upper()})

Input Normalization with __pre_init__
--------------------------------------
Use the :func:`pre_init` decorator alongside ``@dataclass`` to enable the
``__pre_init__`` classmethod hook::

    from collections.abc import Mapping
    from dataclasses import dataclass
    from weakincentives.dataclasses import FrozenDataclassMixin, pre_init

    @pre_init
    @dataclass(slots=True, frozen=True)
    class NormalizedPath(FrozenDataclassMixin):
        path: str

        @classmethod
        def __pre_init__(cls, path: str) -> Mapping[str, object]:
            normalized = path.replace("\\\\", "/").rstrip("/")
            return {"path": normalized}

    p = NormalizedPath("some\\\\path\\\\")
    assert p.path == "some/path"

Derived (Non-Init) Fields
--------------------------
Fields marked with ``init=False`` can be computed in ``__pre_init__``::

    from dataclasses import dataclass, field
    from weakincentives.dataclasses import FrozenDataclassMixin, pre_init

    @pre_init
    @dataclass(slots=True, frozen=True)
    class Rectangle(FrozenDataclassMixin):
        width: float
        height: float
        area: float = field(init=False)

        @classmethod
        def __pre_init__(cls, width: float, height: float) -> Mapping[str, object]:
            return {"width": width, "height": height, "area": width * height}

    rect = Rectangle(3.0, 4.0)
    assert rect.area == 12.0

    rect2 = rect.update(width=5.0)
    assert rect2.area == 20.0

See Also
--------
- :mod:`dataclasses` : Python's standard dataclass module
- :mod:`weakincentives.serde` : Serialization utilities for dataclasses
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, Field, fields
from types import MethodType
from typing import Any, Self, cast

__all__ = ["FrozenDataclassMixin", "pre_init"]


class FrozenDataclassMixin:
    """Mixin providing functional update helpers for frozen dataclasses.

    Add as a base class alongside ``@dataclass(frozen=True, slots=True)``::

        @dataclass(slots=True, frozen=True)
        class Point(FrozenDataclassMixin):
            x: float
            y: float

        p = Point(1.0, 2.0)
        p2 = p.update(x=3.0)    # Point(x=3.0, y=2.0)
        p3 = p.merge({"y": 5.0})  # Point(x=1.0, y=5.0)

    .. note::
        ``__slots__ = ()`` is load-bearing: it prevents this mixin from adding
        a ``__dict__`` that would defeat ``slots=True`` on the dataclass.
    """

    __slots__ = ()

    def update(self, **changes: object) -> Self:
        """Return a modified copy with specified fields replaced."""
        return _apply_changes(self, changes)  # type: ignore[return-value]

    def merge(self, mapping_or_obj: object) -> Self:
        """Merge fields from a mapping or object into a new instance."""
        cls = type(self)
        field_names = [f.name for f in fields(self) if f.init]  # type: ignore[arg-type]
        updates = _extract_updates(cls, field_names, mapping_or_obj)
        return self.update(**updates)

    def map(
        self, transform: Callable[[dict[str, object]], Mapping[str, object]]
    ) -> Self:
        """Apply a transform to current field values and return a modified copy."""
        cls = type(self)
        current = {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[arg-type]
            if f.init
        }
        updates = transform(current)
        _ensure_mapping(cls, updates)
        return self.update(**dict(updates))


def pre_init[T](cls: type[T]) -> type[T]:
    """Wrap ``__init__`` to call ``__pre_init__`` for input normalization.

    Apply this decorator **after** ``@dataclass`` so it wraps the
    dataclass-generated ``__init__``::

        @pre_init
        @dataclass(slots=True, frozen=True)
        class NormalizedPath(FrozenDataclassMixin):
            path: str

            @classmethod
            def __pre_init__(cls, path: str) -> Mapping[str, object]:
                return {"path": path.replace("\\\\\\\\", "/").rstrip("/")}

    The ``__pre_init__`` classmethod receives all init-field values as keyword
    arguments and must return a ``Mapping[str, object]`` covering all init
    fields. Non-init (derived) fields may also be included.
    """
    pre_init_method = getattr(cls, "__pre_init__", None)
    if pre_init_method is None:
        return cls
    original_init = cls.__init__
    wrapper = _build_pre_init_wrapper(
        cls, cast(MethodType, pre_init_method), original_init
    )
    cast(Any, cls).__init__ = wrapper
    return cls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_pre_init_wrapper(
    cls: type[Any],
    pre_init: MethodType,
    original_init: Callable[..., None],
) -> Callable[..., None]:
    all_fields = fields(cls)
    init_field_defs = [f for f in all_fields if f.init]
    init_field_names = [f.name for f in init_field_defs]
    non_init_field_names = {f.name for f in all_fields if not f.init}
    all_field_names = init_field_names + list(non_init_field_names)
    # Get the underlying function from the classmethod to allow calling with
    # the actual subclass type, not the class where __pre_init__ was defined.
    pre_init_func = pre_init.__func__

    def wrapper(self: object, *args: object, **kwargs: object) -> None:
        actual_cls = type(self)
        if len(args) > len(init_field_defs):
            raise TypeError(
                f"{actual_cls.__name__}() takes {len(init_field_defs)} positional arguments but {len(args)} were given"
            )

        provided = dict(kwargs)
        bound = _bind_fields(
            actual_cls, iter(init_field_defs), init_field_names, args, provided
        )
        if provided:
            unexpected = ", ".join(sorted(provided))
            raise TypeError(
                f"{actual_cls.__name__}() got unexpected keyword arguments: {unexpected}"
            )

        normalized = pre_init_func(actual_cls, **bound)
        _ensure_mapping(actual_cls, normalized)
        _validate_normalized_fields(
            actual_cls, all_field_names, init_field_names, normalized
        )

        # Separate init and non-init fields
        init_kwargs = {k: v for k, v in normalized.items() if k in init_field_names}
        non_init_values = {
            k: v for k, v in normalized.items() if k in non_init_field_names
        }

        original_init(self, **init_kwargs)

        # Set non-init fields directly (bypassing frozen restriction)
        for name, value in non_init_values.items():
            object.__setattr__(self, name, value)

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
    cls: type[Any],
    all_field_names: list[str],
    required_field_names: list[str],
    normalized: Mapping[str, object],
) -> None:
    unexpected = normalized.keys() - set(all_field_names)
    missing = {
        name
        for name in required_field_names
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


def _apply_changes(
    instance: FrozenDataclassMixin, changes: Mapping[str, object]
) -> FrozenDataclassMixin:
    cls = type(instance)
    field_defs = fields(instance)  # type: ignore[arg-type]
    init_fields = [f for f in field_defs if f.init]
    non_init_fields = [f for f in field_defs if not f.init]
    init_field_names = {f.name for f in init_fields}
    all_field_names = {f.name for f in field_defs}

    # Validate changes against known fields
    unknown = set(changes) - all_field_names
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise TypeError(f"{cls.__name__}() got unexpected field(s): {joined}")

    # Disallow direct changes to non-init (derived) fields
    non_init_changes = set(changes) - init_field_names
    if non_init_changes:
        joined = ", ".join(sorted(non_init_changes))
        raise TypeError(f"{cls.__name__}() cannot update derived field(s): {joined}")

    # Collect current init-field values and apply changes
    init_values = {f.name: getattr(instance, f.name) for f in init_fields}
    init_values.update(changes)

    # If class has non-init fields, call constructor so __pre_init__ recomputes them
    if non_init_fields:
        return cls(**init_values)

    # Fast path: class has no derived fields, bypass __pre_init__
    new_instance = cls.__new__(cls)
    for name, value in init_values.items():
        object.__setattr__(new_instance, name, value)

    post_init = getattr(new_instance, "__post_init__", None)
    if callable(post_init):
        _ = post_init()

    return new_instance


def _extract_updates(
    cls: type[Any], field_names: list[str], source: object
) -> dict[str, object]:
    from typing import cast

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
        if hasattr(source, name):
            updates[name] = getattr(source, name)

    if not updates:
        raise TypeError(f"{cls.__name__}.merge() source has no matching fields")

    return updates
