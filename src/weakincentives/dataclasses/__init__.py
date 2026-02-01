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

This package provides the :func:`FrozenDataclass` decorator, which extends
Python's standard :func:`dataclasses.dataclass` with sensible defaults for
building immutable, memory-efficient data structures, along with convenient
copy-and-modify helper methods.

Exported Symbols
----------------
FrozenDataclass
    A decorator that creates frozen, slotted dataclasses with additional
    helper methods for functional-style updates.

Benefits Over Standard Dataclasses
----------------------------------
1. **Immutability by default**: Classes are frozen (``frozen=True``), preventing
   accidental mutation and making instances safe for use as dictionary keys or
   in sets.

2. **Memory efficiency**: Slots are enabled by default (``slots=True``),
   reducing memory overhead and improving attribute access speed.

3. **Functional update methods**: Decorated classes gain ``update()``,
   ``merge()``, and ``map()`` methods for creating modified copies without
   mutation.

4. **Input normalization hook**: The optional ``__pre_init__`` classmethod
   allows transforming or validating inputs before the dataclass is constructed.

5. **Full type checker support**: Uses ``@dataclass_transform`` for proper
   static analysis with pyright, mypy, and other type checkers.

Basic Usage
-----------
Create an immutable dataclass with the decorator::

    from weakincentives.dataclasses import FrozenDataclass

    @FrozenDataclass()
    class Point:
        x: float
        y: float

    p1 = Point(1.0, 2.0)
    p2 = p1.update(x=3.0)  # Point(x=3.0, y=2.0)

    # Attempting mutation raises FrozenInstanceError
    p1.x = 5.0  # Raises dataclasses.FrozenInstanceError

Copy Helper Methods
-------------------
All decorated classes receive three helper methods for functional updates:

**update(**changes)**
    Return a new instance with specified fields replaced::

        @FrozenDataclass()
        class User:
            name: str
            email: str
            active: bool = True

        user = User("Alice", "alice@example.com")
        updated = user.update(email="alice@newdomain.com", active=False)

**merge(mapping_or_obj)**
    Merge fields from a dictionary or another object::

        # From a dictionary
        changes = {"email": "new@example.com"}
        merged = user.merge(changes)

        # From another object with matching attributes
        class UserPatch:
            email = "patched@example.com"

        merged = user.merge(UserPatch())

**map(transform)**
    Apply a transformation function that receives current field values
    as a dictionary and returns updates::

        def uppercase_name(fields: dict[str, object]) -> dict[str, object]:
            return {"name": str(fields["name"]).upper()}

        transformed = user.map(uppercase_name)  # User with name="ALICE"

Input Normalization with __pre_init__
-------------------------------------
Define a ``__pre_init__`` classmethod to normalize or validate inputs before
the dataclass is constructed. The method receives all field values as keyword
arguments and must return a mapping of field names to values::

    from collections.abc import Mapping

    @FrozenDataclass()
    class NormalizedPath:
        path: str

        @classmethod
        def __pre_init__(cls, path: str) -> Mapping[str, object]:
            # Normalize path separators and remove trailing slashes
            normalized = path.replace("\\\\", "/").rstrip("/")
            return {"path": normalized}

    p = NormalizedPath("some\\\\path\\\\")
    assert p.path == "some/path"

The ``__pre_init__`` hook is particularly useful for:

- Coercing input types (e.g., converting strings to enums)
- Computing derived fields that depend on other fields
- Validating invariants before construction
- Providing alternative constructors with different signatures

Derived (Non-Init) Fields
-------------------------
Fields marked with ``init=False`` are considered derived fields. When using
``__pre_init__``, you can compute and return values for these fields::

    from dataclasses import field

    @FrozenDataclass()
    class Rectangle:
        width: float
        height: float
        area: float = field(init=False)

        @classmethod
        def __pre_init__(
            cls, width: float, height: float
        ) -> Mapping[str, object]:
            return {"width": width, "height": height, "area": width * height}

    rect = Rectangle(3.0, 4.0)
    assert rect.area == 12.0

    # Updating triggers recomputation
    rect2 = rect.update(width=5.0)
    assert rect2.area == 20.0

Note that derived fields cannot be modified directly via ``update()``; they
are always recomputed through ``__pre_init__``.

Customizing Dataclass Options
-----------------------------
All standard dataclass options can be overridden. The defaults are::

    frozen=True      # Instances are immutable
    slots=True       # Use __slots__ for memory efficiency
    init=True        # Generate __init__
    repr=True        # Generate __repr__
    eq=True          # Generate __eq__
    order=False      # Do not generate comparison methods
    unsafe_hash=False
    match_args=True  # Support pattern matching
    kw_only=False    # Positional arguments allowed

Override as needed::

    @FrozenDataclass(order=True)  # Enable ordering comparisons
    class Version:
        major: int
        minor: int
        patch: int

    v1 = Version(1, 0, 0)
    v2 = Version(2, 0, 0)
    assert v1 < v2

    @FrozenDataclass(frozen=False)  # Mutable (not recommended)
    class MutablePoint:
        x: float
        y: float

See Also
--------
- :mod:`dataclasses` : Python's standard dataclass module
- :mod:`weakincentives.serde` : Serialization utilities for dataclasses
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from typing import (
    Any,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    cast,
    dataclass_transform,
)

__all__ = ["FrozenDataclass"]


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


@dataclass_transform(field_specifiers=(field,))
def FrozenDataclass[T](
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
        dataclass_cls = cast(Callable[[type[T]], type[T]], dataclass(**options))(cls)  # ty: ignore
        _attach_helpers(dataclass_cls)
        return dataclass_cls  # ty: ignore

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
    all_fields = fields(cls)
    init_field_defs = [field for field in all_fields if field.init]
    init_field_names = [field.name for field in init_field_defs]
    non_init_field_names = {field.name for field in all_fields if not field.init}
    all_field_names = init_field_names + list(non_init_field_names)
    # Get the underlying function from the classmethod to allow calling with
    # the actual subclass type, not the class where __pre_init__ was defined.
    pre_init_func = pre_init.__func__  # type: ignore[union-attr]

    def wrapper(self: object, *args: object, **kwargs: object) -> None:
        actual_cls = type(self)
        if len(args) > len(init_field_defs):
            raise TypeError(
                f"{actual_cls.__name__}() takes {len(init_field_defs)} positional arguments but {len(args)} were given"
            )

        provided = dict(kwargs)
        bound = _bind_fields(
            actual_cls, init_field_defs, init_field_names, args, provided
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
        # Only expose init fields to transform (non-init fields are derived)
        current = {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[arg-type]
            if f.init
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
        if hasattr(source, name):
            updates[name] = getattr(source, name)

    if not updates:
        raise TypeError(f"{cls.__name__}.merge() source has no matching fields")

    return updates


def _apply_changes(
    instance: _SupportsUpdate, changes: Mapping[str, object]
) -> _SupportsUpdate:
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
