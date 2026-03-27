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

This package provides the :func:`FrozenDataclass` decorator and the
:class:`Constructable` base class for building immutable, memory-efficient
data structures.

Exported Symbols
----------------
FrozenDataclass
    A decorator that creates frozen, slotted dataclasses with sensible
    defaults.

Constructable
    A base class for frozen dataclasses that require validated construction.
    Subclasses define a ``create()`` classmethod; direct ``__init__`` is
    blocked.  A ``replace()`` method is provided automatically.

allow_construction
    A context manager that temporarily permits direct ``__init__`` calls
    on ``Constructable`` subclasses.  Used inside ``create()`` methods
    and by framework code such as ``serde.parse()``.

Two Tiers
---------
**Tier 1 — Simple value objects.**  Use ``@FrozenDataclass()`` alone::

    @FrozenDataclass()
    class Point:
        x: float
        y: float

    p1 = Point(1.0, 2.0)
    p1.x = 5.0  # Raises FrozenInstanceError

**Tier 2 — Validated / normalized classes.**  Inherit from
``Constructable`` and define a ``create()`` classmethod::

    @FrozenDataclass()
    class Deadline(Constructable):
        expires_at: datetime
        started_at: datetime

        @classmethod
        def create(cls, expires_at: datetime, started_at: datetime | None = None) -> Deadline:
            if started_at is None:
                started_at = datetime.now(UTC)
            with allow_construction():
                return cls(expires_at=expires_at, started_at=started_at)

    Deadline(...)          # TypeError
    Deadline.create(...)   # OK

See Also
--------
- :mod:`dataclasses` : Python's standard dataclass module
- :mod:`weakincentives.serde` : Serialization utilities for dataclasses
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Self,
    TypedDict,
    Unpack,
    dataclass_transform,
)

__all__ = ["Constructable", "FrozenDataclass", "allow_construction"]


_ALLOW_INIT: ContextVar[bool] = ContextVar("_allow_init", default=False)


class DataclassOptions(TypedDict, total=False):
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    match_args: bool
    kw_only: bool
    slots: bool


@contextmanager
def allow_construction() -> Iterator[None]:
    """Temporarily permit direct ``__init__`` on ``Constructable`` subclasses.

    Use inside ``create()`` methods and framework code (e.g. ``serde.parse``)
    that needs to construct guarded instances.

    Example::

        @classmethod
        def create(cls, value: int) -> MyClass:
            validated = _validate(value)
            with allow_construction():
                return cls(value=validated)
    """
    token = _ALLOW_INIT.set(True)
    try:
        yield
    finally:
        _ALLOW_INIT.reset(token)


class Constructable:
    """Base for frozen dataclasses requiring factory construction.

    Subclasses define a fully-typed ``create()`` classmethod that validates,
    normalizes, and derives all field values *before* construction.  Direct
    ``__init__`` is blocked at runtime — construction is only permitted
    inside an :func:`allow_construction` context.

    ``replace(**changes)`` is provided automatically: it introspects
    ``create()``'s signature, copies current values for unchanged fields,
    and delegates to ``create()`` so that all validation re-runs.

    Example::

        @FrozenDataclass()
        class Order(Constructable):
            subtotal: int
            tax: int
            total: int

            @classmethod
            def create(cls, subtotal: int, tax_rate: float = 0.1) -> Order:
                tax = int(subtotal * tax_rate)
                with allow_construction():
                    return cls(subtotal=subtotal, tax=tax, total=subtotal + tax)

        order = Order.create(subtotal=1000)
        order2 = order.replace(subtotal=2000)
    """

    __slots__ = ()

    def replace(self, **changes: object) -> Self:
        """Return a copy with the given fields replaced.

        Only fields that appear as parameters in ``create()`` can be
        replaced.  Derived fields are recomputed automatically because
        ``replace()`` delegates to ``create()``.
        """
        cls = type(self)
        create_method = getattr(cls, "create", None)
        if create_method is None:  # pragma: no cover
            raise TypeError(f"{cls.__name__} does not define create()")

        sig = inspect.signature(create_method)
        create_params = {
            name
            for name, p in sig.parameters.items()
            if name != "cls" and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
        }

        unknown = set(changes) - create_params
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise TypeError(
                f"{cls.__name__}.replace() got unexpected field(s): {joined}"
            )

        current = {name: getattr(self, name) for name in create_params}
        current.update(changes)
        return cls.create(**current)  # type: ignore[return-value]


@dataclass_transform(field_specifiers=(field,))
def FrozenDataclass[T](
    **dataclass_kwargs: Unpack[DataclassOptions],
) -> Callable[[type[T]], type[T]]:
    """Dataclass decorator with frozen, slotted defaults.

    Freezing is mandatory — all ``FrozenDataclass`` instances are immutable.
    Other standard dataclass options can be customized::

        @FrozenDataclass(order=True)
        class Version:
            major: int
            minor: int

    For classes that need validation or normalization at construction time,
    inherit from :class:`Constructable` and define a ``create()`` classmethod.
    The decorator automatically guards ``__init__`` for such classes.
    """

    options: DataclassOptions = {
        "init": True,
        "repr": True,
        "eq": True,
        "order": False,
        "unsafe_hash": False,
        "match_args": True,
        "kw_only": False,
        "slots": True,
        **dataclass_kwargs,
    }

    def decorator(cls: type[T]) -> type[T]:
        dc = dataclass(frozen=True, **options)(cls)

        if issubclass(dc, Constructable):
            if "__post_init__" in dc.__dict__:
                msg = (
                    f"{dc.__name__} inherits from Constructable and must not "
                    f"define __post_init__. Use create() instead."
                )
                raise TypeError(msg)
            _inject_init_guard(dc)

        return dc

    return decorator


def _inject_init_guard(cls: type) -> None:
    """Wrap ``__init__`` to reject direct construction."""
    original_init = cls.__init__

    @functools.wraps(original_init)
    def guarded_init(self: object, *args: object, **kwargs: object) -> None:
        if not _ALLOW_INIT.get():
            msg = (
                f"{type(self).__name__}() is not directly constructable. "
                f"Use {type(self).__name__}.create() instead."
            )
            raise TypeError(msg)
        original_init(self, *args, **kwargs)

    type.__setattr__(cls, "__init__", guarded_init)
