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

"""Tests for frozen dataclass helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import _MISSING_TYPE, MISSING, FrozenInstanceError, field
from typing import ClassVar, Protocol, Self, cast

import pytest

from weakincentives.dataclasses import FrozenDataclass


class HasFrozenOps(Protocol):
    calls: ClassVar[list[str]]
    value: int
    label: str

    def update(self, **changes: object) -> Self: ...

    def merge(self, mapping_or_obj: object) -> Self: ...

    def map(
        self, transform: Callable[[dict[str, object]], Mapping[str, object]]
    ) -> Self: ...


def test_pre_init_shapes_inputs_and_runs_post_init() -> None:
    @FrozenDataclass()
    class Invoice:
        total_cents: int
        tax_rate: float
        tax_cents: int | None = None
        grand_total_cents: int | None = None

        @classmethod
        def __pre_init__(
            cls,
            *,
            total_cents: int,
            tax_rate: float,
            tax_cents: int | _MISSING_TYPE | None = MISSING,
            grand_total_cents: int | _MISSING_TYPE | None = MISSING,
        ) -> dict[str, int | float]:
            computed_tax = (
                int(tax_cents)
                if not isinstance(tax_cents, _MISSING_TYPE) and tax_cents is not None
                else int(total_cents * tax_rate)
            )
            computed_total = (
                int(grand_total_cents)
                if (
                    not isinstance(grand_total_cents, _MISSING_TYPE)
                    and grand_total_cents is not None
                )
                else total_cents + computed_tax
            )
            return {
                "total_cents": total_cents,
                "tax_rate": tax_rate,
                "tax_cents": computed_tax,
                "grand_total_cents": computed_total,
            }

        def __post_init__(self) -> None:
            if self.grand_total_cents is None:
                raise ValueError("grand total missing")

            if self.grand_total_cents < self.total_cents:
                raise ValueError("total cannot decrease")

    invoice = Invoice(total_cents=1000, tax_rate=0.2)

    assert invoice.tax_cents == 200
    assert invoice.grand_total_cents == 1200
    assert invoice.__slots__ == (  # type: ignore[attr-defined]
        "total_cents",
        "tax_rate",
        "tax_cents",
        "grand_total_cents",
    )

    with pytest.raises(ValueError):
        Invoice(total_cents=500, tax_rate=0.3, grand_total_cents=100)


def test_pre_init_validates_field_coverage() -> None:
    @FrozenDataclass()
    class Example:
        value: int

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {}

    with pytest.raises(TypeError):
        Example(value=1)

    @FrozenDataclass()
    class Extra:
        value: int

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {"value": value, "extra": 2}

    with pytest.raises(TypeError):
        Extra(value=1)


def test_copy_helpers_skip_pre_init_and_support_mapping_and_objects() -> None:
    @FrozenDataclass()
    class Tracker:
        calls: ClassVar[list[str]] = []
        value: int
        label: str = field(default="base")

        @classmethod
        def __pre_init__(cls, *, value: int, label: str = "base") -> dict[str, object]:
            cls.calls.append("pre")
            return {"value": value, "label": label}

        def __post_init__(self) -> None:
            type(self).calls.append("post")

    tracker = cast(HasFrozenOps, Tracker(1))

    with pytest.raises(FrozenInstanceError):
        tracker.value = 2

    updated = tracker.update(value=2)
    merged = tracker.merge({"label": "from-mapping"})

    class Source:
        value = 3
        label = "from-object"

    merged_obj = tracker.merge(Source())
    remapped = tracker.map(lambda current: {"value": current["value"] + 10})

    assert tracker.calls == ["pre", "post", "post", "post", "post", "post"]
    assert updated.value == 2
    assert merged.label == "from-mapping"
    assert merged_obj.value == 3
    assert remapped.value == 11

    with pytest.raises(TypeError):
        tracker.merge({"unknown": 1})

    with pytest.raises(TypeError):
        tracker.map(lambda current: {**current, "extra": 1})


def test_pre_init_validates_inputs_and_defaults() -> None:
    @FrozenDataclass()
    class Sample:
        value: int

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {"value": value}

    with pytest.raises(TypeError):
        Sample(1, 2)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Sample(value=1, extra=2)  # type: ignore[call-arg]

    @FrozenDataclass()
    class WithDefaults:
        items: list[int] = field(default_factory=lambda: [1])

        @classmethod
        def __pre_init__(cls, *, items: list[int]) -> dict[str, list[int]]:
            return {"items": list(items)}

    with_defaults = WithDefaults()
    assert with_defaults.items == [1]

    @FrozenDataclass()
    class BadReturn:
        value: int

        @classmethod
        def __pre_init__(cls, *, value: int) -> list[int]:
            return [value]

    with pytest.raises(TypeError):
        BadReturn(value=1)

    @FrozenDataclass()
    class Partial:
        required: int
        derived: int

        @classmethod
        def __pre_init__(
            cls,
            *,
            required: int | _MISSING_TYPE | None = MISSING,
            derived: int | _MISSING_TYPE | None = MISSING,
        ) -> dict[str, int]:
            base = int(required) if not isinstance(required, _MISSING_TYPE) else 0
            derived_value = base if isinstance(derived, _MISSING_TYPE) else int(derived)
            return {"required": base, "derived": derived_value}

    partial = Partial(required=3)  # type: ignore[call-arg]
    assert partial.derived == 3


def test_merge_requires_source_fields() -> None:
    @FrozenDataclass()
    class Target:
        value: int

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {"value": value}

    target = cast(HasFrozenOps, Target(1))

    with pytest.raises(TypeError, match="no matching fields"):
        target.merge(object())


def test_merge_with_partial_object_attributes() -> None:
    """Objects only need matching attributes, not all fields."""

    @FrozenDataclass()
    class Multi:
        a: int
        b: int
        c: int

    multi = Multi(1, 2, 3)

    class PartialSource:
        a = 10

    merged = cast(HasFrozenOps, multi).merge(PartialSource())
    assert merged.a == 10  # type: ignore[attr-defined]
    assert merged.b == 2  # type: ignore[attr-defined]
    assert merged.c == 3  # type: ignore[attr-defined]


def test_frozen_dataclass_without_pre_init() -> None:
    """Classes without __pre_init__ work normally."""

    @FrozenDataclass()
    class Simple:
        value: int
        label: str = "default"

    simple = Simple(42)
    assert simple.value == 42
    assert simple.label == "default"

    simple2 = Simple(value=10, label="custom")
    assert simple2.value == 10
    assert simple2.label == "custom"

    # Copy helpers still work
    simple_ops = cast(HasFrozenOps, simple)
    updated = simple_ops.update(label="updated")
    assert updated.label == "updated"
    assert updated.value == 42


def test_update_rejects_unknown_fields() -> None:
    @FrozenDataclass()
    class Target:
        value: int

    target = cast(HasFrozenOps, Target(1))

    with pytest.raises(TypeError, match="unexpected field"):
        target.update(unknown=99)


def test_pre_init_with_positional_args() -> None:
    @FrozenDataclass()
    class Positional:
        a: int
        b: str
        c: float = 1.5

        @classmethod
        def __pre_init__(
            cls, *, a: int, b: str, c: float = 1.5
        ) -> dict[str, int | str | float]:
            return {"a": a * 2, "b": b.upper(), "c": c}

    # All positional
    p1 = Positional(5, "hello")
    assert p1.a == 10
    assert p1.b == "HELLO"
    assert p1.c == 1.5

    # Mixed positional and keyword
    p2 = Positional(3, b="world", c=2.5)
    assert p2.a == 6
    assert p2.b == "WORLD"
    assert p2.c == 2.5


def test_dataclass_options_passthrough() -> None:
    """Custom dataclass options are respected."""

    @FrozenDataclass(frozen=False, slots=False, order=True)
    class Mutable:
        value: int

    m = Mutable(1)
    m.value = 2
    assert m.value == 2
    assert not hasattr(m, "__slots__")

    # Order should work
    assert Mutable(1) < Mutable(2)  # type: ignore[operator]


class _WithDerivedOps(Protocol):
    value: int
    derived: int

    def update(self, **kwargs: object) -> Self: ...
    def merge(self, source: object) -> Self: ...


def test_copy_helpers_reject_changes_to_non_init_fields() -> None:
    """Copy helpers reject attempts to change derived (non-init) fields."""

    @FrozenDataclass()
    class WithDerived:
        value: int
        derived: int = field(init=False, default=0)

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {"value": value, "derived": value * 2}

    instance = cast(_WithDerivedOps, WithDerived(value=5))
    assert instance.derived == 10

    with pytest.raises(TypeError, match="cannot update derived field"):
        instance.update(derived=20)

    # merge() rejects unknown fields (non-init fields are not in merge's field list)
    with pytest.raises(TypeError, match="received unexpected fields"):
        instance.merge({"derived": 20})


def test_copy_helpers_recompute_derived_fields() -> None:
    """Copy helpers recompute non-init fields via __pre_init__."""

    @FrozenDataclass()
    class WithDerived:
        value: int
        derived: int = field(init=False, default=0)

        @classmethod
        def __pre_init__(cls, *, value: int) -> dict[str, int]:
            return {"value": value, "derived": value * 2}

    instance = cast(_WithDerivedOps, WithDerived(value=5))
    assert instance.value == 5
    assert instance.derived == 10

    updated = instance.update(value=7)
    assert updated.value == 7
    assert updated.derived == 14  # Recomputed via __pre_init__

    merged = instance.merge({"value": 3})
    assert merged.value == 3
    assert merged.derived == 6  # Recomputed via __pre_init__
