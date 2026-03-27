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

from dataclasses import FrozenInstanceError

import pytest

from weakincentives.dataclasses import (
    Constructable,
    FrozenDataclass,
    allow_construction,
)

pytestmark = pytest.mark.core


# --- Tier 1: Plain FrozenDataclass tests ---


def test_frozen_instance_rejects_mutation() -> None:
    @FrozenDataclass()
    class Point:
        x: float
        y: float

    p = Point(1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        p.x = 3.0  # type: ignore[misc]


def test_frozen_dataclass_has_slots_by_default() -> None:
    @FrozenDataclass()
    class Slotted:
        value: int

    assert Slotted.__slots__ == ("value",)  # type: ignore[attr-defined]


def test_frozen_dataclass_plain_construction() -> None:
    """Plain FrozenDataclass (no Constructable) allows direct construction."""

    @FrozenDataclass()
    class Plain:
        value: int
        label: str = "default"

    plain = Plain(42)
    assert plain.value == 42
    assert plain.label == "default"

    with pytest.raises(FrozenInstanceError):
        plain.value = 1  # type: ignore[misc]


def test_frozen_dataclass_order_option() -> None:
    """Custom dataclass options are respected."""

    @FrozenDataclass(order=True)
    class Ordered:
        value: int

    assert Ordered(1) < Ordered(2)  # type: ignore[operator]


# --- Tier 2: Constructable tests ---


def test_constructable_blocks_direct_init() -> None:
    """Direct __init__ on Constructable subclass raises TypeError."""

    @FrozenDataclass()
    class Guarded(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Guarded:
            with allow_construction():
                return cls(value=value)

    with pytest.raises(TypeError, match="not directly constructable"):
        Guarded(value=42)


def test_constructable_create_works() -> None:
    """create() classmethod allows construction via allow_construction()."""

    @FrozenDataclass()
    class Guarded(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Guarded:
            with allow_construction():
                return cls(value=value)

    g = Guarded.create(value=42)
    assert g.value == 42


def test_constructable_create_validates() -> None:
    """create() can perform validation before construction."""

    @FrozenDataclass()
    class Positive(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Positive:
            if value <= 0:
                raise ValueError("must be positive")
            with allow_construction():
                return cls(value=value)

    with pytest.raises(ValueError, match="must be positive"):
        Positive.create(value=-1)


def test_constructable_create_with_derived_fields() -> None:
    """create() can compute derived fields."""

    @FrozenDataclass()
    class Invoice(Constructable):
        total_cents: int
        tax_rate: float
        tax_cents: int
        grand_total_cents: int

        @classmethod
        def create(cls, total_cents: int, tax_rate: float = 0.2) -> Invoice:
            tax_cents = int(total_cents * tax_rate)
            with allow_construction():
                return cls(
                    total_cents=total_cents,
                    tax_rate=tax_rate,
                    tax_cents=tax_cents,
                    grand_total_cents=total_cents + tax_cents,
                )

    invoice = Invoice.create(total_cents=1000)
    assert invoice.tax_cents == 200
    assert invoice.grand_total_cents == 1200


def test_constructable_replace() -> None:
    """replace() delegates to create() and recomputes derived fields."""

    @FrozenDataclass()
    class Invoice(Constructable):
        total_cents: int
        tax_rate: float
        tax_cents: int
        grand_total_cents: int

        @classmethod
        def create(cls, total_cents: int, tax_rate: float = 0.2) -> Invoice:
            tax_cents = int(total_cents * tax_rate)
            with allow_construction():
                return cls(
                    total_cents=total_cents,
                    tax_rate=tax_rate,
                    tax_cents=tax_cents,
                    grand_total_cents=total_cents + tax_cents,
                )

    invoice = Invoice.create(total_cents=1000)
    updated = invoice.replace(total_cents=2000)
    assert updated.total_cents == 2000
    assert updated.tax_cents == 400
    assert updated.grand_total_cents == 2400
    # Original unchanged
    assert invoice.total_cents == 1000


def test_replace_rejects_unknown_fields() -> None:
    """replace() raises TypeError for fields not in create() signature."""

    @FrozenDataclass()
    class Simple(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Simple:
            with allow_construction():
                return cls(value=value)

    s = Simple.create(value=1)
    with pytest.raises(TypeError, match="unexpected field"):
        s.replace(unknown=99)


def test_constructable_bans_post_init() -> None:
    """Constructable subclasses cannot define __post_init__."""
    with pytest.raises(TypeError, match="must not define __post_init__"):

        @FrozenDataclass()
        class Bad(Constructable):
            value: int

            def __post_init__(self) -> None:
                pass


def test_allow_construction_is_reentrant() -> None:
    """allow_construction() context manager is reentrant."""

    @FrozenDataclass()
    class Inner(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Inner:
            with allow_construction():
                return cls(value=value)

    with allow_construction():
        # Nested allow_construction in create() should work
        inner = Inner.create(value=42)
        assert inner.value == 42


def test_allow_construction_resets_on_exception() -> None:
    """allow_construction() resets even if an exception is raised."""

    @FrozenDataclass()
    class Guarded(Constructable):
        value: int

        @classmethod
        def create(cls, value: int) -> Guarded:
            with allow_construction():
                return cls(value=value)

    with pytest.raises(RuntimeError):
        with allow_construction():
            raise RuntimeError("boom")

    # Guard should be back in place
    with pytest.raises(TypeError, match="not directly constructable"):
        Guarded(value=1)
