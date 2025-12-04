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
from dataclasses import MISSING, FrozenInstanceError, field
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
            tax_cents: int | object = MISSING,
            grand_total_cents: int | object = MISSING,
        ) -> dict[str, int | float]:
            computed_tax = (
                int(tax_cents)
                if tax_cents is not MISSING and tax_cents is not None
                else int(total_cents * tax_rate)
            )
            computed_total = (
                int(grand_total_cents)
                if grand_total_cents is not MISSING and grand_total_cents is not None
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
            required: int | object = MISSING,
            derived: int | object = MISSING,
        ) -> dict[str, int]:
            base = int(required) if required is not MISSING else 0
            derived_value = base if derived is MISSING else int(derived)
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

    with pytest.raises(TypeError):
        target.merge(object())
