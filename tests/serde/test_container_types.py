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

"""Tests for container and collection types in serde."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.serde._fixtures import (
    CollectionModel,
    Color,
    Container,
    MappingModel,
    SetHolder,
)
from weakincentives.serde import dump, parse

pytestmark = pytest.mark.core


def test_parse_collection_types() -> None:
    model = parse(
        CollectionModel,
        {
            "unique_tags": "alpha",
            "scores": ["1", "2", "3"],
            "history": ["4", "5", "6"],
            "mapping": {"1": "one", 2: "two"},
        },
    )
    assert model.unique_tags == {"alpha"}
    assert model.scores == (1, 2, 3)
    assert model.history == (4, 5, 6)
    assert model.mapping == {1: "one", 2: "two"}


def test_parse_collection_length_mismatch() -> None:
    with pytest.raises(ValueError) as exc:
        parse(
            CollectionModel,
            {
                "unique_tags": [],
                "scores": ["1", "2"],
                "history": [],
                "mapping": {},
            },
        )
    assert "scores: expected 3 items" in str(exc.value)


def test_parse_dict_key_error_message() -> None:
    with pytest.raises(TypeError) as exc:
        parse(MappingModel, {"mapping": {"bad": "value"}})
    assert "mapping keys" in str(exc.value)


def test_dump_exclude_none_recursively() -> None:
    container = Container(values={"keep": 1, "drop": None}, items=["x", None])
    payload = dump(container, exclude_none=True)
    assert payload["values"] == {"keep": 1}
    assert payload["items"] == ["x"]
    assert "nested" not in payload


def test_dump_serializes_sets_sorted() -> None:
    holder = SetHolder(values={3, 1, 2})
    payload = dump(holder)
    assert payload["values"] == [1, 2, 3]


def test_dump_set_exclude_none_values() -> None:
    @dataclass
    class OptionalSetHolder:
        values: set[int | None]

    holder = OptionalSetHolder(values={1, None})
    payload = dump(holder, exclude_none=True)
    assert payload["values"] == [1]


def test_dump_set_sort_fallback_on_bad_repr() -> None:
    class BadReprStr(str):
        repr_calls = 0

        def __repr__(self) -> str:  # pragma: no cover - executed via dump()
            type(self).repr_calls += 1
            raise TypeError("__repr__ returned non-string")

    @dataclass
    class FancySetHolder:
        values: set[str]

    holder = FancySetHolder({BadReprStr("b"), BadReprStr("a")})
    payload = dump(holder)
    values = payload["values"]
    assert set(values) == {"a", "b"}
    assert BadReprStr.repr_calls > 0


def test_dump_serializes_frozensets_sorted() -> None:
    @dataclass
    class FrozenSetHolder:
        values: frozenset[int]

    holder = FrozenSetHolder(values=frozenset({3, 1, 2}))
    payload = dump(holder)
    assert payload["values"] == [1, 2, 3]


def test_collection_type_errors_and_conversions() -> None:
    @dataclass
    class CollectionErrors:
        items: list[int]
        unique: set[int]
        pair: tuple[int, int]

    with pytest.raises(TypeError):
        parse(CollectionErrors, {"items": 1, "unique": [], "pair": [1, 2]})

    with pytest.raises(TypeError):
        parse(CollectionErrors, {"items": [], "unique": 1, "pair": [1, 2]})

    with pytest.raises(ValueError):
        parse(CollectionErrors, {"items": [], "unique": [], "pair": [1]})

    parsed = parse(
        CollectionErrors,
        {"items": "3", "unique": "4", "pair": ("5", "6")},
    )
    assert parsed.items == [3]
    assert parsed.unique == {4}
    assert parsed.pair == (5, 6)

    with pytest.raises(TypeError):
        parse(CollectionErrors, {"items": [], "unique": [], "pair": 1})

    parsed_iter = parse(
        CollectionErrors,
        {"items": [], "unique": iter([7, 8]), "pair": (1, 2)},
    )
    assert parsed_iter.unique == {7, 8}

    with pytest.raises(TypeError):
        parse(
            CollectionErrors, {"items": [], "unique": 1, "pair": (1, 2)}, coerce=False
        )

    with pytest.raises(ValueError):
        parse(CollectionErrors, {"items": [], "unique": [], "pair": "7"})


def test_mapping_and_enum_branches() -> None:
    @dataclass
    class MappingEnum:
        mapping: dict[int, str]
        color: Color

    parsed = parse(
        MappingEnum,
        {"mapping": {"1": 2}, "color": "GREEN"},
    )
    assert parsed.mapping == {1: "2"}
    assert parsed.color is Color.GREEN

    direct = parse(MappingEnum, {"mapping": {1: "a"}, "color": Color.RED})
    assert direct.color is Color.RED

    with pytest.raises(TypeError):
        parse(MappingEnum, {"mapping": [], "color": "GREEN"})

    with pytest.raises(ValueError):
        parse(MappingEnum, {"mapping": {"1": 2}, "color": "purple"})

    with pytest.raises(ValueError):
        parse(MappingEnum, {"mapping": {"1": 2}, "color": ["GREEN"]})

    with pytest.raises(TypeError):
        parse(MappingEnum, {"mapping": {1: "a"}, "color": "GREEN"}, coerce=False)


def test_serialize_set_sorting_and_extra_policy_noop() -> None:
    @dataclass
    class SetHolder:
        @dataclass(frozen=True)
        class FrozenAddress:
            street: str

        values: set[FrozenAddress]

    holder = SetHolder(
        {
            SetHolder.FrozenAddress("one"),
            SetHolder.FrozenAddress("two"),
        }
    )
    dumped = dump(holder, by_alias=False)
    assert isinstance(dumped["values"], list)
    assert len(dumped["values"]) == 2

    @dataclass
    class Simple:
        name: str

    parsed = parse(Simple, {"name": "Ada"}, extra="allow")
    assert parsed.name == "Ada"
