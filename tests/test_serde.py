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

"""Tests for ``weakincentives.serde``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, cast

import pytest

from weakincentives.serde import (
    ParseError,
    SerdeError,
    dump,
    parse,
    type_identifier,
)


@dataclass(frozen=True)
class Inner:
    label: str
    value: Annotated[int, {"ge": 0, "le": 100}]


@dataclass(frozen=True)
class Outer:
    name: Annotated[str, {"min_length": 1, "max_length": 32}]
    items: tuple[Inner, ...] = ()
    extra: dict[str, int] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class Maybe:
    value: int | None


@dataclass(frozen=True)
class Cat:
    purr: bool


@dataclass(frozen=True)
class Dog:
    bark: bool


@dataclass(frozen=True)
class AnimalHolder:
    animal: Cat | Dog


@dataclass(frozen=True)
class IntCarrier:
    value: int


@dataclass(frozen=True)
class StrCarrier:
    text: str


@dataclass(frozen=True)
class UnionHolder:
    v: IntCarrier | StrCarrier


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_round_trip_through_dump() -> None:
    obj = Outer(
        name="root",
        items=(Inner(label="a", value=1), Inner(label="b", value=2)),
        extra={"k": 3},
    )
    encoded = dump(obj)
    parsed = parse(Outer, encoded)
    assert parsed == obj


def test_parse_rejects_missing_required_field() -> None:
    with pytest.raises(ParseError, match="missing required field"):
        parse(Inner, {"label": "x"})


def test_parse_rejects_unexpected_fields() -> None:
    with pytest.raises(ParseError, match="unexpected fields"):
        parse(Inner, {"label": "x", "value": 1, "rogue": True})


def test_parse_rejects_wrong_type_for_int() -> None:
    with pytest.raises(ParseError, match="expected int"):
        parse(Inner, {"label": "x", "value": "no"})


def test_parse_rejects_bool_for_int() -> None:
    with pytest.raises(ParseError, match="expected int"):
        parse(Inner, {"label": "x", "value": True})


def test_parse_int_for_float() -> None:
    @dataclass(frozen=True)
    class WithFloat:
        x: float

    assert parse(WithFloat, {"x": 3}).x == 3.0


def test_parse_rejects_bool_for_float() -> None:
    @dataclass(frozen=True)
    class WithFloat:
        x: float

    with pytest.raises(ParseError, match="expected number"):
        parse(WithFloat, {"x": True})


def test_parse_string() -> None:
    @dataclass(frozen=True)
    class WithStr:
        x: str

    assert parse(WithStr, {"x": "hi"}).x == "hi"


def test_parse_string_rejects_int() -> None:
    @dataclass(frozen=True)
    class WithStr:
        x: str

    with pytest.raises(ParseError, match="expected string"):
        parse(WithStr, {"x": 3})


def test_parse_bool_rejects_int() -> None:
    @dataclass(frozen=True)
    class WithBool:
        x: bool

    with pytest.raises(ParseError, match="expected bool"):
        parse(WithBool, {"x": 1})


def test_parse_bool_accepts_bool() -> None:
    @dataclass(frozen=True)
    class WithBool:
        x: bool

    assert parse(WithBool, {"x": True}).x is True


def test_parse_float_accepts_float() -> None:
    @dataclass(frozen=True)
    class WithFloat:
        x: float

    assert parse(WithFloat, {"x": 3.5}).x == 3.5


def test_parse_float_rejects_string() -> None:
    @dataclass(frozen=True)
    class WithFloat:
        x: float

    with pytest.raises(ParseError, match="expected number"):
        parse(WithFloat, {"x": "no"})


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_parse_ge_constraint_violation() -> None:
    with pytest.raises(ParseError, match="ge=0"):
        parse(Inner, {"label": "x", "value": -1})


def test_parse_le_constraint_violation() -> None:
    with pytest.raises(ParseError, match="le=100"):
        parse(Inner, {"label": "x", "value": 101})


def test_parse_pattern_constraint() -> None:
    @dataclass(frozen=True)
    class Slug:
        name: Annotated[str, {"pattern": r"^[a-z]+$"}]

    parse(Slug, {"name": "hello"})
    with pytest.raises(ParseError, match="pattern"):
        parse(Slug, {"name": "Hello!"})


def test_parse_length_constraints() -> None:
    @dataclass(frozen=True)
    class Bounded:
        value: Annotated[str, {"min_length": 2, "max_length": 4}]

    parse(Bounded, {"value": "ab"})
    with pytest.raises(ParseError, match="length"):
        parse(Bounded, {"value": "a"})
    with pytest.raises(ParseError, match="length"):
        parse(Bounded, {"value": "abcde"})


def test_parse_gt_lt_constraints() -> None:
    @dataclass(frozen=True)
    class Ranged:
        value: Annotated[int, {"gt": 0, "lt": 10}]

    parse(Ranged, {"value": 5})
    with pytest.raises(ParseError, match="not >"):
        parse(Ranged, {"value": 0})
    with pytest.raises(ParseError, match="not <"):
        parse(Ranged, {"value": 10})


def test_parse_unknown_constraint_raises() -> None:
    @dataclass(frozen=True)
    class Bad:
        value: Annotated[int, {"weird": 1}]

    with pytest.raises(SerdeError, match="unknown constraint"):
        parse(Bad, {"value": 1})


# ---------------------------------------------------------------------------
# Containers and unions
# ---------------------------------------------------------------------------


def test_parse_homogeneous_tuple() -> None:
    parsed = parse(Outer, {"name": "x", "items": [], "extra": {}})
    assert parsed.items == ()


def test_parse_heterogeneous_tuple() -> None:
    @dataclass(frozen=True)
    class Pair:
        kind: tuple[str, int]

    assert parse(Pair, {"kind": ["x", 7]}).kind == ("x", 7)


def test_parse_heterogeneous_tuple_length_mismatch() -> None:
    @dataclass(frozen=True)
    class Pair:
        kind: tuple[str, int]

    with pytest.raises(ParseError, match="tuple expected 2"):
        parse(Pair, {"kind": ["x"]})


def test_parse_list_field() -> None:
    @dataclass(frozen=True)
    class WithList:
        values: list[int]

    assert parse(WithList, {"values": [1, 2]}).values == [1, 2]


def test_parse_mapping_field() -> None:
    @dataclass(frozen=True)
    class WithMap:
        entries: dict[str, int]

    assert parse(WithMap, {"entries": {"a": 1}}).entries == {"a": 1}


def test_parse_optional_none() -> None:
    assert parse(Maybe, {"value": None}).value is None


def test_parse_optional_present() -> None:
    assert parse(Maybe, {"value": 5}).value == 5


def test_parse_polymorphic_union_via_type() -> None:
    payload = {
        "animal": {
            "__type__": type_identifier(Dog),
            "bark": True,
        }
    }
    parsed = parse(AnimalHolder, payload)
    assert isinstance(parsed.animal, Dog)
    assert parsed.animal.bark is True


def test_parse_union_falls_back_when_no_type_tag() -> None:
    parsed = parse(UnionHolder, {"v": {"text": "hello"}})
    assert isinstance(parsed.v, StrCarrier)


def test_parse_union_no_match_raises() -> None:
    with pytest.raises(ParseError, match="no union member"):
        parse(UnionHolder, {"v": {"unknown": 1}})


def test_parse_union_with_unknown_type_tag_raises() -> None:
    @dataclass(frozen=True)
    class _Holder:
        v: IntCarrier | None

    payload = {"v": {"__type__": "x.Y", "value": 1}}
    with pytest.raises(ParseError, match="no union member matches"):
        # Top-level Holder defined at module level for `get_type_hints`.
        parse(UnionHolder, payload)


def test_parse_rejects_null_for_value_type() -> None:
    with pytest.raises(ParseError, match="expected null"):
        parse(type(None), 5)


def test_parse_rejects_non_object_for_dataclass() -> None:
    with pytest.raises(ParseError, match="expected object"):
        parse(Inner, [])


def test_parse_rejects_non_array_for_sequence() -> None:
    @dataclass(frozen=True)
    class WithList:
        values: list[int]

    with pytest.raises(ParseError, match="expected array"):
        parse(WithList, {"values": "no"})


def test_parse_rejects_non_object_for_mapping() -> None:
    @dataclass(frozen=True)
    class WithMap:
        entries: dict[str, int]

    with pytest.raises(ParseError, match="expected object"):
        parse(WithMap, {"entries": []})


def test_parse_rejects_unknown_concrete_type() -> None:
    with pytest.raises(ParseError, match="unsupported concrete type"):
        parse(complex, {})


def test_parse_rejects_unsupported_annotation() -> None:
    bogus = cast("type[object]", 42)
    with pytest.raises(ParseError, match="unsupported annotation"):
        parse(bogus, {})


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def test_dump_primitives() -> None:
    assert dump(None) is None
    assert dump(1) == 1
    assert dump(1.5) == 1.5
    assert dump(True) is True
    assert dump("x") == "x"


def test_dump_lists_and_dicts() -> None:
    assert dump([1, 2]) == [1, 2]
    assert dump((1, 2)) == [1, 2]
    assert dump({"a": 1}) == {"a": 1}


def test_parse_none_annotation_returns_none() -> None:
    assert parse(type(None), None) is None


def test_parse_skips_field_with_default_factory() -> None:
    parsed = parse(Outer, {"name": "x"})
    assert parsed.items == ()
    assert parsed.extra == {}


def test_parse_union_with_array_data_falls_back() -> None:
    @dataclass(frozen=True)
    class Holder:
        v: list[int] | str

    assert parse(Holder, {"v": [1, 2]}).v == [1, 2]


def test_parse_constraint_skipped_for_non_matching_type() -> None:
    @dataclass(frozen=True)
    class Plain:
        value: Annotated[str, {"ge": 0}]

    assert parse(Plain, {"value": "hello"}).value == "hello"


def test_split_annotated_ignores_non_dict_extras() -> None:
    @dataclass(frozen=True)
    class WithExtra:
        value: Annotated[int, "documentation"]

    assert parse(WithExtra, {"value": 5}).value == 5


def test_dump_dataclass_includes_type_tag() -> None:
    encoded = dump(Inner(label="x", value=1))
    assert isinstance(encoded, dict)
    payload = cast("dict[str, object]", encoded)
    assert payload["__type__"] == type_identifier(Inner)
    assert payload["label"] == "x"


def test_dump_rejects_unknown_value() -> None:
    with pytest.raises(SerdeError, match="cannot dump"):
        dump(complex(1, 2))
