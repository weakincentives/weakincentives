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

"""Tests for serde constraints and literal handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID

import pytest

from tests.serde._fixtures import (
    Color,
    LengthModel,
    LiteralBoolModel,
    LiteralModel,
    MembershipModel,
    NumericModel,
    OptionalOnly,
    User,
    WithInitFalse,
    as_dict,
    as_list,
    user_payload,
)
from weakincentives.serde import parse, schema
from weakincentives.types import JSONValue

pytestmark = pytest.mark.core


def test_parse_literal_and_bool_coercion() -> None:
    model = parse(
        LiteralModel,
        {"mode": "manual", "flag": "off", "value": "3.5"},
    )
    assert model.mode == "manual"
    assert model.flag is False
    assert model.value == 3.5


def test_parse_literal_invalid_values() -> None:
    with pytest.raises(ValueError) as exc:
        parse(LiteralModel, {"mode": "unknown", "flag": True, "value": 1})
    assert "mode: expected one of" in str(exc.value)
    with pytest.raises(TypeError) as exc2:
        parse(LiteralModel, {"mode": "auto", "flag": "maybe", "value": 1})
    assert "flag: Cannot interpret" in str(exc2.value)


def test_parse_literal_bool_strings() -> None:
    result = parse(LiteralBoolModel, {"flag": "true"})
    assert result.flag is True
    result_false = parse(LiteralBoolModel, {"flag": "off"})
    assert result_false.flag is False


def test_schema_literal_bool_includes_boolean_type() -> None:
    schema_payload: dict[str, JSONValue] = schema(LiteralBoolModel)
    properties = cast(dict[str, JSONValue], schema_payload["properties"])
    flag_schema = cast(dict[str, JSONValue], properties["flag"])
    assert flag_schema["enum"] == [True, False]
    assert flag_schema["type"] == "boolean"


def test_parse_membership_constraints() -> None:
    model = parse(MembershipModel, {"status": "active", "mode": "modern"})
    assert model.status == "active"
    assert model.mode == "MODERN"

    with pytest.raises(ValueError) as exc:
        parse(MembershipModel, {"status": "paused", "mode": "modern"})
    assert "status: must be one of" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(MembershipModel, {"status": "active", "mode": "legacy"})
    assert "mode: may not be one of" in str(exc2.value)


def test_parse_age_metadata_bounds() -> None:
    with pytest.raises(ValueError) as exc:
        parse(User, user_payload(age="-1"))
    assert "age: must be >= 0" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(User, user_payload(age="150"))
    assert "age: must be <= 130" in str(exc2.value)


def test_schema_reflects_types_constraints_and_aliases() -> None:
    schema_dict = schema(User, extra="forbid")
    assert schema_dict["title"] == "User"
    assert schema_dict["additionalProperties"] is False
    required = cast(list[str], schema_dict["required"])
    assert set(required) == {"id", "name", "email"}

    properties = as_dict(schema_dict["properties"])
    assert as_dict(properties["id"])["format"] == "uuid"
    assert as_dict(properties["name"])["minLength"] == 1
    email_pattern = cast(str, as_dict(properties["email"])["pattern"])
    assert email_pattern.startswith("^[^")
    assert as_dict(properties["created_at"])["format"] == "date-time"

    favorite_anyof = as_list(as_dict(properties["favorite"])["anyOf"])
    assert any(
        as_dict(item).get("enum") == ["red", "green", "blue"] for item in favorite_anyof
    )
    assert any(as_dict(item).get("type") == "null" for item in favorite_anyof)

    price_anyof = as_list(as_dict(properties["price"])["anyOf"])
    assert any(as_dict(item).get("type") == "number" for item in price_anyof)

    birthday_anyof = as_list(as_dict(properties["birthday"])["anyOf"])
    assert any(as_dict(item).get("format") == "date" for item in birthday_anyof)

    wakeup_anyof = as_list(as_dict(properties["wakeup"])["anyOf"])
    assert any(as_dict(item).get("format") == "time" for item in wakeup_anyof)

    avatar_anyof = as_list(as_dict(properties["avatar"])["anyOf"])
    assert any(as_dict(item).get("type") == "string" for item in avatar_anyof)

    assert as_dict(properties["tags"]) == {
        "type": "array",
        "items": {"type": "string"},
    }

    points = as_dict(properties["points"])
    assert points["prefixItems"] == [{"type": "integer"}, {"type": "integer"}]
    assert points["minItems"] == 2 and points["maxItems"] == 2

    attributes = as_dict(properties["attributes"])
    assert attributes["type"] == "object"
    union_values = as_list(as_dict(attributes["additionalProperties"])["anyOf"])
    assert any(as_dict(item).get("type") == "integer" for item in union_values)
    assert any(as_dict(item).get("type") == "string" for item in union_values)

    home_schema = as_dict(properties["home"])
    assert home_schema["type"] == "object"
    assert home_schema["title"] == "Address"
    assert set(cast(list[str], home_schema["required"])) == {"street", "city", "zip"}
    assert as_dict(as_dict(home_schema["properties"])["street"])["minLength"] == 1
    assert "pattern" in as_dict(as_dict(home_schema["properties"])["zip"])


def test_schema_collection_and_literal_models() -> None:
    from tests.serde._fixtures import CollectionModel

    collection_schema = schema(CollectionModel)
    props = as_dict(collection_schema["properties"])
    assert as_dict(props["unique_tags"])["uniqueItems"] is True
    assert as_dict(props["history"])["items"] == {"type": "integer"}

    literal_schema = schema(LiteralModel)
    literal_props = as_dict(literal_schema["properties"])
    assert as_dict(literal_props["mode"])["enum"] == ["auto", "manual"]
    assert as_dict(literal_props["flag"])["type"] == "boolean"
    assert {
        as_dict(item)["type"]
        for item in as_list(as_dict(literal_props["value"])["anyOf"])
    } == {
        "integer",
        "number",
    }

    numeric_schema = schema(NumericModel)
    num_props = as_dict(numeric_schema["properties"])
    assert as_dict(num_props["inclusive"])["minimum"] == 1
    assert as_dict(num_props["inclusive"])["maximum"] == 5
    assert as_dict(num_props["exclusive"])["exclusiveMinimum"] == 1
    assert as_dict(num_props["exclusive"])["exclusiveMaximum"] == 5

    length_schema = schema(LengthModel)
    token_schema = as_dict(as_dict(length_schema["properties"])["token"])
    assert token_schema["minLength"] == 2
    assert token_schema["maxLength"] == 4

    membership_schema = schema(MembershipModel)
    membership_props = as_dict(membership_schema["properties"])
    assert as_dict(membership_props["status"])["enum"] == ["active", "inactive"]
    assert as_dict(membership_props["mode"])["not"] == {"enum": ["legacy"]}

    optional_schema = schema(OptionalOnly)
    assert "required" not in optional_schema

    init_false_schema = schema(WithInitFalse)
    properties_dict = as_dict(init_false_schema["properties"])
    assert "computed" not in properties_dict

    @dataclass
    class LiteralNumbers:
        integer_only: Literal[1, 2]

    literal_numbers_schema = schema(LiteralNumbers)
    literal_numbers_props = as_dict(literal_numbers_schema["properties"])
    assert as_dict(literal_numbers_props["integer_only"])["type"] == "integer"

    @dataclass
    class LiteralFloats:
        ratio: Literal[0.5, 0.75]  # type: ignore[invalid-type-form]

    literal_float_schema = schema(LiteralFloats)
    float_props = as_dict(literal_float_schema["properties"])
    assert as_dict(float_props["ratio"])["type"] == "number"


def test_schema_invalid_extra_value_and_type_errors() -> None:
    with pytest.raises(ValueError):
        schema(User, extra="boom")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        schema(object)


def test_parse_length_constraints_enforced() -> None:
    with pytest.raises(ValueError) as exc:
        parse(LengthModel, {"token": "a"})
    assert "length must be >= 2" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(LengthModel, {"token": "abcde"})
    assert "length must be <= 4" in str(exc2.value)


def test_parse_numeric_model_exclusive_bounds() -> None:
    with pytest.raises(ValueError) as exc:
        parse(NumericModel, {"inclusive": "1", "exclusive": "1"})
    assert "exclusive: must be > 1" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(NumericModel, {"inclusive": "2", "exclusive": "6"})
    assert "exclusive: must be < 5" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        parse(NumericModel, {"inclusive": "0", "exclusive": "3"})
    assert "inclusive: must be >= 1" in str(exc3.value)

    with pytest.raises(ValueError) as exc4:
        parse(NumericModel, {"inclusive": "6", "exclusive": "4"})
    assert "inclusive: must be <= 5" in str(exc4.value)

    model = parse(NumericModel, {"inclusive": "5", "exclusive": "4"})
    assert model.inclusive == 5
    assert model.exclusive == 4


def test_string_normalization_for_membership_constraints() -> None:
    @dataclass
    class NormalizedMembership:
        token: Annotated[
            str,
            {
                "in": {"  KEEP  "},
                "strip": True,
                "lower": True,
            },
        ]

    normalized = parse(NormalizedMembership, {"token": " KEEP "})
    assert normalized.token == "keep"

    @dataclass
    class UpperForbidden:
        mode: Annotated[
            str,
            {
                "not_in": {" legacy "},
                "strip": True,
                "upper": True,
            },
        ]

    with pytest.raises(ValueError) as exc:
        parse(UpperForbidden, {"mode": " legacy "})
    assert "may not be one of" in str(exc.value)

    @dataclass
    class NumericMembership:
        choice: Annotated[int, {"in": {1, 2}}]

    numeric = parse(NumericMembership, {"choice": 1})
    assert numeric.choice == 1


def test_none_branch_and_literal_coercion() -> None:
    from weakincentives.serde._coercers import _coerce_to_type
    from weakincentives.serde._utils import _ParseConfig

    config = _ParseConfig(
        extra="ignore",
        coerce=True,
    )

    with pytest.raises(TypeError):
        _coerce_to_type("value", type(None), None, "field", config)

    with pytest.raises(TypeError):
        _coerce_to_type(None, int, None, "field", config)

    assert _coerce_to_type(None, type(None), None, "field", config) is None

    @dataclass
    class FlagModel:
        flag: bool

    assert parse(FlagModel, {"flag": "true"}).flag is True

    @dataclass
    class LiteralNumber:
        value: Literal[1, 2]

    with pytest.raises(ValueError) as exc:
        parse(LiteralNumber, {"value": "abc"})
    assert "value:" in str(exc.value)


def test_schema_additional_types() -> None:
    @dataclass
    class PrimitiveSchema:
        flag: bool
        count: int
        ratio: float
        text: str
        when: datetime
        day: date
        clock: time
        ident: UUID
        path: Path
        color: Color
        anything: complex

    schema_dict = schema(PrimitiveSchema)
    props = as_dict(schema_dict["properties"])
    assert as_dict(props["flag"]) == {"type": "boolean"}
    assert as_dict(props["count"]) == {"type": "integer"}
    assert as_dict(props["ratio"]) == {"type": "number"}
    assert as_dict(props["text"]) == {"type": "string"}
    assert as_dict(props["when"]) == {"type": "string", "format": "date-time"}
    assert as_dict(props["day"]) == {"type": "string", "format": "date"}
    assert as_dict(props["clock"]) == {"type": "string", "format": "time"}
    assert as_dict(props["ident"]) == {"type": "string", "format": "uuid"}
    assert as_dict(props["path"]) == {"type": "string"}
    assert as_dict(props["color"])["enum"] == ["red", "green", "blue"]
    assert props["anything"] == {}

    @dataclass
    class ObjectSchema:
        payload: object
        none_field: None = None

    object_schema = schema(ObjectSchema)
    object_props = as_dict(object_schema["properties"])
    assert object_props["payload"] == {}
    assert as_dict(object_props["none_field"]) == {"type": "null"}

    class BoolEnum(Enum):
        YES = True
        NO = False

    class IntEnum(Enum):
        ONE = 1
        TWO = 2

    class FloatEnum(Enum):
        HALF = 0.5
        FULL = 1.0

    @dataclass
    class EnumSchema:
        flag: BoolEnum
        count: IntEnum
        ratio: FloatEnum

    globals().update(
        {
            "BoolEnum": BoolEnum,
            "IntEnum": IntEnum,
            "FloatEnum": FloatEnum,
            "EnumSchema": EnumSchema,
        }
    )
    try:
        enum_schema = schema(EnumSchema)
    finally:
        globals().pop("EnumSchema", None)
        globals().pop("FloatEnum", None)
        globals().pop("IntEnum", None)
        globals().pop("BoolEnum", None)

    enum_props = as_dict(enum_schema["properties"])
    assert as_dict(enum_props["flag"])["type"] == "boolean"
    assert as_dict(enum_props["count"])["type"] == "integer"
    assert as_dict(enum_props["ratio"])["type"] == "number"


def test_schema_literal_mixed_types() -> None:
    @dataclass
    class MixedLiteral:
        value: Literal["text", 42]

    mixed_schema = schema(MixedLiteral)
    props = as_dict(mixed_schema["properties"])
    value_schema = as_dict(props["value"])
    assert "type" not in value_schema
    assert value_schema["enum"] == ["text", 42]


def test_schema_enum_mixed_types() -> None:
    class MixedEnum(Enum):
        TEXT = "text"
        NUMBER = 42

    @dataclass
    class MixedEnumModel:
        value: MixedEnum

    globals()["MixedEnum"] = MixedEnum
    globals()["MixedEnumModel"] = MixedEnumModel
    try:
        mixed_schema = schema(MixedEnumModel)
        props = as_dict(mixed_schema["properties"])
        value_schema = as_dict(props["value"])
        assert "type" not in value_schema
        assert set(value_schema["enum"]) == {"text", 42}
    finally:
        globals().pop("MixedEnumModel", None)
        globals().pop("MixedEnum", None)
