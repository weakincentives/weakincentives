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

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import UUID

import pytest

from weakincentives.serde import clone, dump, parse, schema
from weakincentives.serde._utils import (
    _SLOTTED_EXTRAS,
    _ParseConfig,
    _merge_annotated_meta,
    _ordered_values,
)
from weakincentives.serde.parse import _bool_from_str, _coerce_to_type
from weakincentives.types import JSONValue

parse_module = importlib.import_module("weakincentives.serde.parse")


def test_module_exports_align_with_public_api() -> None:
    from weakincentives.serde.dump import clone as module_clone, dump as module_dump
    from weakincentives.serde.parse import parse as module_parse
    from weakincentives.serde.schema import schema as module_schema

    assert clone is module_clone
    assert dump is module_dump
    assert parse is module_parse
    assert schema is module_schema


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


USER_UUID = UUID("a9f95576-7a80-4c79-9b90-6afee4c3f9d9")
TIMESTAMP = "2024-01-01T10:00:00"


def as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def user_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "USER": str(USER_UUID),
        "NAME": "  Ada Lovelace  ",
        "EMAIL": "Ada@Example.COM",
        "AGE": "39",
        "favorite": "GREEN",
        "createdAt": TIMESTAMP,
        "birthday": "1985-12-10",
        "wakeup": "07:30:00",
        "price": "99.95",
        "avatar": "/tmp/avatar.png",
        "HOME": {"street": "1 Example Way", "city": "London", "zip": "12345-6789"},
        "tags": "prolific",
        "points": ["1", "2"],
        "attributes": {"level": "10", 99: "bonus"},
    }
    payload.update(overrides)
    return payload


@dataclass
class Address:
    street: Annotated[str, {"min_length": 1}]
    city: str
    zip: Annotated[str, {"regex": r"^\d{5}(-\d{4})?$"}]


@dataclass
class User:
    __computed__ = ("email_domain",)

    user_id: UUID = field(metadata={"alias": "id"})
    name: Annotated[str, {"min_length": 1, "strip": True}]
    email: Annotated[str, {"regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$", "lower": True}]
    age: int | None = field(default=None, metadata={"ge": 0, "le": 130})
    favorite: Color | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    birthday: date | None = None
    wakeup: time | None = None
    price: Decimal | None = None
    avatar: Path | None = None
    home: Address | None = None
    tags: list[str] = field(default_factory=list)
    points: tuple[int, int] = (0, 0)
    attributes: dict[str, int | str] = field(default_factory=dict)

    def __validate__(self) -> None:
        if self.age is not None and self.age < 13:
            raise ValueError("age must be >= 13")

    @property
    def email_domain(self) -> str:
        return self.email.split("@", 1)[1]


def ensure_code_prefix(value: str) -> str:
    if not value.startswith("ID-"):
        raise ValueError("invalid code")
    return value


def append_done(value: str) -> str:
    return f"{value}-DONE"


def ensure_even(value: int) -> int:
    if value % 2:
        raise ValueError("must be even")
    return value


def ensure_positive(value: int) -> int:
    if value <= 0:
        raise TypeError("must be positive")
    return value


def half(value: int) -> int:
    return value // 2


def ensure_token_prefix(value: str) -> str:
    if not value.startswith("ok"):
        raise ValueError("bad prefix")
    return value


def transform_bang(value: str) -> str:
    return f"{value}!"


def explode_convert(_: int) -> int:
    raise RuntimeError("boom")


def explode_validator(_: str) -> str:
    raise RuntimeError("explode")


def value_error_convert(_: object) -> object:
    raise ValueError("boom")


@dataclass
class HookModel:
    code: Annotated[
        str,
        {
            "strip": True,
            "upper": True,
            "validators": (ensure_code_prefix,),
            "convert": append_done,
        },
    ]
    amount: Annotated[
        int, {"validators": [ensure_even, ensure_positive], "convert": half}
    ]


@dataclass
class SingleValidatorModel:
    token: Annotated[str, {"validate": ensure_token_prefix}]


@dataclass
class TransformModel:
    token: Annotated[str, {"strip": True}] = field(
        metadata={"transform": transform_bang}
    )


@dataclass
class BadConvert:
    value: Annotated[int, {"convert": explode_convert}]


@dataclass
class BadValidator:
    value: Annotated[str, {"validators": (explode_validator,)}]


@dataclass
class MembershipModel:
    status: Annotated[str, {"in": {"active", "inactive"}}]
    mode: Annotated[str, {"not_in": {"legacy"}, "upper": True}]


@dataclass
class AnnotationPrecedence:
    code: Annotated[str, {"strip": True, "upper": True}] = field(
        metadata={"upper": False}
    )


@dataclass
class CollectionModel:
    unique_tags: set[str]
    scores: tuple[int, int, int]
    history: tuple[int, ...]
    mapping: dict[int, str]


@dataclass
class LiteralModel:
    mode: Literal["auto", "manual"]
    flag: bool
    value: int | float


# Use literal bools to exercise coercion and schema branches.
@dataclass
class LiteralBoolModel:
    flag: Literal[True, False]  # noqa: RUF038 Keep literal bools for schema coverage


@dataclass
class MappingModel:
    mapping: dict[int, str]


@dataclass
class NumericModel:
    inclusive: Annotated[int, {"minimum": 1, "maximum": 5}]
    exclusive: Annotated[int, {"gt": 1, "lt": 5}]


@dataclass
class LengthModel:
    token: Annotated[str, {"minLength": 2, "maxLength": 4}]


@dataclass
class WithInitFalse:
    name: str
    computed: str = field(init=False, default="constant")


@dataclass
class OptionalOnly:
    maybe: int | None = None


@dataclass
class Container:
    values: dict[str, int | None]
    items: list[str | None]
    nested: Address | None = None


@dataclass
class SetHolder:
    values: set[int]


@dataclass
class PostValidated:
    value: int

    def __post_validate__(self) -> None:
        if self.value < 0:
            raise ValueError("value must be >= 0")


@dataclass
class Score:
    value: int | float


@dataclass(slots=True)
class Slotted:
    name: str


def test_parse_handles_coercion_aliases_and_normalization() -> None:
    payload = user_payload()
    user = parse(
        User,
        payload,
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
    )

    assert user.user_id == USER_UUID
    assert user.name == "Ada Lovelace"
    assert user.email == "ada@example.com"
    assert user.age == 39
    assert user.favorite == Color.GREEN
    assert user.created_at == datetime.fromisoformat(TIMESTAMP)
    assert user.birthday == date.fromisoformat("1985-12-10")
    assert user.wakeup == time.fromisoformat("07:30:00")
    assert user.price == Decimal("99.95")
    assert user.avatar == Path("/tmp/avatar.png")
    assert isinstance(user.home, Address)
    assert user.home.street == "1 Example Way"
    assert user.home.city == "London"
    assert user.home.zip == "12345-6789"
    assert user.tags == ["prolific"]
    assert user.points == (1, 2)
    assert user.attributes == {"level": 10, "99": "bonus"}
    assert user.email_domain == "example.com"


def test_parse_requires_mapping_and_dataclass() -> None:
    with pytest.raises(TypeError):
        parse(int, {})
    with pytest.raises(TypeError):
        parse(User, cast(Mapping[str, object], []))


def test_parse_strict_type_errors_when_coercion_disabled() -> None:
    payload = {
        "user_id": USER_UUID,
        "name": "Ada",
        "email": "ada@example.com",
        "age": "39",
    }
    with pytest.raises(TypeError) as exc:
        parse(User, payload, coerce=False)
    assert "age: expected int" in str(exc.value)


def test_parse_optional_blank_strings_become_none() -> None:
    payload = user_payload(
        birthday=" ",
        wakeup="",
        price="",
        favorite="",
        avatar=" ",
        HOME=None,
    )
    user = parse(
        User,
        payload,
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
    )
    assert user.birthday is None
    assert user.wakeup is None
    assert user.price is None
    assert user.favorite is None
    assert user.avatar is None
    assert user.home is None


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


def test_parse_nested_dataclass_error_paths() -> None:
    bad_zip = user_payload(HOME={"street": "Main", "city": "Town", "zip": "bad"})
    with pytest.raises(ValueError) as exc:
        parse(
            User,
            bad_zip,
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert "home.zip: does not match pattern" in str(exc.value)

    missing_field = user_payload(HOME={"city": "Town", "zip": "12345"})
    with pytest.raises(ValueError) as exc2:
        parse(
            User,
            missing_field,
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert str(exc2.value) == "home: Missing required field: 'street'"


@pytest.mark.parametrize(
    "payload",
    [
        {"value": "not-a-number"},
        {"value": ["1"]},
    ],
)
def test_parse_union_error_reports_last_branch(payload: dict[str, object]) -> None:
    with pytest.raises(TypeError) as exc:
        parse(Score, payload)
    assert "value" in str(exc.value)


def test_parse_metadata_constraints_and_hooks() -> None:
    model = parse(HookModel, {"code": "  id-007  ", "amount": "8"})
    assert model.code == "ID-007-DONE"
    assert model.amount == 4


def test_parse_metadata_validator_errors() -> None:
    with pytest.raises(ValueError) as exc:
        parse(HookModel, {"code": "ID-100", "amount": "3"})
    assert "amount: must be even" in str(exc.value)

    with pytest.raises(TypeError) as exc2:
        parse(HookModel, {"code": "ID-100", "amount": "-4"})
    assert "amount: must be positive" in str(exc2.value)


@pytest.mark.parametrize(
    "model_cls,payload,expected_message",
    [
        (BadConvert, {"value": "3"}, "converter raised"),
        (BadValidator, {"value": "x"}, "validator raised"),
        (SingleValidatorModel, {"token": "oops"}, "token: bad prefix"),
    ],
)
def test_parse_converter_and_validator_exception_wrapping(
    model_cls: type[Any], payload: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(Exception) as exc:
        parse(model_cls, payload)
    assert expected_message in str(exc.value)


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


def test_parse_annotation_precedence_and_transform() -> None:
    model = parse(AnnotationPrecedence, {"code": " abc "})
    assert model.code == "ABC"

    transformed = parse(TransformModel, {"token": " data "})
    assert transformed.token == "data!"

    single = parse(SingleValidatorModel, {"token": "okay"})
    assert single.token == "okay"


def test_parse_invalid_extra_policy_value() -> None:
    with pytest.raises(ValueError):
        parse(User, {}, extra="boom")  # type: ignore[arg-type]


def test_parse_case_insensitive_lookup() -> None:
    @dataclass
    class CaseModel:
        token: str

    parsed_model = parse(CaseModel, {"TOKEN": "value"}, case_insensitive=True)
    assert parsed_model.token == "value"


def test_parse_skips_non_init_fields() -> None:
    instance = parse(WithInitFalse, {"name": "Ada"})
    assert instance.computed == "constant"


def test_parse_extra_policies() -> None:
    payload = user_payload(nickname="Ada")
    allowed = parse(
        User,
        payload,
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
        extra="allow",
    )
    assert allowed.nickname == "Ada"

    ignored = parse(
        User,
        user_payload(nickname="Ada"),
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
        extra="ignore",
    )
    assert not hasattr(ignored, "nickname")

    with pytest.raises(ValueError) as exc:
        parse(
            User,
            user_payload(nickname="Ada"),
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
            extra="forbid",
        )
    assert str(exc.value) == "Extra keys not permitted: ['nickname']"

    slotted = parse(Slotted, {"name": "Ada", "nickname": "Ace"}, extra="allow")
    assert getattr(slotted, "__extras__", None) == {"nickname": "Ace"}


def test_parse_model_validator_runs() -> None:
    with pytest.raises(ValueError) as exc:
        parse(
            User,
            user_payload(AGE="12"),
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert str(exc.value) == "age must be >= 13"


def test_parse_age_metadata_bounds() -> None:
    with pytest.raises(ValueError) as exc:
        parse(
            User,
            user_payload(AGE="-1"),
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert "age: must be >= 0" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(
            User,
            user_payload(AGE="150"),
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert "age: must be <= 130" in str(exc2.value)


def test_post_validate_hook_runs() -> None:
    instance = parse(PostValidated, {"value": 1})
    assert instance.value == 1
    with pytest.raises(ValueError):
        parse(PostValidated, {"value": -1})
    with pytest.raises(ValueError):
        clone(instance, value=-5)


def test_parse_accepts_nested_dataclass_instance() -> None:
    address = Address(street="1 Road", city="Town", zip="12345")
    payload = {
        "user_id": USER_UUID,
        "name": "Ada",
        "email": "ada@example.com",
        "home": address,
    }
    user = parse(User, payload)
    assert user.home is address


def test_parse_missing_required_field_message() -> None:
    data = {"user_id": USER_UUID, "email": "ada@example.com"}
    with pytest.raises(ValueError) as exc:
        parse(User, data)
    assert str(exc.value) == "Missing required field: 'name'"


def test_parse_rejects_none_for_non_optional() -> None:
    data = {"user_id": USER_UUID, "name": None, "email": "ada@example.com"}
    with pytest.raises(TypeError) as exc:
        parse(User, data)
    assert "name: value cannot be None" in str(exc.value)


def test_parse_dict_key_error_message() -> None:
    with pytest.raises(TypeError) as exc:
        parse(MappingModel, {"mapping": {"bad": "value"}})
    assert "mapping keys" in str(exc.value)


def test_clone_preserves_extras_and_revalidates() -> None:
    payload = user_payload(nickname="Ada")
    user = parse(
        User,
        payload,
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
        extra="allow",
    )
    updated = clone(user, age=40)
    assert updated.age == 40
    assert updated.nickname == "Ada"

    slotted = parse(Slotted, {"name": "Ada", "nickname": "Ace"}, extra="allow")
    cloned_slotted = clone(slotted)
    assert getattr(cloned_slotted, "__extras__", None) == {"nickname": "Ace"}

    with pytest.raises(ValueError):
        clone(user, age=10)

    with pytest.raises(TypeError):
        clone(object())


def test_dump_serializes_with_aliases_and_computed() -> None:
    user = parse(
        User,
        user_payload(),
        aliases={"user_id": "USER"},
        alias_generator=camel,
        case_insensitive=True,
    )
    payload = dump(
        user, by_alias=True, computed=True, exclude_none=True, alias_generator=camel
    )
    assert payload["id"] == str(USER_UUID)
    assert payload["name"] == "Ada Lovelace"
    assert payload["email"] == "ada@example.com"
    assert payload["age"] == 39
    assert payload["favorite"] == "green"
    assert payload["createdAt"] == TIMESTAMP
    assert payload["birthday"] == "1985-12-10"
    assert payload["wakeup"] == "07:30:00"
    assert payload["price"] == "99.95"
    assert payload["avatar"] == "/tmp/avatar.png"
    assert payload["home"] == {
        "street": "1 Example Way",
        "city": "London",
        "zip": "12345-6789",
    }
    assert payload["tags"] == ["prolific"]
    assert payload["points"] == [1, 2]
    assert payload["attributes"] == {"level": 10, "99": "bonus"}
    assert payload["emailDomain"] == "example.com"

    plain = dump(user, by_alias=False, exclude_none=True)
    assert "user_id" in plain
    assert "created_at" in plain

    with pytest.raises(TypeError):
        dump(object())


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
    values = cast(list[str], payload["values"])
    assert set(values) == {"a", "b"}
    assert BadReprStr.repr_calls > 0


def test_dump_computed_none_excluded() -> None:
    @dataclass
    class ComputedNone:
        __computed__ = ("maybe",)

        value: int

        @property
        def maybe(self) -> None:
            return None

    payload = dump(
        ComputedNone(1), computed=True, exclude_none=True, alias_generator=camel
    )
    assert "maybe" not in payload


def test_schema_reflects_types_constraints_and_aliases() -> None:
    schema_dict = schema(User, alias_generator=camel, extra="forbid")
    assert schema_dict["title"] == "User"
    assert schema_dict["additionalProperties"] is False
    required = cast(list[str], schema_dict["required"])
    assert set(required) == {"id", "name", "email"}

    properties = as_dict(schema_dict["properties"])
    assert as_dict(properties["id"])["format"] == "uuid"
    assert as_dict(properties["name"])["minLength"] == 1
    email_pattern = cast(str, as_dict(properties["email"])["pattern"])
    assert email_pattern.startswith("^[^")
    assert as_dict(properties["createdAt"])["format"] == "date-time"

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
    collection_schema = schema(CollectionModel, alias_generator=camel)
    props = as_dict(collection_schema["properties"])
    assert as_dict(props["uniqueTags"])["uniqueItems"] is True
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

    allow_schema = schema(User, extra="allow")
    assert cast(bool, allow_schema["additionalProperties"]) is True

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


def test_internal_helpers_and_extras_descriptor() -> None:
    slotted = parse(Slotted, {"name": "Ada", "nickname": "Ace"}, extra="allow")
    assert getattr(Slotted, "__extras__", None) is None  # descriptor access on class
    assert getattr(slotted, "__extras__", None) == {"nickname": "Ace"}

    descriptor = _SLOTTED_EXTRAS[Slotted]
    descriptor.__set__(slotted, None)
    assert getattr(slotted, "__extras__", None) is None

    unordered = {"a", 1}
    assert _ordered_values(unordered) == sorted(unordered, key=repr)
    assert _ordered_values(["x", "y"]) == ["x", "y"]


def test_merge_annotated_meta_and_bool_parsing() -> None:
    class Placeholder:
        __metadata__ = ()

    base, meta = _merge_annotated_meta(Placeholder, {"x": 1})
    assert base is Placeholder
    assert meta == {"x": 1}

    assert _bool_from_str(" YES ") is True
    assert _bool_from_str("off") is False


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


def test_compiled_regex_and_converter_errors() -> None:
    @dataclass
    class CodeModel:
        code: Annotated[str, {"pattern": re.compile(r"^[A-Z]{3}$")}]

    assert parse(CodeModel, {"code": "ABC"}).code == "ABC"
    with pytest.raises(ValueError):
        parse(CodeModel, {"code": "abc"})

    @dataclass
    class ConvertModel:
        value: Annotated[int, {"convert": value_error_convert}]

    with pytest.raises(ValueError) as exc:
        parse(ConvertModel, {"value": "1"})
    assert "value: boom" in str(exc.value)


def test_object_type_and_union_handling() -> None:
    @dataclass
    class ObjectModel:
        payload: Annotated[object, {"strip": True}]

    parsed = parse(ObjectModel, {"payload": "  data  "})
    assert parsed.payload == "data"

    @dataclass
    class OptionalText:
        token: str | None

    optional = parse(OptionalText, {"token": "   "})
    assert optional.token is None

    @dataclass
    class NumberUnion:
        value: int | float

    with pytest.raises((TypeError, ValueError)) as exc:
        parse(NumberUnion, {"value": "abc"})
    assert "value:" in str(exc.value)

    @dataclass
    class DateUnion:
        value: int | datetime

    with pytest.raises(TypeError) as union_exc:
        parse(DateUnion, {"value": "not-a-date"})
    assert "value: unable to coerce" in str(union_exc.value)

    class CustomWrapper:
        def __init__(self, value: object) -> None:
            self.value = value

        def double(self) -> object:
            return self.value

    @dataclass
    class CustomModel:
        field: CustomWrapper

    globals()["CustomWrapper"] = CustomWrapper
    globals()["CustomModel"] = CustomModel
    try:
        wrapped = parse(CustomModel, {"field": 123})
        assert isinstance(wrapped.field, CustomWrapper)
        assert wrapped.field.value == 123
    finally:
        globals().pop("CustomModel", None)
        globals().pop("CustomWrapper", None)

    class FailOne:
        def __init__(self, value: object) -> None:
            raise ValueError("boom1")

    class FailTwo:
        def __init__(self, value: object) -> None:
            raise ValueError("boom2")

    config = _ParseConfig(
        extra="ignore",
        coerce=True,
        case_insensitive=False,
        alias_generator=None,
        aliases=None,
    )

    with pytest.raises(ValueError) as fail_exc:
        _coerce_to_type("data", FailOne | FailTwo, None, "field", config)
    assert str(fail_exc.value) == "field: boom2"

    class TypeFail:
        def __init__(self, value: object) -> None:
            raise TypeError("type boom")

    with pytest.raises(TypeError) as type_exc:
        _coerce_to_type("data", FailTwo | TypeFail, None, "field", config)
    assert str(type_exc.value) == "field: type boom"


def test_union_without_matching_type_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union_type = int | None
    config = _ParseConfig(
        extra="ignore",
        coerce=True,
        case_insensitive=False,
        alias_generator=None,
        aliases=None,
    )

    original_get_args = parse_module.get_args

    def fake_get_args(typ: object) -> tuple[object, ...]:
        if typ is union_type:
            return (type(None),)
        return original_get_args(typ)

    monkeypatch.setattr(parse_module, "get_args", fake_get_args)

    with pytest.raises(TypeError) as exc:
        _coerce_to_type("value", union_type, None, "field", config)
    assert str(exc.value) == "field: no matching type in Union"


def test_none_branch_and_literal_coercion() -> None:
    config = _ParseConfig(
        extra="ignore",
        coerce=True,
        case_insensitive=False,
        alias_generator=None,
        aliases=None,
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


def test_dataclass_passthrough_and_error_wrapping() -> None:
    @dataclass
    class Holder:
        address: Address

    addr = Address("1 Way", "Town", "12345")
    parsed = parse(Holder, {"address": addr})
    assert parsed.address is addr

    with pytest.raises(TypeError) as exc:
        parse(Holder, {"address": 1})
    assert "address" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse(Holder, {"address": {"street": "", "city": "X", "zip": "bad"}})
    assert "address.street" in str(exc2.value)

    @dataclass
    class Boom:
        value: int

        def __post_init__(self) -> None:
            raise ValueError("boom")

    @dataclass
    class BoomHolder:
        boom: Boom

    globals()["Boom"] = Boom
    globals()["BoomHolder"] = BoomHolder
    try:
        with pytest.raises(ValueError) as exc3:
            parse(BoomHolder, {"boom": {"value": 1}})
        assert str(exc3.value).startswith("boom: boom")
    finally:
        globals().pop("BoomHolder", None)
        globals().pop("Boom", None)


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
        parse(
            MappingEnum,
            {"mapping": {1: "a"}, "color": "GREEN"},
            coerce=False,
        )


def test_bool_coercion_and_errors() -> None:
    @dataclass
    class BoolModel:
        flag: bool

    assert parse(BoolModel, {"flag": True}).flag is True
    assert parse(BoolModel, {"flag": "true"}).flag is True
    assert parse(BoolModel, {"flag": 1}).flag is True

    with pytest.raises(TypeError):
        parse(BoolModel, {"flag": "maybe"})

    with pytest.raises(TypeError):
        parse(BoolModel, {"flag": "true"}, coerce=False)


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
