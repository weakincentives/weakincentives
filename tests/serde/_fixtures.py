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

"""Shared serde test fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID


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
    flag: Literal[True, False]


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


# =============================================================================
# Generic Alias Support Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class _InnerPayload:
    message: str
    priority: int = 1


@dataclass(slots=True, frozen=True)
class _GenericWrapper[T]:
    payload: T
    metadata: str = "default"


@dataclass(slots=True, frozen=True)
class _NestedGenericWrapper[T]:
    """Wrapper for testing deeply nested generics."""

    child: T


@dataclass(slots=True, frozen=True)
class _ConcreteWrapper:
    """Wrapper with concrete (non-TypeVar) nested dataclass field."""

    nested: _InnerPayload


@dataclass(slots=True, frozen=True)
class _InnerGeneric[T]:
    """Inner generic that uses parent's TypeVar."""

    value: T


@dataclass(slots=True, frozen=True)
class _OuterGeneric[T]:
    """Outer generic with nested Inner[T] referencing same TypeVar."""

    inner: _InnerGeneric[T]
    label: str = "default"


@dataclass
class CompiledRegexCodeModel:
    code: Annotated[str, {"pattern": re.compile(r"^[A-Z]{3}$")}]


@dataclass
class ValueErrorConvertModel:
    value: Annotated[int, {"convert": value_error_convert}]
