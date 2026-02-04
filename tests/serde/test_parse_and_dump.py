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

"""Tests for parse/dump and serde helpers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import pytest

from tests.serde._fixtures import (
    USER_UUID,
    Address,
    Score,
    Slotted,
    User,
    WithInitFalse,
    _ConcreteWrapper,
    _GenericWrapper,
    _InnerGeneric,
    _InnerPayload,
    _NestedGenericWrapper,
    _OuterGeneric,
    camel,
    user_payload,
)
from weakincentives.serde import clone, dump, parse, schema
from weakincentives.serde._utils import (
    _SLOTTED_EXTRAS,
    _get_or_create_extras_descriptor,
    _merge_annotated_meta,
    _ordered_values,
    _ParseConfig,
    _set_extras,
)
from weakincentives.serde.parse import (
    _bool_from_str,
    _build_lowered_key_map,
    _coerce_to_type,
)

parse_module = importlib.import_module("weakincentives.serde.parse")

pytestmark = pytest.mark.core


def test_module_exports_align_with_public_api() -> None:
    from weakincentives.serde.dump import clone as module_clone, dump as module_dump
    from weakincentives.serde.parse import parse as module_parse
    from weakincentives.serde.schema import schema as module_schema

    assert clone is module_clone
    assert dump is module_dump
    assert parse is module_parse
    assert schema is module_schema


def test_build_lowered_key_map_skips_non_string_keys() -> None:
    """Test branch 503->502: non-string keys are skipped in _build_lowered_key_map."""
    data: Mapping[Any, Any] = {
        "ValidKey": "value1",
        "AnotherKey": "value2",
        123: "non-string-key-value",
        ("tuple", "key"): "another-non-string",
    }

    result = _build_lowered_key_map(data)

    assert "validkey" in result
    assert "anotherkey" in result
    assert result["validkey"] == "ValidKey"
    assert result["anotherkey"] == "AnotherKey"
    assert len(result) == 2


def test_get_or_create_extras_descriptor_caches_result() -> None:
    """Test line 70: when descriptor exists in cache, return it directly."""

    @dataclass(slots=True, frozen=True)
    class CachedSlottedData:
        value: str

    _SLOTTED_EXTRAS.pop(CachedSlottedData, None)

    descriptor1 = _get_or_create_extras_descriptor(CachedSlottedData)
    assert CachedSlottedData in _SLOTTED_EXTRAS

    descriptor2 = _get_or_create_extras_descriptor(CachedSlottedData)
    assert descriptor1 is descriptor2


def test_set_extras_reuses_descriptor() -> None:
    """Test that _set_extras reuses cached descriptor for slotted classes."""

    @dataclass(slots=True, frozen=True)
    class SlottedData:
        value: str

    _SLOTTED_EXTRAS.pop(SlottedData, None)

    instance1 = SlottedData(value="first")
    _set_extras(instance1, {"key1": "value1"})

    assert SlottedData in _SLOTTED_EXTRAS
    descriptor1 = _SLOTTED_EXTRAS[SlottedData]

    instance2 = SlottedData(value="second")
    _set_extras(instance2, {"key2": "value2"})

    descriptor2 = _SLOTTED_EXTRAS[SlottedData]
    assert descriptor1 is descriptor2

    assert instance1.__extras__ == {"key1": "value1"}  # type: ignore[attr-defined]
    assert instance2.__extras__ == {"key2": "value2"}  # type: ignore[attr-defined]


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
    assert user.favorite.value == "green"
    assert user.home is not None
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

    slotted_no_extras = parse(Slotted, {"name": "Bob"})
    cloned_no_extras = clone(slotted_no_extras)
    assert cloned_no_extras.name == "Bob"

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
    assert payload["createdAt"] == "2024-01-01T10:00:00"
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


def test_parse_generic_alias_resolves_typevar() -> None:
    data = {"payload": {"message": "hello", "priority": 5}, "metadata": "test"}

    restored = parse(_GenericWrapper[_InnerPayload], data)

    assert isinstance(restored, _GenericWrapper)
    assert isinstance(restored.payload, _InnerPayload)
    assert restored.payload.message == "hello"
    assert restored.payload.priority == 5
    assert restored.metadata == "test"


def test_parse_generic_alias_round_trip() -> None:
    inner = _InnerPayload(message="hello", priority=5)
    wrapper = _GenericWrapper(payload=inner, metadata="test")

    data = dump(wrapper)
    restored = parse(_GenericWrapper[_InnerPayload], data)

    assert isinstance(restored, _GenericWrapper)
    assert isinstance(restored.payload, _InnerPayload)
    assert restored.payload.message == "hello"
    assert restored.payload.priority == 5
    assert restored.metadata == "test"


def test_parse_unspecialized_generic_raises_clear_error() -> None:
    data = {"payload": {"message": "hello", "priority": 1}, "metadata": "test"}

    with pytest.raises(TypeError) as exc:
        parse(_GenericWrapper, data)

    assert "cannot parse TypeVar field" in str(exc.value)
    assert "fully specialized generic type" in str(exc.value)


def test_parse_nested_generic_alias() -> None:
    data = {
        "child": {"payload": {"message": "deep", "priority": 99}, "metadata": "mid"}
    }

    restored = parse(_NestedGenericWrapper[_GenericWrapper[_InnerPayload]], data)

    assert isinstance(restored, _NestedGenericWrapper)
    assert isinstance(restored.child, _GenericWrapper)
    assert isinstance(restored.child.payload, _InnerPayload)
    assert restored.child.payload.message == "deep"
    assert restored.child.payload.priority == 99


def test_parse_concrete_nested_dataclass() -> None:
    data = {"nested": {"message": "test", "priority": 1}}

    restored = parse(_ConcreteWrapper, data)

    assert isinstance(restored, _ConcreteWrapper)
    assert isinstance(restored.nested, _InnerPayload)
    assert restored.nested.message == "test"


def test_parse_nested_typevar_resolution() -> None:
    data = {"inner": {"value": 42}, "label": "test"}

    restored = parse(_OuterGeneric[int], data)

    assert isinstance(restored, _OuterGeneric)
    assert isinstance(restored.inner, _InnerGeneric)
    assert restored.inner.value == 42
    assert restored.label == "test"


def test_parse_nested_typevar_with_dataclass_value() -> None:
    data = {
        "inner": {"value": {"message": "hello", "priority": 1}},
        "label": "test",
    }

    restored = parse(_OuterGeneric[_InnerPayload], data)

    assert isinstance(restored, _OuterGeneric)
    assert isinstance(restored.inner, _InnerGeneric)
    assert isinstance(restored.inner.value, _InnerPayload)
    assert restored.inner.value.message == "hello"


def test_get_field_types_with_generic_class() -> None:
    from weakincentives.serde.parse import _get_field_types

    @dataclass(slots=True, frozen=True)
    class GenericClass[T]:
        name: str
        payload: T

    result = _get_field_types(GenericClass)

    assert result["name"] is str
    assert result["payload"] in {object, GenericClass.__type_params__[0]}


def test_build_typevar_map_unresolved_typevar() -> None:
    from typing import TypeVar

    from weakincentives.serde.parse import _build_typevar_map

    UnrelatedTypeVar = TypeVar("UnrelatedTypeVar")

    @dataclass(slots=True, frozen=True)
    class SimpleGeneric[T]:
        value: T

    alias = SimpleGeneric[UnrelatedTypeVar]  # type: ignore[type-arg]

    parent_map: dict[object, type] = {}
    result = _build_typevar_map(alias, parent_typevar_map=parent_map)

    assert result == {}


def test_parse_deeply_nested_generic_with_alias() -> None:
    inner = _InnerPayload(message="deep", priority=99)
    middle = _GenericWrapper(payload=inner, metadata="mid")
    outer = _NestedGenericWrapper(child=middle)

    serialized = dump(outer)

    restored = parse(_NestedGenericWrapper[_GenericWrapper[_InnerPayload]], serialized)

    assert isinstance(restored, _NestedGenericWrapper)
    assert isinstance(restored.child, _GenericWrapper)
    assert isinstance(restored.child.payload, _InnerPayload)
    assert restored.child.payload.message == "deep"
    assert restored.child.payload.priority == 99
    assert restored.child.metadata == "mid"


def test_parse_nested_dataclass_with_extra_forbid() -> None:
    inner = _InnerPayload(message="test", priority=1)
    wrapper = _ConcreteWrapper(nested=inner)

    serialized = dump(wrapper)

    restored = parse(_ConcreteWrapper, serialized, extra="forbid")

    assert isinstance(restored, _ConcreteWrapper)
    assert isinstance(restored.nested, _InnerPayload)
    assert restored.nested.message == "test"


def test_internal_helpers_and_extras_descriptor() -> None:
    slotted = parse(Slotted, {"name": "Ada", "nickname": "Ace"}, extra="allow")
    assert getattr(Slotted, "__extras__", None) is None
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


def test_object_type_and_union_handling() -> None:
    @dataclass
    class ObjectModel:
        payload: object = field(metadata={"strip": True})

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


def test_set_extras_creates_new_descriptor_when_none_exists() -> None:
    @dataclass(slots=True)
    class NewSlotted:
        name: str

    _SLOTTED_EXTRAS.pop(NewSlotted, None)

    instance = parse(NewSlotted, {"name": "Test", "extra": "value"}, extra="allow")
    assert getattr(instance, "__extras__", None) == {"extra": "value"}
    assert NewSlotted in _SLOTTED_EXTRAS


def test_set_extras_reuses_existing_descriptor() -> None:
    """Test branch 75->79: descriptor already exists for the class."""

    @dataclass(slots=True)
    class ReusedSlotted:
        name: str

    _SLOTTED_EXTRAS.pop(ReusedSlotted, None)

    first = parse(ReusedSlotted, {"name": "First", "extra1": "value1"}, extra="allow")
    assert getattr(first, "__extras__", None) == {"extra1": "value1"}
    assert ReusedSlotted in _SLOTTED_EXTRAS

    second = parse(ReusedSlotted, {"name": "Second", "extra2": "value2"}, extra="allow")
    assert getattr(second, "__extras__", None) == {"extra2": "value2"}
    assert getattr(first, "__extras__", None) == {"extra1": "value1"}


def test_merge_annotated_meta_handles_non_mapping_args() -> None:
    from typing import Annotated

    annotated_type = Annotated[int, "string_metadata", {"key": "value"}, 123]
    base, meta = _merge_annotated_meta(annotated_type, None)
    assert base is int
    assert meta == {"key": "value"}


def test_resolve_generic_string_type_simple_name() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    assert _resolve_generic_string_type("str", localns, module_ns) is str
    assert _resolve_generic_string_type("int", localns, module_ns) is int
    assert _resolve_generic_string_type("list", localns, module_ns) is list


def test_resolve_generic_string_type_with_localns() -> None:
    from typing import TypeVar

    from weakincentives.serde.parse import _resolve_generic_string_type

    T = TypeVar("T")
    localns: dict[str, object] = {"T": T}
    module_ns: dict[str, object] = {}

    assert _resolve_generic_string_type("T", localns, module_ns) is T


def test_resolve_generic_string_type_with_module_ns() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    @dataclass(slots=True, frozen=True)
    class MyClass:
        value: str

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {"MyClass": MyClass}

    assert _resolve_generic_string_type("MyClass", localns, module_ns) is MyClass


def test_resolve_generic_string_type_subscript() -> None:
    from typing import TypeVar, get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    @dataclass(slots=True, frozen=True)
    class Container[T]:
        value: T

    T = TypeVar("T")
    localns: dict[str, object] = {"T": T}
    module_ns: dict[str, object] = {"Container": Container}

    result = _resolve_generic_string_type("Container[T]", localns, module_ns)

    assert get_origin(result) is Container
    args = get_args(result)
    assert len(args) == 1
    assert args[0] is T


def test_resolve_generic_string_type_multiple_args() -> None:
    from typing import TypeVar, get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    @dataclass(slots=True, frozen=True)
    class Pair[K, V]:
        key: K
        value: V

    K = TypeVar("K")
    V = TypeVar("V")
    localns: dict[str, object] = {"K": K, "V": V}
    module_ns: dict[str, object] = {"Pair": Pair}

    result = _resolve_generic_string_type("Pair[K, V]", localns, module_ns)

    assert get_origin(result) is Pair
    args = get_args(result)
    assert len(args) == 2
    assert args[0] is K
    assert args[1] is V


def test_resolve_generic_string_type_nested_generic() -> None:
    from typing import TypeVar, get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    @dataclass(slots=True, frozen=True)
    class Wrapper[T]:
        inner: T

    @dataclass(slots=True, frozen=True)
    class Inner[T]:
        value: T

    T = TypeVar("T")
    localns: dict[str, object] = {"T": T}
    module_ns: dict[str, object] = {"Wrapper": Wrapper, "Inner": Inner}

    result = _resolve_generic_string_type("Wrapper[Inner[T]]", localns, module_ns)

    assert get_origin(result) is Wrapper
    outer_args = get_args(result)
    assert len(outer_args) == 1
    inner = outer_args[0]
    assert get_origin(inner) is Inner
    inner_args = get_args(inner)
    assert len(inner_args) == 1
    assert inner_args[0] is T


def test_resolve_generic_string_type_builtin_generic() -> None:
    from typing import Union, get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type("list[str]", localns, module_ns)
    assert get_origin(result) is list
    assert get_args(result) == (str,)

    result = _resolve_generic_string_type("dict[str, int]", localns, module_ns)
    assert get_origin(result) is dict
    assert get_args(result) == (str, int)

    result = _resolve_generic_string_type("Optional[str]", localns, module_ns)
    assert get_origin(result) is Union
    assert get_args(result) == (str, type(None))

    result = _resolve_generic_string_type("tuple[int, ...]", localns, module_ns)
    assert get_origin(result) is tuple
    assert get_args(result) == (int, ...)


def test_resolve_generic_string_type_union() -> None:
    from types import UnionType
    from typing import get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type("str | int", localns, module_ns)
    assert get_origin(result) is UnionType
    assert set(get_args(result)) == {str, int}

    result = _resolve_generic_string_type("str | None", localns, module_ns)
    assert get_origin(result) is UnionType
    assert set(get_args(result)) == {str, type(None)}


def test_resolve_generic_string_type_syntax_error() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    assert _resolve_generic_string_type("[invalid", localns, module_ns) is object


def test_resolve_generic_string_type_unresolvable() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    assert _resolve_generic_string_type("UnknownType", localns, module_ns) is object


def test_resolve_simple_type_returns_none() -> None:
    from weakincentives.serde.parse import _resolve_simple_type

    assert _resolve_simple_type("str") is str
    assert _resolve_simple_type("int") is int

    assert _resolve_simple_type("UnknownType") is None
    assert _resolve_simple_type("T") is None


def test_resolve_subscript_with_object_base() -> None:
    from weakincentives.serde.parse import _resolve_subscript

    result = _resolve_subscript(object, (str,))
    assert result is object


def test_resolve_subscript_with_non_subscriptable_type() -> None:
    from weakincentives.serde.parse import _resolve_subscript

    result = _resolve_subscript(int, (str,))
    assert result is int


def test_resolve_generic_string_type_unknown_node() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type("typing.Optional", localns, module_ns)
    assert result is object


def test_resolve_generic_string_type_constants_outside_literal() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type("'foo'", localns, module_ns)
    assert result is object

    result = _resolve_generic_string_type("42", localns, module_ns)
    assert result is object

    result = _resolve_generic_string_type("True", localns, module_ns)
    assert result is object


def test_resolve_generic_string_type_literal() -> None:
    from typing import Literal, get_args, get_origin

    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type('Literal["foo"]', localns, module_ns)
    assert get_origin(result) is Literal
    assert get_args(result) == ("foo",)

    result = _resolve_generic_string_type("Literal[1, 2, 3]", localns, module_ns)
    assert get_origin(result) is Literal
    assert get_args(result) == (1, 2, 3)

    result = _resolve_generic_string_type("Literal[True, False]", localns, module_ns)
    assert get_origin(result) is Literal
    assert get_args(result) == (True, False)

    result = _resolve_generic_string_type("Literal[-1]", localns, module_ns)
    assert get_origin(result) is Literal
    assert get_args(result) == (-1,)

    result = _resolve_generic_string_type("Literal[-1, 0, +1]", localns, module_ns)
    assert get_origin(result) is Literal
    assert get_args(result) == (-1, 0, 1)


def test_resolve_generic_string_type_unresolvable_subscript() -> None:
    from weakincentives.serde.parse import _resolve_generic_string_type

    localns: dict[str, object] = {}
    module_ns: dict[str, object] = {}

    result = _resolve_generic_string_type("Unknown[str]", localns, module_ns)
    assert result is object
