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
from typing import cast

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
    user_payload,
)
from weakincentives.serde import clone, dump, parse, schema
from weakincentives.serde._coercers import (
    _bool_from_str,
    _coerce_to_type,
)
from weakincentives.serde._utils import (
    _merge_annotated_meta,
    _ordered_values,
    _ParseConfig,
)

coercers_module = importlib.import_module("weakincentives.serde._coercers")

pytestmark = pytest.mark.core


def test_module_exports_align_with_public_api() -> None:
    from weakincentives.serde.dump import clone as module_clone, dump as module_dump
    from weakincentives.serde.parse import parse as module_parse
    from weakincentives.serde.schema import schema as module_schema

    assert clone is module_clone
    assert dump is module_dump
    assert parse is module_parse
    assert schema is module_schema


def test_parse_handles_coercion_and_normalization() -> None:
    payload = user_payload()
    user = parse(User, payload)

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
        home=None,
    )
    user = parse(User, payload)
    assert user.birthday is None
    assert user.wakeup is None
    assert user.price is None
    assert user.favorite is None
    assert user.avatar is None
    assert user.home is None


def test_parse_nested_dataclass_error_paths() -> None:
    bad_zip = user_payload(home={"street": "Main", "city": "Town", "zip": "bad"})
    with pytest.raises(ValueError) as exc:
        parse(User, bad_zip)
    assert "home.zip: does not match pattern" in str(exc.value)

    missing_field = user_payload(home={"city": "Town", "zip": "12345"})
    with pytest.raises(ValueError) as exc2:
        parse(User, missing_field)
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


def test_parse_skips_non_init_fields() -> None:
    instance = parse(WithInitFalse, {"name": "Ada"})
    assert instance.computed == "constant"


def test_parse_extra_policies() -> None:
    payload = user_payload(nickname="Ada")
    ignored = parse(User, payload, extra="ignore")
    assert not hasattr(ignored, "nickname")

    with pytest.raises(ValueError) as exc:
        parse(User, user_payload(nickname="Ada"), extra="forbid")
    assert str(exc.value) == "Extra keys not permitted: ['nickname']"


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


def test_clone_and_revalidates() -> None:
    payload = user_payload()
    user = parse(User, payload)
    updated = clone(user, age=40)
    assert updated.age == 40

    slotted = parse(Slotted, {"name": "Ada"})
    cloned_slotted = clone(slotted)
    assert cloned_slotted.name == "Ada"

    with pytest.raises(ValueError):
        clone(user, age=10)

    with pytest.raises(TypeError):
        clone(object())


def test_dump_serializes_with_aliases_and_computed() -> None:
    user = parse(User, user_payload())
    payload = dump(user, by_alias=True, computed=True, exclude_none=True)
    assert payload["id"] == str(USER_UUID)
    assert payload["name"] == "Ada Lovelace"
    assert payload["email"] == "ada@example.com"
    assert payload["age"] == 39
    assert payload["favorite"] == "green"
    assert payload["created_at"] == "2024-01-01T10:00:00"
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
    assert payload["email_domain"] == "example.com"

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

    payload = dump(ComputedNone(1), computed=True, exclude_none=True)
    assert "maybe" not in payload


def test_dump_computed_respects_by_alias() -> None:
    """Computed fields must honour the caller's by_alias setting."""

    @dataclass
    class Inner:
        value: int = field(metadata={"alias": "v"})

    @dataclass
    class Outer:
        __computed__ = ("nested",)

        x: int

        @property
        def nested(self) -> Inner:
            return Inner(value=self.x)

    obj = Outer(x=42)

    aliased = dump(obj, by_alias=True, computed=True)
    assert aliased["nested"] == {"v": 42}

    plain = dump(obj, by_alias=False, computed=True)
    assert plain["nested"] == {"value": 42}


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
    from weakincentives.serde._generics import _get_field_types

    @dataclass(slots=True, frozen=True)
    class GenericClass[T]:
        name: str
        payload: T

    result = _get_field_types(GenericClass)

    assert result["name"] is str
    assert result["payload"] in {object, GenericClass.__type_params__[0]}


def test_get_field_types_with_type_checking_imports() -> None:
    """Resolves forward references from TYPE_CHECKING imports."""
    # AgentLoopConfig has BundleConfig under TYPE_CHECKING
    from weakincentives.runtime.agent_loop_types import AgentLoopConfig
    from weakincentives.serde._generics import _get_field_types

    result = _get_field_types(AgentLoopConfig)
    assert "debug_bundle" in result


def test_resolve_type_checking_imports_relative() -> None:
    """Resolves a relative TYPE_CHECKING import."""
    from weakincentives.runtime.agent_loop_types import AgentLoopConfig
    from weakincentives.serde._generics import _resolve_type_checking_imports

    resolved = _resolve_type_checking_imports(AgentLoopConfig, "BundleConfig")
    assert resolved is not None
    assert resolved.__name__ == "BundleConfig"  # type: ignore[union-attr]


def test_resolve_type_checking_imports_not_found() -> None:
    """Returns None for a name that doesn't exist in TYPE_CHECKING."""
    from weakincentives.runtime.agent_loop_types import AgentLoopConfig
    from weakincentives.serde._generics import _resolve_type_checking_imports

    resolved = _resolve_type_checking_imports(AgentLoopConfig, "NoSuchType")
    assert resolved is None


def test_is_type_checking_guard_attribute() -> None:
    """Handles typing.TYPE_CHECKING as an Attribute node."""
    import ast

    from weakincentives.serde._generics import _is_type_checking_guard

    # typing.TYPE_CHECKING
    node = ast.parse("typing.TYPE_CHECKING").body[0]
    assert isinstance(node, ast.Expr)
    assert _is_type_checking_guard(node.value) is True

    # something_else.TYPE_CHECKING
    node2 = ast.parse("x.TYPE_CHECKING").body[0]
    assert isinstance(node2, ast.Expr)
    assert _is_type_checking_guard(node2.value) is True


def test_is_type_checking_guard_non_match() -> None:
    """Returns False for non-TYPE_CHECKING guards."""
    import ast

    from weakincentives.serde._generics import _is_type_checking_guard

    node = ast.parse("42").body[0]
    assert isinstance(node, ast.Expr)
    assert _is_type_checking_guard(node.value) is False


def test_find_import_in_block_skips_non_import() -> None:
    """Skips non-ImportFrom statements in TYPE_CHECKING blocks."""
    import ast

    from weakincentives.serde._generics import _find_import_in_block

    # A block containing a pass statement (not an import)
    tree = ast.parse("pass")
    result = _find_import_in_block(tree.body, "SomeType", "test.module")
    assert result is None


def test_find_import_in_block_bare_relative() -> None:
    """Bare relative imports like ``from . import _utils`` resolve correctly."""
    import ast

    from weakincentives.serde._generics import _find_import_in_block

    # ``from . import _utils`` inside weakincentives.serde.parse
    source = "from . import _utils"
    tree = ast.parse(source)
    result = _find_import_in_block(tree.body, "_utils", "weakincentives.serde.parse")
    # Should resolve to the weakincentives.serde._utils module
    import weakincentives.serde._utils as expected_mod

    assert result is expected_mod


def test_find_import_in_block_bare_relative_no_match() -> None:
    """Bare relative import that doesn't match the target name returns None."""
    import ast

    from weakincentives.serde._generics import _find_import_in_block

    source = "from . import other"
    tree = ast.parse(source)
    result = _find_import_in_block(tree.body, "serde", "weakincentives.serde.parse")
    assert result is None


def test_resolve_import_module_absolute() -> None:
    """Absolute imports return the module path unchanged."""
    from weakincentives.serde._generics import _resolve_import_module

    assert _resolve_import_module("foo.bar", 0, "any.module") == "foo.bar"


def test_resolve_import_module_relative() -> None:
    """Relative imports are resolved against the current module."""
    from weakincentives.serde._generics import _resolve_import_module

    result = _resolve_import_module(
        "debug.bundle", 2, "weakincentives.runtime.agent_loop_types"
    )
    assert result == "weakincentives.debug.bundle"


def test_build_typevar_map_unresolved_typevar() -> None:
    from typing import TypeVar

    from weakincentives.serde._generics import _build_typevar_map

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


def test_internal_helpers() -> None:
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
    )

    original_get_args = coercers_module.get_args

    def fake_get_args(typ: object) -> tuple[object, ...]:
        if typ is union_type:
            return (type(None),)
        return original_get_args(typ)

    monkeypatch.setattr(coercers_module, "get_args", fake_get_args)

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


def test_merge_annotated_meta_handles_non_mapping_args() -> None:
    from typing import Annotated

    annotated_type = Annotated[int, "string_metadata", {"key": "value"}, 123]
    base, meta = _merge_annotated_meta(annotated_type, None)
    assert base is int
    assert meta == {"key": "value"}


def test_parse_field_alias_from_metadata() -> None:
    """Field-level aliases still work through field metadata."""

    @dataclass
    class AliasModel:
        api_key: str = field(metadata={"alias": "apiKey"})
        max_retries: int = field(metadata={"alias": "maxRetries"})

    parsed = parse(AliasModel, {"apiKey": "secret", "maxRetries": "3"})
    assert parsed.api_key == "secret"
    assert parsed.max_retries == 3

    # Dump uses aliases by default
    data = dump(parsed)
    assert data == {"apiKey": "secret", "maxRetries": 3}

    # Dump without aliases
    plain = dump(parsed, by_alias=False)
    assert plain == {"api_key": "secret", "max_retries": 3}
