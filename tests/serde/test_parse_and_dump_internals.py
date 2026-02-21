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

"""Tests for serde internal helpers, coercion, and generics utilities."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime

import pytest

from tests.serde._fixtures import (
    Address,
)
from weakincentives.serde import parse
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
