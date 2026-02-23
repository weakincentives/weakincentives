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

"""Tests for TYPE_CHECKING imports, generics helpers, annotated meta, and field aliases."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import pytest

from tests.serde._fixtures import (
    _ConcreteWrapper,
    _GenericWrapper,
    _InnerPayload,
    _NestedGenericWrapper,
)
from weakincentives.serde import dump, parse
from weakincentives.serde._utils import (
    _merge_annotated_meta,
)

coercers_module = importlib.import_module("weakincentives.serde._coercers")

pytestmark = pytest.mark.core


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
