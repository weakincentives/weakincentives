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

"""Tests for type guards and safe casting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.types import ensure_type, is_instance_of, narrow_optional

pytestmark = pytest.mark.core


class TestIsInstanceOf:
    """Tests for is_instance_of type guard."""

    def test_returns_true_for_matching_type(self) -> None:
        assert is_instance_of("hello", str) is True

    def test_returns_false_for_non_matching_type(self) -> None:
        assert is_instance_of("hello", int) is False

    def test_works_with_int(self) -> None:
        assert is_instance_of(42, int) is True
        assert is_instance_of(42, str) is False

    def test_works_with_float(self) -> None:
        assert is_instance_of(3.14, float) is True
        assert is_instance_of(3.14, int) is False

    def test_works_with_none(self) -> None:
        assert is_instance_of(None, type(None)) is True
        assert is_instance_of(None, str) is False

    def test_works_with_list(self) -> None:
        assert is_instance_of([1, 2, 3], list) is True
        assert is_instance_of([1, 2, 3], dict) is False

    def test_works_with_dict(self) -> None:
        assert is_instance_of({"a": 1}, dict) is True
        assert is_instance_of({"a": 1}, list) is False

    def test_works_with_custom_class(self) -> None:
        @dataclass
        class MyClass:
            value: int

        obj = MyClass(42)
        assert is_instance_of(obj, MyClass) is True
        assert is_instance_of(obj, str) is False

    def test_subclass_is_instance(self) -> None:
        class Parent:
            pass

        class Child(Parent):
            pass

        assert is_instance_of(Child(), Parent) is True
        assert is_instance_of(Child(), Child) is True

    def test_narrows_type_in_conditional(self) -> None:
        def process(value: str | int) -> str:
            if is_instance_of(value, str):
                return value.upper()
            return str(value)

        assert process("hello") == "HELLO"
        assert process(42) == "42"


class TestEnsureType:
    """Tests for ensure_type runtime validation."""

    def test_returns_value_when_type_matches(self) -> None:
        result = ensure_type("hello", str)
        assert result == "hello"

    def test_returns_value_with_context(self) -> None:
        result = ensure_type("hello", str, "config.name")
        assert result == "hello"

    def test_raises_type_error_for_wrong_type(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            ensure_type("hello", int)
        assert "Expected int, got str" in str(exc_info.value)

    def test_raises_type_error_with_context(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            ensure_type("hello", int, "config.port")
        assert "config.port: Expected int, got str" in str(exc_info.value)

    def test_works_with_int(self) -> None:
        assert ensure_type(42, int) == 42

    def test_works_with_float(self) -> None:
        assert ensure_type(3.14, float) == 3.14

    def test_works_with_list(self) -> None:
        result = ensure_type([1, 2, 3], list)
        assert result == [1, 2, 3]

    def test_works_with_dict(self) -> None:
        result = ensure_type({"a": 1}, dict)
        assert result == {"a": 1}

    def test_works_with_none_type(self) -> None:
        result = ensure_type(None, type(None))
        assert result is None

    def test_works_with_custom_class(self) -> None:
        @dataclass
        class Config:
            name: str

        config = Config("test")
        result = ensure_type(config, Config)
        assert result.name == "test"

    def test_subclass_passes(self) -> None:
        class Parent:
            pass

        class Child(Parent):
            pass

        child = Child()
        result = ensure_type(child, Parent)
        assert result is child

    def test_rejects_none_when_expecting_str(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            ensure_type(None, str)
        assert "Expected str, got NoneType" in str(exc_info.value)

    def test_error_message_shows_actual_type(self) -> None:
        @dataclass
        class MyClass:
            pass

        with pytest.raises(TypeError) as exc_info:
            ensure_type(MyClass(), str)
        assert "MyClass" in str(exc_info.value)


class TestNarrowOptional:
    """Tests for narrow_optional None elimination."""

    def test_returns_value_when_not_none(self) -> None:
        result = narrow_optional("hello")
        assert result == "hello"

    def test_returns_value_with_context(self) -> None:
        result = narrow_optional("hello", "user.name")
        assert result == "hello"

    def test_raises_value_error_for_none(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            narrow_optional(None)
        assert "Expected non-None value" in str(exc_info.value)

    def test_raises_value_error_with_context(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            narrow_optional(None, "user.email")
        assert "user.email: Expected non-None value" in str(exc_info.value)

    def test_works_with_int(self) -> None:
        value: int | None = 42
        result = narrow_optional(value)
        assert result == 42

    def test_works_with_float(self) -> None:
        value: float | None = 3.14
        result = narrow_optional(value)
        assert result == 3.14

    def test_works_with_list(self) -> None:
        value: list[int] | None = [1, 2, 3]
        result = narrow_optional(value)
        assert result == [1, 2, 3]

    def test_works_with_dict(self) -> None:
        value: dict[str, int] | None = {"a": 1}
        result = narrow_optional(value)
        assert result == {"a": 1}

    def test_works_with_custom_class(self) -> None:
        @dataclass
        class User:
            name: str

        user: User | None = User("Alice")
        result = narrow_optional(user)
        assert result.name == "Alice"

    def test_preserves_empty_string(self) -> None:
        result = narrow_optional("")
        assert result == ""

    def test_preserves_zero(self) -> None:
        result = narrow_optional(0)
        assert result == 0

    def test_preserves_empty_list(self) -> None:
        result = narrow_optional([])
        assert result == []

    def test_preserves_false(self) -> None:
        result = narrow_optional(False)
        assert result is False


class TestIntegration:
    """Integration tests combining guard functions."""

    def test_ensure_type_and_narrow_optional_chain(self) -> None:
        data: dict[str, object] = {"name": "test", "count": 42}
        name = ensure_type(data.get("name"), str, "data.name")
        count = ensure_type(data.get("count"), int, "data.count")
        assert name == "test"
        assert count == 42

    def test_is_instance_of_before_ensure_type(self) -> None:
        def process(value: object) -> str:
            if is_instance_of(value, str):
                return value.upper()
            return ensure_type(str(value), str)

        assert process("hello") == "HELLO"
        assert process(42) == "42"

    def test_narrow_optional_after_dict_get(self) -> None:
        data: dict[str, str | None] = {"key": "value"}
        result = narrow_optional(data.get("key"), "data.key")
        assert result == "value"

    def test_ensure_type_with_dataclass(self) -> None:
        @dataclass(frozen=True)
        class Config:
            name: str
            value: int

        raw: object = Config("test", 42)
        config = ensure_type(raw, Config, "loaded config")
        assert config.name == "test"
        assert config.value == 42
