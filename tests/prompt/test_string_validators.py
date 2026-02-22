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

import re

import pytest

from weakincentives.prompt._validation import (
    EXAMPLE_DESCRIPTION_VALIDATOR,
    TOOL_DESCRIPTION_VALIDATOR,
    TOOL_NAME_VALIDATOR,
    StringValidator,
    ValidationError,
)


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_validation_error_stores_field_and_message(self) -> None:
        error = ValidationError(field="name", message="invalid")
        assert error.field == "name"
        assert error.message == "invalid"
        assert error.value is None

    def test_validation_error_stores_optional_value(self) -> None:
        error = ValidationError(field="name", message="invalid", value="bad_value")
        assert error.field == "name"
        assert error.message == "invalid"
        assert error.value == "bad_value"

    def test_validation_error_is_frozen(self) -> None:
        error = ValidationError(field="name", message="invalid")
        with pytest.raises(AttributeError):
            error.field = "other"  # type: ignore[misc]


class TestStringValidator:
    """Tests for StringValidator."""

    def test_valid_string_returns_none(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=10)
        assert validator.validate("hello") is None

    def test_string_too_short_returns_error(self) -> None:
        validator = StringValidator(field_name="test", min_length=3, max_length=10)
        error = validator.validate("ab")
        assert error is not None
        assert error.field == "test"
        assert "3-10 characters" in error.message
        assert error.value == "ab"

    def test_string_too_long_returns_error(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=5)
        error = validator.validate("too long")
        assert error is not None
        assert error.field == "test"
        assert "1-5 characters" in error.message
        assert error.value == "too long"

    def test_empty_string_returns_error(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=10)
        error = validator.validate("")
        assert error is not None
        assert error.field == "test"
        assert "1-10 characters" in error.message

    def test_whitespace_only_string_returns_error_when_stripped(self) -> None:
        validator = StringValidator(
            field_name="test", min_length=1, max_length=10, strip_whitespace=True
        )
        error = validator.validate("   ")
        assert error is not None
        assert "1-10 characters" in error.message

    def test_strips_whitespace_by_default(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=5)
        assert validator.validate("  hi  ") is None

    def test_no_strip_when_strip_whitespace_false(self) -> None:
        validator = StringValidator(
            field_name="test", min_length=1, max_length=5, strip_whitespace=False
        )
        error = validator.validate("  hi  ")
        assert error is not None
        assert "1-5 characters" in error.message

    def test_pattern_match_succeeds(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=20,
            pattern=re.compile(r"^[a-z]+$"),
        )
        assert validator.validate("hello") is None

    def test_pattern_mismatch_returns_error(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=20,
            pattern=re.compile(r"^[a-z]+$"),
        )
        error = validator.validate("Hello123")
        assert error is not None
        assert error.field == "test"
        assert "pattern" in error.message
        assert "^[a-z]+$" in error.message

    def test_require_ascii_accepts_ascii(self) -> None:
        validator = StringValidator(
            field_name="test", min_length=1, max_length=20, require_ascii=True
        )
        assert validator.validate("hello world!") is None

    def test_require_ascii_rejects_non_ascii(self) -> None:
        validator = StringValidator(
            field_name="test", min_length=1, max_length=20, require_ascii=True
        )
        error = validator.validate("hello \u00e9")  # e with accent
        assert error is not None
        assert error.field == "test"
        assert "ASCII" in error.message

    def test_surrounding_whitespace_allowed_by_default(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=10)
        assert validator.validate("  hello  ") is None

    def test_surrounding_whitespace_rejected_when_disallowed(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=10,
            allow_surrounding_whitespace=False,
        )
        error = validator.validate("  hello  ")
        assert error is not None
        assert error.field == "test"
        assert "whitespace" in error.message
        assert error.value == "  hello  "

    def test_leading_whitespace_rejected_when_disallowed(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=10,
            allow_surrounding_whitespace=False,
        )
        error = validator.validate("  hello")
        assert error is not None
        assert "whitespace" in error.message

    def test_trailing_whitespace_rejected_when_disallowed(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=10,
            allow_surrounding_whitespace=False,
        )
        error = validator.validate("hello  ")
        assert error is not None
        assert "whitespace" in error.message

    def test_no_whitespace_passes_when_disallowed(self) -> None:
        validator = StringValidator(
            field_name="test",
            min_length=1,
            max_length=10,
            allow_surrounding_whitespace=False,
        )
        assert validator.validate("hello") is None

    def test_validator_is_frozen(self) -> None:
        validator = StringValidator(field_name="test", min_length=1, max_length=10)
        with pytest.raises(AttributeError):
            validator.field_name = "other"  # type: ignore[misc]


class TestToolNameValidator:
    """Tests for the pre-configured TOOL_NAME_VALIDATOR."""

    def test_valid_tool_name(self) -> None:
        assert TOOL_NAME_VALIDATOR.validate("my_tool") is None
        assert TOOL_NAME_VALIDATOR.validate("tool-name") is None
        assert TOOL_NAME_VALIDATOR.validate("tool123") is None
        assert TOOL_NAME_VALIDATOR.validate("a") is None

    def test_rejects_uppercase(self) -> None:
        error = TOOL_NAME_VALIDATOR.validate("MyTool")
        assert error is not None
        assert "pattern" in error.message

    def test_rejects_spaces(self) -> None:
        error = TOOL_NAME_VALIDATOR.validate("my tool")
        assert error is not None
        assert "pattern" in error.message

    def test_rejects_special_characters(self) -> None:
        error = TOOL_NAME_VALIDATOR.validate("tool@name")
        assert error is not None
        assert "pattern" in error.message

    def test_rejects_surrounding_whitespace(self) -> None:
        error = TOOL_NAME_VALIDATOR.validate("  tool  ")
        assert error is not None
        assert "whitespace" in error.message

    def test_rejects_empty_string(self) -> None:
        error = TOOL_NAME_VALIDATOR.validate("")
        assert error is not None
        assert "1-64 characters" in error.message

    def test_rejects_too_long_name(self) -> None:
        long_name = "a" * 65
        error = TOOL_NAME_VALIDATOR.validate(long_name)
        assert error is not None
        # Length check happens before pattern check
        assert "1-64 characters" in error.message

    def test_accepts_max_length_name(self) -> None:
        max_name = "a" * 64
        assert TOOL_NAME_VALIDATOR.validate(max_name) is None


class TestToolDescriptionValidator:
    """Tests for the pre-configured TOOL_DESCRIPTION_VALIDATOR."""

    def test_valid_description(self) -> None:
        assert (
            TOOL_DESCRIPTION_VALIDATOR.validate("This is a valid description.") is None
        )

    def test_strips_whitespace(self) -> None:
        assert TOOL_DESCRIPTION_VALIDATOR.validate("  description  ") is None

    def test_rejects_empty_description(self) -> None:
        error = TOOL_DESCRIPTION_VALIDATOR.validate("")
        assert error is not None
        assert "1-200 characters" in error.message

    def test_rejects_whitespace_only(self) -> None:
        error = TOOL_DESCRIPTION_VALIDATOR.validate("   ")
        assert error is not None
        assert "1-200 characters" in error.message

    def test_rejects_non_ascii(self) -> None:
        error = TOOL_DESCRIPTION_VALIDATOR.validate(
            "Description with \u00e9"
        )  # e with accent
        assert error is not None
        assert "ASCII" in error.message

    def test_rejects_too_long_description(self) -> None:
        long_desc = "a" * 201
        error = TOOL_DESCRIPTION_VALIDATOR.validate(long_desc)
        assert error is not None
        assert "1-200 characters" in error.message

    def test_accepts_max_length_description(self) -> None:
        max_desc = "a" * 200
        assert TOOL_DESCRIPTION_VALIDATOR.validate(max_desc) is None


class TestExampleDescriptionValidator:
    """Tests for the pre-configured EXAMPLE_DESCRIPTION_VALIDATOR."""

    def test_valid_example_description(self) -> None:
        assert EXAMPLE_DESCRIPTION_VALIDATOR.validate("Simple lookup example") is None

    def test_rejects_empty(self) -> None:
        error = EXAMPLE_DESCRIPTION_VALIDATOR.validate("")
        assert error is not None

    def test_rejects_non_ascii(self) -> None:
        error = EXAMPLE_DESCRIPTION_VALIDATOR.validate(
            "example \u00e9"
        )  # e with accent
        assert error is not None
        assert "ASCII" in error.message

    def test_rejects_too_long(self) -> None:
        long_desc = "a" * 201
        error = EXAMPLE_DESCRIPTION_VALIDATOR.validate(long_desc)
        assert error is not None
