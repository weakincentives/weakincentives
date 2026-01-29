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

"""Composable validation framework for prompt components.

This module provides reusable validators for string fields with multiple
constraints, reducing duplication across tool and section validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Protocol


@dataclass(slots=True, frozen=True)
class ValidationError:
    """Structured validation failure.

    Captures the field name, error message, and optionally the invalid value
    to provide detailed diagnostics when validation fails.
    """

    field: str
    message: str
    value: object = None


class Validator[T](Protocol):
    """Protocol for composable validators.

    Validators check a single value and return either None (valid) or a
    ValidationError describing the failure. This enables composition of
    multiple validation rules.
    """

    def validate(self, value: T) -> ValidationError | None:
        """Return None if valid, ValidationError if invalid."""
        ...


@dataclass(slots=True, frozen=True)
class StringValidator:
    """Validates string fields with multiple constraints.

    Supports length bounds, pattern matching, ASCII-only requirements, and
    whitespace handling. Pre-configured instances are available for common
    use cases like tool names and descriptions.

    Attributes:
        field_name: Name of the field being validated (used in error messages).
        min_length: Minimum allowed length after stripping (default: 1).
        max_length: Maximum allowed length after stripping (default: 200).
        pattern: Optional compiled regex that the value must match.
        require_ascii: If True, value must be ASCII-encodable (default: False).
        strip_whitespace: If True, strip whitespace before validation (default: True).
        allow_surrounding_whitespace: If False, reject values with leading/trailing
            whitespace even after stripping (default: True).
    """

    field_name: str
    min_length: int = 1
    max_length: int = 200
    pattern: re.Pattern[str] | None = None
    require_ascii: bool = False
    strip_whitespace: bool = True
    allow_surrounding_whitespace: bool = True

    def validate(self, value: str) -> ValidationError | None:
        """Validate a string value against all configured constraints.

        Args:
            value: The string to validate.

        Returns:
            None if the value passes all checks, or a ValidationError
            describing the first constraint that failed.
        """
        clean = value.strip() if self.strip_whitespace else value

        if not self.allow_surrounding_whitespace and value != clean:
            return ValidationError(
                self.field_name,
                f"{self.field_name} must not have surrounding whitespace",
                value,
            )

        if len(clean) < self.min_length or len(clean) > self.max_length:
            return ValidationError(
                self.field_name,
                f"{self.field_name} must be {self.min_length}-{self.max_length} characters",
                clean,
            )

        if self.pattern and not self.pattern.fullmatch(clean):
            return ValidationError(
                self.field_name,
                f"{self.field_name} must match pattern {self.pattern.pattern}",
                clean,
            )

        if self.require_ascii:
            try:
                _ = clean.encode("ascii")
            except UnicodeEncodeError:
                return ValidationError(
                    self.field_name,
                    f"{self.field_name} must be ASCII",
                    clean,
                )

        return None


# Pre-configured validators for common cases

_NAME_MIN_LENGTH: Final = 1
_NAME_MAX_LENGTH: Final = 64
_DESCRIPTION_MAX_LENGTH: Final = 200

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)

TOOL_NAME_VALIDATOR: Final = StringValidator(
    field_name="name",
    min_length=_NAME_MIN_LENGTH,
    max_length=_NAME_MAX_LENGTH,
    pattern=_NAME_PATTERN,
    allow_surrounding_whitespace=False,
)
"""Validator for tool names: 1-64 lowercase ASCII letters, digits, underscores, or hyphens."""

TOOL_DESCRIPTION_VALIDATOR: Final = StringValidator(
    field_name="description",
    min_length=1,
    max_length=_DESCRIPTION_MAX_LENGTH,
    require_ascii=True,
)
"""Validator for tool descriptions: 1-200 ASCII characters."""

EXAMPLE_DESCRIPTION_VALIDATOR: Final = StringValidator(
    field_name="description",
    min_length=1,
    max_length=_DESCRIPTION_MAX_LENGTH,
    require_ascii=True,
)
"""Validator for tool example descriptions: 1-200 ASCII characters."""


__all__ = [
    "EXAMPLE_DESCRIPTION_VALIDATOR",
    "TOOL_DESCRIPTION_VALIDATOR",
    "TOOL_NAME_VALIDATOR",
    "StringValidator",
    "ValidationError",
    "Validator",
]
