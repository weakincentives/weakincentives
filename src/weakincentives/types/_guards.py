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

"""Type guards and safe casting utilities.

This module provides type-safe alternatives to :func:`typing.cast` that perform
runtime validation. Unlike ``cast()``, these functions verify types at runtime,
catching bugs early while still satisfying the type checker.

Example usage::

    from weakincentives.types import ensure_type, narrow_optional, is_instance_of

    # Instead of: cast(str, value)
    text = ensure_type(value, str, "config.name")

    # Instead of: cast(T, optional_value) with assertion
    required = narrow_optional(optional_value, "user.email")

    # Type guard for conditional narrowing
    if is_instance_of(response, SuccessResponse):
        print(response.data)  # type narrowed
"""

from __future__ import annotations

from typing import TypeGuard


def is_instance_of[T](value: object, type_: type[T]) -> TypeGuard[T]:
    """Type guard that checks if a value is an instance of a type.

    This is a thin wrapper around ``isinstance()`` that provides proper type
    narrowing through ``TypeGuard``. Use this in conditional expressions where
    you need the type checker to understand the narrowed type.

    Args:
        value: The value to check.
        type_: The type to check against.

    Returns:
        True if value is an instance of type_, False otherwise.

    Example::

        def process(item: str | int) -> str:
            if is_instance_of(item, str):
                return item.upper()  # type is narrowed to str
            return str(item)
    """
    return isinstance(value, type_)


def ensure_type[T](value: object, type_: type[T], context: str = "") -> T:
    """Assert a value is of the expected type and return it with correct typing.

    Unlike ``cast()``, this performs runtime validation, catching type mismatches
    immediately rather than allowing them to propagate. Use this when you need
    to convert from ``object`` or ``Any`` to a specific type with verification.

    Args:
        value: The value to check and return.
        type_: The expected type.
        context: Optional description of where this check occurs, included in
            error messages for easier debugging.

    Returns:
        The value, typed as T.

    Raises:
        TypeError: If value is not an instance of type_.

    Example::

        data: dict[str, object] = load_config()
        name = ensure_type(data["name"], str, "config.name")
    """
    if not isinstance(value, type_):
        msg = f"Expected {type_.__name__}, got {type(value).__name__}"
        if context:
            msg = f"{context}: {msg}"
        raise TypeError(msg)
    return value


def narrow_optional[T](value: T | None, context: str = "") -> T:
    """Narrow an optional value to its non-None type.

    This function asserts that a value is not None and returns it with the
    narrowed type. Use this when you have an ``Optional[T]`` but know from
    context that the value must be present.

    Args:
        value: The potentially-None value.
        context: Optional description of where this check occurs, included in
            error messages for easier debugging.

    Returns:
        The value, with None removed from its type.

    Raises:
        ValueError: If value is None.

    Example::

        user: User | None = find_user(id)
        # After validation that user must exist:
        current_user = narrow_optional(user, "authenticated user")
    """
    if value is None:
        msg = "Expected non-None value"
        if context:
            msg = f"{context}: {msg}"
        raise ValueError(msg)
    return value


__all__ = [
    "ensure_type",
    "is_instance_of",
    "narrow_optional",
]
