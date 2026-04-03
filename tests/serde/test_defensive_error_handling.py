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

"""Tests for defensive error-handling paths in serde."""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import patch

import pytest

from weakincentives.serde import parse
from weakincentives.serde._coercers import (  # pyright: ignore[reportPrivateUsage]
    _PRIMITIVE_COERCERS,
    _coerce_to_type,
    _ParseConfig,
)
from weakincentives.serde._generics import (  # pyright: ignore[reportPrivateUsage]
    _get_field_types,
    _resolve_type_checking_imports,
)

pytestmark = pytest.mark.core


# ---------------------------------------------------------------------------
# _utils.py: validator raising unexpected exception
# ---------------------------------------------------------------------------


def _validator_raises_runtime(value: object) -> object:
    raise RuntimeError("unexpected validator failure")


@dataclass
class _ValidatorRuntimeError:
    value: Annotated[str, {"validators": (_validator_raises_runtime,)}]


def test_validator_unexpected_exception_logs_and_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="weakincentives.serde._utils"):
        with pytest.raises(ValueError, match="validator raised"):
            parse(_ValidatorRuntimeError, {"value": "hello"})
    assert "unexpected" in caplog.text.lower() or "RuntimeError" in caplog.text


# ---------------------------------------------------------------------------
# _utils.py: converter raising unexpected exception
# ---------------------------------------------------------------------------


def _converter_raises_runtime(_: object) -> object:
    raise RuntimeError("unexpected converter failure")


@dataclass
class _ConverterRuntimeError:
    value: Annotated[int, {"convert": _converter_raises_runtime}]


def test_converter_unexpected_exception_logs_and_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="weakincentives.serde._utils"):
        with pytest.raises(ValueError, match="converter raised"):
            parse(_ConverterRuntimeError, {"value": 42})
    assert "unexpected" in caplog.text.lower() or "RuntimeError" in caplog.text


# ---------------------------------------------------------------------------
# _coercers.py: primitive coercion unexpected exception (non-TypeError/ValueError)
# ---------------------------------------------------------------------------


class _ExplodingPrimitive:
    """A type registered as a primitive coercer that raises unexpectedly."""


def _exploding_coercer(_value: object) -> object:
    raise RuntimeError("coercion kaboom")


def test_primitive_coercion_unexpected_exception_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_coerce_primitive catches non-TypeError/ValueError from registered coercers."""
    config = _ParseConfig(extra="ignore", coerce=True)
    _PRIMITIVE_COERCERS[_ExplodingPrimitive] = _exploding_coercer
    try:
        with caplog.at_level(logging.WARNING, logger="weakincentives.serde._coercers"):
            with pytest.raises(TypeError, match="unable to coerce"):
                _coerce_to_type("hello", _ExplodingPrimitive, None, "test", config)
        assert "RuntimeError" in caplog.text
    finally:
        del _PRIMITIVE_COERCERS[_ExplodingPrimitive]


# ---------------------------------------------------------------------------
# _coercers.py: fallback coercion in _coerce_to_type (line ~511)
# ---------------------------------------------------------------------------


class _FallbackExploding:
    """Not a standard type, triggers the fallback coercion path."""

    def __init__(self, _value: object) -> None:
        raise AttributeError("no such attribute")


def test_fallback_coercion_unexpected_exception_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _ParseConfig(extra="ignore", coerce=True)
    with caplog.at_level(logging.WARNING, logger="weakincentives.serde._coercers"):
        with pytest.raises(TypeError, match="no such attribute"):
            _coerce_to_type("hello", _FallbackExploding, None, "test", config)
    assert "AttributeError" in caplog.text


# ---------------------------------------------------------------------------
# _generics.py: _resolve_type_checking_imports source parse failure
# ---------------------------------------------------------------------------


def test_resolve_type_checking_imports_source_failure_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When inspect.getsource fails, the function logs and returns None."""
    # Create a module that exists in sys.modules but has no source
    fake_module = types.ModuleType("_test_no_source_module")
    sys.modules["_test_no_source_module"] = fake_module

    @dataclass
    class _FakeClass:
        pass

    _FakeClass.__module__ = "_test_no_source_module"

    try:
        with caplog.at_level(logging.WARNING, logger="weakincentives.serde._generics"):
            result = _resolve_type_checking_imports(_FakeClass, "MissingName")
        assert result is None
        assert "Could not parse source" in caplog.text
    finally:
        del sys.modules["_test_no_source_module"]


# ---------------------------------------------------------------------------
# _generics.py: _get_field_types defensive re-raise paths
# ---------------------------------------------------------------------------


def test_get_field_types_unresolvable_name_error_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When _resolve_type_checking_imports returns None, logs and re-raises."""

    @dataclass
    class _HasBadAnnotation:
        pass

    # Simulate a NameError from get_type_hints where the name can't be resolved
    with (
        caplog.at_level(logging.WARNING, logger="weakincentives.serde._generics"),
        patch(
            "weakincentives.serde._generics.get_type_hints",
            side_effect=NameError("name 'Bogus' is not defined"),
        ),
        patch(
            "weakincentives.serde._generics._resolve_type_checking_imports",
            return_value=None,
        ),
        pytest.raises(NameError, match="Bogus"),
    ):
        _get_field_types(_HasBadAnnotation)
    assert "Could not resolve TYPE_CHECKING import" in caplog.text


def test_get_field_types_unparseable_name_error_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When _extract_missing_name returns None, logs and re-raises."""
    with (
        caplog.at_level(logging.WARNING, logger="weakincentives.serde._generics"),
        patch(
            "weakincentives.serde._generics.get_type_hints",
            side_effect=NameError("weird format"),
        ),
        pytest.raises(NameError, match="weird format"),
    ):
        _get_field_types(dataclass(type("_Dummy", (), {"__annotations__": {}})))
    assert "Could not extract" in caplog.text
