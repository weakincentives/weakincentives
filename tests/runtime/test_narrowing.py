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

"""Tests for type narrowing helpers in weakincentives.runtime.session.narrowing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.runtime.session import (
    as_dataclass_type,
    as_slice_type,
    extract_event_value,
    is_dataclass_type,
)


@dataclass(slots=True)
class _SampleDataclass:
    value: str


@dataclass(slots=True)
class _EventWithValue:
    value: _SampleDataclass | None


class _PlainClass:
    pass


def test_is_dataclass_type_returns_true_for_dataclass() -> None:
    assert is_dataclass_type(_SampleDataclass) is True


def test_is_dataclass_type_returns_false_for_plain_class() -> None:
    assert is_dataclass_type(_PlainClass) is False


def test_is_dataclass_type_returns_false_for_builtin() -> None:
    assert is_dataclass_type(str) is False


def test_as_slice_type_returns_type_for_dataclass() -> None:
    result = as_slice_type(_SampleDataclass)
    assert result is _SampleDataclass


def test_as_slice_type_raises_for_non_dataclass() -> None:
    with pytest.raises(TypeError, match="is not a dataclass type"):
        as_slice_type(_PlainClass)


def test_as_dataclass_type_returns_type_for_dataclass() -> None:
    result = as_dataclass_type(_SampleDataclass)
    assert result is _SampleDataclass


def test_as_dataclass_type_raises_for_non_dataclass() -> None:
    with pytest.raises(TypeError, match="is not a dataclass type"):
        as_dataclass_type(_PlainClass)


def test_extract_event_value_returns_typed_value() -> None:
    event = _EventWithValue(value=_SampleDataclass("test"))
    result = extract_event_value(event, _SampleDataclass)
    assert result == _SampleDataclass("test")


def test_extract_event_value_raises_for_wrong_type() -> None:
    event = _EventWithValue(value=_SampleDataclass("test"))

    @dataclass(slots=True)
    class _OtherDataclass:
        value: str

    with pytest.raises(TypeError, match="Expected _OtherDataclass"):
        extract_event_value(event, _OtherDataclass)


def test_extract_event_value_raises_for_none_value() -> None:
    event = _EventWithValue(value=None)
    with pytest.raises(TypeError, match="has a None value"):
        extract_event_value(event, _SampleDataclass)


@dataclass(slots=True)
class _NoValueEvent:
    """An event without a value attribute."""

    data: str


def test_extract_event_value_raises_for_non_reducer_event() -> None:
    plain_event = _NoValueEvent("not an event")
    with pytest.raises(TypeError, match="does not have a value attribute"):
        extract_event_value(plain_event, _SampleDataclass)
