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

"""Tests covering bytes handling in serde parsing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.serde.dump import dump
from weakincentives.serde.parse import parse


@dataclass(slots=True, frozen=True)
class BytesPayload:
    payload: bytes


class CustomBytes(bytes):
    """Subclass used to exercise bytes coercion branches."""


@dataclass(slots=True, frozen=True)
class CustomBytesPayload:
    payload: CustomBytes


def test_parse_bytes_from_base64_round_trip() -> None:
    original = BytesPayload(payload=b"hello")
    encoded = dump(original)
    restored = parse(BytesPayload, encoded)
    assert restored == original


def test_parse_bytes_accepts_raw_bytes() -> None:
    restored = parse(BytesPayload, {"payload": b"raw"})
    assert restored.payload == b"raw"


def test_parse_bytes_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="payload: invalid base64"):
        parse(BytesPayload, {"payload": "not-base64!"})


def test_parse_bytes_rejects_unexpected_type() -> None:
    with pytest.raises(TypeError, match="payload: expected base64"):
        parse(BytesPayload, {"payload": 123})


def test_parse_bytes_subclass_coerces_value() -> None:
    restored = parse(
        CustomBytesPayload,
        {"payload": "aGVsbG8="},
    )
    assert isinstance(restored.payload, bytearray)
    assert bytes(restored.payload) == b"hello"
