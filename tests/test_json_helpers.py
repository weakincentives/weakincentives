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

"""Tests for JSON type narrowing helpers."""

from __future__ import annotations

import pytest

from weakincentives.types.json import as_json_array, as_json_object


class TestAsJsonObject:
    def test_narrows_dict(self) -> None:
        value: object = {"key": "value"}
        result = as_json_object(value)
        assert result["key"] == "value"

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON object"):
            as_json_object("not a mapping")

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON object"):
            as_json_object(None)

    def test_rejects_list(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON object"):
            as_json_object([1, 2, 3])


class TestAsJsonArray:
    def test_narrows_list(self) -> None:
        value: object = [1, 2, 3]
        result = as_json_array(value)
        assert list(result) == [1, 2, 3]

    def test_rejects_string(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array"):
            as_json_array("not an array")

    def test_rejects_bytes(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array"):
            as_json_array(b"bytes")

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array"):
            as_json_array(None)

    def test_rejects_dict(self) -> None:
        with pytest.raises(TypeError, match="Expected JSON array"):
            as_json_array({"key": "value"})
