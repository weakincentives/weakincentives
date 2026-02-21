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

"""Tests for the Codex App Server adapter schema utilities."""

from __future__ import annotations

from weakincentives.adapters.codex_app_server._schema import (
    openai_strict_schema,
)


class TestOpenaiStrictSchema:
    def test_sets_additional_properties_false(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False

    def test_all_properties_required(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        result = openai_strict_schema(s)
        assert sorted(result["required"]) == ["a", "b"]

    def test_nested_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert result["properties"]["inner"]["additionalProperties"] is False
        assert result["properties"]["inner"]["required"] == ["x"]

    def test_non_object_unchanged(self) -> None:
        s = {"type": "array", "items": {"type": "string"}}
        result = openai_strict_schema(s)
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_object_without_properties(self) -> None:
        s = {"type": "object", "additionalProperties": True}
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert "required" not in result

    def test_preserves_other_fields(self) -> None:
        s = {"type": "object", "title": "Foo", "properties": {"a": {"type": "string"}}}
        result = openai_strict_schema(s)
        assert result["title"] == "Foo"
        assert result["required"] == ["a"]

    def test_array_items_strictified(self) -> None:
        s = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": True,
            },
        }
        result = openai_strict_schema(s)
        assert result["items"]["additionalProperties"] is False
        assert result["items"]["required"] == ["x"]

    def test_anyof_strictified(self) -> None:
        s = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": True,
                },
                {"type": "string"},
            ]
        }
        result = openai_strict_schema(s)
        assert result["anyOf"][0]["additionalProperties"] is False
        assert result["anyOf"][0]["required"] == ["a"]
        assert result["anyOf"][1] == {"type": "string"}

    def test_oneof_strictified(self) -> None:
        s = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = openai_strict_schema(s)
        assert result["oneOf"][0]["additionalProperties"] is False

    def test_allof_strictified(self) -> None:
        s = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"c": {"type": "boolean"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = openai_strict_schema(s)
        assert result["allOf"][0]["additionalProperties"] is False

    def test_defs_strictified(self) -> None:
        s = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                    "additionalProperties": True,
                }
            },
        }
        result = openai_strict_schema(s)
        assert result["$defs"]["Inner"]["additionalProperties"] is False
        assert result["$defs"]["Inner"]["required"] == ["val"]

    def test_definitions_strictified(self) -> None:
        s = {
            "definitions": {
                "Foo": {
                    "type": "object",
                    "properties": {"z": {"type": "number"}},
                    "additionalProperties": True,
                }
            }
        }
        result = openai_strict_schema(s)
        assert result["definitions"]["Foo"]["additionalProperties"] is False

    def test_deeply_nested_array_of_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nested": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"v": {"type": "integer"}},
                                    "additionalProperties": True,
                                },
                            }
                        },
                        "additionalProperties": True,
                    },
                }
            },
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        # Top-level object
        assert result["additionalProperties"] is False
        # Array items object
        items_obj = result["properties"]["items"]["items"]
        assert items_obj["additionalProperties"] is False
        assert items_obj["required"] == ["nested"]
        # Deeply nested array items object
        deep_obj = items_obj["properties"]["nested"]["items"]
        assert deep_obj["additionalProperties"] is False
        assert deep_obj["required"] == ["v"]
