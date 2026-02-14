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

"""Tests for schema normalization utilities."""

from __future__ import annotations

from typing import Any


class TestCollapseNullableAnyOf:
    """Tests for _collapse_nullable_any_of edge cases."""

    def test_non_dict_entries_returns_none(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _collapse_nullable_any_of,
        )

        result = _collapse_nullable_any_of(["string", {"type": "null"}])
        assert result is None

    def test_no_null_entry_returns_none(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _collapse_nullable_any_of,
        )

        result = _collapse_nullable_any_of([{"type": "string"}, {"type": "integer"}])
        assert result is None

    def test_list_type_collapses_with_null(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _collapse_nullable_any_of,
        )

        result = _collapse_nullable_any_of(
            [{"type": ["string", "integer"]}, {"type": "null"}]
        )
        assert result is not None
        assert result["type"] == ["string", "integer", "null"]

    def test_list_type_already_has_null(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _collapse_nullable_any_of,
        )

        result = _collapse_nullable_any_of(
            [{"type": ["string", "null"]}, {"type": "null"}]
        )
        assert result is not None
        assert result["type"] == ["string", "null"]

    def test_unknown_type_returns_none(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _collapse_nullable_any_of,
        )

        result = _collapse_nullable_any_of([{"type": 42}, {"type": "null"}])
        assert result is None


class TestNormalizeClaudeOutputSchema:
    """Tests for _normalize_claude_output_schema recursive cases."""

    def test_object_without_properties(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _normalize_claude_output_schema,
        )

        schema: dict[str, Any] = {"type": "object"}
        result = _normalize_claude_output_schema(schema)
        assert result == {"type": "object"}

    def test_array_items_normalized(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _normalize_claude_output_schema,
        )

        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
            },
        }
        result = _normalize_claude_output_schema(schema)
        assert result["items"]["type"] == ["integer", "null"]
        assert "anyOf" not in result["items"]

    def test_defs_normalized(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._schema_normalization import (
            _normalize_claude_output_schema,
        )

        schema: dict[str, Any] = {
            "$defs": {
                "Inner": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
            },
        }
        result = _normalize_claude_output_schema(schema)
        assert result["$defs"]["Inner"]["type"] == ["string", "null"]
