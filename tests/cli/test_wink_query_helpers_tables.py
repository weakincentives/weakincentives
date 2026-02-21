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

"""Unit tests for wink query table-related helper functions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from weakincentives.cli._query_tables import (
    _create_dynamic_slice_table,
    _extract_slices_from_snapshot,
    _extract_tool_call_from_entry,
    _is_tool_event,
    _process_session_line,
)


class TestExtractToolCall:
    """Tests for _extract_tool_call_from_entry helper function."""

    def test_no_context(self) -> None:
        entry: dict[str, Any] = {"timestamp": "2024-01-01T00:00:00Z"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_context_not_dict(self) -> None:
        entry: dict[str, Any] = {"context": "not a dict", "timestamp": "2024-01-01"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_no_tool_name(self) -> None:
        entry: dict[str, Any] = {"context": {"params": {}}, "timestamp": "2024-01-01"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_with_error_legacy_fallback(self) -> None:
        """Test legacy fallback: infer failure from error text presence."""
        entry: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {
                "tool_name": "read_file",
                "error": "File not found",
                "error_code": "E001",
            },
        }
        result = _extract_tool_call_from_entry(entry)
        assert result is not None
        assert result[4] == 0  # success = 0
        assert result[5] == "E001"  # error_code

    def test_explicit_success_true(self) -> None:
        """Test explicit success=True field from tool.execution.complete logs."""
        entry: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {
                "tool_name": "read_file",
                "success": True,
                "message": "File read successfully",
            },
        }
        result = _extract_tool_call_from_entry(entry)
        assert result is not None
        assert result[4] == 1  # success = 1
        assert result[5] == ""  # no error_code

    def test_explicit_success_false(self) -> None:
        """Test explicit success=False field takes precedence over error text."""
        # Even without "error" in context, success=False should be detected
        entry: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {
                "tool_name": "write_file",
                "success": False,
                "message": "Permission denied",
            },
        }
        result = _extract_tool_call_from_entry(entry)
        assert result is not None
        assert result[4] == 0  # success = 0
        assert result[5] == "Permission denied"  # message used as error_code

    def test_arguments_field_used_for_params(self) -> None:
        """Test that 'arguments' field from current logs is used for params."""
        entry: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {
                "tool_name": "read_file",
                "arguments": {"path": "/test.txt", "encoding": "utf-8"},
                "success": True,
            },
        }
        result = _extract_tool_call_from_entry(entry)
        assert result is not None
        # result[2] is params (JSON string)
        params = json.loads(result[2])
        assert params == {"path": "/test.txt", "encoding": "utf-8"}

    def test_value_field_used_for_result(self) -> None:
        """Test that 'value' field from current logs is used for result."""
        entry: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {
                "tool_name": "read_file",
                "value": "file contents here",
                "success": True,
            },
        }
        result = _extract_tool_call_from_entry(entry)
        assert result is not None
        # result[3] is result (JSON string)
        result_val = json.loads(result[3])
        assert result_val == "file contents here"


class TestIsToolEvent:
    """Tests for _is_tool_event helper function."""

    def test_tool_execution_events(self) -> None:
        # Actual event names from tool_executor
        assert _is_tool_event("tool.execution.start") is True
        assert _is_tool_event("tool.execution.complete") is True

    def test_tool_call_event(self) -> None:
        # Alternative event formats
        assert _is_tool_event("tool.call.start") is True
        assert _is_tool_event("tool.result.end") is True

    def test_not_tool_event(self) -> None:
        assert _is_tool_event("session.start") is False
        assert _is_tool_event("request.complete") is False


class TestExtractSlicesFromSnapshot:
    """Tests for _extract_slices_from_snapshot helper function."""

    def test_no_slices_field(self) -> None:
        entry: dict[str, object] = {"other": "data"}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_slices_not_list(self) -> None:
        entry: dict[str, object] = {"slices": "not a list"}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_slice_obj_not_mapping(self) -> None:
        entry: dict[str, object] = {"slices": ["not a mapping"]}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_items_not_list(self) -> None:
        entry: dict[str, object] = {
            "slices": [{"slice_type": "MyType", "items": "not a list"}]
        }
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_item_not_mapping(self) -> None:
        entry: dict[str, object] = {
            "slices": [{"slice_type": "MyType", "items": ["not a mapping"]}]
        }
        result = _extract_slices_from_snapshot(entry)
        assert result == []


class TestProcessSessionLine:
    """Tests for _process_session_line helper function."""

    def test_invalid_json(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """
        )
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        _process_session_line(conn, "not valid json", slices_by_type)

        cursor = conn.execute("SELECT COUNT(*) FROM session_slices")
        assert cursor.fetchone()[0] == 0

    def test_non_mapping_entry(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """
        )
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        _process_session_line(conn, '"just a string"', slices_by_type)

        cursor = conn.execute("SELECT COUNT(*) FROM session_slices")
        assert cursor.fetchone()[0] == 0


class TestDirectSliceFormat:
    """Tests for direct slice format in _process_session_line."""

    def test_direct_slice_format_with_type(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """
        )
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        # Direct slice format with __type__ field
        line = json.dumps({"__type__": "myapp.state:AgentPlan", "goal": "test"})
        _process_session_line(conn, line, slices_by_type)

        cursor = conn.execute("SELECT slice_type, data FROM session_slices")
        row = cursor.fetchone()
        assert row[0] == "myapp.state:AgentPlan"

    def test_direct_slice_format_without_type(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """
        )
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        # Direct slice format without __type__ field
        line = json.dumps({"goal": "test", "steps": 3})
        _process_session_line(conn, line, slices_by_type)

        cursor = conn.execute("SELECT slice_type FROM session_slices")
        row = cursor.fetchone()
        assert row[0] == "unknown"


class TestDynamicSliceTableEdgeCases:
    """Tests for dynamic slice table creation edge cases."""

    def test_slice_with_no_columns(self) -> None:
        """Test creating a dynamic slice table with only __type__ (no extra columns)."""
        import sqlite3

        conn = sqlite3.connect(":memory:")

        # Slice data with only __type__ field
        slices: list[Mapping[str, Any]] = [{"__type__": "EmptySlice"}]

        _create_dynamic_slice_table(conn, "EmptySlice", slices)

        # Table should exist with only rowid
        cursor = conn.execute("SELECT * FROM slice_emptyslice")
        rows = cursor.fetchall()
        assert len(rows) == 0  # No data inserted since no columns

    def test_slice_with_duplicate_keys_uses_first_type(self) -> None:
        """Test that when same key appears in multiple slices, first type is used."""
        import sqlite3

        conn = sqlite3.connect(":memory:")

        # First slice has integer, second has string
        slices: list[Mapping[str, Any]] = [{"value": 42}, {"value": "text"}]

        _create_dynamic_slice_table(conn, "MixedSlice", slices)

        cursor = conn.execute("PRAGMA table_info(slice_mixedslice)")
        cols = {row[1]: row[2] for row in cursor.fetchall()}
        assert cols["value"] == "INTEGER"  # First value type wins
