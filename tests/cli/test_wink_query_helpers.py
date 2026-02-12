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

"""Unit tests for wink query helper functions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weakincentives.cli._query_helpers import (
    _flatten_json,
    _get_table_description,
    _infer_sqlite_type,
    _json_to_sql_value,
    _normalize_slice_type,
    _safe_json_dumps,
)
from weakincentives.cli._query_tables import (
    _create_dynamic_slice_table,
    _extract_slices_from_snapshot,
    _extract_tool_call_from_entry,
    _is_tool_event,
    _process_session_line,
)
from weakincentives.cli._query_transcript import (
    _apply_bridged_tool_details,
    _apply_notification_item_details,
    _apply_split_block_details,
    _apply_tool_result_details,
    _apply_transcript_content_fallbacks,
    _extract_tool_use_from_content,
    _extract_transcript_details,
    _extract_transcript_message_details,
    _extract_transcript_parsed_obj,
    _extract_transcript_row,
    _stringify_transcript_content,
    _stringify_transcript_mapping,
    _stringify_transcript_tool_use,
)


class TestNormalizeSliceType:
    """Tests for _normalize_slice_type function."""

    def test_with_colon(self) -> None:
        result = _normalize_slice_type("myapp.state:AgentPlan")
        assert result == "slice_agentplan"

    def test_with_dots(self) -> None:
        result = _normalize_slice_type("myapp.state.AgentPlan")
        assert result == "slice_agentplan"

    def test_simple_name(self) -> None:
        result = _normalize_slice_type("AgentPlan")
        assert result == "slice_agentplan"

    def test_mixed_case(self) -> None:
        result = _normalize_slice_type("MyApp.State:MyPlan")
        assert result == "slice_myplan"


class TestFlattenJson:
    """Tests for _flatten_json function."""

    def test_simple_object(self) -> None:
        result = _flatten_json({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_nested_object(self) -> None:
        result = _flatten_json({"outer": {"inner": "value"}})
        assert result == {"outer_inner": "value"}

    def test_list_value(self) -> None:
        result = _flatten_json({"items": [1, 2, 3]})
        assert result == {"items": "[1, 2, 3]"}

    def test_nested_with_list(self) -> None:
        result = _flatten_json({"a": {"b": [1, 2]}})
        assert result == {"a_b": "[1, 2]"}

    def test_primitive_with_prefix(self) -> None:
        result = _flatten_json("value", prefix="key")
        assert result == {"key": "value"}


class TestFlattenJsonTopLevelList:
    """Tests for _flatten_json with top-level list."""

    def test_top_level_list(self) -> None:
        result = _flatten_json([1, 2, 3], prefix="items")
        assert result == {"items": "[1, 2, 3]"}


class TestInferSqliteType:
    """Tests for _infer_sqlite_type function."""

    def test_none(self) -> None:
        assert _infer_sqlite_type(None) == "TEXT"

    def test_bool(self) -> None:
        assert _infer_sqlite_type(True) == "INTEGER"
        assert _infer_sqlite_type(False) == "INTEGER"

    def test_int(self) -> None:
        assert _infer_sqlite_type(42) == "INTEGER"

    def test_float(self) -> None:
        assert _infer_sqlite_type(3.14) == "REAL"

    def test_string(self) -> None:
        assert _infer_sqlite_type("hello") == "TEXT"

    def test_other(self) -> None:
        assert _infer_sqlite_type({"a": 1}) == "TEXT"


class TestJsonToSqlValue:
    """Tests for _json_to_sql_value function."""

    def test_none(self) -> None:
        assert _json_to_sql_value(None) is None

    def test_bool_true(self) -> None:
        assert _json_to_sql_value(True) == 1

    def test_bool_false(self) -> None:
        assert _json_to_sql_value(False) == 0

    def test_int(self) -> None:
        assert _json_to_sql_value(42) == 42

    def test_float(self) -> None:
        assert _json_to_sql_value(3.14) == 3.14

    def test_string(self) -> None:
        assert _json_to_sql_value("hello") == "hello"

    def test_list(self) -> None:
        assert _json_to_sql_value([1, 2, 3]) == "[1, 2, 3]"

    def test_dict(self) -> None:
        result = _json_to_sql_value({"a": 1})
        assert result == '{"a": 1}'


class TestGetTableDescription:
    """Tests for _get_table_description function."""

    def test_known_table(self) -> None:
        assert _get_table_description("manifest") == "Bundle metadata"
        assert (
            _get_table_description("logs")
            == "Log entries (seq extracted from context.sequence_number when present)"
        )
        assert (
            _get_table_description("transcript")
            == "Transcript entries extracted from logs"
        )
        assert _get_table_description("errors") == "Aggregated errors"

    def test_slice_table(self) -> None:
        assert _get_table_description("slice_agentplan") == "Session slice: agentplan"

    def test_unknown_table(self) -> None:
        assert _get_table_description("unknown_table") == ""


class TestGetTableDescriptionViews:
    """Tests for _get_table_description with views."""

    def test_view_description(self) -> None:
        """Test that views get appropriate descriptions."""
        assert "View:" in _get_table_description("tool_timeline", is_view=True)
        assert "View:" in _get_table_description("native_tool_calls", is_view=True)
        assert "View:" in _get_table_description("transcript_entries", is_view=True)
        assert "View:" in _get_table_description("error_summary", is_view=True)

    def test_unknown_view(self) -> None:
        """Test that unknown views get a generic description."""
        desc = _get_table_description("unknown_view", is_view=True)
        assert desc == "View"


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


class TestTranscriptHelperCoverage:
    """Targeted unit tests for transcript helper branches in query.py."""

    def test_safe_json_dumps_falls_back_to_str(self) -> None:
        class _Unserializable:
            def __str__(self) -> str:
                return "unserializable"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, _Unserializable)

        assert _safe_json_dumps(_Unserializable()) == "unserializable"

    def test_stringify_transcript_tool_use_formats_components(self) -> None:
        full = _stringify_transcript_tool_use(
            {"name": "read_file", "id": "toolu_123", "input": {"path": "/tmp/a.txt"}}
        )
        assert full == '[tool_use] read_file toolu_123 {"path": "/tmp/a.txt"}'

        minimal = _stringify_transcript_tool_use({})
        assert minimal == "[tool_use]"

    def test_stringify_transcript_mapping_covers_paths(self) -> None:
        assert _stringify_transcript_mapping({"text": "Hello"}) == "Hello"

        # Empty extracted content should fall through to JSON rendering.
        assert _stringify_transcript_mapping({"text": ""}) == '{"text": ""}'

        tool_use = _stringify_transcript_mapping({"type": "tool_use", "name": "search"})
        assert tool_use.startswith("[tool_use] search")

    def test_stringify_transcript_content_handles_common_types(self) -> None:
        assert _stringify_transcript_content(None) == ""
        assert _stringify_transcript_content(123) == "123"
        assert _stringify_transcript_content({"text": "A"}) == "A"
        assert _stringify_transcript_content([{"text": "A"}, "B", None]) == "A\nB"

    def test_extract_tool_use_from_content_handles_mapping_and_list(self) -> None:
        assert _extract_tool_use_from_content(
            {"type": "tool_use", "name": "search", "id": "toolu_1"}
        ) == ("search", "toolu_1")

        assert _extract_tool_use_from_content([{"type": "text", "text": "hi"}]) == (
            "",
            "",
        )

    def test_extract_transcript_message_details_handles_shapes(self) -> None:
        assert _extract_transcript_message_details({"message": "not-a-mapping"}) == (
            "",
            "",
            "",
            "",
        )

        role, content, tool_name, tool_use_id = _extract_transcript_message_details(
            {"message": {"role": 123, "content": [{"type": "text", "text": "Hello"}]}}
        )
        assert role == ""
        assert content == "Hello"
        assert tool_name == ""
        assert tool_use_id == ""

        role2, content2, tool_name2, tool_use_id2 = _extract_transcript_message_details(
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "read_file",
                            "tool_use_id": "toolu_2",
                            "input": {"path": "/tmp/a.txt"},
                        }
                    ],
                }
            }
        )
        assert role2 == "assistant"
        assert content2.startswith("[tool_use]")
        assert tool_name2 == "read_file"
        assert tool_use_id2 == "toolu_2"

    def test_apply_tool_result_details_prefers_existing_and_fills_missing(self) -> None:
        parsed = {"tool_use_id": "toolu_99", "tool_name": "search", "content": "OK"}
        content, tool_name, tool_use_id = _apply_tool_result_details(
            parsed,
            content="",
            tool_name="",
            tool_use_id="",
        )
        assert content == "OK"
        assert tool_name == "search"
        assert tool_use_id == "toolu_99"

        parsed2 = {"tool_use_id": 123, "tool_name": 456, "content": "ignored"}
        content2, tool_name2, tool_use_id2 = _apply_tool_result_details(
            parsed2,
            content="already",
            tool_name="existing",
            tool_use_id="toolu_existing",
        )
        assert content2 == "already"
        assert tool_name2 == "existing"
        assert tool_use_id2 == "toolu_existing"

    def test_extract_transcript_details_tool_result_path(self) -> None:
        parsed = {
            "tool_use_id": "toolu_1",
            "tool_name": "search",
            "content": "RESULT",
            "message": {"role": "assistant", "content": ""},
        }
        role, content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert role == "assistant"
        assert content == "RESULT"
        assert tool_name == "search"
        assert tool_use_id == "toolu_1"

    def test_apply_transcript_content_fallbacks_covers_entry_types(self) -> None:
        assert (
            _apply_transcript_content_fallbacks(
                {"thinking": "Thinking"}, "thinking", ""
            )
            == "Thinking"
        )
        assert (
            _apply_transcript_content_fallbacks({"summary": "Summary"}, "summary", "")
            == "Summary"
        )
        assert (
            _apply_transcript_content_fallbacks({"details": "Details"}, "system", "")
            == "Details"
        )
        assert (
            _apply_transcript_content_fallbacks({"event": "Event"}, "system", "")
            == "Event"
        )
        assert (
            _apply_transcript_content_fallbacks({"content": "Content"}, "assistant", "")
            == "Content"
        )
        assert (
            _apply_transcript_content_fallbacks({}, "assistant", "") == "{}"
        )  # Fallback to stringify(parsed)

        assert (
            _apply_transcript_content_fallbacks(
                {"content": "Ignored"}, "assistant", "ok"
            )
            == "ok"
        )

    def test_apply_notification_item_details_extracts_codex_fields(self) -> None:
        """Codex notification.item fields are extracted."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "commandExecution",
                    "id": "call_abc123",
                    "command": "/bin/zsh -lc 'ls'",
                    "aggregatedOutput": "file1.py\nfile2.py",
                }
            }
        }
        tool_use_id, tool_name, content = _apply_notification_item_details(
            parsed, tool_use_id="", tool_name="", content=""
        )
        assert tool_use_id == "call_abc123"
        assert tool_name == "/bin/zsh -lc 'ls'"
        assert content == "file1.py\nfile2.py"

    def test_apply_notification_item_details_uses_type_when_no_command(self) -> None:
        """Falls back to item type when no command field."""
        parsed: dict[str, object] = {
            "notification": {"item": {"type": "fileChange", "id": "call_xyz"}}
        }
        tool_use_id, tool_name, _content = _apply_notification_item_details(
            parsed, tool_use_id="", tool_name="", content=""
        )
        assert tool_use_id == "call_xyz"
        assert tool_name == "fileChange"

    def test_apply_notification_item_details_preserves_existing(self) -> None:
        """Existing values are not overwritten."""
        parsed: dict[str, object] = {
            "notification": {"item": {"id": "new_id", "command": "new_cmd"}}
        }
        tool_use_id, tool_name, content = _apply_notification_item_details(
            parsed, tool_use_id="existing", tool_name="existing", content="existing"
        )
        assert tool_use_id == "existing"
        assert tool_name == "existing"
        assert content == "existing"

    def test_apply_notification_item_details_no_notification(self) -> None:
        """Returns defaults when no notification key."""
        tool_use_id, tool_name, content = _apply_notification_item_details(
            {}, tool_use_id="", tool_name="", content=""
        )
        assert tool_use_id == ""
        assert tool_name == ""
        assert content == ""

    def test_apply_notification_item_details_non_mapping_item(self) -> None:
        """Returns defaults when notification.item is not a mapping."""
        parsed: dict[str, object] = {"notification": {"item": "not-a-dict"}}
        tool_use_id, tool_name, content = _apply_notification_item_details(
            parsed, tool_use_id="", tool_name="", content=""
        )
        assert tool_use_id == ""
        assert tool_name == ""
        assert content == ""

    def test_extract_transcript_parsed_obj_parses_raw_json(self) -> None:
        assert _extract_transcript_parsed_obj({}, None) is None
        assert _extract_transcript_parsed_obj({}, "{") is None
        assert _extract_transcript_parsed_obj({}, '{"a": 1}') == {"a": 1}
        assert _extract_transcript_parsed_obj({}, '["x"]') is None

    def test_extract_transcript_parsed_obj_unwraps_detail_sdk_entry(self) -> None:
        """Unified format: detail.sdk_entry is unwrapped."""
        sdk = {"type": "user", "message": {"role": "user", "content": "Hi"}}
        ctx: dict[str, object] = {"detail": {"sdk_entry": sdk}}
        assert _extract_transcript_parsed_obj(ctx, None) == sdk

    def test_extract_transcript_parsed_obj_uses_detail_directly(self) -> None:
        """detail without sdk_entry is returned as-is (e.g. Codex)."""
        detail: dict[str, object] = {"text": "hello"}
        ctx: dict[str, object] = {"detail": detail}
        assert _extract_transcript_parsed_obj(ctx, None) == detail

    def test_extract_transcript_row_ignores_bad_context(self) -> None:
        assert (
            _extract_transcript_row(
                {"event": "transcript.entry", "context": "not-a-mapping"}
            )
            is None
        )

    def test_extract_transcript_row_rejects_legacy_event(self) -> None:
        """Old event name is no longer supported."""
        assert (
            _extract_transcript_row(
                {
                    "event": "transcript.collector.entry",
                    "context": {"source": "main", "entry_type": "user_message"},
                }
            )
            is None
        )


class TestIsCacheValid:
    """Tests for _is_cache_valid function."""

    def test_cache_not_exists(self, tmp_path: Path) -> None:
        from weakincentives.cli.query import _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_newer_but_no_schema_version(self, tmp_path: Path) -> None:
        """Cache without schema version table is invalid."""
        import sqlite3

        from weakincentives.cli.query import _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        # Create empty SQLite file (no schema version table)
        conn = sqlite3.connect(cache)
        conn.close()
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_with_old_schema_version(self, tmp_path: Path) -> None:
        """Cache with old schema version is invalid."""
        import sqlite3

        from weakincentives.cli.query import _SCHEMA_VERSION, _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        conn = sqlite3.connect(cache)
        conn.execute("CREATE TABLE _schema_version (version INTEGER)")
        conn.execute("INSERT INTO _schema_version VALUES (?)", (_SCHEMA_VERSION - 1,))
        conn.commit()
        conn.close()
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_with_current_schema_version(self, tmp_path: Path) -> None:
        """Cache with current schema version is valid."""
        import sqlite3

        from weakincentives.cli.query import _SCHEMA_VERSION, _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        conn = sqlite3.connect(cache)
        conn.execute("CREATE TABLE _schema_version (version INTEGER)")
        conn.execute("INSERT INTO _schema_version VALUES (?)", (_SCHEMA_VERSION,))
        conn.commit()
        conn.close()
        assert _is_cache_valid(bundle, cache) is True

    def test_cache_corrupted_file(self, tmp_path: Path) -> None:
        """Corrupted cache file is invalid."""
        from weakincentives.cli.query import _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        # Write invalid data that's not a SQLite file
        cache.write_text("not a valid sqlite database")
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_older(self, tmp_path: Path) -> None:
        import time

        from weakincentives.cli.query import _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        cache = tmp_path / "bundle.zip.sqlite"
        cache.touch()
        time.sleep(0.01)  # Ensure different mtime
        bundle.touch()
        assert _is_cache_valid(bundle, cache) is False


class TestExtractTranscriptDetailsSplitToolUse:
    """Tests for top-level tool_use block extraction in _extract_transcript_details."""

    def test_top_level_tool_use_extracts_name_and_id(self) -> None:
        """Split tool_use entries have parsed as the block itself."""
        parsed: dict[str, object] = {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "read_file",
            "input": {"path": "/tmp/a.txt"},
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_name == "read_file"
        assert tool_use_id == "toolu_abc"

    def test_top_level_tool_use_with_tool_use_id_key(self) -> None:
        """tool_use_id key is also accepted (alternative to id)."""
        parsed: dict[str, object] = {
            "type": "tool_use",
            "tool_use_id": "toolu_xyz",
            "name": "search",
            "input": {},
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_name == "search"
        assert tool_use_id == "toolu_xyz"

    def test_skipped_when_tool_name_already_found(self) -> None:
        """If message-based extraction already found tool info, skip top-level."""
        parsed: dict[str, object] = {
            "type": "tool_use",
            "id": "toolu_new",
            "name": "new_tool",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "existing_tool", "id": "toolu_old"}
                ],
            },
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        # Message-based extraction found the tool, so top-level is skipped
        assert tool_name == "existing_tool"
        assert tool_use_id == "toolu_old"

    def test_non_tool_use_type_not_extracted(self) -> None:
        """Non tool_use entry_type does not trigger extraction."""
        parsed: dict[str, object] = {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "read_file",
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "assistant_message"
        )
        assert tool_name == ""
        assert tool_use_id == ""

    def test_non_string_name_and_id_ignored(self) -> None:
        """Non-string name/id values are not extracted."""
        parsed: dict[str, object] = {
            "type": "tool_use",
            "id": 12345,
            "name": None,
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_name == ""
        assert tool_use_id == ""


class TestApplySplitBlockDetails:
    """Direct tests for _apply_split_block_details."""

    def test_tool_result_extracts_id_and_content(self) -> None:
        """Exercises the tool_result branch directly."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_split",
            "content": "result text",
        }
        content, tool_name, tool_use_id = _apply_split_block_details(
            parsed,
            "tool_result",
            content="",
            tool_name="",
            tool_use_id="",
        )
        assert tool_use_id == "toolu_split"
        assert content == "result text"
        assert tool_name == ""

    def test_tool_result_skips_when_id_already_set(self) -> None:
        """Skips tool_result extraction when tool_use_id already populated."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_new",
            "content": "new",
        }
        content, _tool_name, tool_use_id = _apply_split_block_details(
            parsed,
            "tool_result",
            content="existing",
            tool_name="",
            tool_use_id="toolu_existing",
        )
        assert tool_use_id == "toolu_existing"
        assert content == "existing"

    def test_tool_result_non_string_content_skipped(self) -> None:
        """Non-string content in tool_result block is not extracted."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": ["complex", "content"],
        }
        content, _tool_name, tool_use_id = _apply_split_block_details(
            parsed,
            "tool_result",
            content="",
            tool_name="",
            tool_use_id="",
        )
        assert tool_use_id == "toolu_1"
        assert content == ""


class TestExtractTranscriptDetailsSplitToolResult:
    """Tests for top-level tool_result block extraction."""

    def test_top_level_tool_result_extracts_id_and_content(self) -> None:
        """Split tool_result entries have parsed as the block itself."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_abc",
            "content": "file contents",
        }
        _role, content, _tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert tool_use_id == "toolu_abc"
        assert content == "file contents"

    def test_non_string_id_ignored(self) -> None:
        """Non-string tool_use_id is not extracted."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": 12345,
            "content": "result",
        }
        _role, _content, _tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert tool_use_id == ""

    def test_skipped_when_already_found(self) -> None:
        """If _apply_tool_result_details already found tool_use_id, skip."""
        parsed: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_new",
            "content": "new_content",
        }
        _role, _content, _tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        # _apply_tool_result_details runs first and picks up tool_use_id
        assert tool_use_id == "toolu_new"


class TestExtractTranscriptParsedObjSplitToolUse:
    """Tests for _extract_transcript_parsed_obj with split tool_use sdk_entry."""

    def test_unwraps_tool_use_block_from_sdk_entry(self) -> None:
        """Split tool_use entry has sdk_entry as the tool_use block."""
        block: dict[str, object] = {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "read_file",
            "input": {"path": "/a.txt"},
        }
        ctx: dict[str, object] = {"detail": {"sdk_entry": block}}
        result = _extract_transcript_parsed_obj(ctx, None)
        assert result == block
        assert result is not None
        assert result.get("name") == "read_file"

    def test_unwraps_tool_result_block_from_sdk_entry(self) -> None:
        """Split tool_result entry has sdk_entry as the tool_result block."""
        block: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "result text",
        }
        ctx: dict[str, object] = {"detail": {"sdk_entry": block}}
        result = _extract_transcript_parsed_obj(ctx, None)
        assert result == block
        assert result is not None
        assert result.get("tool_use_id") == "toolu_1"


class TestP1AgentMessageNotTreatedAsToolCall:
    """P1: assistant_message entries from Codex agentMessage must NOT extract tool fields."""

    def test_assistant_message_with_agent_notification_no_tool_fields(self) -> None:
        """agentMessage notification should not contaminate tool_name/tool_use_id."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "agentMessage",
                    "id": "msg_abc123",
                    "text": "Hello from assistant",
                }
            }
        }
        _role, content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "assistant_message"
        )
        # Must NOT extract tool fields from agentMessage
        assert tool_name == ""
        assert tool_use_id == ""
        # Content should come from notification.item.text via fallback
        assert "Hello from assistant" in content

    def test_tool_use_with_notification_still_extracts(self) -> None:
        """tool_use entries with notification.item should still extract tool fields."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "commandExecution",
                    "id": "call_xyz",
                    "command": "ls -la",
                    "aggregatedOutput": "file1\nfile2",
                }
            }
        }
        _role, content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_use_id == "call_xyz"
        assert tool_name == "ls -la"
        assert content == "file1\nfile2"

    def test_tool_result_with_notification_still_extracts(self) -> None:
        """tool_result entries with notification.item should still extract tool fields."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "commandExecution",
                    "id": "call_xyz",
                    "command": "ls -la",
                    "aggregatedOutput": "file1\nfile2",
                }
            }
        }
        _role, content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert tool_use_id == "call_xyz"
        assert tool_name == "ls -la"
        assert content == "file1\nfile2"


class TestP2aNotificationItemTextFallback:
    """P2a: assistant_message content extracted from notification.item.text."""

    def test_content_fallback_reads_notification_item_text(self) -> None:
        """_apply_transcript_content_fallbacks extracts notification.item.text."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "agentMessage",
                    "id": "msg_1",
                    "text": "The assistant said something",
                }
            }
        }
        result = _apply_transcript_content_fallbacks(parsed, "assistant_message", "")
        assert result == "The assistant said something"

    def test_content_fallback_skips_empty_text(self) -> None:
        """Empty text in notification.item falls through to stringify."""
        parsed: dict[str, object] = {
            "notification": {
                "item": {
                    "type": "agentMessage",
                    "id": "msg_2",
                    "text": "",
                }
            }
        }
        result = _apply_transcript_content_fallbacks(parsed, "assistant_message", "")
        # Falls through to _stringify_transcript_content(parsed)
        assert result  # Some stringified representation


class TestP2bBridgedToolDetails:
    """P2b: bridged WINK tool events (no .item key) should extract tool metadata."""

    def test_apply_bridged_tool_details_extracts_tool_and_call_id(self) -> None:
        """Bridged notification with tool/callId is extracted."""
        notification: dict[str, object] = {
            "tool": "read_file",
            "callId": "call_bridge_1",
            "arguments": {"path": "/tmp/a.txt"},
        }
        parsed: dict[str, object] = {"notification": notification}
        tool_use_id, tool_name, _content = _apply_bridged_tool_details(
            parsed,
            notification,
            tool_use_id="",
            tool_name="",
            content="",
        )
        assert tool_use_id == "call_bridge_1"
        assert tool_name == "read_file"

    def test_apply_bridged_tool_details_extracts_result_string(self) -> None:
        """Bridged tool_result with string result in parsed.result."""
        notification: dict[str, object] = {
            "tool": "read_file",
            "callId": "call_bridge_2",
        }
        parsed: dict[str, object] = {
            "notification": notification,
            "result": "file contents here",
        }
        _tool_use_id, _tool_name, content = _apply_bridged_tool_details(
            parsed,
            notification,
            tool_use_id="",
            tool_name="",
            content="",
        )
        assert content == "file contents here"

    def test_apply_bridged_tool_details_extracts_result_dict(self) -> None:
        """Bridged tool_result with dict result is JSON-dumped."""
        notification: dict[str, object] = {
            "tool": "search",
            "callId": "call_bridge_3",
        }
        parsed: dict[str, object] = {
            "notification": notification,
            "result": {"matches": 5},
        }
        _tool_use_id, _tool_name, content = _apply_bridged_tool_details(
            parsed,
            notification,
            tool_use_id="",
            tool_name="",
            content="",
        )
        assert '"matches"' in content

    def test_apply_bridged_tool_details_preserves_existing(self) -> None:
        """Existing values are not overwritten."""
        notification: dict[str, object] = {
            "tool": "new_tool",
            "callId": "new_id",
        }
        parsed: dict[str, object] = {
            "notification": notification,
            "result": "new_result",
        }
        tool_use_id, tool_name, content = _apply_bridged_tool_details(
            parsed,
            notification,
            tool_use_id="existing_id",
            tool_name="existing_tool",
            content="existing_content",
        )
        assert tool_use_id == "existing_id"
        assert tool_name == "existing_tool"
        assert content == "existing_content"

    def test_notification_item_details_delegates_to_bridged(self) -> None:
        """_apply_notification_item_details delegates to bridged when no .item."""
        parsed: dict[str, object] = {
            "notification": {
                "tool": "write_file",
                "callId": "call_bridge_4",
                "arguments": {"path": "/tmp/b.txt"},
            }
        }
        tool_use_id, tool_name, _content = _apply_notification_item_details(
            parsed, tool_use_id="", tool_name="", content=""
        )
        assert tool_use_id == "call_bridge_4"
        assert tool_name == "write_file"

    def test_end_to_end_bridged_tool_use(self) -> None:
        """End-to-end: _extract_transcript_details with bridged tool_use entry."""
        parsed: dict[str, object] = {
            "notification": {
                "tool": "bash",
                "callId": "call_bridge_5",
                "arguments": {"command": "echo hello"},
            }
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_name == "bash"
        assert tool_use_id == "call_bridge_5"

    def test_end_to_end_bridged_tool_result(self) -> None:
        """End-to-end: _extract_transcript_details with bridged tool_result entry."""
        parsed: dict[str, object] = {
            "notification": {
                "tool": "bash",
                "callId": "call_bridge_6",
            },
            "result": "hello\n",
        }
        _role, content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert tool_name == "bash"
        assert tool_use_id == "call_bridge_6"
        assert content == "hello\n"
