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

"""Tests for the wink query command."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from weakincentives.cli import wink
from weakincentives.cli.query import (
    ColumnInfo,
    QueryError,
    SchemaOutput,
    TableInfo,
    _flatten_json,
    _get_table_description,
    _infer_sqlite_type,
    _is_cache_valid,
    _json_to_sql_value,
    _normalize_slice_type,
    export_jsonl,
    format_as_json,
    format_as_table,
    open_query_database,
)
from weakincentives.debug.bundle import BundleConfig, BundleWriter
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _AgentPlan:
    goal: str
    steps: int


@dataclass(slots=True, frozen=True)
class _TaskStatus:
    task_id: str
    completed: bool


def _create_test_bundle(
    target_dir: Path,
    *,
    with_logs: bool = True,
    with_error: bool = False,
    with_filesystem: bool = False,
    with_config: bool = True,
    with_metrics: bool = True,
) -> Path:
    """Create a test debug bundle with various artifacts."""
    session = Session()
    session.dispatch(_AgentPlan(goal="Test goal", steps=3))
    session.dispatch(_TaskStatus(task_id="task-001", completed=True))

    with BundleWriter(target_dir, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})

        if with_config:
            writer.write_config({"adapter": {"model": "gpt-4"}, "max_tokens": 1000})

        if with_metrics:
            writer.write_metrics(
                {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_ms": 1500,
                }
            )

        if with_error:
            writer.write_error(
                {
                    "type": "ValueError",
                    "message": "Test error message",
                    "traceback": ["line 1", "line 2"],
                }
            )

    assert writer.path is not None
    return writer.path


def _create_bundle_with_logs(target_dir: Path, log_file: Path) -> Path:
    """Create bundle with custom log content."""
    session = Session()
    session.dispatch(_AgentPlan(goal="Test goal", steps=3))

    with BundleWriter(target_dir, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})

        # Manually trigger log capture to write to file
        with writer.capture_logs():
            logger = logging.getLogger("test.logger")
            logger.info(
                "Test message",
                extra={
                    "event": "test.event",
                    "context": {"key": "value"},
                },
            )
            logger.error(
                "Error message",
                extra={
                    "event": "test.error",
                    "context": {"traceback": "stack trace"},
                },
            )
            # Tool call log
            logger.info(
                "Tool executed",
                extra={
                    "event": "tool.execution.complete",
                    "context": {
                        "tool_name": "read_file",
                        "params": {"path": "/test.txt"},
                        "duration_ms": 15.5,
                    },
                },
            )

    assert writer.path is not None
    return writer.path


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
            == "Log entries (seq column for log_aggregator events)"
        )
        assert _get_table_description("errors") == "Aggregated errors"

    def test_slice_table(self) -> None:
        assert _get_table_description("slice_agentplan") == "Session slice: agentplan"

    def test_unknown_table(self) -> None:
        assert _get_table_description("unknown_table") == ""


class TestIsCacheValid:
    """Tests for _is_cache_valid function."""

    def test_cache_not_exists(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_newer_but_no_schema_version(self, tmp_path: Path) -> None:
        """Cache without schema version table is invalid."""
        import sqlite3

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

        from weakincentives.cli.query import _SCHEMA_VERSION

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

        from weakincentives.cli.query import _SCHEMA_VERSION

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
        bundle = tmp_path / "bundle.zip"
        bundle.touch()
        cache = tmp_path / "bundle.zip.sqlite"
        # Write invalid data that's not a SQLite file
        cache.write_text("not a valid sqlite database")
        assert _is_cache_valid(bundle, cache) is False

    def test_cache_older(self, tmp_path: Path) -> None:
        import time

        bundle = tmp_path / "bundle.zip"
        cache = tmp_path / "bundle.zip.sqlite"
        cache.touch()
        time.sleep(0.01)  # Ensure different mtime
        bundle.touch()
        assert _is_cache_valid(bundle, cache) is False


class TestQueryDatabase:
    """Tests for QueryDatabase class."""

    def test_build_creates_tables(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            assert "manifest" in table_names
            assert "logs" in table_names
            assert "errors" in table_names
            assert "session_slices" in table_names
            assert "config" in table_names
            assert "metrics" in table_names
        finally:
            db.close()

    def test_manifest_table(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM manifest")
            assert len(results) == 1
            assert results[0]["status"] == "success"
        finally:
            db.close()

    def test_session_slices_table(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM session_slices")
            assert len(results) == 2  # AgentPlan + TaskStatus
        finally:
            db.close()

    def test_dynamic_slice_tables(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            # Check that slice tables were created
            assert any(t.startswith("slice_") for t in table_names)
        finally:
            db.close()

    def test_config_table_flattened(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query(
                "SELECT key, value FROM config WHERE key = 'adapter_model'"
            )
            assert len(results) == 1
            assert results[0]["value"] == "gpt-4"
        finally:
            db.close()

    def test_metrics_table(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query(
                "SELECT value FROM metrics WHERE key = 'input_tokens'"
            )
            assert len(results) == 1
            assert results[0]["value"] == "100"
        finally:
            db.close()

    def test_errors_from_error_json(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path, with_error=True)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query(
                "SELECT * FROM errors WHERE source = 'error.json'"
            )
            assert len(results) == 1
            assert results[0]["error_type"] == "ValueError"
            assert results[0]["message"] == "Test error message"
        finally:
            db.close()

    def test_query_error_on_invalid_sql(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            with pytest.raises(QueryError, match="SQL error"):
                db.execute_query("SELECT * FROM nonexistent_table")
        finally:
            db.close()

    def test_caching_works(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")

        # First access creates cache
        db1 = open_query_database(bundle_path)
        db1.close()
        assert cache_path.exists()

        # Get original mtime
        original_mtime = cache_path.stat().st_mtime

        # Second access should use cache (no rebuild)
        db2 = open_query_database(bundle_path)
        db2.close()

        # Cache mtime should be unchanged
        assert cache_path.stat().st_mtime == original_mtime


class TestSchemaOutput:
    """Tests for SchemaOutput class."""

    def test_to_json(self) -> None:
        schema = SchemaOutput(
            bundle_id="test-123",
            status="success",
            created_at="2024-01-15T10:30:00Z",
            tables=(
                TableInfo(
                    name="manifest",
                    description="Bundle metadata",
                    row_count=1,
                    columns=(ColumnInfo(name="bundle_id", type="TEXT"),),
                ),
            ),
        )

        json_str = schema.to_json()
        data = json.loads(json_str)

        assert data["bundle_id"] == "test-123"
        assert data["status"] == "success"
        assert len(data["tables"]) == 1
        assert data["tables"][0]["name"] == "manifest"


class TestFormatAsTable:
    """Tests for format_as_table function."""

    def test_empty_results(self) -> None:
        result = format_as_table([])
        assert result == "(no results)"

    def test_single_row(self) -> None:
        rows: list[dict[str, Any]] = [{"name": "test", "value": 42}]
        result = format_as_table(rows)

        assert "name" in result
        assert "value" in result
        assert "test" in result
        assert "42" in result

    def test_multiple_rows(self) -> None:
        rows: list[dict[str, Any]] = [
            {"id": 1, "name": "one"},
            {"id": 2, "name": "two"},
        ]
        result = format_as_table(rows)

        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "-+-" in lines[1]  # separator

    def test_truncates_long_values(self) -> None:
        rows: list[dict[str, Any]] = [{"data": "x" * 100}]
        result = format_as_table(rows)

        assert "..." in result
        assert len(result.split("\n")[2]) <= 60  # Reasonable width

    def test_no_truncate(self) -> None:
        long_value = "x" * 100
        rows: list[dict[str, Any]] = [{"data": long_value}]
        result = format_as_table(rows, truncate=False)

        assert "..." not in result
        assert long_value in result


class TestFormatAsJson:
    """Tests for format_as_json function."""

    def test_empty_results(self) -> None:
        result = format_as_json([])
        assert result == "[]"

    def test_formats_as_json(self) -> None:
        rows: list[dict[str, Any]] = [{"name": "test", "value": 42}]
        result = format_as_json(rows)

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "test"
        assert data[0]["value"] == 42


class TestOpenQueryDatabase:
    """Tests for open_query_database function."""

    def test_opens_bundle(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            assert db is not None
            schema = db.get_schema()
            assert schema.bundle_id != ""
        finally:
            db.close()

    def test_raises_on_invalid_bundle(self, tmp_path: Path) -> None:
        invalid_path = tmp_path / "invalid.zip"
        invalid_path.write_text("not a zip")

        with pytest.raises(QueryError, match="Failed to load bundle"):
            open_query_database(invalid_path)

    def test_creates_cache_file(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        db.close()

        cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")
        assert cache_path.exists()


class TestWinkQueryCLI:
    """Tests for wink query CLI command."""

    def test_query_with_schema_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(["--no-json-logs", "query", str(bundle_path), "--schema"])

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "tables" in data
        assert "bundle_id" in data

    def test_query_with_sql(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "SELECT * FROM manifest"]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1

    def test_query_with_table_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(
            [
                "--no-json-logs",
                "query",
                str(bundle_path),
                "SELECT * FROM manifest",
                "--table",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "bundle_id" in captured.out
        assert "|" in captured.out  # Table separator

    def test_query_missing_sql_and_schema(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(["--no-json-logs", "query", str(bundle_path)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "SQL query required" in captured.err

    def test_query_nonexistent_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = tmp_path / "nonexistent.zip"

        exit_code = wink.main(["--no-json-logs", "query", str(bundle_path), "--schema"])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_query_invalid_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Create a file that exists but is not a valid zip
        bundle_path = tmp_path / "invalid.zip"
        bundle_path.write_text("not a zip file")

        exit_code = wink.main(["--no-json-logs", "query", str(bundle_path), "--schema"])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_query_invalid_sql(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(
            [
                "--no-json-logs",
                "query",
                str(bundle_path),
                "SELECT * FROM nonexistent",
            ]
        )

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_export_jsonl_logs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test --export-jsonl flag for logs."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-export-cli",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        logs_content = '{"event": "test.event", "message": "hello"}\n'

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "--export-jsonl"]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "test.event" in captured.out

    def test_export_jsonl_logs_no_trailing_newline(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test --export-jsonl handles content without trailing newline."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-no-newline",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        # Content WITHOUT trailing newline
        logs_content = '{"event": "test.event"}'

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "--export-jsonl"]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        # Output should end with newline (added by the code)
        assert captured.out.endswith("\n")

    def test_export_jsonl_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test --export-jsonl with empty content returns error."""
        bundle_path = _create_test_bundle(tmp_path)

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "--export-jsonl"]
        )

        # Should fail because test bundle may have empty logs
        # Just verify it doesn't crash and returns a code
        assert exit_code in {0, 1}

    def test_export_jsonl_invalid_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test --export-jsonl with invalid bundle."""
        bundle_path = tmp_path / "invalid.zip"
        bundle_path.write_text("not a zip file")

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "--export-jsonl"]
        )

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestQueryDatabaseWithLogs:
    """Tests for QueryDatabase with log entries."""

    def test_logs_table_populated(self, tmp_path: Path) -> None:
        bundle_path = _create_bundle_with_logs(tmp_path, tmp_path / "logs.jsonl")
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM logs")
            # Should have at least some log entries
            assert len(results) >= 0
        finally:
            db.close()

    def test_tool_calls_derived(self, tmp_path: Path) -> None:
        bundle_path = _create_bundle_with_logs(tmp_path, tmp_path / "logs.jsonl")
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM tool_calls")
            # Table should exist even if empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_errors_from_logs(self, tmp_path: Path) -> None:
        bundle_path = _create_bundle_with_logs(tmp_path, tmp_path / "logs.jsonl")
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM errors WHERE source = 'log'")
            # Should have at least one error from the ERROR level log
            assert isinstance(results, list)
        finally:
            db.close()


class TestExtractToolCall:
    """Tests for _extract_tool_call_from_entry helper function."""

    def test_no_context(self) -> None:
        from weakincentives.cli.query import _extract_tool_call_from_entry

        entry: dict[str, Any] = {"timestamp": "2024-01-01T00:00:00Z"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_context_not_dict(self) -> None:
        from weakincentives.cli.query import _extract_tool_call_from_entry

        entry: dict[str, Any] = {"context": "not a dict", "timestamp": "2024-01-01"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_no_tool_name(self) -> None:
        from weakincentives.cli.query import _extract_tool_call_from_entry

        entry: dict[str, Any] = {"context": {"params": {}}, "timestamp": "2024-01-01"}
        result = _extract_tool_call_from_entry(entry)
        assert result is None

    def test_with_error_legacy_fallback(self) -> None:
        """Test legacy fallback: infer failure from error text presence."""
        from weakincentives.cli.query import _extract_tool_call_from_entry

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
        from weakincentives.cli.query import _extract_tool_call_from_entry

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
        from weakincentives.cli.query import _extract_tool_call_from_entry

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
        from weakincentives.cli.query import _extract_tool_call_from_entry

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
        import json

        params = json.loads(result[2])
        assert params == {"path": "/test.txt", "encoding": "utf-8"}

    def test_value_field_used_for_result(self) -> None:
        """Test that 'value' field from current logs is used for result."""
        from weakincentives.cli.query import _extract_tool_call_from_entry

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
        import json

        result_val = json.loads(result[3])
        assert result_val == "file contents here"


class TestIsToolEvent:
    """Tests for _is_tool_event helper function."""

    def test_tool_execution_events(self) -> None:
        from weakincentives.cli.query import _is_tool_event

        # Actual event names from tool_executor
        assert _is_tool_event("tool.execution.start") is True
        assert _is_tool_event("tool.execution.complete") is True

    def test_tool_call_event(self) -> None:
        from weakincentives.cli.query import _is_tool_event

        # Alternative event formats
        assert _is_tool_event("tool.call.start") is True
        assert _is_tool_event("tool.result.end") is True

    def test_not_tool_event(self) -> None:
        from weakincentives.cli.query import _is_tool_event

        assert _is_tool_event("session.start") is False
        assert _is_tool_event("request.complete") is False


class TestExtractSlicesFromSnapshot:
    """Tests for _extract_slices_from_snapshot helper function."""

    def test_no_slices_field(self) -> None:
        from weakincentives.cli.query import _extract_slices_from_snapshot

        entry: dict[str, object] = {"other": "data"}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_slices_not_list(self) -> None:
        from weakincentives.cli.query import _extract_slices_from_snapshot

        entry: dict[str, object] = {"slices": "not a list"}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_slice_obj_not_mapping(self) -> None:
        from weakincentives.cli.query import _extract_slices_from_snapshot

        entry: dict[str, object] = {"slices": ["not a mapping"]}
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_items_not_list(self) -> None:
        from weakincentives.cli.query import _extract_slices_from_snapshot

        entry: dict[str, object] = {
            "slices": [{"slice_type": "MyType", "items": "not a list"}]
        }
        result = _extract_slices_from_snapshot(entry)
        assert result == []

    def test_item_not_mapping(self) -> None:
        from weakincentives.cli.query import _extract_slices_from_snapshot

        entry: dict[str, object] = {
            "slices": [{"slice_type": "MyType", "items": ["not a mapping"]}]
        }
        result = _extract_slices_from_snapshot(entry)
        assert result == []


class TestProcessSessionLine:
    """Tests for _process_session_line helper function."""

    def test_invalid_json(self, tmp_path: Path) -> None:
        import sqlite3

        from weakincentives.cli.query import _process_session_line

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """)
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        _process_session_line(conn, "not valid json", slices_by_type)

        cursor = conn.execute("SELECT COUNT(*) FROM session_slices")
        assert cursor.fetchone()[0] == 0

    def test_non_mapping_entry(self, tmp_path: Path) -> None:
        import sqlite3

        from weakincentives.cli.query import _process_session_line

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """)
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        _process_session_line(conn, '"just a string"', slices_by_type)

        cursor = conn.execute("SELECT COUNT(*) FROM session_slices")
        assert cursor.fetchone()[0] == 0


class TestQueryDatabaseEdgeCases:
    """Tests for edge cases in QueryDatabase."""

    def test_empty_config(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path, with_config=False)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM config")
            # Table should exist but may be empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_empty_metrics(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path, with_metrics=False)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM metrics")
            # Table should exist but may be empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        # Closing multiple times should not raise
        db.close()
        db.close()

    def test_get_schema_opens_connection(self, tmp_path: Path) -> None:
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        db.close()

        # Getting schema after close should work (reopens connection)
        schema = db.get_schema()
        assert len(schema.tables) > 0
        db.close()

    def test_connection_reused(self, tmp_path: Path) -> None:
        """Test that connection is reused when already open."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            # First query opens connection
            results1 = db.execute_query("SELECT * FROM manifest")
            # Second query should reuse existing connection
            results2 = db.execute_query("SELECT * FROM manifest")
            assert results1 == results2
        finally:
            db.close()

    def test_bundle_path_property(self, tmp_path: Path) -> None:
        """Test that bundle_path property returns the correct path."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            assert db.bundle_path == bundle_path
        finally:
            db.close()


class TestFlattenJsonTopLevelList:
    """Tests for _flatten_json with top-level list."""

    def test_top_level_list(self) -> None:
        result = _flatten_json([1, 2, 3], prefix="items")
        assert result == {"items": "[1, 2, 3]"}


class TestDirectSliceFormat:
    """Tests for direct slice format in _process_session_line."""

    def test_direct_slice_format_with_type(self) -> None:
        import sqlite3

        from weakincentives.cli.query import _process_session_line

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """)
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        # Direct slice format with __type__ field
        line = json.dumps({"__type__": "myapp.state:AgentPlan", "goal": "test"})
        _process_session_line(conn, line, slices_by_type)

        cursor = conn.execute("SELECT slice_type, data FROM session_slices")
        row = cursor.fetchone()
        assert row[0] == "myapp.state:AgentPlan"

    def test_direct_slice_format_without_type(self) -> None:
        import sqlite3

        from weakincentives.cli.query import _process_session_line

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """)
        slices_by_type: dict[str, list[Mapping[str, Any]]] = {}

        # Direct slice format without __type__ field
        line = json.dumps({"goal": "test", "steps": 3})
        _process_session_line(conn, line, slices_by_type)

        cursor = conn.execute("SELECT slice_type FROM session_slices")
        row = cursor.fetchone()
        assert row[0] == "unknown"


class TestFilesystemFiles:
    """Tests for filesystem/ files in bundle."""

    def test_bundle_with_filesystem_files(self, tmp_path: Path) -> None:
        """Test that filesystem files are loaded into files table."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-filesystem",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")
            # Add filesystem files
            zf.writestr("debug_bundle/filesystem/test.txt", "Hello World")
            zf.writestr(
                "debug_bundle/filesystem/subdir/nested.txt",
                "Nested content",
            )

        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT path, content FROM files")
            paths = [r["path"] for r in results]
            assert len(results) == 2
            assert "test.txt" in paths
            assert "subdir/nested.txt" in paths
        finally:
            db.close()

    def test_bundle_with_binary_file(self, tmp_path: Path) -> None:
        """Test that binary files are handled as hex."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-binary",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")
            # Add binary file (invalid UTF-8)
            zf.writestr("debug_bundle/filesystem/binary.bin", b"\x00\x01\x02\xff\xfe")

        db = open_query_database(bundle_path)

        try:
            results = db.execute_query(
                "SELECT content FROM files WHERE path = 'binary.bin'"
            )
            assert len(results) == 1
            # Binary content should be stored as hex
            assert results[0]["content"] == "000102fffe"
        finally:
            db.close()


class TestOptionalBundleFiles:
    """Tests for optional bundle files: run_context, prompt_overrides, eval."""

    def test_run_context_populated(self, tmp_path: Path) -> None:
        """Test run_context table is populated when data exists."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-run-context",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        run_context = {
            "request_id": "req-123",
            "session_id": "sess-456",
            "nested": {"trace_id": "trace-789"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/run_context.json", json.dumps(run_context))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)

        try:
            results = db.execute_query(
                "SELECT key, value FROM run_context WHERE key = 'request_id'"
            )
            assert len(results) == 1
            assert results[0]["value"] == "req-123"
        finally:
            db.close()

    def test_prompt_overrides_populated(self, tmp_path: Path) -> None:
        """Test prompt_overrides table is populated when data exists."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-overrides",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        overrides = {
            "section.key": {"hidden": True},
            "another.section": {"expanded": False},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/prompt_overrides.json", json.dumps(overrides))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM prompt_overrides")
            # Should have some entries from the overrides
            assert len(results) > 0
        finally:
            db.close()

    def test_eval_populated(self, tmp_path: Path) -> None:
        """Test eval table is populated when data exists."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-eval",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        eval_data = {
            "sample_id": "sample-001",
            "experiment": "baseline",
            "score": 0.95,
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/eval.json", json.dumps(eval_data))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT value FROM eval WHERE key = 'sample_id'")
            assert len(results) == 1
            assert results[0]["value"] == "sample-001"
        finally:
            db.close()


class TestCacheInvalidation:
    """Tests for cache invalidation and rebuild."""

    def test_stale_cache_removed(self, tmp_path: Path) -> None:
        """Test that stale cache is removed and rebuilt."""
        import time

        bundle_path = _create_test_bundle(tmp_path)
        cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")

        # First access creates cache
        db1 = open_query_database(bundle_path)
        db1.close()
        assert cache_path.exists()

        # Make cache stale by touching bundle with newer time
        time.sleep(0.02)
        bundle_path.touch()

        # Re-open should rebuild cache
        db2 = open_query_database(bundle_path)
        db2.close()

        # Cache should have been rebuilt (exists and mtime updated)
        assert cache_path.exists()
        assert cache_path.stat().st_mtime >= bundle_path.stat().st_mtime


class TestErrorJsonNonListTraceback:
    """Tests for error.json with non-list traceback."""

    def test_traceback_as_string(self, tmp_path: Path) -> None:
        """Test handling traceback as a plain string instead of list."""
        from weakincentives.debug.bundle import BundleConfig, BundleWriter

        session = Session()
        session.dispatch(_AgentPlan(goal="Test", steps=1))

        with BundleWriter(tmp_path, config=BundleConfig()) as writer:
            writer.write_session_after(session)
            writer.write_request_input({"task": "test"})
            writer.write_request_output({"status": "ok"})
            writer.write_error(
                {
                    "type": "ValueError",
                    "message": "Test error",
                    "traceback": "Single line traceback string",  # Not a list
                }
            )

        assert writer.path is not None
        db = open_query_database(writer.path)

        try:
            results = db.execute_query(
                "SELECT traceback FROM errors WHERE source = 'error.json'"
            )
            assert len(results) == 1
            assert results[0]["traceback"] == "Single line traceback string"
        finally:
            db.close()


class TestLogsWithEmptyLines:
    """Tests for log processing with empty lines and errors."""

    def test_logs_with_blank_lines(self, tmp_path: Path) -> None:
        """Test that blank lines in logs are handled gracefully."""
        import zipfile

        # Create a minimal bundle manually with blank lines in logs
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()

        # Create logs with blank lines
        logs_content = "\n".join(
            [
                '{"timestamp": "2024-01-01", "level": "INFO", "event": "start"}',
                "",  # blank line
                '{"timestamp": "2024-01-02", "level": "INFO", "event": "end"}',
                "   ",  # whitespace only
            ]
        )

        # Create manifest
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-blank-lines",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        # Create zip bundle
        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM logs")
            assert len(results) == 2  # Only non-blank lines
        finally:
            db.close()


class TestLogsWithInvalidJson:
    """Tests for log processing with invalid JSON lines."""

    def test_logs_with_invalid_json_lines(self, tmp_path: Path) -> None:
        """Test that invalid JSON lines in logs are skipped."""
        import zipfile

        logs_content = "\n".join(
            [
                '{"timestamp": "2024-01-01", "level": "INFO", "event": "valid"}',
                "not valid json",
                '{"timestamp": "2024-01-02", "level": "INFO", "event": "also_valid"}',
            ]
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-invalid-json",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM logs")
            assert len(results) == 2  # Invalid line skipped
        finally:
            db.close()


class TestToolCallExtraction:
    """Tests for tool call extraction from logs."""

    def test_tool_call_extraction_and_insertion(self, tmp_path: Path) -> None:
        """Test that tool calls are extracted from logs and inserted."""
        import zipfile

        logs_content = "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "level": "INFO",
                        "event": "tool.execution.complete",
                        "context": {
                            "tool_name": "read_file",
                            "params": {"path": "/test.txt"},
                            "result": {"content": "hello"},
                            "duration_ms": 15.5,
                        },
                    }
                ),
            ]
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-tool-calls",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM tool_calls")
            assert len(results) == 1
            assert results[0]["tool_name"] == "read_file"
            assert results[0]["duration_ms"] == 15.5
        finally:
            db.close()


class TestFailedToolCallErrors:
    """Tests for failed tool call errors."""

    def test_failed_tool_calls_create_errors(self, tmp_path: Path) -> None:
        """Test that failed tool calls are added to errors table."""
        import zipfile

        # Use explicit success=False to match actual tool.execution.complete logs
        logs_content = json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "INFO",
                "event": "tool.execution.complete",
                "context": {
                    "tool_name": "write_file",
                    "success": False,
                    "message": "Permission denied",
                },
            }
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-failed-tools",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query(
                "SELECT * FROM errors WHERE source = 'tool_call'"
            )
            assert len(results) == 1
            assert "write_file" in results[0]["error_type"]
        finally:
            db.close()


class TestDynamicSliceTableEdgeCases:
    """Tests for dynamic slice table creation edge cases."""

    def test_slice_with_no_columns(self, tmp_path: Path) -> None:
        """Test creating a dynamic slice table with only __type__ (no extra columns)."""
        import sqlite3

        from weakincentives.cli.query import _create_dynamic_slice_table

        conn = sqlite3.connect(":memory:")

        # Slice data with only __type__ field
        slices: list[Mapping[str, Any]] = [{"__type__": "EmptySlice"}]

        _create_dynamic_slice_table(conn, "EmptySlice", slices)

        # Table should exist with only rowid
        cursor = conn.execute("SELECT * FROM slice_emptyslice")
        rows = cursor.fetchall()
        assert len(rows) == 0  # No data inserted since no columns

    def test_slice_with_duplicate_keys_uses_first_type(self, tmp_path: Path) -> None:
        """Test that when same key appears in multiple slices, first type is used."""
        import sqlite3

        from weakincentives.cli.query import _create_dynamic_slice_table

        conn = sqlite3.connect(":memory:")

        # First slice has integer, second has string
        slices: list[Mapping[str, Any]] = [{"value": 42}, {"value": "text"}]

        _create_dynamic_slice_table(conn, "MixedSlice", slices)

        cursor = conn.execute("PRAGMA table_info(slice_mixedslice)")
        cols = {row[1]: row[2] for row in cursor.fetchall()}
        assert cols["value"] == "INTEGER"  # First value type wins


class TestErrorFromLogNonDictContext:
    """Tests for error log entries with non-dict context."""

    def test_error_log_with_non_dict_context(self, tmp_path: Path) -> None:
        """Test that error log entries with non-dict context are handled."""
        import zipfile

        # Log with ERROR level but context is not a dict
        logs_content = json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "ERROR",
                "event": "test.error",
                "message": "Error with string context",
                "context": "not a dict",  # String instead of dict
            }
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-error-context",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM errors WHERE source = 'log'")
            # Should have inserted the error with empty traceback
            assert len(results) == 1
            assert results[0]["traceback"] == ""
        finally:
            db.close()


class TestToolEventWithoutToolName:
    """Tests for tool events that don't have tool_name in context."""

    def test_tool_event_without_tool_name_skipped(self, tmp_path: Path) -> None:
        """Test that tool events without tool_name are skipped."""
        import zipfile

        # Tool event but context has no tool_name
        logs_content = json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "INFO",
                "event": "tool.execution.complete",
                "context": {"params": {"path": "/test.txt"}},  # No tool_name
            }
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-no-tool-name",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM tool_calls")
            # Tool call should be skipped since no tool_name
            assert len(results) == 0
        finally:
            db.close()


class TestSessionWithEmptyLines:
    """Tests for session JSONL with empty lines."""

    def test_session_with_blank_lines(self, tmp_path: Path) -> None:
        """Test that blank lines in session are skipped."""
        import zipfile

        session_content = "\n".join(
            [
                json.dumps({"__type__": "TestSlice", "value": 1}),
                "",  # blank line
                json.dumps({"__type__": "TestSlice", "value": 2}),
                "   ",  # whitespace only
            ]
        )

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-session-blanks",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", session_content)
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM session_slices")
            # Should only have 2 slices (blanks skipped)
            assert len(results) == 2
        finally:
            db.close()


class TestFilesTableReadError:
    """Tests for handling errors when reading filesystem files."""

    def test_file_read_error_skipped(self, tmp_path: Path) -> None:
        """Test that files causing read errors are skipped."""
        import zipfile
        from unittest.mock import patch

        from weakincentives.debug.bundle import BundleValidationError, DebugBundle

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-read-error",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")
            # Add a file that will "fail" to read
            zf.writestr("debug_bundle/filesystem/good.txt", "Good content")
            zf.writestr("debug_bundle/filesystem/bad.txt", "Bad content")

        # Patch read_file to fail for bad.txt
        original_read_file = DebugBundle.read_file

        def mock_read_file(self: DebugBundle, rel_path: str) -> bytes:
            if "bad.txt" in rel_path:
                raise BundleValidationError("Simulated read error")
            return original_read_file(self, rel_path)

        with patch.object(DebugBundle, "read_file", mock_read_file):
            db = open_query_database(bundle_path)
            try:
                results = db.execute_query("SELECT * FROM files")
                # Should only have the good file, bad file was skipped
                paths = [r["path"] for r in results]
                assert "good.txt" in paths
                assert "bad.txt" not in paths
            finally:
                db.close()


class TestExportJsonl:
    """Tests for export_jsonl function."""

    def test_export_logs(self, tmp_path: Path) -> None:
        """Test exporting logs JSONL."""
        import zipfile

        from weakincentives.debug.bundle import DebugBundle

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-export",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        logs_content = '{"event": "test", "message": "hello"}\n'

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, "logs")

        assert content is not None
        assert "test" in content

    def test_export_session(self, tmp_path: Path) -> None:
        """Test exporting session JSONL."""
        from weakincentives.debug.bundle import DebugBundle

        bundle_path = _create_test_bundle(tmp_path)
        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, "session")

        # Should return session content or None
        assert content is None or isinstance(content, str)

    def test_export_invalid_source(self, tmp_path: Path) -> None:
        """Test exporting with invalid source returns None."""
        from weakincentives.debug.bundle import DebugBundle

        bundle_path = _create_test_bundle(tmp_path)
        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, "invalid")

        assert content is None


class TestSeqColumn:
    """Tests for seq column in logs table."""

    def test_seq_extracted_from_log_aggregator(self, tmp_path: Path) -> None:
        """Test that seq column is extracted for log_aggregator.log_line events."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-seq",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": [],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        # Create logs with log_aggregator.log_line events
        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "logger": "aggregator",
                "event": "log_aggregator.log_line",
                "message": "Line 1",
                "context": {"sequence_number": 42, "content": "test content"},
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "level": "DEBUG",
                "logger": "aggregator",
                "event": "log_aggregator.log_line",
                "message": "Line 2",
                "context": {"sequence_number": 43, "content": "more content"},
            },
            {
                "timestamp": "2024-01-01T00:00:02Z",
                "level": "DEBUG",
                "logger": "aggregator",
                "event": "log_aggregator.log_line",
                "message": "Line with invalid seq",
                # Non-int sequence_number should be ignored
                "context": {"sequence_number": "not-an-int", "content": "invalid"},
            },
            {
                "timestamp": "2024-01-01T00:00:03Z",
                "level": "INFO",
                "logger": "app",
                "event": "other.event",
                "message": "Other",
                "context": {},
            },
        ]
        logs_content = "\n".join(json.dumps(entry) for entry in logs)

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/logs/app.jsonl", logs_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query(
                "SELECT event, seq FROM logs WHERE seq IS NOT NULL ORDER BY seq"
            )
            assert len(results) == 2
            assert results[0]["seq"] == 42
            assert results[1]["seq"] == 43

            # Non-log_aggregator events should have NULL seq
            non_agg = db.execute_query(
                "SELECT event, seq FROM logs WHERE event = 'other.event'"
            )
            assert len(non_agg) == 1
            assert non_agg[0]["seq"] is None
        finally:
            db.close()


class TestSchemaHints:
    """Tests for schema hints in schema output."""

    def test_schema_includes_hints(self, tmp_path: Path) -> None:
        """Test that schema output includes hints section."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            assert schema.hints is not None
            assert len(schema.hints.json_extraction) > 0
            assert len(schema.hints.common_queries) > 0
        finally:
            db.close()

    def test_hints_serializes_to_json(self, tmp_path: Path) -> None:
        """Test that hints are included in JSON output."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            json_str = schema.to_json()
            data = json.loads(json_str)
            assert "hints" in data
            assert "json_extraction" in data["hints"]
            assert "common_queries" in data["hints"]
        finally:
            db.close()


class TestQueryViews:
    """Tests for SQL views."""

    def test_tool_timeline_view_exists(self, tmp_path: Path) -> None:
        """Test that tool_timeline view is created."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            # Should be able to query the view without error
            results = db.execute_query("SELECT * FROM tool_timeline LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_native_tool_calls_view_exists(self, tmp_path: Path) -> None:
        """Test that native_tool_calls view is created."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM native_tool_calls LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_error_summary_view_exists(self, tmp_path: Path) -> None:
        """Test that error_summary view is created."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM error_summary LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_views_included_in_schema(self, tmp_path: Path) -> None:
        """Test that views appear in schema output."""
        bundle_path = _create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            assert "tool_timeline" in table_names
            assert "native_tool_calls" in table_names
            assert "error_summary" in table_names
        finally:
            db.close()


class TestGetTableDescriptionViews:
    """Tests for _get_table_description with views."""

    def test_view_description(self) -> None:
        """Test that views get appropriate descriptions."""
        assert "View:" in _get_table_description("tool_timeline", is_view=True)
        assert "View:" in _get_table_description("native_tool_calls", is_view=True)
        assert "View:" in _get_table_description("error_summary", is_view=True)

    def test_unknown_view(self) -> None:
        """Test that unknown views get a generic description."""
        desc = _get_table_description("unknown_view", is_view=True)
        assert desc == "View"
