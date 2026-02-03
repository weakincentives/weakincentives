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

"""CLI-focused tests for wink query."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.cli._query_fixtures import create_test_bundle
from weakincentives.cli import wink
from weakincentives.cli.query import format_as_json, format_as_table


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


class TestWinkQueryCLI:
    """Tests for wink query CLI command."""

    def test_query_with_schema_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = create_test_bundle(tmp_path)

        exit_code = wink.main(["--no-json-logs", "query", str(bundle_path), "--schema"])

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "tables" in data
        assert "bundle_id" in data

    def test_query_with_sql(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle_path = create_test_bundle(tmp_path)

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
        bundle_path = create_test_bundle(tmp_path)

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
        bundle_path = create_test_bundle(tmp_path)

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
        bundle_path = create_test_bundle(tmp_path)

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
        bundle_path = create_test_bundle(tmp_path, with_logs=False)

        exit_code = wink.main(
            ["--no-json-logs", "query", str(bundle_path), "--export-jsonl"]
        )

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No logs content" in captured.err

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
