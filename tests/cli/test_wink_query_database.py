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

"""Database-oriented tests for the wink query command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli._query_fixtures import (
    _AgentPlan,
    create_bundle_with_logs,
    create_test_bundle,
)
from weakincentives.cli._query_helpers import _get_table_description
from weakincentives.cli.query import (
    ColumnInfo,
    QueryError,
    SchemaOutput,
    TableInfo,
    export_jsonl,
    open_query_database,
)
from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.bundle import (
    BundleConfig,
    BundleValidationError,
)
from weakincentives.runtime.session import Session


class TestQueryDatabase:
    """Tests for QueryDatabase class."""

    def test_build_creates_tables(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            assert "manifest" in table_names
            assert "logs" in table_names
            assert "transcript" in table_names
            assert "errors" in table_names
            assert "session_slices" in table_names
            assert "config" in table_names
            assert "metrics" in table_names
        finally:
            db.close()

    def test_manifest_table(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM manifest")
            assert len(results) == 1
            assert results[0]["status"] == "success"
        finally:
            db.close()

    def test_session_slices_table(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM session_slices")
            assert len(results) == 2  # AgentPlan + TaskStatus
        finally:
            db.close()

    def test_dynamic_slice_tables(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            # Check that slice tables were created
            assert any(t.startswith("slice_") for t in table_names)
        finally:
            db.close()

    def test_config_table_flattened(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path, with_error=True)
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
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            with pytest.raises(QueryError, match="SQL error"):
                db.execute_query("SELECT * FROM nonexistent_table")
        finally:
            db.close()

    def test_caching_works(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
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


class TestOpenQueryDatabase:
    """Tests for open_query_database function."""

    def test_opens_bundle(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        db.close()

        cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")
        assert cache_path.exists()


class TestQueryDatabaseWithLogs:
    """Tests for QueryDatabase with log entries."""

    def test_logs_table_populated(self, tmp_path: Path) -> None:
        bundle_path = create_bundle_with_logs(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM logs")
            # Should have at least some log entries
            assert len(results) >= 0
        finally:
            db.close()

    def test_tool_calls_derived(self, tmp_path: Path) -> None:
        bundle_path = create_bundle_with_logs(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM tool_calls")
            # Table should exist even if empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_errors_from_logs(self, tmp_path: Path) -> None:
        bundle_path = create_bundle_with_logs(tmp_path)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM errors WHERE source = 'log'")
            # Should have at least one error from the ERROR level log
            assert isinstance(results, list)
        finally:
            db.close()


def _create_agents_view_bundle(tmp_path: Path) -> Path:
    """Create a debug bundle zip with main + subagent transcript entries."""
    import zipfile

    manifest = {
        "format_version": "1.0.0",
        "bundle_id": "test-agents",
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

    logs = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "level": "DEBUG",
            "logger": "collector",
            "event": "transcript.entry",
            "message": "User",
            "context": {
                "prompt_name": "test",
                "source": "main",
                "sequence_number": 1,
                "entry_type": "user_message",
                "detail": {
                    "type": "user",
                    "message": {"role": "user", "content": "Hi"},
                },
            },
        },
        {
            "timestamp": "2024-01-01T00:00:01Z",
            "level": "DEBUG",
            "logger": "collector",
            "event": "transcript.entry",
            "message": "Assistant",
            "context": {
                "prompt_name": "test",
                "source": "main",
                "sequence_number": 2,
                "entry_type": "assistant_message",
                "detail": {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": "Hello"},
                },
            },
        },
        {
            "timestamp": "2024-01-01T00:00:02Z",
            "level": "DEBUG",
            "logger": "collector",
            "event": "transcript.entry",
            "message": "Subagent thinking",
            "context": {
                "prompt_name": "test",
                "source": "subagent:001",
                "sequence_number": 1,
                "entry_type": "thinking",
                "detail": {"type": "thinking", "thinking": "Processing..."},
            },
        },
        {
            "timestamp": "2024-01-01T00:00:03Z",
            "level": "DEBUG",
            "logger": "collector",
            "event": "transcript.entry",
            "message": "Subagent assistant",
            "context": {
                "prompt_name": "test",
                "source": "subagent:001",
                "sequence_number": 2,
                "entry_type": "assistant_message",
                "detail": {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Running tool"},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "read",
                                "input": {},
                            },
                        ],
                    },
                },
            },
        },
    ]

    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
        zf.writestr(
            "debug_bundle/logs/app.jsonl",
            "\n".join(json.dumps(log) for log in logs),
        )
        zf.writestr("debug_bundle/session/after.jsonl", "")
        zf.writestr("debug_bundle/request/input.json", "{}")
        zf.writestr("debug_bundle/request/output.json", "{}")
    return bundle_path


class TestTranscriptTable:
    """Tests for transcript table extraction."""

    def test_transcript_entries_extracted(self, tmp_path: Path) -> None:
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-transcript",
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

        parsed_entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running tool"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        }

        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Transcript entry",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 7,
                    "entry_type": "assistant_message",
                    "raw": json.dumps(parsed_entry),
                    "detail": parsed_entry,
                },
            }
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
            rows = db.execute_query(
                "SELECT transcript_source, entry_type, role, content, tool_name, tool_use_id "
                "FROM transcript"
            )
            assert len(rows) == 1
            row = rows[0]
            assert row["transcript_source"] == "main"
            assert row["entry_type"] == "assistant_message"
            assert row["role"] == "assistant"
            assert "Running tool" in row["content"]
            assert "bash" in row["content"]
            assert row["tool_name"] == "bash"
            assert row["tool_use_id"] == "tool-1"
        finally:
            db.close()

    def test_transcript_flow_view(self, tmp_path: Path) -> None:
        """Test transcript_flow view provides conversation preview."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-flow",
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

        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "User message",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 1,
                    "entry_type": "user_message",
                    "detail": {
                        "type": "user",
                        "message": {"role": "user", "content": "Hello, assistant!"},
                    },
                },
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Assistant response",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 2,
                    "entry_type": "assistant_message",
                    "detail": {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help?",
                        },
                    },
                },
            },
            {
                "timestamp": "2024-01-01T00:00:02Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Thinking",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 3,
                    "entry_type": "thinking",
                    "detail": {
                        "type": "thinking",
                        "thinking": "The user is greeting me. I should respond politely.",
                    },
                },
            },
        ]

        bundle_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/logs/app.jsonl",
                "\n".join(json.dumps(log) for log in logs),
            )
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            # Query the transcript_flow view
            rows = db.execute_query(
                "SELECT * FROM transcript_flow ORDER BY sequence_number"
            )

            assert len(rows) == 3

            # Check user message
            assert rows[0]["sequence_number"] == 1
            assert rows[0]["entry_type"] == "user_message"
            assert rows[0]["message_preview"] == "Hello, assistant!"

            # Check assistant message
            assert rows[1]["sequence_number"] == 2
            assert rows[1]["entry_type"] == "assistant_message"
            assert rows[1]["message_preview"] == "Hello! How can I help?"

            # Check thinking block
            assert rows[2]["sequence_number"] == 3
            assert rows[2]["entry_type"] == "thinking"
            assert rows[2]["message_preview"] == "[THINKING]"
        finally:
            db.close()

    def test_transcript_tools_view(self, tmp_path: Path) -> None:
        """Test transcript_tools view pairs tool calls with results."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-tools",
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

        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Tool call",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 1,
                    "entry_type": "assistant_message",
                    "detail": {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Running bash"},
                                {
                                    "type": "tool_use",
                                    "id": "tool-123",
                                    "name": "bash",
                                    "input": {"command": "echo hello"},
                                },
                            ],
                        },
                    },
                },
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Tool result",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "sequence_number": 2,
                    "entry_type": "tool_result",
                    "detail": {
                        "type": "tool_result",
                        "tool_use_id": "tool-123",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                },
            },
        ]

        bundle_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/logs/app.jsonl",
                "\n".join(json.dumps(log) for log in logs),
            )
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            # Query the transcript_tools view
            rows = db.execute_query("SELECT * FROM transcript_tools")

            assert len(rows) == 1

            # Check tool call paired with result
            assert rows[0]["tool_name"] == "bash"
            assert rows[0]["tool_use_id"] == "tool-123"
            assert rows[0]["call_seq"] == 1
            assert rows[0]["result_seq"] == 2
            assert "Running bash" in rows[0]["tool_params"]
            assert "hello" in rows[0]["tool_result"]
        finally:
            db.close()

    def test_transcript_agents_view(self, tmp_path: Path) -> None:
        """Test transcript_agents view aggregates agent metrics."""
        bundle_path = _create_agents_view_bundle(tmp_path)

        db = open_query_database(bundle_path)
        try:
            rows = db.execute_query(
                "SELECT * FROM transcript_agents ORDER BY transcript_source"
            )

            assert len(rows) == 2

            # Check main agent metrics
            main = rows[0]
            assert main["transcript_source"] == "main"
            assert main["agent_id"] is None
            assert main["total_entries"] == 2
            assert main["user_messages"] == 1
            assert main["assistant_messages"] == 1
            assert main["thinking_blocks"] == 0

            # Check subagent metrics
            subagent = rows[1]
            assert subagent["transcript_source"] == "subagent:001"
            assert subagent["agent_id"] == "001"
            assert subagent["total_entries"] == 2
            assert subagent["thinking_blocks"] == 1
            assert subagent["assistant_messages"] == 1
            assert subagent["unique_tools"] == 1
            assert subagent["total_tool_calls"] == 1
        finally:
            db.close()

    def test_transcript_artifact_with_non_matching_entries(
        self, tmp_path: Path
    ) -> None:
        """transcript.jsonl with non-transcript entries skips them gracefully."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-artifact",
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

        # transcript.jsonl with one valid and one non-matching entry
        valid_entry = json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "logger": "transcript",
                "event": "transcript.entry",
                "message": "transcript entry: user_message",
                "context": {
                    "prompt_name": "test",
                    "source": "main",
                    "entry_type": "user_message",
                    "sequence_number": 1,
                    "detail": {"text": "hello"},
                },
            }
        )
        non_matching = json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "DEBUG",
                "event": "transcript.start",
                "message": "start",
                "context": {},
            }
        )
        transcript_content = f"{valid_entry}\n{non_matching}\n"

        bundle_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/transcript.jsonl", transcript_content)
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            rows = db.execute_query("SELECT entry_type FROM transcript")
            assert len(rows) == 1
            assert rows[0]["entry_type"] == "user_message"
        finally:
            db.close()


class TestQueryDatabaseEdgeCases:
    """Tests for edge cases in QueryDatabase."""

    def test_empty_config(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path, with_config=False)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM config")
            # Table should exist but may be empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_empty_metrics(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path, with_metrics=False)
        db = open_query_database(bundle_path)

        try:
            results = db.execute_query("SELECT * FROM metrics")
            # Table should exist but may be empty
            assert isinstance(results, list)
        finally:
            db.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        # Closing multiple times should not raise
        db.close()
        db.close()

    def test_get_schema_opens_connection(self, tmp_path: Path) -> None:
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        db.close()

        # Getting schema after close should work (reopens connection)
        schema = db.get_schema()
        assert len(schema.tables) > 0
        db.close()

    def test_connection_reused(self, tmp_path: Path) -> None:
        """Test that connection is reused when already open."""
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)

        try:
            assert db.bundle_path == bundle_path
        finally:
            db.close()


class TestCacheInvalidation:
    """Tests for cache invalidation and rebuild."""

    def test_stale_cache_removed(self, tmp_path: Path) -> None:
        """Test that stale cache is removed and rebuilt."""
        import time

        bundle_path = create_test_bundle(tmp_path)
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


class TestFilesTableReadError:
    """Tests for handling errors when reading filesystem files."""

    def test_file_read_error_skipped(self, tmp_path: Path) -> None:
        """Test that files causing read errors are skipped."""
        import zipfile
        from unittest.mock import patch

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


class TestExportJsonl:
    """Tests for export_jsonl function."""

    def test_export_logs(self, tmp_path: Path) -> None:
        """Test exporting logs JSONL."""
        import zipfile

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
        bundle_path = create_test_bundle(tmp_path)
        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, "session")

        # Should return session content or None
        assert content is None or isinstance(content, str)

    def test_export_invalid_source(self, tmp_path: Path) -> None:
        """Test exporting with invalid source returns None."""
        bundle_path = create_test_bundle(tmp_path)
        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, "invalid")

        assert content is None


class TestSeqColumn:
    """Tests for seq column in logs table."""

    def test_seq_extracted_from_context_sequence_number(self, tmp_path: Path) -> None:
        """Test that seq column is extracted from context.sequence_number."""
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

        # Create logs with sequence_number context values
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
                "timestamp": "2024-01-01T00:00:01.500Z",
                "level": "DEBUG",
                "logger": "collector",
                "event": "transcript.entry",
                "message": "Transcript entry",
                "context": {
                    "sequence_number": 44,
                    "entry_type": "assistant_message",
                    "source": "main",
                },
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
            assert len(results) == 3
            assert results[0]["seq"] == 42
            assert results[1]["seq"] == 43
            assert results[2]["seq"] == 44

            # Non sequence events should have NULL seq
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
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path)
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
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            # Should be able to query the view without error
            results = db.execute_query("SELECT * FROM tool_timeline LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_native_tool_calls_view_exists(self, tmp_path: Path) -> None:
        """Test that native_tool_calls view is created."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM native_tool_calls LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_error_summary_view_exists(self, tmp_path: Path) -> None:
        """Test that error_summary view is created."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM error_summary LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_transcript_entries_view_extracts_fields(self, tmp_path: Path) -> None:
        """Test that transcript_entries view extracts key fields from logs."""
        bundle_path = create_bundle_with_logs(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query(
                """
                SELECT prompt_name, transcript_source, sequence_number, entry_type, role, content
                FROM transcript_entries
                ORDER BY transcript_source, sequence_number
            """
            )
            assert len(results) == 1
            assert results[0]["prompt_name"] == "test-prompt"
            assert results[0]["transcript_source"] == "main"
            assert results[0]["sequence_number"] == 1
            assert results[0]["entry_type"] == "user_message"
            assert results[0]["role"] == "user"
            assert results[0]["content"] == "Hello"
        finally:
            db.close()

    def test_views_included_in_schema(self, tmp_path: Path) -> None:
        """Test that views appear in schema output."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            assert "tool_timeline" in table_names
            assert "native_tool_calls" in table_names
            assert "transcript_entries" in table_names
            assert "error_summary" in table_names
        finally:
            db.close()


class TestOpenQueryDatabaseLogSkipping:
    """Tests for log parsing fallback behavior."""

    def test_open_query_database_skips_non_mapping_log_lines(
        self, tmp_path: Path
    ) -> None:
        import zipfile

        base_bundle = create_bundle_with_logs(tmp_path)

        with zipfile.ZipFile(base_bundle, "r") as zf:
            manifest = zf.read("debug_bundle/manifest.json")
            session_after = zf.read("debug_bundle/session/after.jsonl")
            request_input = zf.read("debug_bundle/request/input.json")
            request_output = zf.read("debug_bundle/request/output.json")
            logs = zf.read("debug_bundle/logs/app.jsonl").decode("utf-8")

        # Prepend a valid JSON line that is *not* a mapping; it should be skipped.
        new_logs = json.dumps(["not", "a", "mapping"]) + "\n" + logs

        bundle_path = tmp_path / "non_mapping_logs.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", manifest)
            zf.writestr("debug_bundle/session/after.jsonl", session_after)
            zf.writestr("debug_bundle/request/input.json", request_input)
            zf.writestr("debug_bundle/request/output.json", request_output)
            zf.writestr("debug_bundle/logs/app.jsonl", new_logs)

        db = open_query_database(bundle_path)
        try:
            results = db.execute_query(
                """
                SELECT transcript_source, sequence_number, entry_type, role, content
                FROM transcript_entries
            """
            )
            assert len(results) == 1
            assert results[0]["transcript_source"] == "main"
            assert results[0]["sequence_number"] == 1
            assert results[0]["entry_type"] == "user_message"
            assert results[0]["role"] == "user"
            assert results[0]["content"] == "Hello"
        finally:
            db.close()


class TestEnvironmentTables:
    """Tests for environment data tables."""

    def test_environment_tables_created_without_env_data(self, tmp_path: Path) -> None:
        """Test that environment tables are created even without data."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            # Tables should exist but be empty
            for table in [
                "env_system",
                "env_python",
                "env_git",
                "env_container",
                "env_vars",
                "environment",
            ]:
                results = db.execute_query(f"SELECT * FROM {table}")
                assert isinstance(results, list)
        finally:
            db.close()

    def test_environment_tables_populated(self, tmp_path: Path) -> None:
        """Test that environment tables are populated when data exists."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env",
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
            "files": ["environment/system.json", "environment/python.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        system_data = {
            "os_name": "Linux",
            "os_release": "5.15.0",
            "kernel_version": "5.15.0-generic",
            "architecture": "x86_64",
            "processor": "x86_64",
            "cpu_count": 8,
            "memory_total_bytes": 16000000000,
            "hostname": "testhost",
        }

        python_data = {
            "version": "3.11.5",
            "version_info": [3, 11, 5],
            "implementation": "CPython",
            "executable": "/usr/bin/python3",
            "prefix": "/usr",
            "base_prefix": "/usr",
            "is_virtualenv": False,
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/system.json", json.dumps(system_data))
            zf.writestr("debug_bundle/environment/python.json", json.dumps(python_data))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            # Check env_system table
            system_results = db.execute_query("SELECT * FROM env_system")
            assert len(system_results) == 1
            assert system_results[0]["os_name"] == "Linux"
            assert system_results[0]["architecture"] == "x86_64"
            assert system_results[0]["cpu_count"] == 8

            # Check env_python table
            python_results = db.execute_query("SELECT * FROM env_python")
            assert len(python_results) == 1
            assert python_results[0]["implementation"] == "CPython"
            assert python_results[0]["is_virtualenv"] == 0

            # Check flat environment table
            env_results = db.execute_query(
                "SELECT key, value FROM environment WHERE key = 'system_os_name'"
            )
            assert len(env_results) == 1
            assert env_results[0]["value"] == "Linux"
        finally:
            db.close()

    def test_env_git_table_populated(self, tmp_path: Path) -> None:
        """Test that env_git table is populated with git info."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-git",
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
            "files": ["environment/git.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        git_data = {
            "repo_root": "/home/user/project",
            "commit_sha": "abc123def456",
            "commit_short": "abc123de",
            "branch": "main",
            "is_dirty": True,
            "remotes": {"origin": "https://github.com/user/project.git"},
            "tags": ["v1.0.0"],
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/git.json", json.dumps(git_data))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            git_results = db.execute_query("SELECT * FROM env_git")
            assert len(git_results) == 1
            assert git_results[0]["branch"] == "main"
            assert git_results[0]["is_dirty"] == 1
            assert git_results[0]["commit_short"] == "abc123de"
        finally:
            db.close()

    def test_env_vars_table_populated(self, tmp_path: Path) -> None:
        """Test that env_vars table is populated with environment variables."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-vars",
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
            "files": ["environment/env_vars.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        env_vars_data = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHON_VERSION": "3.11.5",
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/environment/env_vars.json", json.dumps(env_vars_data)
            )
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            env_vars_results = db.execute_query(
                "SELECT name, value FROM env_vars ORDER BY name"
            )
            assert len(env_vars_results) == 3
            names = [r["name"] for r in env_vars_results]
            assert "HOME" in names
            assert "PATH" in names
        finally:
            db.close()

    def test_environment_flat_table_includes_packages(self, tmp_path: Path) -> None:
        """Test that flat environment table includes packages."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-packages",
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
            "files": ["environment/packages.txt"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        packages = "fastapi==0.100.0\nuvicorn==0.23.0\npytest==7.4.0"

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/packages.txt", packages)
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            pkg_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'packages'"
            )
            assert len(pkg_results) == 1
            assert "fastapi==0.100.0" in pkg_results[0]["value"]
        finally:
            db.close()

    def test_environment_table_descriptions(self) -> None:
        """Test that environment tables have descriptions."""
        assert "System/OS" in _get_table_description("env_system")
        assert "Python" in _get_table_description("env_python")
        assert "Git" in _get_table_description("env_git")
        assert "Container" in _get_table_description("env_container")
        assert "environment" in _get_table_description("env_vars").lower()
        assert "environment" in _get_table_description("environment").lower()

    def test_schema_includes_environment_queries(self, tmp_path: Path) -> None:
        """Test that schema hints include environment-related queries."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            assert schema.hints is not None
            assert "system_info" in schema.hints.common_queries
            assert "python_info" in schema.hints.common_queries
            assert "git_info" in schema.hints.common_queries
            assert "env_vars" in schema.hints.common_queries
        finally:
            db.close()

    def test_env_container_table_populated(self, tmp_path: Path) -> None:
        """Test that env_container table is populated with container info."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-container",
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
            "files": ["environment/container.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        container_data = {
            "runtime": "docker",
            "container_id": "abc123def456",
            "image": "python:3.11",
            "image_digest": "sha256:abc123",
            "cgroup_path": "/docker/abc123def456",
            "is_containerized": True,
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/environment/container.json", json.dumps(container_data)
            )
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            container_results = db.execute_query("SELECT * FROM env_container")
            assert len(container_results) == 1
            assert container_results[0]["runtime"] == "docker"
            assert container_results[0]["is_containerized"] == 1
        finally:
            db.close()

    def test_environment_flat_table_includes_command_and_git_diff(
        self, tmp_path: Path
    ) -> None:
        """Test that flat environment table includes command and git_diff."""
        import zipfile

        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-command-diff",
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
            "files": ["environment/command.txt", "environment/git.diff"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        command_data = "python script.py --arg value"
        git_diff_data = "diff --git a/file.py b/file.py\n+new line"

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/command.txt", command_data)
            zf.writestr("debug_bundle/environment/git.diff", git_diff_data)
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            cmd_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'command'"
            )
            assert len(cmd_results) == 1
            assert "python script.py" in cmd_results[0]["value"]

            diff_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'git_diff'"
            )
            assert len(diff_results) == 1
            assert "diff --git" in diff_results[0]["value"]
        finally:
            db.close()
