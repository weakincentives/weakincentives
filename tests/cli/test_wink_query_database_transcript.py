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

"""Transcript table tests for the wink query command."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from weakincentives.cli.query import open_query_database


def _create_agents_view_bundle(tmp_path: Path) -> Path:
    """Create a debug bundle zip with main + subagent transcript entries."""
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
