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

"""Tests for transcript capture functionality."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from weakincentives.runtime.transcript import (
    TranscriptEntry,
    TranscriptRole,
    convert_claude_transcript_entry,
    format_transcript,
)


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_create_basic_entry(self) -> None:
        """Test creating a basic transcript entry."""
        entry = TranscriptEntry(
            role="user",
            content="Hello, world!",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )

        assert entry.role == "user"
        assert entry.content == "Hello, world!"
        assert entry.created_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        assert entry.tool_name is None
        assert entry.tool_call_id is None
        assert entry.tool_input is None
        assert entry.metadata is None
        assert entry.source == "primary"
        assert entry.sequence == 0
        assert isinstance(entry.entry_id, UUID)

    def test_create_tool_entry(self) -> None:
        """Test creating a tool transcript entry."""
        entry = TranscriptEntry(
            role="tool",
            content='{"result": "success"}',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            tool_name="Read",
            tool_call_id="call_123",
            metadata={"success": True},
        )

        assert entry.role == "tool"
        assert entry.tool_name == "Read"
        assert entry.tool_call_id == "call_123"
        assert entry.metadata == {"success": True}

    def test_create_assistant_tool_call_entry(self) -> None:
        """Test creating an assistant entry with tool call."""
        entry = TranscriptEntry(
            role="assistant",
            content="Calling tool: Read",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            tool_name="Read",
            tool_call_id="call_456",
            tool_input={"file_path": "/path/to/file"},
        )

        assert entry.role == "assistant"
        assert entry.tool_name == "Read"
        assert entry.tool_input == {"file_path": "/path/to/file"}

    def test_entry_with_custom_source(self) -> None:
        """Test creating entry with custom source."""
        entry = TranscriptEntry(
            role="assistant",
            content="Subagent response",
            created_at=datetime.now(UTC),
            source="subagent:abc123",
            sequence=5,
        )

        assert entry.source == "subagent:abc123"
        assert entry.sequence == 5

    def test_entry_is_frozen(self) -> None:
        """Test that TranscriptEntry is immutable."""
        entry = TranscriptEntry(
            role="user",
            content="Test",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            entry.content = "Modified"  # type: ignore[misc]


class TestConvertClaudeTranscriptEntry:
    """Tests for convert_claude_transcript_entry function."""

    def test_convert_user_message(self) -> None:
        """Test converting a user message."""
        raw = {
            "type": "user",
            "content": "Hello!",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.role == "user"
        assert entry.content == "Hello!"
        assert entry.source == "primary"
        assert entry.sequence == 0

    def test_convert_assistant_message(self) -> None:
        """Test converting an assistant message."""
        raw = {
            "type": "assistant",
            "content": "I can help with that.",
            "timestamp": "2024-01-01T12:00:01Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.role == "assistant"
        assert entry.content == "I can help with that."

    def test_convert_tool_use(self) -> None:
        """Test converting a tool_use entry."""
        raw = {
            "type": "tool_use",
            "tool_name": "Read",
            "input": {"file_path": "/path/to/file"},
            "tool_use_id": "call_123",
            "timestamp": "2024-01-01T12:00:02Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.role == "assistant"  # tool_use comes from assistant
        assert entry.tool_name == "Read"
        assert entry.tool_call_id == "call_123"
        assert entry.tool_input == {"file_path": "/path/to/file"}

    def test_convert_tool_result(self) -> None:
        """Test converting a tool_result entry."""
        raw = {
            "type": "tool_result",
            "tool_name": "Read",
            "content": "File contents here",
            "tool_use_id": "call_123",
            "timestamp": "2024-01-01T12:00:03Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.role == "tool"
        assert entry.tool_name == "Read"
        assert entry.tool_call_id == "call_123"
        assert entry.content == "File contents here"

    def test_convert_with_content_blocks(self) -> None:
        """Test converting entry with content blocks (list format)."""
        raw = {
            "type": "assistant",
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ],
            "timestamp": "2024-01-01T12:00:00Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.content == "First part.\nSecond part."

    def test_convert_with_dict_content(self) -> None:
        """Test converting entry with dict content."""
        raw = {
            "type": "assistant",
            "content": {"key": "value", "nested": {"data": 123}},
            "timestamp": "2024-01-01T12:00:00Z",
        }

        entry = convert_claude_transcript_entry(raw)

        # Dict content is serialized to JSON
        assert '"key": "value"' in entry.content
        assert '"nested"' in entry.content

    def test_convert_with_custom_source_and_sequence(self) -> None:
        """Test converting with custom source and sequence."""
        raw = {
            "type": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        entry = convert_claude_transcript_entry(
            raw,
            source="subagent:test",
            sequence=10,
        )

        assert entry.source == "subagent:test"
        assert entry.sequence == 10

    def test_convert_with_invalid_timestamp(self) -> None:
        """Test converting with invalid timestamp falls back to now."""
        raw = {
            "type": "user",
            "content": "Hello",
            "timestamp": "not-a-valid-timestamp",
        }

        entry = convert_claude_transcript_entry(raw)

        # Should use datetime.now(UTC) as fallback
        assert entry.created_at is not None
        assert entry.created_at.tzinfo is not None

    def test_convert_with_missing_timestamp(self) -> None:
        """Test converting with missing timestamp."""
        raw = {
            "type": "user",
            "content": "Hello",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.created_at is not None

    def test_convert_unknown_type(self) -> None:
        """Test converting unknown type defaults to assistant."""
        raw = {
            "type": "unknown_type",
            "content": "Some content",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.role == "assistant"

    def test_convert_preserves_metadata(self) -> None:
        """Test that conversion preserves raw_type and session_id in metadata."""
        raw = {
            "type": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T12:00:00Z",
            "session_id": "sess_abc123",
        }

        entry = convert_claude_transcript_entry(raw)

        assert entry.metadata is not None
        assert entry.metadata["raw_type"] == "user"
        assert entry.metadata["claude_session_id"] == "sess_abc123"


class TestFormatTranscript:
    """Tests for format_transcript function."""

    def test_format_empty_entries(self) -> None:
        """Test formatting empty list."""
        result = format_transcript([])
        assert result == ""

    def test_format_single_user_entry(self) -> None:
        """Test formatting a single user entry."""
        entries = [
            TranscriptEntry(
                role="user",
                content="Hello!",
                created_at=datetime(2024, 1, 1, 9, 15, 32, tzinfo=UTC),
            )
        ]

        result = format_transcript(entries)

        assert "[09:15:32]" in result
        assert "👤 USER" in result
        assert "Hello!" in result

    def test_format_multiple_roles(self) -> None:
        """Test formatting entries with different roles."""
        entries = [
            TranscriptEntry(
                role="system",
                content="You are a helpful assistant.",
                created_at=datetime(2024, 1, 1, 9, 15, 30, tzinfo=UTC),
            ),
            TranscriptEntry(
                role="user",
                content="What is 2+2?",
                created_at=datetime(2024, 1, 1, 9, 15, 32, tzinfo=UTC),
            ),
            TranscriptEntry(
                role="assistant",
                content="The answer is 4.",
                created_at=datetime(2024, 1, 1, 9, 15, 35, tzinfo=UTC),
            ),
        ]

        result = format_transcript(entries)

        assert "⚙️ SYSTEM" in result
        assert "👤 USER" in result
        assert "🤖 ASSISTANT" in result
        assert "You are a helpful assistant." in result
        assert "What is 2+2?" in result
        assert "The answer is 4." in result

    def test_format_tool_entry_with_name(self) -> None:
        """Test formatting tool entry shows tool name."""
        entries = [
            TranscriptEntry(
                role="tool",
                content='{"path": "src/main.py"}',
                created_at=datetime(2024, 1, 1, 9, 15, 36, tzinfo=UTC),
                tool_name="Read",
            )
        ]

        result = format_transcript(entries)

        assert "🔧 Read" in result
        assert '{"path": "src/main.py"}' in result

    def test_format_tool_entry_without_name(self) -> None:
        """Test formatting tool entry without name falls back to TOOL."""
        entries = [
            TranscriptEntry(
                role="tool",
                content="Result",
                created_at=datetime(2024, 1, 1, 9, 15, 36, tzinfo=UTC),
            )
        ]

        result = format_transcript(entries)

        assert "🔧 TOOL" in result

    def test_format_with_subagent_source(self) -> None:
        """Test formatting entries from subagent includes source prefix."""
        entries = [
            TranscriptEntry(
                role="assistant",
                content="Subagent response",
                created_at=datetime(2024, 1, 1, 9, 15, 38, tzinfo=UTC),
                source="subagent:abc123",
            )
        ]

        result = format_transcript(entries)

        assert "[subagent:abc123]" in result
        assert "🤖 ASSISTANT" in result

    def test_format_primary_source_no_prefix(self) -> None:
        """Test that primary source doesn't add prefix."""
        entries = [
            TranscriptEntry(
                role="user",
                content="Hello",
                created_at=datetime(2024, 1, 1, 9, 15, 32, tzinfo=UTC),
                source="primary",
            )
        ]

        result = format_transcript(entries)

        assert "[primary]" not in result

    def test_format_entries_separated_by_blank_lines(self) -> None:
        """Test that entries are separated by blank lines."""
        entries = [
            TranscriptEntry(
                role="user",
                content="First",
                created_at=datetime(2024, 1, 1, 9, 15, 32, tzinfo=UTC),
            ),
            TranscriptEntry(
                role="assistant",
                content="Second",
                created_at=datetime(2024, 1, 1, 9, 15, 35, tzinfo=UTC),
            ),
        ]

        result = format_transcript(entries)

        # Check structure with blank lines
        lines = result.split("\n")
        # Each entry has: header, content, blank line
        # Last entry still has trailing blank line
        assert "" in lines  # Has blank lines


class TestTranscriptRole:
    """Tests for TranscriptRole type."""

    def test_valid_roles(self) -> None:
        """Test that all valid roles can be used."""
        roles: list[TranscriptRole] = ["system", "user", "assistant", "tool"]

        for role in roles:
            entry = TranscriptEntry(
                role=role,
                content="test",
                created_at=datetime.now(UTC),
            )
            assert entry.role == role
