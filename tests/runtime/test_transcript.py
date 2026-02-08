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

"""Tests for runtime transcript types and emitter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from weakincentives.runtime.transcript import (
    CANONICAL_ENTRY_TYPES,
    TranscriptEmitter,
    TranscriptEntry,
    TranscriptSummary,
    reconstruct_transcript,
)


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_basic_creation(self) -> None:
        """TranscriptEntry can be constructed with required fields."""
        ts = datetime.now(UTC)
        entry = TranscriptEntry(
            prompt_name="test",
            adapter="claude_agent_sdk",
            entry_type="user_message",
            sequence_number=1,
            source="main",
            timestamp=ts,
        )
        assert entry.prompt_name == "test"
        assert entry.adapter == "claude_agent_sdk"
        assert entry.entry_type == "user_message"
        assert entry.sequence_number == 1
        assert entry.source == "main"
        assert entry.timestamp == ts
        assert entry.session_id is None
        assert entry.detail == {}
        assert entry.raw is None

    def test_with_optional_fields(self) -> None:
        """TranscriptEntry accepts optional fields."""
        entry = TranscriptEntry(
            prompt_name="test",
            adapter="codex_app_server",
            entry_type="tool_use",
            sequence_number=5,
            source="subagent:001",
            timestamp=datetime.now(UTC),
            session_id="abc-123",
            detail={"tool": "bash"},
            raw='{"method": "item/tool/call"}',
        )
        assert entry.session_id == "abc-123"
        assert entry.detail == {"tool": "bash"}
        assert entry.raw is not None

    def test_frozen(self) -> None:
        """TranscriptEntry is immutable."""
        entry = TranscriptEntry(
            prompt_name="test",
            adapter="test",
            entry_type="user_message",
            sequence_number=1,
            source="main",
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            entry.prompt_name = "changed"  # type: ignore[misc]


class TestTranscriptSummary:
    """Tests for TranscriptSummary dataclass."""

    def test_basic_creation(self) -> None:
        """TranscriptSummary can be constructed."""
        summary = TranscriptSummary(
            total_entries=10,
            entries_by_type={"user_message": 3, "assistant_message": 7},
            entries_by_source={"main": 10},
            sources=("main",),
            adapter="claude_agent_sdk",
            prompt_name="test",
            first_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            last_timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        )
        assert summary.total_entries == 10
        assert summary.sources == ("main",)

    def test_frozen(self) -> None:
        """TranscriptSummary is immutable."""
        summary = TranscriptSummary(
            total_entries=0,
            entries_by_type={},
            entries_by_source={},
            sources=(),
            adapter="test",
            prompt_name="test",
            first_timestamp=None,
            last_timestamp=None,
        )
        with pytest.raises(AttributeError):
            summary.total_entries = 5  # type: ignore[misc]


class TestCanonicalEntryTypes:
    """Tests for CANONICAL_ENTRY_TYPES."""

    def test_expected_types_present(self) -> None:
        """All canonical entry types are defined."""
        expected = {
            "user_message",
            "assistant_message",
            "tool_use",
            "tool_result",
            "thinking",
            "system_event",
            "token_usage",
            "error",
            "unknown",
        }
        assert expected == CANONICAL_ENTRY_TYPES

    def test_is_frozenset(self) -> None:
        """CANONICAL_ENTRY_TYPES is a frozenset."""
        assert isinstance(CANONICAL_ENTRY_TYPES, frozenset)


class TestTranscriptEmitter:
    """Tests for TranscriptEmitter."""

    def test_emit_basic(self) -> None:
        """Emitter emits entries via structured logger."""
        emitter = TranscriptEmitter(
            prompt_name="test-prompt",
            adapter="test-adapter",
        )

        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("user_message", source="main")

            entry_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if call[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            context = entry_calls[0][1]["context"]
            assert context["prompt_name"] == "test-prompt"
            assert context["adapter"] == "test-adapter"
            assert context["entry_type"] == "user_message"
            assert context["sequence_number"] == 1
            assert context["source"] == "main"
            assert "timestamp" in context
            assert "session_id" not in context

    def test_emit_with_session_id(self) -> None:
        """Emitter includes session_id when provided."""
        emitter = TranscriptEmitter(
            prompt_name="test",
            adapter="test",
            session_id="sess-42",
        )
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("assistant_message")
            context = mock_logger.debug.call_args[1]["context"]
            assert context["session_id"] == "sess-42"

    def test_emit_with_detail(self) -> None:
        """Emitter includes detail when provided."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a")
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("tool_use", detail={"tool": "bash"})
            context = mock_logger.debug.call_args[1]["context"]
            assert context["detail"] == {"tool": "bash"}

    def test_emit_with_raw(self) -> None:
        """Emitter includes raw when emit_raw=True."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a", emit_raw=True)
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("tool_result", raw='{"type": "test"}')
            context = mock_logger.debug.call_args[1]["context"]
            assert context["raw"] == '{"type": "test"}'

    def test_emit_without_raw(self) -> None:
        """Emitter omits raw when emit_raw=False."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a", emit_raw=False)
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("tool_result", raw='{"type": "test"}')
            context = mock_logger.debug.call_args[1]["context"]
            assert "raw" not in context

    def test_sequence_numbers_increase(self) -> None:
        """Sequence numbers increase per source."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a")
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("user_message", source="main")
            emitter.emit("assistant_message", source="main")
            emitter.emit("user_message", source="subagent:1")

            calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(calls) == 3
            assert calls[0][1]["context"]["sequence_number"] == 1
            assert calls[1][1]["context"]["sequence_number"] == 2
            assert calls[2][1]["context"]["sequence_number"] == 1

    def test_total_entries(self) -> None:
        """Total entries count tracks all emitted entries."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a")
        assert emitter.total_entries == 0

        with patch("weakincentives.runtime.transcript._logger"):
            emitter.emit("user_message", source="main")
            emitter.emit("assistant_message", source="main")
            emitter.emit("user_message", source="sub:1")

        assert emitter.total_entries == 3

    def test_source_count(self) -> None:
        """source_count tracks per-source entries."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a")

        with patch("weakincentives.runtime.transcript._logger"):
            emitter.emit("user_message", source="main")
            emitter.emit("assistant_message", source="main")
            emitter.emit("user_message", source="sub:1")

        assert emitter.source_count("main") == 2
        assert emitter.source_count("sub:1") == 1
        assert emitter.source_count("nonexistent") == 0

    def test_start_event(self) -> None:
        """start() emits transcript.start event."""
        emitter = TranscriptEmitter(prompt_name="test", adapter="myad")
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.start()
            start_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.start"
            ]
            assert len(start_calls) == 1
            ctx = start_calls[0][1]["context"]
            assert ctx["prompt_name"] == "test"
            assert ctx["adapter"] == "myad"

    def test_stop_event(self) -> None:
        """stop() emits transcript.stop event with summary."""
        emitter = TranscriptEmitter(prompt_name="test", adapter="myad")
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            emitter.emit("user_message", source="main")
            emitter.emit("assistant_message", source="main")
            emitter.stop()

            stop_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.stop"
            ]
            assert len(stop_calls) == 1
            ctx = stop_calls[0][1]["context"]
            assert ctx["total_entries"] == 2
            assert ctx["entries_by_source"]["main"] == 2

    def test_emit_never_raises(self) -> None:
        """Emitter catches exceptions from the logger."""
        emitter = TranscriptEmitter(prompt_name="t", adapter="a")
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            mock_logger.debug.side_effect = RuntimeError("boom")
            # Should not raise
            emitter.emit("user_message")
            # Should have logged a warning about the failure
            mock_logger.warning.assert_called()


class TestReconstructTranscript:
    """Tests for reconstruct_transcript."""

    def test_basic_reconstruction(self) -> None:
        """Reconstruct entries from log records."""
        ts = datetime.now(UTC).isoformat()
        records = [
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    "adapter": "sdk",
                    "entry_type": "user_message",
                    "sequence_number": 1,
                    "source": "main",
                    "timestamp": ts,
                    "detail": {"sdk_entry": {"type": "user"}},
                    "raw": '{"type": "user"}',
                },
            },
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    "adapter": "sdk",
                    "entry_type": "assistant_message",
                    "sequence_number": 2,
                    "source": "main",
                    "timestamp": ts,
                },
            },
        ]
        entries = reconstruct_transcript(records)
        assert len(entries) == 2
        assert entries[0].entry_type == "user_message"
        assert entries[0].detail == {"sdk_entry": {"type": "user"}}
        assert entries[0].raw == '{"type": "user"}'
        assert entries[1].entry_type == "assistant_message"
        assert entries[1].raw is None

    def test_filters_non_transcript_events(self) -> None:
        """Records with wrong event name are skipped."""
        records = [
            {"event": "transcript.start", "context": {}},
            {"event": "other.event", "context": {}},
        ]
        assert reconstruct_transcript(records) == []

    def test_skips_missing_context(self) -> None:
        """Records without context are skipped."""
        records = [
            {"event": "transcript.entry"},
            {"event": "transcript.entry", "context": "not a dict"},
        ]
        assert reconstruct_transcript(records) == []

    def test_skips_malformed_entries(self) -> None:
        """Records missing required fields are skipped."""
        records = [
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    # Missing other required fields
                },
            },
        ]
        assert reconstruct_transcript(records) == []

    def test_with_session_id(self) -> None:
        """Entries with session_id are preserved."""
        ts = datetime.now(UTC).isoformat()
        records = [
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    "adapter": "sdk",
                    "entry_type": "thinking",
                    "sequence_number": 1,
                    "source": "main",
                    "timestamp": ts,
                    "session_id": "sess-42",
                },
            },
        ]
        entries = reconstruct_transcript(records)
        assert len(entries) == 1
        assert entries[0].session_id == "sess-42"

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert reconstruct_transcript([]) == []
