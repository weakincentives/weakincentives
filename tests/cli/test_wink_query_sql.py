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

"""Unit tests for wink query — cache validation, transcript detail split/ACP tests."""

from __future__ import annotations

from pathlib import Path

from weakincentives.cli._query_transcript import (
    _apply_bridged_tool_details,
    _apply_notification_item_details,
    _apply_split_block_details,
    _extract_transcript_details,
    _extract_transcript_parsed_obj,
    _extract_transcript_row,
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
        import os

        from weakincentives.cli.query import _is_cache_valid

        bundle = tmp_path / "bundle.zip"
        cache = tmp_path / "bundle.zip.sqlite"
        cache.touch()
        bundle.touch()
        # Ensure bundle has a newer mtime than cache
        cache_mtime = cache.stat().st_mtime
        os.utime(bundle, (cache_mtime + 1, cache_mtime + 1))
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


class TestExtractTranscriptDetailsAcpFallback:
    """Tests for ACP adapter-style detail dicts using tool_call_id."""

    def test_acp_tool_use_extracts_tool_call_id_and_tool_name(self) -> None:
        """ACP adapter emits tool_call_id and tool_name (no 'type' key)."""
        parsed: dict[str, object] = {
            "tool_name": "read",
            "tool_call_id": "call_abc123",
            "input": {"path": "/tmp/a.txt"},
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_use"
        )
        assert tool_name == "read"
        assert tool_use_id == "call_abc123"

    def test_acp_tool_result_extracts_tool_call_id(self) -> None:
        """ACP adapter tool_result entries also use tool_call_id."""
        parsed: dict[str, object] = {
            "tool_name": "glob",
            "tool_call_id": "call_xyz789",
            "status": "completed",
            "output": {"files": ["a.py"]},
        }
        _role, _content, tool_name, tool_use_id = _extract_transcript_details(
            parsed, "tool_result"
        )
        assert tool_name == "glob"
        assert tool_use_id == "call_xyz789"


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
        from weakincentives.cli._query_transcript import (
            _apply_transcript_content_fallbacks,
        )

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
        from weakincentives.cli._query_transcript import (
            _apply_transcript_content_fallbacks,
        )

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


class TestExtractTranscriptRow:
    """Tests for _extract_transcript_row edge cases."""

    def test_ignores_bad_context(self) -> None:
        assert (
            _extract_transcript_row(
                {"event": "transcript.entry", "context": "not-a-mapping"}
            )
            is None
        )

    def test_rejects_legacy_event(self) -> None:
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
