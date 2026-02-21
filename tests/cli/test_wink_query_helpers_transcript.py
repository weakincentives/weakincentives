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

"""Unit tests for wink query transcript helper functions."""

from __future__ import annotations

from weakincentives.cli._query_helpers import (
    _safe_json_dumps,
)
from weakincentives.cli._query_transcript import (
    _apply_notification_item_details,
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
