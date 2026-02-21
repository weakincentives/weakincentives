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

"""Tests for TranscriptCollector message splitting, helpers, and fallback branches."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from weakincentives.adapters.claude_agent_sdk._transcript_collector import (
    TranscriptCollector,
    TranscriptCollectorConfig,
    _extract_assistant_text,
    _extract_text_from_content,
)
from weakincentives.adapters.claude_agent_sdk._transcript_parser import emit_entry


class TestTranscriptCollectorSplitting:
    """Tests for assistant/user message splitting into sub-entries."""

    def test_assistant_split_text_and_tool_use(self, tmp_path: Path) -> None:
        """Assistant message with text + 2 tool_use blocks emits 1 + 2 entries."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="split_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me help."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "write_file",
                            "input": {},
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["assistant_message", "tool_use", "tool_use"]

            # assistant_message has text-only content
            am_detail = collected[0]["detail"]["sdk_entry"]
            assert len(am_detail["message"]["content"]) == 1
            assert am_detail["message"]["content"][0]["type"] == "text"

            # tool_use entries have the block directly
            assert collected[1]["detail"]["sdk_entry"]["name"] == "read_file"
            assert collected[2]["detail"]["sdk_entry"]["name"] == "write_file"

            # raw is only on the assistant_message
            assert collected[0].get("raw") is not None
            assert "raw" not in collected[1]
            assert "raw" not in collected[2]

            # entry_count = 3
            assert collector.main_entry_count == 3

        asyncio.run(run_test())

    def test_assistant_split_only_tool_use(self, tmp_path: Path) -> None:
        """Assistant message with only tool_use blocks emits no assistant_message."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="split_only_tools_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "search",
                            "input": {},
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["tool_use"]
            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_assistant_no_split_text_only(self, tmp_path: Path) -> None:
        """Assistant message with only text emits single assistant_message."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="no_split_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Just text."}],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["assistant_message"]
            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_assistant_no_split_string_content(self, tmp_path: Path) -> None:
        """Assistant message with string content (not list) is not split."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="string_content_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": "Simple string content",
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["assistant_message"]
            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_assistant_split_entry_count(self, tmp_path: Path) -> None:
        """Entry count reflects total emitted entries including splits."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="count_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entries = [
                # user message: 1 entry
                {"type": "user", "message": {"role": "user", "content": "Hi"}},
                # assistant with text + 2 tools: 3 entries
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Ok"},
                            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
                        ],
                    },
                },
                # tool result: 1 entry
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
            ]
            transcript = tmp_path / "test.jsonl"
            transcript.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

            async with collector.run():
                await collector._remember_transcript_path(str(transcript))
                await asyncio.sleep(0.05)

            assert collector.main_entry_count == 5

        asyncio.run(run_test())

    def test_user_message_split_tool_results(self, tmp_path: Path) -> None:
        """User message with tool_result blocks emits tool_result entries."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="user_split_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "file contents here",
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["tool_result"]

            # tool_result entry has the block as sdk_entry
            assert collected[0]["detail"]["sdk_entry"]["tool_use_id"] == "toolu_1"
            assert (
                collected[0]["detail"]["sdk_entry"]["content"] == "file contents here"
            )

            # raw is attached since no other blocks remain
            assert collected[0].get("raw") is not None

            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_user_message_split_mixed_content(self, tmp_path: Path) -> None:
        """User message with text + tool_result -> user_message + tool_result."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="user_mixed_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are the results:"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "result A",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": "result B",
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["user_message", "tool_result", "tool_result"]

            # user_message has text only
            um = collected[0]["detail"]["sdk_entry"]
            assert len(um["message"]["content"]) == 1
            assert um["message"]["content"][0]["type"] == "text"
            assert collected[0].get("raw") is not None

            # tool_results have no raw (user_message got it)
            assert "raw" not in collected[1]
            assert "raw" not in collected[2]

            assert collected[1]["detail"]["sdk_entry"]["tool_use_id"] == "toolu_1"
            assert collected[2]["detail"]["sdk_entry"]["tool_use_id"] == "toolu_2"

            assert collector.main_entry_count == 3

        asyncio.run(run_test())

    def test_tool_result_correlates_tool_name(self, tmp_path: Path) -> None:
        """tool_result entries include tool_name from earlier tool_use blocks."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="tool_name_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            assistant_entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "read_file",
                            "input": {"path": "foo.py"},
                        },
                    ],
                },
            }
            user_entry = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "file contents",
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(
                json.dumps(assistant_entry) + "\n" + json.dumps(user_entry) + "\n"
            )

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["tool_use", "tool_result"]

            # tool_result has tool_name from the earlier tool_use
            assert collected[1]["detail"]["tool_name"] == "read_file"

        asyncio.run(run_test())

    def test_tool_use_block_missing_id_skips_name_recording(
        self, tmp_path: Path
    ) -> None:
        """tool_use block without id does not record a name mapping."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="no_id_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "orphan_tool",
                            "input": {},
                        },
                    ],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["tool_use"]

            # No name mapping was recorded (block had no id)
            tailer = collector._tailers["main"]
            assert tailer.tool_names == {}

        asyncio.run(run_test())

    def test_user_message_no_split_without_tool_result(self, tmp_path: Path) -> None:
        """User message without tool_result is not split."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="user_no_split_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entry = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            }
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == ["user_message"]
            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_emit_entry_without_raw(self, tmp_path: Path) -> None:
        """Emit path omits raw when emit_raw is disabled."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="emit_raw_false_test",
                config=TranscriptCollectorConfig(emit_raw=False),
            )

            transcript_path = tmp_path / "session123.jsonl"
            transcript_path.write_text('{"type": "user"}\n')
            await collector._remember_transcript_path(str(transcript_path))
            tailer = collector._tailers["main"]
            tailer.entry_count = 0

            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                emit_entry(collector._get_emitter(), tailer, '{"type": "user"}')
                entry_calls = [
                    call
                    for call in mock_logger.debug.call_args_list
                    if call[1].get("event") == "transcript.entry"
                ]
                assert len(entry_calls) == 1
                context = entry_calls[0][1]["context"]
                assert "raw" not in context
                assert context["entry_type"] == "user_message"

        asyncio.run(run_test())

    def test_inode_change_triggers_position_reset(
        self,
        transcript_collector: TranscriptCollector,
        transcript_temp_dir: Path,
    ) -> None:
        """Tailer resets position when file inode changes (rotation)."""
        # Create initial transcript
        transcript_path = transcript_temp_dir / "session123.jsonl"
        transcript_path.write_text('{"type": "user", "message": "first"}\n')

        # Set up collector
        asyncio.run(
            transcript_collector._remember_transcript_path(str(transcript_path))
        )
        tailer = transcript_collector._tailers["main"]

        # Read initial content
        asyncio.run(transcript_collector._poll_once())
        original_inode = tailer.inode
        assert tailer.entry_count == 1

        # Simulate inode change by modifying the tailer's stored inode
        tailer.inode = original_inode + 12345
        tailer.partial_line = "incomplete"

        # Write new content
        with transcript_path.open("a") as f:
            f.write('{"type": "assistant", "message": "second"}\n')

        # Poll - should detect inode mismatch and reset
        asyncio.run(transcript_collector._poll_once())

        # Verify position and partial_line were reset, inode updated
        assert tailer.inode == original_inode
        assert tailer.partial_line == ""


class TestExtractTextFromContent:
    """Tests for _extract_text_from_content helper."""

    def test_string_content(self) -> None:
        assert _extract_text_from_content("hello") == "hello"

    def test_non_list_non_string_returns_empty(self) -> None:
        assert _extract_text_from_content(42) == ""
        assert _extract_text_from_content(None) == ""

    def test_dict_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _extract_text_from_content(content) == "first\nsecond"

    def test_object_text_attributes(self) -> None:
        @dataclass
        class Block:
            text: str

        content = [Block("alpha"), Block("beta")]
        assert _extract_text_from_content(content) == "alpha\nbeta"

    def test_mixed_blocks_and_empty(self) -> None:
        content = [
            {"type": "text", "text": ""},
            {"type": "image", "url": "x"},
            {"type": "text", "text": "valid"},
        ]
        assert _extract_text_from_content(content) == "valid"


class TestExtractAssistantText:
    """Tests for _extract_assistant_text helper."""

    def test_empty_messages(self) -> None:
        assert _extract_assistant_text([]) == ""

    def test_jsonl_transcript_format(self) -> None:
        @dataclass
        class Msg:
            message: dict

        msg = Msg({"role": "assistant", "content": "from jsonl"})
        assert _extract_assistant_text([msg]) == "from jsonl"

    def test_no_matching_message(self) -> None:
        @dataclass
        class Msg:
            message: dict = field(
                default_factory=lambda: {"role": "user", "content": "not assistant"}
            )

        assert _extract_assistant_text([Msg()]) == ""

    def test_sdk_api_format(self) -> None:
        class FakeAssistantMessage:
            content = "sdk response"

        FakeAssistantMessage.__name__ = "AssistantMessage"
        assert _extract_assistant_text([FakeAssistantMessage()]) == "sdk response"


class TestEmitFallbacksBranches:
    """Tests for edge cases in _emit_fallbacks."""

    def test_fallback_assistant_skipped_when_text_empty(self, tmp_path: Path) -> None:
        """Fallback assistant_message is NOT emitted when text extraction is empty."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="empty_assistant_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            # Transcript with no assistant entry.
            transcript_path = tmp_path / "session.jsonl"
            entry = {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
            }
            transcript_path.write_text(json.dumps(entry) + "\n")

            # Set fallback with messages that yield empty text.
            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript_path))
                    collector.set_assistant_message_fallback([{"role": "user"}])
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            # Only user_message from file; no assistant_message fallback.
            assert types == ["user_message"]

        asyncio.run(run_test())

    def test_fallbacks_with_no_main_tailer(self) -> None:
        """_emit_fallbacks handles missing main tailer gracefully."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="no_tailer_test",
                config=TranscriptCollectorConfig(poll_interval=60),
            )
            collector.set_user_message_fallback("hello")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    pass  # No transcript path set -> no main tailer.

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            # Fallback should fire since observed_types is empty set.
            types = [c["entry_type"] for c in collected]
            assert "user_message" in types

        asyncio.run(run_test())
