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

"""Tests for transcript text extraction helpers."""

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
