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

"""Tests for TranscriptCollector: pending tailers, late creation, shutdown, fallbacks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from weakincentives.adapters.claude_agent_sdk._transcript_collector import (
    TranscriptCollector,
    TranscriptCollectorConfig,
)


class TestTranscriptCollectorPending:
    """Tests for pending tailers, late file creation, and shutdown drain."""

    def test_late_file_creation_retried(
        self,
        transcript_collector: TranscriptCollector,
        transcript_temp_dir: Path,
    ) -> None:
        """File that doesn't exist at hook time is tailed once it appears."""
        transcript_path = transcript_temp_dir / "session123.jsonl"

        # Hook fires before the CLI creates the file on disk.
        asyncio.run(
            transcript_collector._remember_transcript_path(str(transcript_path))
        )

        # No active tailer, but the path is pending for retry.
        assert "main" not in transcript_collector._tailers
        assert "main" in transcript_collector._pending_tailers

        # CLI creates the file a moment later.
        transcript_path.write_text('{"type": "user", "message": "hello"}\n')

        # Next poll cycle retries the pending tailer and reads content.
        asyncio.run(transcript_collector._poll_once())

        assert "main" in transcript_collector._tailers
        assert "main" not in transcript_collector._pending_tailers
        assert transcript_collector.main_entry_count == 1

    def test_pending_tailer_stays_pending_until_file_exists(
        self,
        transcript_collector: TranscriptCollector,
        transcript_temp_dir: Path,
    ) -> None:
        """Pending tailer remains pending while file is still absent."""
        transcript_path = transcript_temp_dir / "session123.jsonl"

        asyncio.run(
            transcript_collector._remember_transcript_path(str(transcript_path))
        )
        assert "main" in transcript_collector._pending_tailers

        # Poll several times - file still missing.
        asyncio.run(transcript_collector._poll_once())
        asyncio.run(transcript_collector._poll_once())

        assert "main" not in transcript_collector._tailers
        assert "main" in transcript_collector._pending_tailers

    def test_pending_warning_logged_once(
        self,
        transcript_collector: TranscriptCollector,
        transcript_temp_dir: Path,
    ) -> None:
        """Warning is logged on first failure only, not on every retry."""
        transcript_path = transcript_temp_dir / "session123.jsonl"

        with patch(
            "weakincentives.adapters.claude_agent_sdk._transcript_collector.logger"
        ) as mock_logger:
            # First call logs warning.
            asyncio.run(
                transcript_collector._remember_transcript_path(str(transcript_path))
            )
            assert mock_logger.warning.call_count == 1

            # Retries via _poll_once do NOT log additional warnings.
            asyncio.run(transcript_collector._poll_once())
            asyncio.run(transcript_collector._poll_once())
            assert mock_logger.warning.call_count == 1

    def test_late_file_creation_with_context_manager(self, tmp_path: Path) -> None:
        """End-to-end: file appears after collector.run() starts."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="late_file_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            transcript_path = tmp_path / "session123.jsonl"

            async with collector.run():
                # Hook fires - file does not exist yet.
                await collector._remember_transcript_path(str(transcript_path))
                assert "main" not in collector._tailers

                # Wait one poll cycle.
                await asyncio.sleep(0.02)

                # File appears.
                transcript_path.write_text('{"type": "user", "message": "late"}\n')

                # Wait for another poll cycle to pick it up.
                await asyncio.sleep(0.05)

            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_shutdown_drain_resolves_pending_tailer(self, tmp_path: Path) -> None:
        """Shutdown drain polls resolve pending tailers when file appears."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="shutdown_drain_test",
                config=TranscriptCollectorConfig(poll_interval=60),
            )

            transcript_path = tmp_path / "session456.jsonl"

            async with collector.run():
                # Hook fires - file does not exist yet.
                await collector._remember_transcript_path(str(transcript_path))
                assert "main" not in collector._tailers
                assert "main" in collector._pending_tailers

                # File appears just before context exit (simulating SDK flush).
                transcript_path.write_text('{"type": "user", "message": "shutdown"}\n')

            # Shutdown drain should have picked up the file.
            assert collector.main_entry_count == 1
            assert "main" not in collector._pending_tailers

        asyncio.run(run_test())

    def test_shutdown_drain_completes_without_error(self, tmp_path: Path) -> None:
        """Shutdown drain completes gracefully when file never appears."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="drain_no_file_test",
                config=TranscriptCollectorConfig(poll_interval=60),
            )

            transcript_path = tmp_path / "never_created.jsonl"

            async with collector.run():
                # Hook fires - file does not exist.
                await collector._remember_transcript_path(str(transcript_path))
                assert "main" in collector._pending_tailers

            # All drain polls completed, file never appeared.
            assert "main" not in collector._tailers
            assert collector.main_entry_count == 0

        asyncio.run(run_test())

    def test_shutdown_drain_skips_when_no_pending(self, tmp_path: Path) -> None:
        """Drain loop exits immediately when there are no pending tailers."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="drain_skip_test",
                config=TranscriptCollectorConfig(poll_interval=60),
            )

            transcript_path = tmp_path / "session789.jsonl"
            transcript_path.write_text('{"type": "user", "message": "hi"}\n')

            async with collector.run():
                await collector._remember_transcript_path(str(transcript_path))
                # File exists -> no pending tailers.
                assert not collector._pending_tailers

            # Drain loop should have exited immediately (no sleep delay).
            assert collector.main_entry_count == 1

        asyncio.run(run_test())


class TestTranscriptCollectorFallbacks:
    """Tests for user/assistant message fallback emission."""

    def test_fallback_user_message_emitted_when_not_in_file(
        self, tmp_path: Path
    ) -> None:
        """Fallback user_message is emitted when file has no user entry."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="fallback_user_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            # Transcript with only an assistant entry (no user entry).
            transcript_path = tmp_path / "session.jsonl"
            entry = {
                "type": "assistant",
                "message": {"role": "assistant", "content": "reply"},
            }
            transcript_path.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript_path))
                    collector.set_user_message_fallback("Hello from user")
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            # File had assistant_message; fallback adds user_message.
            assert "assistant_message" in types
            assert "user_message" in types

        asyncio.run(run_test())

    def test_fallback_user_message_not_emitted_when_in_file(
        self, tmp_path: Path
    ) -> None:
        """Fallback user_message is NOT emitted when file already has user entry."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="no_fallback_user_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            # Transcript with a user entry.
            transcript_path = tmp_path / "session.jsonl"
            entry = {
                "type": "user",
                "message": {"role": "user", "content": "from file"},
            }
            transcript_path.write_text(json.dumps(entry) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript_path))
                    collector.set_user_message_fallback("Hello from user")
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            # Only one user_message (from file), not two.
            assert types.count("user_message") == 1

        asyncio.run(run_test())

    def test_fallback_assistant_message_emitted_when_not_in_file(
        self, tmp_path: Path
    ) -> None:
        """Fallback assistant_message emitted when file has no assistant entry."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="fallback_assistant_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            # Transcript with only a user entry (no assistant entry).
            transcript_path = tmp_path / "session.jsonl"
            entry = {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
            }
            transcript_path.write_text(json.dumps(entry) + "\n")

            # Create a fake SDK message for fallback.
            class FakeAssistantMessage:
                content = "I can help with that"

            FakeAssistantMessage.__name__ = "AssistantMessage"

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript_path))
                    collector.set_assistant_message_fallback([FakeAssistantMessage()])
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert "user_message" in types
            assert "assistant_message" in types

        asyncio.run(run_test())
