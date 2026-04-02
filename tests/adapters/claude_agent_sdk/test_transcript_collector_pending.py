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

"""Tests for TranscriptCollector pending tailer retry and inode reset."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.adapters.claude_agent_sdk._transcript_collector import (
    TranscriptCollector,
    TranscriptCollectorConfig,
)
from weakincentives.adapters.claude_agent_sdk._transcript_parser import emit_entry


class TestTranscriptCollectorPending:
    """Tests for TranscriptCollector pending tailer retry and drain behaviour."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test transcripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TranscriptCollectorConfig(
            poll_interval=0.01,  # Fast polling for tests
            subagent_discovery_interval=0.02,
            max_read_bytes=1024,
            emit_raw=True,
        )

    @pytest.fixture
    def collector(self, config: TranscriptCollectorConfig) -> TranscriptCollector:
        """Create a test collector."""
        return TranscriptCollector(
            prompt_name="test-prompt",
            config=config,
        )

    def test_late_file_creation_retried(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """File that doesn't exist at hook time is tailed once it appears."""
        transcript_path = temp_dir / "session123.jsonl"

        # Hook fires before the CLI creates the file on disk.
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))

        # No active tailer, but the path is pending for retry.
        assert "main" not in collector._tailers
        assert "main" in collector._pending_tailers

        # CLI creates the file a moment later.
        transcript_path.write_text('{"type": "user", "message": "hello"}\n')

        # Next poll cycle retries the pending tailer and reads content.
        asyncio.run(collector._poll_once())

        assert "main" in collector._tailers
        assert "main" not in collector._pending_tailers
        assert collector.main_entry_count == 1

    def test_pending_tailer_stays_pending_until_file_exists(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Pending tailer remains pending while file is still absent."""
        transcript_path = temp_dir / "session123.jsonl"

        asyncio.run(collector._remember_transcript_path(str(transcript_path)))
        assert "main" in collector._pending_tailers

        # Poll several times — file still missing.
        asyncio.run(collector._poll_once())
        asyncio.run(collector._poll_once())

        assert "main" not in collector._tailers
        assert "main" in collector._pending_tailers

    def test_pending_warning_logged_once(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Warning is logged on first failure only, not on every retry."""
        transcript_path = temp_dir / "session123.jsonl"

        with patch(
            "weakincentives.adapters.claude_agent_sdk._transcript_collector.logger"
        ) as mock_logger:
            # First call logs warning.
            asyncio.run(collector._remember_transcript_path(str(transcript_path)))
            assert mock_logger.warning.call_count == 1

            # Retries via _poll_once do NOT log additional warnings.
            asyncio.run(collector._poll_once())
            asyncio.run(collector._poll_once())
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
                # Hook fires — file does not exist yet.
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
                # Hook fires — file does not exist yet.
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
                # Hook fires — file does not exist.
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
                # File exists → no pending tailers.
                assert not collector._pending_tailers

            # Drain loop should have exited immediately (no sleep delay).
            assert collector.main_entry_count == 1

        asyncio.run(run_test())

    def test_inode_change_triggers_position_reset(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Tailer resets position when file inode changes (rotation)."""
        # Create initial transcript
        transcript_path = temp_dir / "session123.jsonl"
        transcript_path.write_text('{"type": "user", "message": "first"}\n')

        # Set up collector
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))
        tailer = collector._tailers["main"]

        # Read initial content
        asyncio.run(collector._poll_once())
        original_inode = tailer.inode
        assert tailer.entry_count == 1

        # Simulate inode change by modifying the tailer's stored inode
        tailer.inode = original_inode + 12345
        tailer.partial_line = "incomplete"

        # Write new content
        with transcript_path.open("a") as f:
            f.write('{"type": "assistant", "message": "second"}\n')

        # Poll - should detect inode mismatch and reset
        asyncio.run(collector._poll_once())

        # Verify position and partial_line were reset, inode updated
        assert tailer.inode == original_inode
        assert tailer.partial_line == ""

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
