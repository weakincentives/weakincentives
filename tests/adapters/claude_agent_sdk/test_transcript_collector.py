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

"""Tests for transcript collection from Claude Agent SDK execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.adapters.claude_agent_sdk._transcript_collector import (
    TranscriptCollector,
    TranscriptCollectorConfig,
    _extract_assistant_text,
    _extract_text_from_content,
)
from weakincentives.adapters.claude_agent_sdk._transcript_parser import emit_entry


class TestTranscriptCollector:
    """Tests for TranscriptCollector."""

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

    def test_main_transcript_discovery(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Collector discovers main transcript from hook."""
        # Create a mock transcript file
        transcript_path = temp_dir / "session123.jsonl"
        transcript_path.touch()

        # Call hook callback with transcript path
        result = asyncio.run(
            collector.hook_callback(
                input_data={"transcript_path": str(transcript_path)},
                tool_use_id=None,
                context=None,
            )
        )

        # Should return empty dict (no modifications)
        assert result == {}

        # Should have discovered the transcript
        assert collector._main_transcript_path == transcript_path
        assert collector._session_dir == temp_dir / "session123"
        assert "main" in collector._tailers

    def test_subagent_discovery(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Collector discovers sub-agent transcripts."""
        # Set up main transcript first
        main_path = temp_dir / "session123.jsonl"
        main_path.touch()
        asyncio.run(collector._remember_transcript_path(str(main_path)))

        # Create subagent directory and transcripts
        subagents_dir = temp_dir / "session123" / "subagents"
        subagents_dir.mkdir(parents=True)

        agent1 = subagents_dir / "agent-001.jsonl"
        agent2 = subagents_dir / "agent-002.jsonl"
        agent1.touch()
        agent2.touch()

        # Run discovery
        asyncio.run(collector._discover_subagents())

        # Should have discovered both subagents
        assert "subagent:001" in collector._tailers
        assert "subagent:002" in collector._tailers
        assert collector.subagent_count == 2

    def test_entry_emission(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Entries emitted via TranscriptEmitter with correct canonical types."""
        # Create transcript with test entry
        transcript_path = temp_dir / "session123.jsonl"
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
        }
        transcript_path.write_text(json.dumps(entry) + "\n")

        # Set up collector with the transcript
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))

        # Mock the emitter's logger to capture emissions
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            # Poll for content
            asyncio.run(collector._poll_once())

            # Should have emitted the entry via TranscriptEmitter
            entry_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if call[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            context = entry_calls[0][1]["context"]
            assert context["prompt_name"] == "test-prompt"
            assert context["adapter"] == "claude_agent_sdk"
            assert context["source"] == "main"
            assert context["entry_type"] == "user_message"
            assert context["sequence_number"] == 1
            assert context["detail"]["sdk_entry"]["type"] == "user"

    def test_rotation_handling(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Tailer handles file rotation correctly."""
        # Create initial transcript
        transcript_path = temp_dir / "session123.jsonl"
        transcript_path.write_text('{"type": "user", "message": "first"}\n')

        # Set up collector
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))
        tailer = collector._tailers["main"]

        # Read initial content
        asyncio.run(collector._poll_once())
        assert tailer.entry_count == 1

        # Simulate rotation (new file with same name)
        transcript_path.unlink()
        transcript_path.write_text('{"type": "assistant", "message": "rotated"}\n')

        # Poll again - should detect rotation and reset
        asyncio.run(collector._poll_once())

        # Should have read from the new file
        assert tailer.entry_count == 2  # 1 from old + 1 from new
        assert tailer.position > 0

    def test_compaction_handling(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """PreCompact hook triggers snapshot."""
        # Create transcript with initial content
        transcript_path = temp_dir / "session123.jsonl"
        transcript_path.write_text('{"type": "user", "message": "before"}\n' * 10)

        # Set up collector
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))
        tailer = collector._tailers["main"]

        # Read initial content
        asyncio.run(collector._poll_once())
        initial_count = tailer.entry_count

        # Simulate compaction (truncate file)
        transcript_path.write_text(
            '{"type": "summary", "dropped_count": 9}\n{"type": "user", "message": "after"}\n'
        )

        # Poll again - should detect truncation and reset position
        asyncio.run(collector._poll_once())

        # Should have reset position and read new content
        assert tailer.position > 0
        assert tailer.entry_count == initial_count + 2  # summary + new entry

    def test_nonexistent_path_error_handling(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """File errors logged, not raised for nonexistent paths."""
        # Set up with non-existent path
        fake_path = temp_dir / "nonexistent" / "session.jsonl"

        with patch(
            "weakincentives.adapters.claude_agent_sdk._transcript_collector.logger"
        ) as mock_logger:
            # Should not raise, just log warning
            asyncio.run(collector._remember_transcript_path(str(fake_path)))

            # No tailer should be created
            assert "main" not in collector._tailers

            # Should have logged warning
            mock_logger.warning.assert_called()

    def test_hooks_config(self, collector: TranscriptCollector) -> None:
        """Hooks configuration returns correct structure."""
        hooks = collector.hooks_config()

        # Should have all supported hook types
        expected_hooks = [
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "SubagentStop",
            "Stop",
            "PreCompact",
        ]

        for hook_type in expected_hooks:
            assert hook_type in hooks
            # Each should have a HookMatcher with a hook function
            matchers = hooks[hook_type]
            assert len(matchers) == 1
            # The hook should be a callable (async function)
            assert callable(matchers[0].hooks[0])

    def test_partial_line_buffering(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Partial lines are buffered until complete."""
        transcript_path = temp_dir / "session123.jsonl"

        # Write partial entry
        transcript_path.write_text('{"type": "user", ')

        # Set up collector
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))
        tailer = collector._tailers["main"]

        # Poll - should buffer partial line
        asyncio.run(collector._poll_once())
        assert tailer.entry_count == 0
        assert tailer.partial_line == '{"type": "user", '

        # Complete the entry
        with transcript_path.open("a") as f:
            f.write('"message": "complete"}\n')

        # Poll again - should emit complete entry
        asyncio.run(collector._poll_once())
        assert tailer.entry_count == 1
        assert tailer.partial_line == ""

    def test_context_manager_lifecycle(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Collector starts and stops correctly as context manager."""
        transcript_path = temp_dir / "session123.jsonl"
        transcript_path.write_text('{"type": "user", "message": "test"}\n')

        # Set up transcript before running
        asyncio.run(collector._remember_transcript_path(str(transcript_path)))

        async def run_test() -> None:
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    # Should be running
                    assert collector._running
                    assert collector._poll_task is not None
                    assert collector._discovery_task is not None

                    # Wait a bit for polling
                    await asyncio.sleep(0.05)

                # Should have stopped
                assert not collector._running

                # Should have logged start and stop
                start_calls = [
                    call
                    for call in mock_logger.debug.call_args_list
                    if call[1].get("event") == "transcript.start"
                ]
                stop_calls = [
                    call
                    for call in mock_logger.debug.call_args_list
                    if call[1].get("event") == "transcript.stop"
                ]
                assert len(start_calls) == 1
                assert len(stop_calls) == 1

        asyncio.run(run_test())

    def test_multiple_transcripts_parallel(
        self, collector: TranscriptCollector, temp_dir: Path
    ) -> None:
        """Multiple transcripts can be tailed in parallel."""
        # Set up main transcript
        main_path = temp_dir / "session123.jsonl"
        main_path.write_text('{"type": "user", "message": "main"}\n')
        asyncio.run(collector._remember_transcript_path(str(main_path)))

        # Set up subagent transcripts
        subagents_dir = temp_dir / "session123" / "subagents"
        subagents_dir.mkdir(parents=True)

        agent1 = subagents_dir / "agent-001.jsonl"
        agent2 = subagents_dir / "agent-002.jsonl"
        agent1.write_text('{"type": "assistant", "message": "agent1"}\n')
        agent2.write_text('{"type": "tool_result", "content": "result"}\n')

        # Discover subagents
        asyncio.run(collector._discover_subagents())

        # Poll all transcripts
        asyncio.run(collector._poll_once())

        # Check statistics
        assert collector.main_entry_count == 1
        assert collector.subagent_count == 2
        assert collector.total_entries == 3
        assert len(collector.transcript_paths) == 3

    def test_invalid_json_handling(self, tmp_path: Path) -> None:
        """Test handling of invalid JSON in transcript files."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="json_error_test",
                config=TranscriptCollectorConfig(
                    poll_interval=0.01,
                ),
            )

            # Create transcript with invalid JSON
            transcript = tmp_path / "test.jsonl"
            transcript.write_text(
                '{"valid": "entry"}\ninvalid json here\n{"another": "valid"}\n'
            )

            async with collector.run():
                await collector._remember_transcript_path(str(transcript))
                await asyncio.sleep(0.05)

            # Should have processed 3 entries (1 valid, 1 invalid, 1 valid)
            assert collector.main_entry_count == 3

        asyncio.run(run_test())

    def test_duplicate_transcript_path(self, tmp_path: Path) -> None:
        """Test that duplicate transcript paths are ignored."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="duplicate_test",
                config=TranscriptCollectorConfig(),
            )

            transcript = tmp_path / "test.jsonl"
            transcript.write_text('{"type": "test"}\n')

            # Remember the same path twice
            await collector._remember_transcript_path(str(transcript))
            await collector._remember_transcript_path(str(transcript))

            # Should only have one tailer
            assert len(collector._tailers) == 1
            assert "main" in collector._tailers

        asyncio.run(run_test())

    def test_duplicate_tailer_start(self, tmp_path: Path) -> None:
        """Test that starting the same tailer twice is handled."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="duplicate_tailer_test",
                config=TranscriptCollectorConfig(),
            )

            transcript = tmp_path / "test.jsonl"
            transcript.write_text('{"type": "test"}\n')

            # Start the same tailer twice
            await collector._start_tailer(transcript, "main")
            await collector._start_tailer(transcript, "main")

            # Should only have one tailer
            assert len(collector._tailers) == 1

        asyncio.run(run_test())

    def test_subagent_discovery_with_missing_dir(self, tmp_path: Path) -> None:
        """Test subagent discovery when subagents directory doesn't exist."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="missing_dir_test",
                config=TranscriptCollectorConfig(),
            )

            # Set session dir but don't create subagents directory
            collector._session_dir = tmp_path / "session"

            # Should handle gracefully
            await collector._discover_subagents()

            # No subagents discovered
            assert collector.subagent_count == 0

        asyncio.run(run_test())

    def test_hook_callback_wrapper(self, tmp_path: Path) -> None:
        """Test the hook callback wrapper function."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="hook_wrapper_test",
                config=TranscriptCollectorConfig(),
            )

            # Get the hooks config
            hooks = collector.hooks_config()

            # Get a hook function (e.g., UserPromptSubmit)
            hook_matchers = hooks.get("UserPromptSubmit", [])
            assert len(hook_matchers) > 0

            # Get the hook function from the matcher
            hook_fn = hook_matchers[0].hooks[0]

            # Call the wrapper function
            input_data = {"transcript_path": str(tmp_path / "test.jsonl")}
            result = await hook_fn(input_data, None, None)

            # Should return empty dict
            assert result == {}

            # Transcript path should be remembered
            assert collector._main_transcript_path == tmp_path / "test.jsonl"

        asyncio.run(run_test())

    def test_graceful_error_handling(self, tmp_path: Path) -> None:
        """Test graceful handling of file access errors."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="error_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            # Create a transcript file
            transcript = tmp_path / "test.jsonl"
            transcript.write_text('{"type": "test"}\n')

            async with collector.run():
                await collector._remember_transcript_path(str(transcript))
                await asyncio.sleep(0.02)

                # Delete the file to trigger error handling
                transcript.unlink()
                await collector._poll_once()  # Should handle missing file gracefully

            # Collector should still be functional
            assert collector.main_entry_count >= 1

        asyncio.run(run_test())

    def test_subagent_discovery_with_oserror(self, tmp_path: Path) -> None:
        """Test subagent discovery handles OSError gracefully."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="oserror_test",
                config=TranscriptCollectorConfig(),
            )

            # Create a session dir that exists
            session_dir = tmp_path / "session"
            session_dir.mkdir()
            subagents_dir = session_dir / "subagents"
            subagents_dir.mkdir()

            # Create a file that will cause an error when globbing
            bad_file = subagents_dir / "agent-bad.jsonl"
            bad_file.touch()

            collector._session_dir = session_dir

            # Mock glob to raise OSError
            with patch("pathlib.Path.glob", side_effect=OSError("Permission denied")):
                await collector._discover_subagents()

            # No subagents discovered due to error
            assert collector.subagent_count == 0

        asyncio.run(run_test())

    def test_file_read_error(self, tmp_path: Path) -> None:
        """Test handling of file read errors."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="read_error_test",
                config=TranscriptCollectorConfig(),
            )

            # Create a transcript file
            transcript = tmp_path / "test.jsonl"
            transcript.write_text('{"type": "test"}\n')

            # Start tailing
            await collector._remember_transcript_path(str(transcript))
            tailer = collector._tailers["main"]

            # Mock the _read_bytes method to raise OSError
            with patch.object(
                TranscriptCollector, "_read_bytes", side_effect=OSError("Read error")
            ):
                with patch(
                    "weakincentives.adapters.claude_agent_sdk._transcript_collector.logger"
                ) as mock_logger:
                    await collector._read_transcript_content(tailer)

                    # Should have logged warning
                    mock_logger.warning.assert_called()

        asyncio.run(run_test())

    def test_no_session_dir(self, tmp_path: Path) -> None:
        """Test subagent discovery when session_dir is None."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="no_session_test",
                config=TranscriptCollectorConfig(),
            )

            # Session dir is None by default
            assert collector._session_dir is None

            # Should return early
            await collector._discover_subagents()

            # No subagents discovered
            assert collector.subagent_count == 0

        asyncio.run(run_test())

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="empty_lines_test",
                config=TranscriptCollectorConfig(),
            )

            # Create transcript with empty lines
            transcript = tmp_path / "test.jsonl"
            transcript.write_text('{"type": "entry1"}\n\n{"type": "entry2"}\n\n\n')

            async with collector.run():
                await collector._remember_transcript_path(str(transcript))
                await asyncio.sleep(0.05)

            # Should have processed only 2 entries (skipping empty lines)
            assert collector.main_entry_count == 2

        asyncio.run(run_test())

    def test_sdk_type_mapping(self, tmp_path: Path) -> None:
        """Verify SDK transcript types map to canonical entry types."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="type_map_test",
                config=TranscriptCollectorConfig(poll_interval=0.01),
            )

            entries = [
                '{"type": "user", "message": {"role": "user"}}',
                '{"type": "assistant", "message": {"role": "assistant"}}',
                '{"type": "tool_result", "tool_use_id": "123"}',
                '{"type": "thinking", "thinking": "hmm"}',
                '{"type": "summary", "dropped_count": 5}',
                '{"type": "system", "event": "lifecycle"}',
                '{"type": "never_seen_before"}',
            ]
            transcript = tmp_path / "test.jsonl"
            transcript.write_text("\n".join(entries) + "\n")

            collected: list[dict] = []
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                async with collector.run():
                    await collector._remember_transcript_path(str(transcript))
                    await asyncio.sleep(0.05)

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            types = [c["entry_type"] for c in collected]
            assert types == [
                "user_message",
                "assistant_message",
                "tool_result",
                "thinking",
                "system_event",
                "system_event",
                "unknown",
            ]

        asyncio.run(run_test())

    def test_hook_callback_without_transcript_path(
        self, collector: TranscriptCollector
    ) -> None:
        """Hook callback returns and does not discover without transcript_path."""
        result = asyncio.run(
            collector.hook_callback(
                input_data={},
                tool_use_id=None,
                context=None,
            )
        )
        assert result == {}
        assert collector._main_transcript_path is None

    def test_poll_loop_exits_when_not_running(
        self, collector: TranscriptCollector
    ) -> None:
        """Polling loop should exit immediately when _running is False."""
        assert collector._running is False
        asyncio.run(collector._poll_loop())

    def test_context_manager_handles_missing_task_refs(
        self, collector: TranscriptCollector
    ) -> None:
        """Context manager stop path handles missing task refs."""

        async def run_test() -> None:
            async with collector.run():
                poll_task = collector._poll_task
                discovery_task = collector._discovery_task
                assert poll_task is not None
                assert discovery_task is not None

                # Simulate losing references to the tasks before exit.
                collector._poll_task = None
                collector._discovery_task = None

            # Clean up tasks since collector.run() couldn't cancel them.
            for task in (poll_task, discovery_task):
                if task is None or task.done():
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        asyncio.run(run_test())

    def test_read_transcript_content_no_bytes_read(self, tmp_path: Path) -> None:
        """Read path should handle empty reads without emitting entries."""

        async def run_test() -> None:
            collector = TranscriptCollector(
                prompt_name="max_read_bytes_zero_test",
                config=TranscriptCollectorConfig(max_read_bytes=0),
            )

            transcript_path = tmp_path / "session123.jsonl"
            transcript_path.write_text('{"type": "user"}\n')
            await collector._remember_transcript_path(str(transcript_path))
            tailer = collector._tailers["main"]
            assert tailer.position == 0
            assert tailer.entry_count == 0

            await collector._read_transcript_content(tailer)

            # With max_read_bytes=0, no bytes are read and no updates occur.
            assert tailer.position == 0
            assert tailer.entry_count == 0

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
        """User message with text + tool_result → user_message + tool_result."""

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
                    pass  # No transcript path set → no main tailer.

                for call in mock_logger.debug.call_args_list:
                    if call[1].get("event") == "transcript.entry":
                        collected.append(call[1]["context"])

            # Fallback should fire since observed_types is empty set.
            types = [c["entry_type"] for c in collected]
            assert "user_message" in types

        asyncio.run(run_test())
