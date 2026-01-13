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

"""Tests for Claude Agent SDK log aggregation module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

from weakincentives.adapters.claude_agent_sdk._log_aggregator import (
    _EXCLUDED_PATHS,
    _LOG_EXTENSIONS,
    ClaudeLogAggregator,
    _FileState,
)


class TestFileState:
    def test_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "test.log"
        path.write_text("content")
        state = _FileState(path=path)
        assert state.path == path
        assert state.position == 0
        assert state.inode == 0
        assert state.partial_line == ""

    def test_with_position(self, tmp_path: Path) -> None:
        path = tmp_path / "test.log"
        path.write_text("content")
        state = _FileState(path=path, position=100, inode=12345, partial_line="partial")
        assert state.position == 100
        assert state.inode == 12345
        assert state.partial_line == "partial"


class TestClaudeLogAggregatorConstants:
    def test_excluded_paths_contains_settings(self) -> None:
        assert "settings.json" in _EXCLUDED_PATHS
        assert "skills" in _EXCLUDED_PATHS

    def test_log_extensions_includes_common_types(self) -> None:
        assert ".log" in _LOG_EXTENSIONS
        assert ".jsonl" in _LOG_EXTENSIONS
        assert ".txt" in _LOG_EXTENSIONS
        assert ".json" in _LOG_EXTENSIONS


class TestClaudeLogAggregatorConstruction:
    def test_basic_construction(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        aggregator = ClaudeLogAggregator(
            claude_dir=claude_dir,
            prompt_name="test-prompt",
        )
        assert aggregator.claude_dir == claude_dir
        assert aggregator.prompt_name == "test-prompt"
        assert aggregator._running is False
        assert aggregator._total_bytes_read == 0
        assert aggregator._total_lines_emitted == 0

    def test_custom_poll_interval(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        aggregator = ClaudeLogAggregator(
            claude_dir=claude_dir,
            prompt_name="test-prompt",
            poll_interval=0.1,
        )
        assert aggregator.poll_interval == 0.1


class TestClaudeLogAggregatorExclusion:
    def test_excludes_settings_json(self) -> None:
        assert ClaudeLogAggregator._is_excluded(Path("settings.json")) is True

    def test_excludes_skills_directory(self) -> None:
        assert ClaudeLogAggregator._is_excluded(Path("skills")) is True
        assert ClaudeLogAggregator._is_excluded(Path("skills/my-skill")) is True
        assert (
            ClaudeLogAggregator._is_excluded(Path("skills/my-skill/SKILL.md")) is True
        )

    def test_does_not_exclude_log_files(self) -> None:
        assert ClaudeLogAggregator._is_excluded(Path("debug.log")) is False
        assert ClaudeLogAggregator._is_excluded(Path("transcript.jsonl")) is False


class TestClaudeLogAggregatorLogFileDetection:
    def test_detects_log_extension(self) -> None:
        assert ClaudeLogAggregator._is_log_file(Path("test.log")) is True
        assert ClaudeLogAggregator._is_log_file(Path("test.jsonl")) is True
        assert ClaudeLogAggregator._is_log_file(Path("test.txt")) is True
        assert ClaudeLogAggregator._is_log_file(Path("test.json")) is True

    def test_detects_log_in_name(self) -> None:
        assert ClaudeLogAggregator._is_log_file(Path("debug_log")) is True
        assert ClaudeLogAggregator._is_log_file(Path("transcript_file")) is True

    def test_rejects_non_log_files(self) -> None:
        assert ClaudeLogAggregator._is_log_file(Path("config.yaml")) is False
        assert ClaudeLogAggregator._is_log_file(Path("data.bin")) is False


class TestClaudeLogAggregatorPollOnce:
    def test_poll_once_with_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test poll_once returns early when directory doesn't exist."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"  # Does not exist

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Should not raise, just return early
            await aggregator._poll_once()
            assert len(aggregator._file_states) == 0

        asyncio.run(_test())

    def test_poll_once_discovers_and_reads(self, tmp_path: Path) -> None:
        """Test poll_once discovers files and reads content."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            (claude_dir / "test.log").write_text("test content\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._poll_once()

            assert len(aggregator._file_states) == 1
            assert aggregator._total_bytes_read > 0

        asyncio.run(_test())


class TestClaudeLogAggregatorDiscovery:
    def test_discovers_new_log_files(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Create log file after aggregator construction
            log_file = claude_dir / "debug.log"
            log_file.write_text("test line")

            await aggregator._discover_files()

            assert log_file in aggregator._file_states
            assert aggregator._file_states[log_file].position == 0

        asyncio.run(_test())

    def test_does_not_discover_excluded_files(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            # Create excluded files
            (claude_dir / "settings.json").write_text("{}")
            skills_dir = claude_dir / "skills"
            skills_dir.mkdir()
            (skills_dir / "test-skill").mkdir()
            (skills_dir / "test-skill" / "SKILL.md").write_text("# Skill")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()

            assert claude_dir / "settings.json" not in aggregator._file_states
            for path in aggregator._file_states:
                assert "skills" not in str(path)

        asyncio.run(_test())

    def test_skips_directories(self, tmp_path: Path) -> None:
        """Test that directories are skipped during discovery."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            # Create a directory that might match log patterns
            log_dir = claude_dir / "logs"
            log_dir.mkdir()

            # And a file inside it
            (log_dir / "test.log").write_text("content")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()

            # Should only track the file, not the directory
            assert len(aggregator._file_states) == 1
            assert claude_dir / "logs" / "test.log" in aggregator._file_states

        asyncio.run(_test())

    def test_skips_non_log_files(self, tmp_path: Path) -> None:
        """Test that non-log files are skipped during discovery."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            # Create non-log files
            (claude_dir / "config.yaml").write_text("key: value")
            (claude_dir / "data.bin").write_bytes(b"\x00\x01\x02")

            # And a log file
            (claude_dir / "debug.log").write_text("log content")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()

            # Should only track the log file
            assert len(aggregator._file_states) == 1
            assert claude_dir / "debug.log" in aggregator._file_states

        asyncio.run(_test())

    def test_handles_nonexistent_directory(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"  # Does not exist

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Should not raise
            await aggregator._discover_files()
            assert len(aggregator._file_states) == 0

        asyncio.run(_test())


class TestClaudeLogAggregatorReading:
    def test_reads_new_content(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("line 1\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()
            await aggregator._read_new_content()

            assert aggregator._file_states[log_file].position > 0
            assert aggregator._total_bytes_read > 0

        asyncio.run(_test())

    def test_tracks_file_position(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("line 1\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()
            await aggregator._read_new_content()
            initial_position = aggregator._file_states[log_file].position

            # Append more content
            with log_file.open("a") as f:
                f.write("line 2\n")

            await aggregator._read_new_content()

            # Position should have advanced
            assert aggregator._file_states[log_file].position > initial_position

        asyncio.run(_test())

    def test_handles_file_rotation(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("old content\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            await aggregator._discover_files()
            await aggregator._read_new_content()
            old_inode = aggregator._file_states[log_file].inode

            # Simulate file rotation (delete and recreate)
            log_file.unlink()
            log_file.write_text("new content\n")

            await aggregator._read_new_content()

            # Inode should have changed and position should be reset
            new_state = aggregator._file_states[log_file]
            if new_state.inode != old_inode:
                # Position was reset due to rotation
                assert new_state.position <= len("new content\n")

        asyncio.run(_test())

    def test_inode_change_resets_position(self, tmp_path: Path) -> None:
        """Test that inode change detection resets position."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("content\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Discover and read the file
            await aggregator._discover_files()
            await aggregator._read_new_content()

            state = aggregator._file_states[log_file]

            # Artificially set a different inode to simulate rotation
            # This tests the inode check code path directly
            state.inode = 0  # Set to invalid inode

            # Write more content to file
            with log_file.open("a") as f:
                f.write("more content\n")

            # Read again - inode mismatch should reset position
            await aggregator._read_new_content()

            # Position should have been reset and then read the whole file
            # The new inode should be captured
            new_state = aggregator._file_states[log_file]
            assert new_state.inode != 0  # Should have new inode now
            assert new_state.position > 0  # Should have read content

        asyncio.run(_test())

    def test_handles_file_deleted_before_read(self, tmp_path: Path) -> None:
        """Test handling of file deleted after tracking but before read."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("initial content\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Discover and track the file
            await aggregator._discover_files()
            assert log_file in aggregator._file_states

            # Delete the file before reading
            log_file.unlink()

            # Should not raise
            await aggregator._read_new_content()

            # Position should remain unchanged (file not read)
            assert aggregator._file_states[log_file].position == 0

        asyncio.run(_test())

    def test_no_read_when_no_new_content(self, tmp_path: Path) -> None:
        """Test that no read occurs when file hasn't grown."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("content\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Discover and read initial content
            await aggregator._discover_files()
            await aggregator._read_new_content()
            initial_bytes = aggregator._total_bytes_read

            # Read again without new content
            await aggregator._read_new_content()

            # No additional bytes should have been read
            assert aggregator._total_bytes_read == initial_bytes

        asyncio.run(_test())

    def test_handles_in_place_truncation(self, tmp_path: Path) -> None:
        """Test handling of file truncated in place (e.g., copytruncate rotation)."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("original content that is long\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Discover and read the file
            await aggregator._discover_files()
            await aggregator._read_new_content()
            old_position = aggregator._file_states[log_file].position
            assert old_position > 0

            # Simulate in-place truncation (file size becomes smaller)
            log_file.write_text("new\n")  # Smaller than original

            # Read again - should detect truncation and reset position
            await aggregator._read_new_content()

            # New content should be captured
            new_state = aggregator._file_states[log_file]
            # Position should be reset to capture the new content
            assert new_state.position == len("new\n")

        asyncio.run(_test())

    def test_truncation_clears_partial_line_buffer(self, tmp_path: Path) -> None:
        """Test that truncation clears the partial line buffer."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            log_file = claude_dir / "debug.log"
            log_file.write_text("line1\npartial")  # Ends without newline

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )

            # Discover and read the file
            await aggregator._discover_files()
            await aggregator._read_new_content()

            # Should have buffered the partial line
            state = aggregator._file_states[log_file]
            assert state.partial_line == "partial"

            # Truncate the file
            log_file.write_text("new\n")

            # Read again - truncation should clear partial buffer
            await aggregator._read_new_content()

            # Partial line should be cleared and new content read
            assert state.partial_line == ""
            # Should have emitted the new line
            assert aggregator._total_lines_emitted == 2  # line1 + new

        asyncio.run(_test())


class TestClaudeLogAggregatorEmission:
    def test_emits_log_lines(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "debug.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ):
                await aggregator._emit_content(
                    Path("debug.log"),
                    b"line 1\nline 2\n",
                    state,
                )

            # Should have emitted 2 lines
            assert aggregator._total_lines_emitted == 2

        asyncio.run(_test())

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "debug.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            await aggregator._emit_content(
                Path("debug.log"),
                b"line 1\n\n\nline 2\n",
                state,
            )

            # Should have emitted only 2 non-empty lines
            assert aggregator._total_lines_emitted == 2

        asyncio.run(_test())

    def test_handles_binary_content(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "binary.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            # Binary content with invalid UTF-8 - should be handled gracefully
            # with replacement characters and still emit lines
            binary_content = b"\x80\x81\x82invalid utf-8\n"

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ):
                # Should not raise - invalid bytes are replaced with replacement char
                await aggregator._emit_content(
                    Path("binary.log"), binary_content, state
                )

            # Should still emit a line (with replacement characters)
            assert aggregator._total_lines_emitted == 1

        asyncio.run(_test())

    def test_buffers_partial_lines(self, tmp_path: Path) -> None:
        """Test that partial lines are buffered until newline is seen."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "debug.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            # First read: complete line + partial line
            await aggregator._emit_content(
                Path("debug.log"),
                b"complete line\npartial",
                state,
            )

            # Should have emitted only 1 complete line
            assert aggregator._total_lines_emitted == 1
            # Partial line should be buffered
            assert state.partial_line == "partial"

            # Second read: rest of partial line + another complete line
            await aggregator._emit_content(
                Path("debug.log"),
                b" continued\nanother line\n",
                state,
            )

            # Should have emitted 2 more lines (the completed partial + another)
            assert aggregator._total_lines_emitted == 3
            assert state.partial_line == ""

        asyncio.run(_test())

    def test_partial_line_across_multiple_reads(self, tmp_path: Path) -> None:
        """Test partial line buffering across multiple read cycles."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "debug.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            # Read 1: just a partial
            await aggregator._emit_content(Path("debug.log"), b"part1", state)
            assert aggregator._total_lines_emitted == 0
            assert state.partial_line == "part1"

            # Read 2: more partial
            await aggregator._emit_content(Path("debug.log"), b"part2", state)
            assert aggregator._total_lines_emitted == 0
            assert state.partial_line == "part1part2"

            # Read 3: complete the line
            await aggregator._emit_content(Path("debug.log"), b"part3\n", state)
            assert aggregator._total_lines_emitted == 1
            assert state.partial_line == ""

        asyncio.run(_test())

    def test_content_ending_with_carriage_return(self, tmp_path: Path) -> None:
        """Test that carriage return is treated as line ending."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            log_file = claude_dir / "debug.log"
            log_file.write_text("")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
            )
            state = _FileState(path=log_file)

            # Content ending with \r should be treated as complete
            await aggregator._emit_content(Path("debug.log"), b"line\r", state)
            assert aggregator._total_lines_emitted == 1
            assert state.partial_line == ""

        asyncio.run(_test())


class TestClaudeLogAggregatorContextManager:
    def test_context_manager_starts_and_stops(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
                poll_interval=0.01,
            )

            async with aggregator.run():
                assert aggregator._running is True

            assert aggregator._running is False

        asyncio.run(_test())

    def test_context_manager_captures_logs_during_execution(
        self, tmp_path: Path
    ) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
                poll_interval=0.01,
            )

            async with aggregator.run():
                # Create log file during execution
                log_file = claude_dir / "runtime.log"
                log_file.write_text("runtime log entry\n")

                # Wait for a poll cycle
                await asyncio.sleep(0.05)

            # Should have discovered and read the file
            assert log_file in aggregator._file_states
            assert aggregator._total_bytes_read > 0

        asyncio.run(_test())

    def test_final_poll_captures_remaining_content(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="test-prompt",
                poll_interval=1.0,  # Long interval to ensure final poll is needed
            )

            async with aggregator.run():
                # Create log file at the very end (won't be caught by regular polling)
                log_file = claude_dir / "final.log"
                log_file.write_text("final log entry\n")

            # Final poll should have captured it
            assert log_file in aggregator._file_states

        asyncio.run(_test())


class TestClaudeLogAggregatorIntegration:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Test complete lifecycle: start, discover, read, stop."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            # Create initial file
            initial_log = claude_dir / "initial.log"
            initial_log.write_text("initial line\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="integration-test",
                poll_interval=0.01,
            )

            async with aggregator.run():
                # Wait for initial file to be discovered
                await asyncio.sleep(0.03)

                # Create new file during execution
                runtime_log = claude_dir / "runtime.jsonl"
                runtime_log.write_text('{"event": "test"}\n')

                # Append to initial file
                with initial_log.open("a") as f:
                    f.write("appended line\n")

                # Wait for changes to be picked up
                await asyncio.sleep(0.03)

            # Verify both files were tracked
            assert initial_log in aggregator._file_states
            assert runtime_log in aggregator._file_states

            # Verify stats were updated
            assert aggregator._total_bytes_read > 0
            assert aggregator._total_lines_emitted > 0
            assert len(aggregator._file_states) == 2

        asyncio.run(_test())

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Test discovery of log files in nested directories."""

        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            # Create nested structure
            subdir = claude_dir / "logs" / "subdir"
            subdir.mkdir(parents=True)
            nested_log = subdir / "nested.log"
            nested_log.write_text("nested log entry\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="nested-test",
                poll_interval=0.01,
            )

            async with aggregator.run():
                await asyncio.sleep(0.03)

            assert nested_log in aggregator._file_states

        asyncio.run(_test())


class TestClaudeLogAggregatorLogging:
    def test_logs_start_event(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="logging-test",
                poll_interval=0.01,
            )

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ) as mock_debug:
                async with aggregator.run():
                    pass

                # Check start event was logged
                start_calls = [
                    call
                    for call in mock_debug.call_args_list
                    if call[1].get("event") == "log_aggregator.start"
                ]
                assert len(start_calls) >= 1

        asyncio.run(_test())

    def test_logs_stop_event(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="logging-test",
                poll_interval=0.01,
            )

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ) as mock_debug:
                async with aggregator.run():
                    pass

                # Check stop event was logged
                stop_calls = [
                    call
                    for call in mock_debug.call_args_list
                    if call[1].get("event") == "log_aggregator.stop"
                ]
                assert len(stop_calls) >= 1

        asyncio.run(_test())

    def test_logs_file_discovered_event(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            (claude_dir / "test.log").write_text("content")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="logging-test",
                poll_interval=0.01,
            )

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ) as mock_debug:
                async with aggregator.run():
                    await asyncio.sleep(0.03)

                # Check file discovered event was logged
                discovery_calls = [
                    call
                    for call in mock_debug.call_args_list
                    if call[1].get("event") == "log_aggregator.file_discovered"
                ]
                assert len(discovery_calls) >= 1

        asyncio.run(_test())

    def test_logs_line_content(self, tmp_path: Path) -> None:
        async def _test() -> None:
            claude_dir = tmp_path / ".claude"
            claude_dir.mkdir()
            (claude_dir / "test.log").write_text("test log line\n")

            aggregator = ClaudeLogAggregator(
                claude_dir=claude_dir,
                prompt_name="logging-test",
                poll_interval=0.01,
            )

            with mock.patch(
                "weakincentives.adapters.claude_agent_sdk._log_aggregator.logger.debug"
            ) as mock_debug:
                async with aggregator.run():
                    await asyncio.sleep(0.03)

                # Check log line event was logged
                line_calls = [
                    call
                    for call in mock_debug.call_args_list
                    if call[1].get("event") == "log_aggregator.log_line"
                ]
                assert len(line_calls) >= 1

        asyncio.run(_test())
