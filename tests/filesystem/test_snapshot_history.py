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

"""Tests for filesystem snapshot history and git bundle export."""

from __future__ import annotations

from pathlib import Path

from weakincentives.filesystem import HostFilesystem, SnapshotHistoryEntry
from weakincentives.filesystem._git_ops import (
    _parse_numstat_line,
    _parse_snapshot_tag,
    export_history_bundle,
    snapshot_history,
)


class TestParseSnapshotTag:
    """Tests for _parse_snapshot_tag."""

    def test_valid_tag(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag("pre:write_file:call_001")
        assert tool_name == "write_file"
        assert tool_call_id == "call_001"

    def test_none_tag(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag(None)
        assert tool_name is None
        assert tool_call_id is None

    def test_empty_tag(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag("")
        assert tool_name is None
        assert tool_call_id is None

    def test_non_pre_tag(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag("snapshot-2024-01-01")
        assert tool_name is None
        assert tool_call_id is None

    def test_pre_with_only_two_parts(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag("pre:write_file")
        assert tool_name is None
        assert tool_call_id is None

    def test_pre_with_colon_in_call_id(self) -> None:
        tool_name, tool_call_id = _parse_snapshot_tag("pre:tool:id:with:colons")
        assert tool_name == "tool"
        assert tool_call_id == "id:with:colons"


class TestParseNumstatLine:
    """Tests for _parse_numstat_line."""

    def test_normal_line(self) -> None:
        ins, dels, path = _parse_numstat_line("10\t5\tsrc/main.py")
        assert ins == 10
        assert dels == 5
        assert path == "src/main.py"

    def test_binary_file(self) -> None:
        ins, dels, path = _parse_numstat_line("-\t-\timage.png")
        assert ins == 0
        assert dels == 0
        assert path == "image.png"

    def test_malformed_line(self) -> None:
        ins, dels, path = _parse_numstat_line("no tabs here")
        assert ins == 0
        assert dels == 0
        assert path == ""


class TestSnapshotHistory:
    """Integration tests for snapshot_history with real git operations."""

    def test_empty_repo_returns_empty(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.snapshot_history() == []

    def test_history_after_snapshots(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        # Take initial snapshot
        _ = fs.snapshot(tag="pre:write_file:call_001")

        # Write a file and take another snapshot
        (workspace / "hello.txt").write_text("Hello")
        _ = fs.snapshot(tag="pre:read_file:call_002")

        history = fs.snapshot_history()
        assert len(history) >= 2
        assert all(isinstance(e, SnapshotHistoryEntry) for e in history)

        # First entry should have parsed tag
        first = history[0]
        assert first.tool_name == "write_file"
        assert first.tool_call_id == "call_001"

    def test_history_records_files_changed(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        # Initial empty snapshot
        _ = fs.snapshot(tag="pre:init:call_000")

        # Write a file and snapshot
        (workspace / "new_file.txt").write_text("content")
        _ = fs.snapshot(tag="pre:write_file:call_001")

        history = fs.snapshot_history()
        # The second snapshot should have new_file.txt in files_changed
        second = history[1]
        assert "new_file.txt" in second.files_changed

    def test_history_detects_rollback(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        # Take initial snapshot
        snap_before = fs.snapshot(tag="pre:init:call_000")

        # Write file and snapshot (this will be rolled back)
        (workspace / "bad.txt").write_text("bad")
        _ = fs.snapshot(tag="pre:write_file:call_001")

        # Restore to before the write
        fs.restore(snap_before)

        # Take a new snapshot after restore
        (workspace / "good.txt").write_text("good")
        _ = fs.snapshot(tag="pre:write_file:call_002")

        history = fs.snapshot_history()
        # The rolled back entry (call_001) should be marked as rolled back
        rolled_back_entries = [e for e in history if e.rolled_back]
        assert len(rolled_back_entries) >= 1

    def test_history_includes_insertions_deletions(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        _ = fs.snapshot(tag="pre:init:call_000")

        (workspace / "file.txt").write_text("line1\nline2\nline3\n")
        _ = fs.snapshot(tag="pre:write_file:call_001")

        history = fs.snapshot_history()
        write_entry = next((e for e in history if e.tool_call_id == "call_001"), None)
        assert write_entry is not None
        assert write_entry.insertions > 0


class TestExportHistoryBundle:
    """Tests for export_history_bundle."""

    def test_no_history_returns_none(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.export_history_bundle(tmp_path / "out") is None

    def test_export_creates_bundle_file(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        # Create a snapshot to have history
        _ = fs.snapshot(tag="pre:init:call_000")

        out_dir = tmp_path / "out"
        result = fs.export_history_bundle(out_dir)

        assert result is not None
        assert result.exists()
        assert result.name == "history.bundle"
        assert result.stat().st_size > 0

    def test_export_returns_none_without_git(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        # Don't initialize git
        result = fs.export_history_bundle(tmp_path / "out")
        assert result is None

    def test_raw_export_no_head(self, tmp_path: Path) -> None:
        """export_history_bundle returns None when git repo has no HEAD."""
        from weakincentives.filesystem._git_ops import init_git_repo

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        git_dir = init_git_repo(str(tmp_path / "git"))

        result = export_history_bundle(str(workspace), git_dir, tmp_path / "out")
        assert result is None

    def test_raw_snapshot_history_no_head(self, tmp_path: Path) -> None:
        """snapshot_history returns empty list when git repo has no HEAD."""
        from weakincentives.filesystem._git_ops import init_git_repo

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        git_dir = init_git_repo(str(tmp_path / "git"))

        result = snapshot_history(str(workspace), git_dir)
        assert result == []
