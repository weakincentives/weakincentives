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

"""Tests for Filesystem implementations.

This module uses the generic FilesystemValidationSuite to test protocol
compliance for both InMemoryFilesystem and HostFilesystem, plus
implementation-specific tests for features unique to each backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from tests.helpers.filesystem import (
    FilesystemValidationSuite,
    ReadOnlyFilesystemValidationSuite,
    SnapshotableFilesystemValidationSuite,
)
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.errors import SnapshotError, SnapshotRestoreError
from weakincentives.filesystem import (
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    HostFilesystem,
    ReadResult,
    WriteResult,
)

# =============================================================================
# InMemoryFilesystem Tests using Validation Suites
# =============================================================================


class TestInMemoryFilesystemProtocol(FilesystemValidationSuite):
    """Test InMemoryFilesystem against the generic Filesystem protocol suite."""

    @pytest.fixture
    def fs(self) -> InMemoryFilesystem:
        """Provide a fresh InMemoryFilesystem instance."""
        return InMemoryFilesystem()


class TestInMemoryFilesystemReadOnly(ReadOnlyFilesystemValidationSuite):
    """Test InMemoryFilesystem read-only behavior."""

    @pytest.fixture
    def fs_readonly(self) -> InMemoryFilesystem:
        """Provide a read-only InMemoryFilesystem instance."""
        return InMemoryFilesystem(_read_only=True)


class TestInMemoryFilesystemSnapshots(SnapshotableFilesystemValidationSuite):
    """Test InMemoryFilesystem snapshot behavior."""

    @pytest.fixture
    def fs_snapshotable(self) -> InMemoryFilesystem:
        """Provide a snapshotable InMemoryFilesystem instance."""
        return InMemoryFilesystem()


class TestInMemoryFilesystemSpecific:
    """Implementation-specific tests for InMemoryFilesystem."""

    def test_root_property_is_slash(self) -> None:
        """InMemoryFilesystem root should be '/'."""
        fs = InMemoryFilesystem()
        assert fs.root == "/"

    def test_snapshot_commit_ref_prefix(self) -> None:
        """InMemoryFilesystem snapshot commit_ref should start with 'mem-'."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()
        assert snapshot.commit_ref.startswith("mem-")

    def test_snapshot_root_path_is_slash(self) -> None:
        """InMemoryFilesystem snapshot root_path should be '/'."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()
        assert snapshot.root_path == "/"

    def test_restore_unknown_snapshot_raises(self) -> None:
        """restore() should raise SnapshotRestoreError for unknown snapshot."""
        from datetime import UTC, datetime

        fs = InMemoryFilesystem()
        fake_snapshot = FilesystemSnapshot(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            commit_ref="mem-nonexistent",
            root_path="/",
        )
        with pytest.raises(SnapshotRestoreError, match="Unknown snapshot"):
            fs.restore(fake_snapshot)

    def test_list_with_implicit_subdirectories(self) -> None:
        """Listing should show directories implicitly created by file writes."""
        fs = InMemoryFilesystem()
        fs.write("dir1/subdir/file1.txt", "content1")
        fs.write("dir1/subdir/file2.txt", "content2")
        fs.write("dir1/other/file3.txt", "content3")

        entries = fs.list("dir1")
        dir_names = [e.name for e in entries if e.is_directory]
        assert "subdir" in dir_names
        assert "other" in dir_names

    def test_delete_root_keeps_root_exists(self) -> None:
        """Deleting root recursively should keep root existing but empty."""
        fs = InMemoryFilesystem()
        fs.write("/dir/file.txt", "content")
        fs.delete("/", recursive=True)
        assert fs.exists("/")
        assert not fs.exists("/dir")

    def test_glob_with_recursive_pattern(self) -> None:
        """Glob with ** pattern should match files in subdirectories."""
        fs = InMemoryFilesystem()
        fs.write("dir/file.py", "code")
        fs.write("subdir/nested/deep.py", "more code")
        matches = fs.glob("**/*.py")
        assert len(matches) >= 1

    def test_glob_in_root_includes_nested_files(self) -> None:
        """Glob in root with * should match files at root level."""
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        fs.write("mydir/file.txt", "content")
        fs.write("root.txt", "root content")
        matches = fs.glob("*")
        assert len(matches) == 2
        assert all(m.is_file for m in matches)


# =============================================================================
# HostFilesystem Tests using Validation Suites
# =============================================================================


class TestHostFilesystemProtocol(FilesystemValidationSuite):
    """Test HostFilesystem against the generic Filesystem protocol suite."""

    @pytest.fixture
    def fs(self, tmp_path: Path) -> HostFilesystem:
        """Provide a fresh HostFilesystem instance backed by a temp directory."""
        return HostFilesystem(_root=str(tmp_path))


class TestHostFilesystemReadOnly(ReadOnlyFilesystemValidationSuite):
    """Test HostFilesystem read-only behavior."""

    @pytest.fixture
    def fs_readonly(self, tmp_path: Path) -> HostFilesystem:
        """Provide a read-only HostFilesystem instance."""
        return HostFilesystem(_root=str(tmp_path), _read_only=True)


class TestHostFilesystemSnapshots(SnapshotableFilesystemValidationSuite):
    """Test HostFilesystem snapshot behavior."""

    @pytest.fixture
    def fs_snapshotable(self, tmp_path: Path) -> HostFilesystem:
        """Provide a snapshotable HostFilesystem instance."""
        return HostFilesystem(_root=str(tmp_path))


class TestHostFilesystemSpecific:
    """Implementation-specific tests for HostFilesystem."""

    def test_root_property_matches_path(self, tmp_path: Path) -> None:
        """HostFilesystem root should match the provided path."""
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.root == str(tmp_path)

    def test_snapshot_commit_ref_is_git_hash(self, tmp_path: Path) -> None:
        """HostFilesystem snapshot commit_ref should be a 40-char git hash."""
        fs = HostFilesystem(_root=str(tmp_path))
        (tmp_path / "file.txt").write_text("content")
        snapshot = fs.snapshot()
        assert len(snapshot.commit_ref) == 40
        assert all(c in "0123456789abcdef" for c in snapshot.commit_ref)

    def test_snapshot_root_path_matches(self, tmp_path: Path) -> None:
        """HostFilesystem snapshot root_path should match filesystem root."""
        fs = HostFilesystem(_root=str(tmp_path))
        (tmp_path / "file.txt").write_text("content")
        snapshot = fs.snapshot()
        assert snapshot.root_path == str(tmp_path)

    def test_restore_invalid_commit_raises(self, tmp_path: Path) -> None:
        """restore() should raise SnapshotRestoreError for invalid commit."""
        from datetime import UTC, datetime

        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")
        _ = fs.snapshot()  # Initialize git

        fake_snapshot = FilesystemSnapshot(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            commit_ref="0" * 40,
            root_path=str(tmp_path),
        )
        with pytest.raises(SnapshotRestoreError, match="Failed to restore"):
            fs.restore(fake_snapshot)


class TestHostFilesystemPathSecurity:
    """Tests for HostFilesystem path security."""

    def test_path_escape_prevented(self, tmp_path: Path) -> None:
        """Attempting to escape the root should raise PermissionError."""
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(PermissionError, match="escapes root"):
            fs.read("../etc/passwd")

    def test_exists_returns_false_for_escape_attempt(self, tmp_path: Path) -> None:
        """exists() should return False for escape attempts rather than raise."""
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.exists("../etc/passwd") is False


class TestHostFilesystemEdgeCases:
    """Tests for HostFilesystem edge cases."""

    def test_glob_skips_directories(self, tmp_path: Path) -> None:
        """Glob should only return files, not directories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*")
        assert len(matches) == 1
        assert matches[0].path == "file.txt"

    def test_grep_skips_directories(self, tmp_path: Path) -> None:
        """Grep should only search files, not directories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("hello")
        assert len(matches) == 1

    def test_grep_skips_binary_files(self, tmp_path: Path) -> None:
        """Grep should skip files that fail to decode as UTF-8."""
        (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
        (tmp_path / "text.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep(".*")
        assert len(matches) == 1
        assert matches[0].path == "text.txt"

    def test_glob_nonexistent_path(self, tmp_path: Path) -> None:
        """Glob on nonexistent path should return empty list."""
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*.py", path="missing")
        assert len(matches) == 0

    def test_grep_nonexistent_path(self, tmp_path: Path) -> None:
        """Grep on nonexistent path should return empty list."""
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("pattern", path="missing")
        assert len(matches) == 0

    def test_delete_root_fails(self, tmp_path: Path) -> None:
        """Deleting root directory should fail."""
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(PermissionError, match="Cannot delete root"):
            fs.delete(".")


class TestHostFilesystemGitSnapshots:
    """Tests for HostFilesystem git-based snapshot features."""

    def test_git_initialized_once(self, tmp_path: Path) -> None:
        """Git should only be initialized once."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        assert fs._git_initialized
        assert fs.git_dir is not None
        git_dir = Path(fs.git_dir)
        assert git_dir.exists()

        first_git_dir = fs.git_dir
        _ = fs.snapshot()
        assert fs.git_dir == first_git_dir

    def test_idempotent_empty_snapshot(self, tmp_path: Path) -> None:
        """Creating snapshots without changes should work (allow-empty)."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")

        snapshot1 = fs.snapshot()
        snapshot2 = fs.snapshot()

        assert snapshot1.commit_ref != snapshot2.commit_ref
        assert len(snapshot1.commit_ref) == 40
        assert len(snapshot2.commit_ref) == 40

    def test_restore_removes_gitignored_files(self, tmp_path: Path) -> None:
        """restore() should remove gitignored files for strict rollback."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("src/main.py", "print('hello')")
        fs.write(".gitignore", "*.pyc\n__pycache__/\n*.log\n")
        snapshot = fs.snapshot()

        (tmp_path / "src" / "main.pyc").write_bytes(b"compiled")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"cache")
        (tmp_path / "debug.log").write_text("log output")

        fs.restore(snapshot)

        assert not (tmp_path / "src" / "main.pyc").exists()
        assert not (tmp_path / "__pycache__").exists()
        assert not (tmp_path / "debug.log").exists()
        assert fs.exists("src/main.py")
        assert fs.exists(".gitignore")

    def test_snapshot_empty_repo_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot should handle edge case where commit fails on empty repo."""
        import subprocess
        from unittest.mock import MagicMock

        fs = HostFilesystem(_root=str(tmp_path))
        original_run = subprocess.run
        commit_call_count = 0

        def mock_subprocess_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
            nonlocal commit_call_count

            subcommand = None
            for arg in args:
                if arg in {"init", "--bare", "config", "add", "commit", "rev-parse"}:
                    subcommand = arg
                    break

            if subcommand in {"init", "--bare", "config", "add", "rev-parse"}:
                return original_run(args, **kwargs)

            if subcommand == "commit":
                commit_call_count += 1
                if commit_call_count == 1:
                    mock_result = MagicMock()
                    mock_result.returncode = 1
                    mock_result.stderr = "simulated failure"
                    return mock_result
                return original_run(args, **kwargs)

            return original_run(args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        snapshot = fs.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert len(snapshot.commit_ref) == 40

    def test_commit_fails_but_head_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Commit failure with existing HEAD should raise SnapshotError."""
        import subprocess
        from unittest.mock import MagicMock

        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "initial")
        _ = fs.snapshot(tag="initial")

        original_run = subprocess.run
        commit_call_count = 0

        def mock_subprocess_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
            nonlocal commit_call_count

            subcommand = None
            for arg in args:
                if arg in {"init", "--bare", "config", "add", "commit", "rev-parse"}:
                    subcommand = arg
                    break

            if subcommand in {"init", "--bare", "config", "add", "rev-parse"}:
                return original_run(args, **kwargs)

            if subcommand == "commit":
                commit_call_count += 1
                if commit_call_count == 1:
                    mock_result = MagicMock()
                    mock_result.returncode = 1
                    mock_result.stderr = "simulated failure"
                    return mock_result
                return original_run(args, **kwargs)

            return original_run(args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        with pytest.raises(SnapshotError, match="Failed to create snapshot commit"):
            fs.snapshot(tag="second")


class TestHostFilesystemExternalGitDir:
    """Tests for HostFilesystem external git directory feature."""

    def test_git_dir_outside_workspace_root(self, tmp_path: Path) -> None:
        """Git directory should be created outside workspace root."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.write("file.txt", "content")

        snapshot = fs.snapshot()

        assert fs.git_dir is not None
        assert Path(fs.git_dir).exists()
        assert not (workspace / ".git").exists()
        assert Path(fs.git_dir).parent == Path(tempfile.gettempdir())
        assert Path(fs.git_dir).name.startswith("wink-git-")
        assert snapshot.git_dir == fs.git_dir

    def test_git_dir_not_accessible_in_workspace(self, tmp_path: Path) -> None:
        """Workspace should not contain .git folder after snapshot."""
        workspace = tmp_path / "project"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.write("src/main.py", "print('hello')")

        _ = fs.snapshot()

        assert not (workspace / ".git").exists()
        assert (workspace / "src" / "main.py").exists()

    def test_custom_git_dir(self, tmp_path: Path) -> None:
        """Custom git directory can be specified."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        custom_git = tmp_path / "custom-git-storage"

        fs = HostFilesystem(_root=str(workspace), _git_dir=str(custom_git))
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()

        assert fs.git_dir == str(custom_git)
        assert custom_git.exists()
        assert snapshot.git_dir == str(custom_git)

    def test_cleanup_removes_git_dir(self, tmp_path: Path) -> None:
        """cleanup() should remove the external git directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        git_dir_path = Path(fs.git_dir)
        assert git_dir_path.exists()

        fs.cleanup()

        assert not git_dir_path.exists()
        assert fs._git_initialized is False

    def test_cleanup_idempotent(self, tmp_path: Path) -> None:
        """cleanup() should be safe to call multiple times."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        fs.cleanup()
        fs.cleanup()
        fs.cleanup()

    def test_cleanup_before_snapshot(self, tmp_path: Path) -> None:
        """cleanup() should be safe to call before any snapshot."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.cleanup()

    def test_snapshot_contains_git_dir(self, tmp_path: Path) -> None:
        """Snapshot should contain the git_dir path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        fs.write("file.txt", "content")

        snapshot = fs.snapshot()

        assert snapshot.git_dir is not None
        assert snapshot.git_dir == fs.git_dir

    def test_restore_with_snapshot_git_dir(self, tmp_path: Path) -> None:
        """Restore should work when filesystem git_dir is not set but snapshot has it."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        git_dir = tmp_path / "git-storage"

        fs1 = HostFilesystem(_root=str(workspace), _git_dir=str(git_dir))
        fs1.write("file.txt", "original")
        snapshot = fs1.snapshot()

        fs1.write("file.txt", "modified")
        assert fs1.read("file.txt").content == "modified"

        fs2 = HostFilesystem(_root=str(workspace))
        fs2.restore(snapshot)

        assert fs2.git_dir == str(git_dir)
        assert fs2.read("file.txt").content == "original"

    def test_git_dir_property(self, tmp_path: Path) -> None:
        """git_dir property should return the external git directory path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        assert fs.git_dir is None

        fs.write("file.txt", "content")
        _ = fs.snapshot()

        assert fs.git_dir is not None
        assert Path(fs.git_dir).exists()

    def test_multiple_filesystems_independent_git_dirs(self, tmp_path: Path) -> None:
        """Multiple HostFilesystem instances should have independent git dirs."""
        ws1 = tmp_path / "workspace1"
        ws2 = tmp_path / "workspace2"
        ws1.mkdir()
        ws2.mkdir()

        fs1 = HostFilesystem(_root=str(ws1))
        fs2 = HostFilesystem(_root=str(ws2))

        fs1.write("file.txt", "content1")
        fs2.write("file.txt", "content2")

        _ = fs1.snapshot()
        _ = fs2.snapshot()

        assert fs1.git_dir != fs2.git_dir
        assert Path(fs1.git_dir).exists()
        assert Path(fs2.git_dir).exists()


# =============================================================================
# Dataclass Rendering Tests
# =============================================================================


class TestDataclassRendering:
    """Tests for dataclass string representations."""

    def test_file_stat_str(self) -> None:
        stat = FileStat(
            path="file.txt", is_file=True, is_directory=False, size_bytes=100
        )
        rendered = str(stat)
        assert "file.txt" in rendered

    def test_file_entry_str(self) -> None:
        entry = FileEntry(
            name="file.txt", path="file.txt", is_file=True, is_directory=False
        )
        rendered = str(entry)
        assert "file.txt" in rendered

    def test_glob_match_str(self) -> None:
        match = GlobMatch(path="file.txt", is_file=True)
        rendered = str(match)
        assert "file.txt" in rendered

    def test_grep_match_str(self) -> None:
        match = GrepMatch(
            path="file.txt",
            line_number=1,
            line_content="hello world",
            match_start=0,
            match_end=5,
        )
        rendered = str(match)
        assert "file.txt" in rendered
        assert "hello" in rendered

    def test_read_result_str(self) -> None:
        result = ReadResult(
            content="hello",
            path="file.txt",
            total_lines=1,
            offset=0,
            limit=100,
            truncated=False,
        )
        rendered = str(result)
        assert "file.txt" in rendered

    def test_write_result_str(self) -> None:
        result = WriteResult(path="file.txt", bytes_written=5, mode="create")
        rendered = str(result)
        assert "file.txt" in rendered


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestNormalizePath:
    """Tests for normalize_path utility function."""

    def test_normalize_path_with_parent_refs_and_result(self) -> None:
        """normalize_path with .. when result is not empty."""
        from weakincentives.filesystem import normalize_path

        result = normalize_path("a/b/../c")
        assert result == "a/c"

        result = normalize_path("a/b/c/../../d")
        assert result == "a/d"

    def test_normalize_path_with_leading_dotdot(self) -> None:
        """normalize_path when .. appears without parent to pop."""
        from weakincentives.filesystem import normalize_path

        result = normalize_path("../file.txt")
        assert result == "file.txt"

        result = normalize_path("../../file.txt")
        assert result == "file.txt"

        result = normalize_path("dir/../../../file.txt")
        assert result == "file.txt"
