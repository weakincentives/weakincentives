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

"""HostBackend-specific tests: sandboxing, git snapshots, restore contract."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from weakincentives.errors import SnapshotError, SnapshotRestoreError
from weakincentives.filesystem import Filesystem, HostBackend, SnapshotRef


def _host(tmp_path: Path) -> tuple[Filesystem, HostBackend]:
    backend = HostBackend(tmp_path)
    return Filesystem(backend), backend


class TestPathSecurity:
    """Sandbox containment for the host backend."""

    def test_dotdot_paths_stay_in_sandbox(self, tmp_path: Path) -> None:
        """Normalization collapses .. segments so reads stay inside the root."""
        fs, _ = _host(tmp_path)
        with pytest.raises(FileNotFoundError):
            fs.read("../etc/passwd")  # normalizes to etc/passwd inside the root

    def test_symlink_escape_raises_permission_error(self, tmp_path: Path) -> None:
        """A symlink pointing outside the root is rejected on access."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (workspace / "link.txt").symlink_to(outside)

        fs, _ = _host(workspace)
        with pytest.raises(PermissionError, match="escapes root"):
            fs.read("link.txt")

    def test_exists_returns_false_for_symlink_escape(self, tmp_path: Path) -> None:
        """exists() reports False rather than raising for escaping symlinks."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (workspace / "link.txt").symlink_to(outside)

        fs, _ = _host(workspace)
        assert fs.exists("link.txt") is False

    def test_validation_applies_to_reads(self, tmp_path: Path) -> None:
        """Path constraints are enforced on read operations, not only writes."""
        fs, _ = _host(tmp_path)
        deep_path = "/".join(["d"] * 20)
        with pytest.raises(ValueError, match=r"[Pp]ath depth exceeds"):
            fs.read(deep_path)
        with pytest.raises(ValueError, match=r"[Pp]ath depth exceeds"):
            fs.stat(deep_path)

    def test_root_resolved_once_at_construction(self, tmp_path: Path) -> None:
        """The backend root is the resolved path of the constructor argument."""
        nested = tmp_path / "a" / ".." / "a"
        (tmp_path / "a").mkdir()
        backend = HostBackend(nested)
        assert backend.root == str((tmp_path / "a").resolve())


class TestHostProperties:
    """Root and read_only surface."""

    def test_root_property_matches_resolved_path(self, tmp_path: Path) -> None:
        fs, _ = _host(tmp_path)
        assert fs.root == str(tmp_path.resolve())

    def test_delete_root_fails(self, tmp_path: Path) -> None:
        fs, _ = _host(tmp_path)
        with pytest.raises(PermissionError, match="Cannot delete root"):
            fs.delete(".")


class TestEdgeCases:
    """Host-specific glob/grep behaviors."""

    def test_glob_skips_directories(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("content")
        fs, _ = _host(tmp_path)
        matches = fs.glob("*")
        assert [m.path for m in matches] == ["file.txt"]

    def test_grep_skips_directories(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hello")
        fs, _ = _host(tmp_path)
        assert len(fs.grep("hello")) == 1

    def test_glob_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        fs, _ = _host(tmp_path)
        assert len(fs.glob("*.py", path="missing")) == 0

    def test_grep_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        fs, _ = _host(tmp_path)
        assert len(fs.grep("pattern", path="missing")) == 0

    def test_glob_on_file_base_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs, _ = _host(tmp_path)
        assert len(fs.glob("*", path="file.txt")) == 0


class TestSnapshotRefs:
    """Snapshot ref shape for the host backend."""

    def test_snapshot_token_embeds_commit_and_git_dir(self, tmp_path: Path) -> None:
        fs, backend = _host(tmp_path)
        fs.write("file.txt", "content")
        ref = fs.snapshot()
        commit, _, git_dir = ref.token.partition(":")
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)
        assert git_dir == backend.git_dir

    def test_restore_invalid_commit_raises(self, tmp_path: Path) -> None:
        fs, backend = _host(tmp_path)
        fs.write("file.txt", "content")
        _ = fs.snapshot()  # Initialize git

        fake_ref = SnapshotRef(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            token=f"{'0' * 40}:{backend.git_dir}",
        )
        with pytest.raises(SnapshotRestoreError, match="Failed to restore"):
            fs.restore(fake_ref)


class TestRestoreContract:
    """Strict rollback: restore returns the exact snapshot-time state."""

    def test_ignored_files_created_after_snapshot_are_removed(
        self, tmp_path: Path
    ) -> None:
        fs, _ = _host(tmp_path)
        fs.write("src/main.py", "print('hello')")
        fs.write(".gitignore", "*.pyc\n__pycache__/\n*.log\n")
        ref = fs.snapshot()

        (tmp_path / "src" / "main.pyc").write_bytes(b"compiled")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"cache")
        (tmp_path / "debug.log").write_text("log output")

        fs.restore(ref)

        assert not (tmp_path / "src" / "main.pyc").exists()
        assert not (tmp_path / "__pycache__").exists()
        assert not (tmp_path / "debug.log").exists()
        assert fs.exists("src/main.py")
        assert fs.exists(".gitignore")

    def test_ignored_files_present_at_snapshot_are_restored(
        self, tmp_path: Path
    ) -> None:
        """Ignored files are captured too: restore brings them back."""
        fs, _ = _host(tmp_path)
        fs.write(".gitignore", "*.log\n")
        fs.write("debug.log", "precious log state")
        ref = fs.snapshot()

        fs.delete("debug.log")
        assert not fs.exists("debug.log")

        fs.restore(ref)

        assert fs.read("debug.log").content == "precious log state"

    def test_ignored_file_contents_roll_back(self, tmp_path: Path) -> None:
        fs, _ = _host(tmp_path)
        fs.write(".gitignore", "*.log\n")
        fs.write("debug.log", "v1")
        ref = fs.snapshot()

        fs.write("debug.log", "v2")
        fs.restore(ref)

        assert fs.read("debug.log").content == "v1"


def _patch_first_commit_to_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch subprocess.run so the next git-commit call fails once."""
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

        if subcommand == "commit":
            commit_call_count += 1
            if commit_call_count == 1:
                mock_result = MagicMock()
                mock_result.returncode = 1
                mock_result.stderr = "simulated failure"
                return mock_result

        return original_run(args, **kwargs)  # ty: ignore[missing-argument]

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)


class TestGitSnapshots:
    """Git-based snapshot mechanics."""

    def test_git_initialized_once(self, tmp_path: Path) -> None:
        fs, backend = _host(tmp_path)
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        assert backend.git_dir is not None
        assert Path(backend.git_dir).exists()

        first_git_dir = backend.git_dir
        _ = fs.snapshot()
        assert backend.git_dir == first_git_dir

    def test_idempotent_empty_snapshot(self, tmp_path: Path) -> None:
        """Creating snapshots without changes should work (allow-empty)."""
        fs, _ = _host(tmp_path)
        fs.write("file.txt", "content")

        ref1 = fs.snapshot()
        ref2 = fs.snapshot()

        assert ref1.token != ref2.token

    def test_snapshot_empty_repo_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot should handle edge case where commit fails on empty repo."""
        fs, _ = _host(tmp_path)
        _patch_first_commit_to_fail(monkeypatch)
        ref = fs.snapshot()
        assert isinstance(ref, SnapshotRef)
        commit, _, _ = ref.token.partition(":")
        assert len(commit) == 40

    def test_commit_fails_but_head_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Commit failure with existing HEAD should raise SnapshotError."""
        fs, _ = _host(tmp_path)
        fs.write("file.txt", "initial")
        _ = fs.snapshot(tag="initial")

        _patch_first_commit_to_fail(monkeypatch)
        with pytest.raises(SnapshotError, match="Failed to create snapshot commit"):
            fs.snapshot(tag="second")


class TestExternalGitDir:
    """External git directory lifecycle."""

    def test_git_dir_outside_workspace_root(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs, backend = _host(workspace)
        fs.write("file.txt", "content")

        _ = fs.snapshot()

        assert backend.git_dir is not None
        assert Path(backend.git_dir).exists()
        assert not (workspace / ".git").exists()
        assert Path(backend.git_dir).parent == Path(tempfile.gettempdir())
        assert Path(backend.git_dir).name.startswith("wink-git-")

    def test_custom_git_dir(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        custom_git = tmp_path / "custom-git-storage"

        backend = HostBackend(workspace, git_dir=str(custom_git))
        fs = Filesystem(backend)
        fs.write("file.txt", "content")
        ref = fs.snapshot()

        assert backend.git_dir == str(custom_git)
        assert custom_git.exists()
        assert ref.token.endswith(f":{custom_git}")

    def test_cleanup_removes_git_dir(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs, backend = _host(workspace)
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        assert backend.git_dir is not None
        git_dir_path = Path(backend.git_dir)
        assert git_dir_path.exists()

        backend.cleanup()

        assert not git_dir_path.exists()

    def test_cleanup_idempotent(self, tmp_path: Path) -> None:
        fs, backend = _host(tmp_path)
        fs.write("file.txt", "content")

        _ = fs.snapshot()
        backend.cleanup()
        backend.cleanup()
        backend.cleanup()

    def test_cleanup_before_snapshot(self, tmp_path: Path) -> None:
        _, backend = _host(tmp_path)
        backend.cleanup()

    def test_restore_adopts_ref_git_dir(self, tmp_path: Path) -> None:
        """Restore works on a fresh backend using the git dir from the ref."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        git_dir = tmp_path / "git-storage"

        fs1 = Filesystem(HostBackend(workspace, git_dir=str(git_dir)))
        fs1.write("file.txt", "original")
        ref = fs1.snapshot()

        fs1.write("file.txt", "modified")
        assert fs1.read("file.txt").content == "modified"

        backend2 = HostBackend(workspace)
        fs2 = Filesystem(backend2)
        fs2.restore(ref)

        assert backend2.git_dir == str(git_dir)
        assert fs2.read("file.txt").content == "original"

    def test_git_dir_property(self, tmp_path: Path) -> None:
        _, backend = _host(tmp_path)
        assert backend.git_dir is None

        fs = Filesystem(backend)
        fs.write("file.txt", "content")
        _ = fs.snapshot()

        assert backend.git_dir is not None
        assert Path(backend.git_dir).exists()

    def test_multiple_backends_independent_git_dirs(self, tmp_path: Path) -> None:
        ws1 = tmp_path / "workspace1"
        ws2 = tmp_path / "workspace2"
        ws1.mkdir()
        ws2.mkdir()

        fs1, backend1 = _host(ws1)
        fs2, backend2 = _host(ws2)

        fs1.write("file.txt", "content1")
        fs2.write("file.txt", "content2")

        _ = fs1.snapshot()
        _ = fs2.snapshot()

        assert backend1.git_dir != backend2.git_dir
        assert backend1.git_dir is not None and Path(backend1.git_dir).exists()
        assert backend2.git_dir is not None and Path(backend2.git_dir).exists()
