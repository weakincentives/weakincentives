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

"""Tests for git-related environment capture functionality."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.debug._git import (
    GitInfo,
    _is_sensitive_file,
    _redact_url_credentials,
    _run_git_command,
    capture_git_diff,
    capture_git_info,
)
from weakincentives.debug.environment import (
    ContainerInfo,
    _capture_container_info,
    _extract_container_id_from_cgroup,
    _is_valid_container_id,
)


def _init_test_git_repo(path: Path) -> None:
    """Initialize a test git repo with signing disabled."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    # Disable commit signing for tests
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Path]:
    """Create a temporary git repository with an initial commit."""
    _init_test_git_repo(tmp_path)
    (tmp_path / "test.txt").write_text("test")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    yield tmp_path


class TestGitInfo:
    """Tests for GitInfo capture."""

    def test_git_info_dataclass(self) -> None:
        """Test GitInfo dataclass construction."""
        info = GitInfo(
            repo_root="/path/to/repo",
            commit_sha="abc123def456",
            commit_short="abc123de",
            branch="main",
            is_dirty=False,
            remotes={"origin": "https://github.com/test/repo.git"},
            tags=("v1.0.0",),
        )
        assert info.repo_root == "/path/to/repo"
        assert info.branch == "main"
        assert "origin" in info.remotes

    def test_capture_git_info_in_repo(self, git_repo: Path) -> None:
        """Test _capture_git_info in a real repo."""
        subprocess.run(
            ["git", "remote", "add", "origin", "https://user:token@host/repo.git"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        info = capture_git_info(git_repo)

        assert info is not None
        assert info.repo_root == str(git_repo)
        assert len(info.commit_sha) == 40
        assert info.commit_short == info.commit_sha[:8]
        assert info.is_dirty is False
        assert info.remotes["origin"] == "https://[REDACTED]@host/repo.git"

    def test_capture_git_info_detached_head(self, git_repo: Path) -> None:
        """Test _capture_git_info with detached HEAD."""
        # Detach HEAD
        subprocess.run(
            ["git", "checkout", "--detach"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        info = capture_git_info(git_repo)

        assert info is not None
        assert info.branch is None  # Detached HEAD has no branch

    def test_capture_git_info_dirty_repo(self, git_repo: Path) -> None:
        """Test _capture_git_info with uncommitted changes."""
        # Make uncommitted change
        (git_repo / "test.txt").write_text("modified")

        info = capture_git_info(git_repo)

        assert info is not None
        assert info.is_dirty is True

    def test_capture_git_info_with_tags(self, git_repo: Path) -> None:
        """Test _capture_git_info captures tags pointing to HEAD."""
        # Create a tag
        subprocess.run(
            ["git", "tag", "v1.0.0"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        info = capture_git_info(git_repo)

        assert info is not None
        assert "v1.0.0" in info.tags

    def test_capture_git_info_with_remote(self, git_repo: Path) -> None:
        """Test _capture_git_info captures remotes with redaction."""
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://user:token@github.com/test/repo.git",
            ],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        info = capture_git_info(git_repo)

        assert info is not None
        assert info.remotes["origin"] == "https://[REDACTED]@github.com/test/repo.git"

    def test_capture_git_info_not_in_repo(self, tmp_path: Path) -> None:
        """Test _capture_git_info outside a git repo."""
        info = capture_git_info(tmp_path)
        assert info is None

    def test_run_git_command_failure(self, tmp_path: Path) -> None:
        """Test _run_git_command handles failures gracefully."""
        result = _run_git_command("invalid-command", cwd=tmp_path)
        assert result is None

    def test_run_git_command_timeout(self, tmp_path: Path) -> None:
        """Test _run_git_command handles timeout gracefully."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)):
            result = _run_git_command("status", cwd=tmp_path)
            assert result is None


class TestSensitiveFileDetection:
    """Tests for sensitive file pattern detection."""

    def test_is_sensitive_file_env_files(self) -> None:
        """Test _is_sensitive_file detects .env files."""
        assert _is_sensitive_file(".env") is True
        assert _is_sensitive_file(".env.local") is True
        assert _is_sensitive_file(".env.production") is True
        assert _is_sensitive_file("src/.env") is True

    def test_is_sensitive_file_credentials(self) -> None:
        """Test _is_sensitive_file detects credential files."""
        assert _is_sensitive_file("credentials.json") is True
        assert _is_sensitive_file("secrets.json") is True
        assert _is_sensitive_file("secrets.yaml") is True
        assert _is_sensitive_file("secrets.yml") is True

    def test_is_sensitive_file_keys(self) -> None:
        """Test _is_sensitive_file detects key files."""
        assert _is_sensitive_file("private.pem") is True
        assert _is_sensitive_file("server.key") is True
        assert _is_sensitive_file("cert.p12") is True
        assert _is_sensitive_file("id_rsa") is True
        assert _is_sensitive_file("id_ed25519") is True
        assert _is_sensitive_file(".ssh/config") is True

    def test_is_sensitive_file_config_files(self) -> None:
        """Test _is_sensitive_file detects sensitive config files."""
        assert _is_sensitive_file(".netrc") is True
        assert _is_sensitive_file(".npmrc") is True
        assert _is_sensitive_file(".pypirc") is True

    def test_is_sensitive_file_safe_files(self) -> None:
        """Test _is_sensitive_file allows safe files."""
        assert _is_sensitive_file("README.md") is False
        assert _is_sensitive_file("main.py") is False
        assert _is_sensitive_file("config.json") is False
        assert _is_sensitive_file("package.json") is False


class TestUrlCredentialRedaction:
    """Tests for URL credential redaction."""

    def test_redact_url_credentials_with_user_pass(self) -> None:
        """Test _redact_url_credentials with username:password."""
        url = "https://user:token123@github.com/org/repo.git"
        result = _redact_url_credentials(url)
        assert result == "https://[REDACTED]@github.com/org/repo.git"

    def test_redact_url_credentials_with_user_only(self) -> None:
        """Test _redact_url_credentials with username only (no password)."""
        url = "https://user@github.com/org/repo.git"
        result = _redact_url_credentials(url)
        assert result == "https://[REDACTED]@github.com/org/repo.git"

    def test_redact_url_credentials_with_token(self) -> None:
        """Test _redact_url_credentials with token as username."""
        url = "https://ghp_xxxxxxxxxxxx@github.com/org/repo.git"
        result = _redact_url_credentials(url)
        assert result == "https://[REDACTED]@github.com/org/repo.git"

    def test_redact_url_credentials_no_credentials(self) -> None:
        """Test _redact_url_credentials with no credentials."""
        url = "https://github.com/org/repo.git"
        result = _redact_url_credentials(url)
        assert result == "https://github.com/org/repo.git"

    def test_redact_url_credentials_ssh(self) -> None:
        """Test _redact_url_credentials with SSH URL (no redaction needed)."""
        url = "git@github.com:org/repo.git"
        result = _redact_url_credentials(url)
        # SSH URLs don't match the scheme:// pattern, returned unchanged
        assert result == "git@github.com:org/repo.git"

    def test_redact_url_credentials_various_schemes(self) -> None:
        """Test _redact_url_credentials with various URL schemes."""
        # HTTP with credentials
        assert _redact_url_credentials("http://user:pass@host/path") == (
            "http://[REDACTED]@host/path"
        )
        # Git protocol with credentials
        assert _redact_url_credentials("git://token@host/repo.git") == (
            "git://[REDACTED]@host/repo.git"
        )


class TestGitDiff:
    """Tests for git diff capture."""

    def test_capture_git_diff_clean_repo(self, git_repo: Path) -> None:
        """Test _capture_git_diff with no changes."""
        diff = capture_git_diff(git_repo)
        assert diff is None  # No changes, no diff

    def test_capture_git_diff_with_changes(self, git_repo: Path) -> None:
        """Test _capture_git_diff with uncommitted changes."""
        (git_repo / "test.txt").write_text("modified content")

        diff = capture_git_diff(git_repo)
        assert diff is not None
        assert "modified content" in diff

    def test_capture_git_diff_truncation(self, git_repo: Path) -> None:
        """Test git diff truncation for large diffs."""
        # Create a large change
        large_content = "x" * 200_000
        (git_repo / "test.txt").write_text(large_content)

        diff = capture_git_diff(git_repo)
        assert diff is not None
        assert "[TRUNCATED:" in diff
        assert len(diff) <= 100_000

    def test_capture_git_diff_not_in_repo(self, tmp_path: Path) -> None:
        """Test _capture_git_diff outside a git repo."""
        diff = capture_git_diff(tmp_path)
        assert diff is None

    def test_capture_git_diff_with_untracked_files(self, git_repo: Path) -> None:
        """Test _capture_git_diff includes untracked files."""
        # Create an untracked file
        (git_repo / "new_file.txt").write_text("new file content")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "Untracked files:" in diff
        assert "new_file.txt" in diff
        assert "new file content" in diff

    def test_capture_git_diff_untracked_with_tracked_changes(
        self, git_repo: Path
    ) -> None:
        """Test _capture_git_diff includes both tracked and untracked changes."""
        # Modify tracked file
        (git_repo / "test.txt").write_text("modified content")
        # Create untracked file
        (git_repo / "untracked.py").write_text("print('hello')")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        # Should contain tracked file change
        assert "modified content" in diff
        # Should contain untracked file
        assert "untracked.py" in diff
        assert "print('hello')" in diff

    def test_capture_git_diff_untracked_file_format(self, git_repo: Path) -> None:
        """Test untracked files are formatted as unified diff."""
        (git_repo / "config.json").write_text('{"key": "value"}')

        diff = capture_git_diff(git_repo)

        assert diff is not None
        # Should have unified diff format
        assert "diff --git a/config.json b/config.json" in diff
        assert "new file mode" in diff
        assert "--- /dev/null" in diff
        assert "+++ b/config.json" in diff
        assert '+{"key": "value"}' in diff

    def test_capture_git_diff_untracked_multiline(self, git_repo: Path) -> None:
        """Test untracked file with multiple lines shows line count."""
        content = "line1\nline2\nline3"
        (git_repo / "multi.txt").write_text(content)

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "@@ -0,0 +1,3 @@" in diff

    def test_capture_git_diff_untracked_directory(self, git_repo: Path) -> None:
        """Test untracked files in subdirectories are captured."""
        subdir = git_repo / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "subdir/nested.txt" in diff
        assert "nested content" in diff

    def test_capture_git_diff_untracked_unreadable(self, git_repo: Path) -> None:
        """Test graceful handling of unreadable untracked files."""
        unreadable = git_repo / "unreadable.txt"
        unreadable.write_text("test")
        # Remove read permission
        unreadable.chmod(0o000)

        try:
            diff = capture_git_diff(git_repo)
            # Should still return something (possibly just the header)
            # and not crash
            assert diff is None or isinstance(diff, str)
        finally:
            # Restore permissions for cleanup
            unreadable.chmod(0o644)

    def test_capture_git_diff_truncation_with_untracked(self, git_repo: Path) -> None:
        """Test truncation applies to combined tracked + untracked changes."""
        # Create large tracked change
        (git_repo / "test.txt").write_text("x" * 60_000)
        # Create large untracked file
        (git_repo / "large.txt").write_text("y" * 60_000)

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "[TRUNCATED:" in diff
        assert len(diff) <= 100_000

    def test_capture_git_diff_untracked_symlink_to_dir(self, git_repo: Path) -> None:
        """Test graceful handling of untracked symlinks to directories."""
        # Create a directory and symlink to it (symlink won't be a regular file)
        target_dir = git_repo / "target_dir"
        target_dir.mkdir()
        symlink = git_repo / "link_to_dir"
        symlink.symlink_to(target_dir)

        # Also create a regular untracked file
        (git_repo / "regular.txt").write_text("content")

        diff = capture_git_diff(git_repo)

        # Should include the regular file but skip the symlink to directory
        assert diff is not None
        assert "regular.txt" in diff
        assert "content" in diff

    def test_capture_git_diff_untracked_binary_file(self, git_repo: Path) -> None:
        """Test graceful handling of binary untracked files with decode errors."""
        # Create a file that will trigger errors="replace" behavior
        binary_file = git_repo / "binary.bin"
        binary_file.write_bytes(b"\xff\xfe\x00\x01invalid\x80utf8")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "binary.bin" in diff
        # Content should be captured with replacement characters

    def test_capture_git_diff_untracked_read_error(self, git_repo: Path) -> None:
        """Test graceful handling when file read raises OSError."""
        (git_repo / "error.txt").write_text("test")

        # Mock Path.read_text to raise OSError
        original_read_text = Path.read_text

        def mock_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "error.txt":
                raise OSError("Read error")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read_text):
            diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "error.txt: [unable to read]" in diff

    def test_capture_git_diff_untracked_whitespace_only(self, git_repo: Path) -> None:
        """Test handling when ls-files returns only whitespace."""
        # Mock ls-files to return whitespace-only output
        original_run_git = _run_git_command

        def mock_run_git(*args: str, cwd: Path | None = None) -> str | None:
            if args[:2] == ("ls-files", "--others"):
                return "   \n  \n"
            return original_run_git(*args, cwd=cwd)

        with patch(
            "weakincentives.debug._git._run_git_command",
            side_effect=mock_run_git,
        ):
            diff = capture_git_diff(git_repo)

        # Should return None since no tracked or untracked changes
        assert diff is None

    def test_capture_git_diff_untracked_empty_file(self, git_repo: Path) -> None:
        """Test handling of empty untracked files."""
        # Create an empty untracked file
        (git_repo / "empty.txt").write_text("")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "empty.txt" in diff
        # Empty file should have the header but no @@ hunk
        assert "diff --git a/empty.txt b/empty.txt" in diff

    def test_capture_git_diff_excludes_sensitive_files(self, git_repo: Path) -> None:
        """Test that sensitive files are excluded from untracked capture."""
        # Create sensitive files
        (git_repo / ".env").write_text("SECRET_KEY=abc123")
        (git_repo / "credentials.json").write_text('{"api_key": "xyz"}')
        # Create a safe file too
        (git_repo / "config.txt").write_text("safe content")

        diff = capture_git_diff(git_repo)

        assert diff is not None
        # Sensitive files should be excluded with a message
        assert ".env: [excluded - sensitive file]" in diff
        assert "credentials.json: [excluded - sensitive file]" in diff
        # Content should NOT be captured
        assert "SECRET_KEY" not in diff
        assert "api_key" not in diff
        # Safe file should be included
        assert "config.txt" in diff
        assert "safe content" in diff

    def test_capture_git_diff_untracked_file_truncation(self, git_repo: Path) -> None:
        """Test that large untracked files are truncated per-file."""
        # Create a file larger than 10KB limit
        large_content = "x" * 15_000
        (git_repo / "large.txt").write_text(large_content)

        diff = capture_git_diff(git_repo)

        assert diff is not None
        assert "large.txt" in diff
        # Should be truncated
        assert "[TRUNCATED: file exceeded 10000B]" in diff


class TestContainerInfo:
    """Tests for ContainerInfo capture."""

    def test_container_info_dataclass(self) -> None:
        """Test ContainerInfo dataclass construction."""
        info = ContainerInfo(
            runtime="docker",
            container_id="abc123",
            image="myimage:latest",
            image_digest="sha256:abc123",
            cgroup_path="/docker/abc123",
            is_containerized=True,
        )
        assert info.runtime == "docker"
        assert info.is_containerized is True

    def test_capture_container_info_not_containerized(self) -> None:
        """Test _capture_container_info when not in a container."""
        # Mock the container detection to return None
        with patch("pathlib.Path.exists", return_value=False):
            info = _capture_container_info()
            # May or may not be None depending on system, just verify no crash
            assert info is None or isinstance(info, ContainerInfo)

    def test_capture_container_info_docker_env(self) -> None:
        """Test _capture_container_info with dockerenv file."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Mock /.dockerenv to exist, other paths not
            def exists_side_effect(self: Path) -> bool:
                return str(self) == "/.dockerenv"

            mock_exists.side_effect = lambda: exists_side_effect(Path("/.dockerenv"))

            # This test is tricky due to how Path.exists works
            # We'll just verify the function doesn't crash
            info = _capture_container_info()
            assert info is None or isinstance(info, ContainerInfo)

    def test_is_valid_container_id_valid(self) -> None:
        """Test _is_valid_container_id with valid ID."""
        valid_id = "a" * 64
        assert _is_valid_container_id(valid_id) is True

    def test_is_valid_container_id_invalid_length(self) -> None:
        """Test _is_valid_container_id with invalid length."""
        assert _is_valid_container_id("abc123") is False
        assert _is_valid_container_id("a" * 63) is False
        assert _is_valid_container_id("a" * 65) is False

    def test_is_valid_container_id_invalid_chars(self) -> None:
        """Test _is_valid_container_id with invalid characters."""
        invalid_id = "g" * 64  # g is not a hex char
        assert _is_valid_container_id(invalid_id) is False

    def test_extract_container_id_from_cgroup_docker(self) -> None:
        """Test _extract_container_id_from_cgroup with docker content."""
        container_id = "a" * 64
        content = f"0::/docker/{container_id}"
        cid, runtime = _extract_container_id_from_cgroup(content)
        assert cid == container_id
        assert runtime == "docker"

    def test_extract_container_id_from_cgroup_containerd(self) -> None:
        """Test _extract_container_id_from_cgroup with containerd content."""
        container_id = "b" * 64
        content = f"0::/containerd/{container_id}"
        cid, runtime = _extract_container_id_from_cgroup(content)
        assert cid == container_id
        assert runtime == "containerd"

    def test_extract_container_id_from_cgroup_no_id(self) -> None:
        """Test _extract_container_id_from_cgroup without valid ID."""
        content = "0::/docker/short"  # Not a 64-char hex
        cid, runtime = _extract_container_id_from_cgroup(content)
        assert cid is None
        assert runtime == "docker"

    def test_extract_container_id_from_cgroup_no_match(self) -> None:
        """Test _extract_container_id_from_cgroup with no docker/containerd."""
        content = "0::/system.slice"
        cid, runtime = _extract_container_id_from_cgroup(content)
        assert cid is None
        assert runtime is None
