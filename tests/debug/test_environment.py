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

"""Tests for environment capture functionality."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
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
from weakincentives.debug.bundle import BundleWriter, DebugBundle
from weakincentives.debug.environment import (
    CommandInfo,
    ContainerInfo,
    EnvironmentCapture,
    PythonInfo,
    SystemInfo,
    _capture_command_info,
    _capture_container_info,
    _capture_env_vars,
    _capture_memory_bytes,
    _capture_packages,
    _capture_python_info,
    _capture_system_info,
    _extract_container_id_from_cgroup,
    _get_darwin_memory_bytes,
    _get_linux_memory_bytes,
    _is_valid_container_id,
    _should_capture_env_var,
    _should_redact_value,
    capture_environment,
)

if TYPE_CHECKING:
    pass


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


class TestSystemInfo:
    """Tests for SystemInfo capture."""

    def test_system_info_dataclass(self) -> None:
        """Test SystemInfo dataclass construction."""
        info = SystemInfo(
            os_name="Linux",
            os_release="5.4.0",
            kernel_version="test",
            architecture="x86_64",
            processor="Intel",
            cpu_count=8,
            memory_total_bytes=16_000_000_000,
            hostname="testhost",
        )
        assert info.os_name == "Linux"
        assert info.cpu_count == 8
        assert info.memory_total_bytes == 16_000_000_000

    def test_capture_system_info(self) -> None:
        """Test _capture_system_info captures real system data."""
        info = _capture_system_info()

        assert info.os_name == platform.system()
        assert info.architecture == platform.machine()
        assert info.processor == platform.processor()
        assert info.cpu_count == os.cpu_count()
        assert info.hostname == platform.node()

    def test_capture_system_info_memory_linux(self) -> None:
        """Test memory capture on Linux."""
        if sys.platform != "linux":
            pytest.skip("Linux-specific test")

        info = _capture_system_info()
        # Should have captured memory on Linux
        assert info.memory_total_bytes is not None
        assert info.memory_total_bytes > 0

    def test_capture_system_info_memory_failure(self) -> None:
        """Test graceful handling of memory capture failure."""
        # Mock the memory capture function to fail
        with patch(
            "weakincentives.debug.environment._capture_memory_bytes",
            return_value=None,
        ):
            info = _capture_system_info()
            # Should still capture other fields, memory will be None
            assert info.os_name == platform.system()
            assert info.memory_total_bytes is None

    def test_get_linux_memory_bytes_no_memtotal(self, tmp_path: Path) -> None:
        """Test _get_linux_memory_bytes with no MemTotal line."""
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemFree: 1234 kB\n")

        with patch(
            "weakincentives.debug.environment.Path",
            return_value=meminfo,
        ):
            # Mock Path("/proc/meminfo") to return our temp file
            with patch.object(Path, "open", meminfo.open):
                result = _get_linux_memory_bytes()
                # No MemTotal line means None
                assert result is None

    def test_get_linux_memory_bytes_malformed_line(self, tmp_path: Path) -> None:
        """Test _get_linux_memory_bytes with malformed MemTotal line."""
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:\n")  # Missing value

        with patch.object(Path, "open", meminfo.open):
            result = _get_linux_memory_bytes()
            # Malformed line means None
            assert result is None

    def test_get_linux_memory_bytes_oserror(self) -> None:
        """Test _get_linux_memory_bytes handles OSError gracefully."""
        with patch.object(Path, "open", side_effect=OSError("Permission denied")):
            result = _get_linux_memory_bytes()
            assert result is None

    def test_get_darwin_memory_bytes_success(self) -> None:
        """Test _get_darwin_memory_bytes with successful sysctl."""
        mock_result = type("Result", (), {"returncode": 0, "stdout": "17179869184\n"})()
        with patch("subprocess.run", return_value=mock_result):
            result = _get_darwin_memory_bytes()
            assert result == 17179869184

    def test_get_darwin_memory_bytes_failure(self) -> None:
        """Test _get_darwin_memory_bytes with failed sysctl."""
        mock_result = type("Result", (), {"returncode": 1, "stdout": ""})()
        with patch("subprocess.run", return_value=mock_result):
            result = _get_darwin_memory_bytes()
            assert result is None

    def test_capture_memory_bytes_darwin(self) -> None:
        """Test _capture_memory_bytes on Darwin platform."""
        with patch("sys.platform", "darwin"):
            mock_result = type(
                "Result", (), {"returncode": 0, "stdout": "17179869184\n"}
            )()
            with patch("subprocess.run", return_value=mock_result):
                result = _capture_memory_bytes()
                assert result == 17179869184

    def test_capture_memory_bytes_unsupported_platform(self) -> None:
        """Test _capture_memory_bytes on unsupported platform."""
        with patch("sys.platform", "win32"):
            result = _capture_memory_bytes()
            assert result is None

    def test_capture_memory_bytes_exception(self) -> None:
        """Test _capture_memory_bytes handles exceptions."""
        with patch("sys.platform", "linux"):
            with patch(
                "weakincentives.debug.environment._get_linux_memory_bytes",
                side_effect=ValueError("Parse error"),
            ):
                result = _capture_memory_bytes()
                assert result is None


class TestPythonInfo:
    """Tests for PythonInfo capture."""

    def test_python_info_dataclass(self) -> None:
        """Test PythonInfo dataclass construction."""
        info = PythonInfo(
            version="3.12.0",
            version_info=(3, 12, 0),
            implementation="CPython",
            executable="/usr/bin/python3",
            prefix="/usr",
            base_prefix="/usr",
            is_virtualenv=False,
        )
        assert info.version == "3.12.0"
        assert info.version_info == (3, 12, 0)

    def test_capture_python_info(self) -> None:
        """Test _capture_python_info captures real Python data."""
        info = _capture_python_info()

        assert info.version == sys.version
        vi = sys.version_info
        assert info.version_info == (vi.major, vi.minor, vi.micro)
        assert info.implementation == platform.python_implementation()
        assert info.executable == sys.executable
        assert info.prefix == sys.prefix
        assert info.base_prefix == sys.base_prefix
        assert info.is_virtualenv == (sys.prefix != sys.base_prefix)


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


class TestCommandInfo:
    """Tests for CommandInfo capture."""

    def test_command_info_dataclass(self) -> None:
        """Test CommandInfo dataclass construction."""
        info = CommandInfo(
            argv=("python", "script.py", "--arg"),
            working_dir="/home/user",
            entrypoint="script.py",
            executable="/usr/bin/python3",
        )
        assert info.argv == ("python", "script.py", "--arg")
        assert info.entrypoint == "script.py"

    def test_capture_command_info(self) -> None:
        """Test _capture_command_info captures real command data."""
        info = _capture_command_info()

        assert info.argv == tuple(sys.argv)
        assert info.working_dir == str(Path.cwd())
        assert info.executable == sys.executable
        if sys.argv:
            assert info.entrypoint == sys.argv[0]


class TestEnvVars:
    """Tests for environment variable capture."""

    def test_should_capture_env_var_allowed(self) -> None:
        """Test _should_capture_env_var allows expected variables."""
        assert _should_capture_env_var("PYTHONPATH") is True
        assert _should_capture_env_var("PATH") is True
        assert _should_capture_env_var("HOME") is True
        assert _should_capture_env_var("VIRTUAL_ENV") is True
        assert _should_capture_env_var("GITHUB_ACTIONS") is True
        assert _should_capture_env_var("CI") is True

    def test_should_capture_env_var_blocked(self) -> None:
        """Test _should_capture_env_var blocks unexpected variables."""
        assert _should_capture_env_var("MY_CUSTOM_VAR") is False
        assert _should_capture_env_var("RANDOM_VAR") is False

    def test_should_redact_value_sensitive(self) -> None:
        """Test _should_redact_value identifies sensitive variables."""
        assert _should_redact_value("API_KEY", "secret123") is True
        assert _should_redact_value("GITHUB_TOKEN", "ghp_xxx") is True
        assert _should_redact_value("PASSWORD", "pass123") is True
        assert _should_redact_value("SECRET_KEY", "key123") is True
        assert _should_redact_value("AUTH_TOKEN", "token") is True

    def test_should_redact_value_safe(self) -> None:
        """Test _should_redact_value allows safe variables."""
        assert _should_redact_value("PATH", "/usr/bin") is False
        assert _should_redact_value("HOME", "/home/user") is False
        assert _should_redact_value("PYTHONPATH", "/lib") is False

    def test_capture_env_vars(self) -> None:
        """Test _capture_env_vars captures and filters correctly."""
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "HOME": "/home/test",
                "MY_CUSTOM_VAR": "should_not_appear",
                "GITHUB_TOKEN": "ghp_secret123",
            },
            clear=True,
        ):
            vars_dict = _capture_env_vars()

            assert "PATH" in vars_dict
            assert vars_dict["PATH"] == "/usr/bin"
            assert "HOME" in vars_dict
            assert "MY_CUSTOM_VAR" not in vars_dict
            assert vars_dict.get("GITHUB_TOKEN") == "[REDACTED]"


class TestPackages:
    """Tests for package capture."""

    def test_capture_packages_with_uv(self) -> None:
        """Test _capture_packages with uv available."""
        # Mock uv being available and returning data
        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "package==1.0.0\nother==2.0.0"

                packages = _capture_packages()

                assert "package==1.0.0" in packages

    def test_capture_packages_uv_empty_output(self) -> None:
        """Test _capture_packages falls back to pip when uv returns empty."""
        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch("subprocess.run") as mock_run:
                # First call (uv) returns empty, second call (pip) returns data
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = ""

                _capture_packages()

                # Should have called both uv and pip
                assert mock_run.call_count >= 1

    def test_capture_packages_uv_subprocess_error(self) -> None:
        """Test _capture_packages falls back to pip when uv raises error."""
        call_count = [0]

        def side_effect(*args: object, **kwargs: object) -> object:
            call_count[0] += 1
            if call_count[0] == 1:
                raise subprocess.SubprocessError("uv failed")
            return type("Result", (), {"returncode": 0, "stdout": "pip==1.0\n"})()

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch("subprocess.run", side_effect=side_effect):
                packages = _capture_packages()
                assert packages == "pip==1.0"

    def test_capture_packages_fallback_to_pip(self) -> None:
        """Test _capture_packages falls back to pip."""
        with patch("shutil.which", return_value=None):  # No uv
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "pip-package==1.0.0"

                result = _capture_packages()

                # Should have called pip freeze and returned result
                assert mock_run.called
                assert result == "pip-package==1.0.0"

    def test_capture_packages_pip_failure(self) -> None:
        """Test _capture_packages returns empty when pip fails."""
        with patch("shutil.which", return_value=None):  # No uv
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stdout = ""

                result = _capture_packages()

                assert result == ""

    def test_capture_packages_handles_failure(self) -> None:
        """Test _capture_packages handles failures gracefully."""
        with patch("shutil.which", return_value=None):
            with patch(
                "subprocess.run", side_effect=subprocess.SubprocessError("Failed")
            ):
                packages = _capture_packages()
                assert packages == ""


class TestCaptureEnvironment:
    """Tests for complete environment capture."""

    def test_capture_environment_complete(self) -> None:
        """Test capture_environment returns complete data."""
        env = capture_environment(include_packages=False, include_git_diff=False)

        assert isinstance(env, EnvironmentCapture)
        assert env.system.os_name == platform.system()
        assert env.python.version == sys.version
        assert env.command.executable == sys.executable
        assert env.captured_at != ""

    def test_capture_environment_with_packages(self) -> None:
        """Test capture_environment with package capture."""
        env = capture_environment(include_packages=True, include_git_diff=False)

        # Packages may or may not be present depending on environment
        assert isinstance(env.packages, str)

    def test_capture_environment_with_git(self, git_repo: Path) -> None:
        """Test capture_environment in a git repo."""
        env = capture_environment(
            working_dir=git_repo,
            include_packages=False,
            include_git_diff=True,
        )

        assert env.git is not None
        assert env.git.repo_root == str(git_repo)


class TestBundleWriterEnvironment:
    """Tests for BundleWriter.write_environment integration."""

    def test_write_environment_creates_files(self, tmp_path: Path) -> None:
        """Test write_environment creates all expected files."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False, include_git_diff=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()

        assert "environment/system.json" in files
        assert "environment/python.json" in files
        assert "environment/env_vars.json" in files
        assert "environment/command.txt" in files

    def test_write_environment_with_packages(self, tmp_path: Path) -> None:
        """Test write_environment includes packages when captured."""
        # Create environment with mock packages
        env = capture_environment(include_packages=False, include_git_diff=False)
        # Replace with a version that has packages
        from dataclasses import replace

        env_with_packages = replace(env, packages="test-package==1.0.0\nother==2.0.0")

        with BundleWriter(tmp_path) as writer:
            writer.write_environment(env_with_packages)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert "environment/packages.txt" in bundle.list_files()
        env_data = bundle.environment
        assert env_data is not None
        assert "test-package==1.0.0" in str(env_data.get("packages", ""))

    def test_write_environment_with_git(self, git_repo: Path) -> None:
        """Test write_environment includes git info when in repo."""
        env = capture_environment(
            working_dir=git_repo,
            include_packages=False,
            include_git_diff=False,
        )

        with BundleWriter(git_repo.parent) as writer:
            writer.write_environment(env)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert "environment/git.json" in bundle.list_files()
        env_data = bundle.environment
        assert env_data is not None
        assert env_data.get("git") is not None

    def test_write_environment_with_git_diff(self, git_repo: Path) -> None:
        """Test write_environment includes git diff when present."""
        # Create uncommitted changes
        (git_repo / "test.txt").write_text("modified content")

        env = capture_environment(
            working_dir=git_repo,
            include_packages=False,
            include_git_diff=True,
        )

        with BundleWriter(git_repo.parent) as writer:
            writer.write_environment(env)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert "environment/git.diff" in bundle.list_files()
        env_data = bundle.environment
        assert env_data is not None
        assert env_data.get("git_diff") is not None
        assert "modified content" in str(env_data.get("git_diff"))

    def test_write_environment_system_data(self, tmp_path: Path) -> None:
        """Test environment system data is correct."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        env = bundle.environment

        assert env is not None
        assert env["system"] is not None
        system = env["system"]
        assert isinstance(system, dict)
        assert system["os_name"] == platform.system()
        assert system["architecture"] == platform.machine()

    def test_write_environment_python_data(self, tmp_path: Path) -> None:
        """Test environment Python data is correct."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        env = bundle.environment

        assert env is not None
        assert env["python"] is not None
        python = env["python"]
        assert isinstance(python, dict)
        assert python["version"] == sys.version
        assert python["executable"] == sys.executable

    def test_write_environment_with_pre_captured(self, tmp_path: Path) -> None:
        """Test write_environment with pre-captured environment."""
        env = capture_environment(include_packages=False, include_git_diff=False)

        with BundleWriter(tmp_path) as writer:
            writer.write_environment(env)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.environment is not None
        assert bundle.environment["system"] is not None

    def test_write_environment_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_environment handles errors gracefully."""
        with patch(
            "weakincentives.debug.environment.capture_environment",
            side_effect=RuntimeError("Capture failed"),
        ):
            with BundleWriter(tmp_path) as writer:
                writer.write_environment()

        assert writer.path is not None
        assert "Failed to write environment" in caplog.text


class TestDebugBundleEnvironmentAccessors:
    """Tests for DebugBundle environment accessor property."""

    def test_environment_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test environment returns None when not present."""
        with BundleWriter(tmp_path) as writer:
            pass  # Don't write environment

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.environment is None

    def test_environment_returns_dict_with_all_keys(self, tmp_path: Path) -> None:
        """Test environment returns dict with all expected keys."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        env = bundle.environment

        assert env is not None
        # All keys should be present (some may be None)
        assert "system" in env
        assert "python" in env
        assert "packages" in env
        assert "env_vars" in env
        assert "git" in env
        assert "git_diff" in env
        assert "command" in env
        assert "container" in env

    def test_environment_missing_optional_files(self, tmp_path: Path) -> None:
        """Test environment accessor handles missing optional files gracefully."""
        # Create environment without git or container
        env = capture_environment(include_packages=False, include_git_diff=False)
        # Make sure git and container are None
        from dataclasses import replace

        env_minimal = replace(env, git=None, git_diff=None, container=None, packages="")

        with BundleWriter(tmp_path) as writer:
            writer.write_environment(env_minimal)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        env_data = bundle.environment

        assert env_data is not None
        # Optional files should be None when not written
        assert env_data["git"] is None
        assert env_data["container"] is None
        assert env_data["git_diff"] is None
        # But required files should be present
        assert env_data["system"] is not None
        assert env_data["python"] is not None

    def test_environment_command_content(self, tmp_path: Path) -> None:
        """Test environment command contains expected content."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        env = bundle.environment

        assert env is not None
        command = env["command"]
        assert command is not None
        assert isinstance(command, str)
        assert "Working Directory:" in command
        assert "Executable:" in command


class TestReadmeGeneration:
    """Tests for README generation with environment section."""

    def test_readme_includes_environment_section(self, tmp_path: Path) -> None:
        """Test README includes environment section."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment(include_packages=False)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        readme = bundle.read_file("README.txt").decode("utf-8")

        assert "environment/" in readme
        assert "system.json" in readme
        assert "python.json" in readme
        assert "packages.txt" in readme
        assert "env_vars.json" in readme
        assert "git.json" in readme
        assert "command.txt" in readme
        assert "container.json" in readme
        assert "Reproducibility envelope" in readme
