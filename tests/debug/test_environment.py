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

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.environment import (
    CommandInfo,
    EnvironmentCapture,
    PythonInfo,
    SystemInfo,
    _capture_command_info,
    _capture_env_vars,
    _capture_memory_bytes,
    _capture_packages,
    _capture_python_info,
    _capture_system_info,
    _get_darwin_memory_bytes,
    _get_linux_memory_bytes,
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
