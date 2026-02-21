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

"""Tests for environment capture, bundle writer, accessors, and README generation."""

from __future__ import annotations

import platform
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.environment import (
    EnvironmentCapture,
    capture_environment,
)


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
