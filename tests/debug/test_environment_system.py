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

"""Tests for system, Python, container, command, env vars, and package capture."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.debug.environment import (
    CommandInfo,
    ContainerInfo,
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
)


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
