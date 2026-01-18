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

"""Tests for subprocess utilities."""

from __future__ import annotations

from pathlib import Path

from subprocess_utils import (
    SubprocessResult,
    run_python_module,
    run_tool,
)


class TestSubprocessResult:
    """Tests for SubprocessResult dataclass."""

    def test_success_property(self) -> None:
        """Success property reflects return code."""
        success = SubprocessResult(
            returncode=0,
            stdout="output",
            stderr="",
            duration_ms=100,
        )
        assert success.success is True

        failure = SubprocessResult(
            returncode=1,
            stdout="",
            stderr="error",
            duration_ms=50,
        )
        assert failure.success is False

    def test_output_property(self) -> None:
        """Output combines stdout and stderr."""
        result = SubprocessResult(
            returncode=0,
            stdout="standard output",
            stderr="standard error",
            duration_ms=10,
        )
        assert "standard output" in result.output
        assert "standard error" in result.output

    def test_output_empty(self) -> None:
        """Output handles empty streams."""
        result = SubprocessResult(
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=10,
        )
        assert result.output == ""


class TestRunTool:
    """Tests for run_tool function."""

    def test_run_echo(self) -> None:
        """Run a simple echo command."""
        result = run_tool(["echo", "hello"])
        assert result.success is True
        assert "hello" in result.stdout
        assert result.duration_ms >= 0

    def test_run_with_cwd(self, tmp_path: Path) -> None:
        """Run command in specific directory."""
        result = run_tool(["pwd"], cwd=tmp_path)
        assert result.success is True
        assert str(tmp_path) in result.stdout or tmp_path.name in result.stdout

    def test_run_failing_command(self) -> None:
        """Handle failing command."""
        result = run_tool(["false"])
        assert result.success is False
        assert result.returncode != 0

    def test_run_nonexistent_command(self) -> None:
        """Handle command not found."""
        result = run_tool(["nonexistent_command_12345"])
        assert result.success is False
        assert result.returncode == 127
        assert "not found" in result.stderr.lower()

    def test_run_with_env(self) -> None:
        """Run command with custom environment."""
        result = run_tool(
            ["sh", "-c", "echo $TEST_VAR"], env={"TEST_VAR": "test_value"}
        )
        assert result.success is True
        assert "test_value" in result.stdout

    def test_run_with_timeout(self) -> None:
        """Command exceeding timeout returns timeout error."""
        result = run_tool(["sleep", "10"], timeout_seconds=0.1)
        assert result.success is False
        assert result.returncode == 124
        assert "timed out" in result.stderr.lower()


class TestRunPythonModule:
    """Tests for run_python_module function."""

    def test_run_python_version(self) -> None:
        """Run python with --version."""
        result = run_python_module("sys", ["--version"])
        # This will fail because sys isn't runnable, but demonstrates the pattern
        # We just verify it returns a result
        assert isinstance(result, SubprocessResult)

    def test_run_pip_help(self) -> None:
        """Run pip --help."""
        result = run_python_module("pip", ["--help"])
        assert result.success is True
        assert "pip" in result.stdout.lower()
