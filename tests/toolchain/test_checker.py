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

"""Tests for toolchain checker module."""

from __future__ import annotations

import os
from unittest import mock

from toolchain.checker import AutoFormatChecker, SubprocessChecker, is_ci_environment
from toolchain.result import Diagnostic, Location


class TestSubprocessChecker:
    """Tests for SubprocessChecker."""

    def test_successful_command(self) -> None:
        checker = SubprocessChecker(
            name="echo",
            description="Echo test",
            command=["echo", "hello"],
        )
        result = checker.run()
        assert result.name == "echo"
        assert result.status == "passed"
        assert result.duration_ms >= 0
        assert "hello" in result.output

    def test_failing_command(self) -> None:
        checker = SubprocessChecker(
            name="false",
            description="Always fails",
            command=["false"],
        )
        result = checker.run()
        assert result.name == "false"
        assert result.status == "failed"

    def test_command_with_stderr(self) -> None:
        checker = SubprocessChecker(
            name="stderr",
            description="Writes to stderr",
            command=["bash", "-c", "echo error >&2"],
        )
        result = checker.run()
        assert "error" in result.output

    def test_command_with_stdout_and_stderr(self) -> None:
        checker = SubprocessChecker(
            name="both",
            description="Writes to both",
            command=["bash", "-c", "echo out; echo err >&2"],
        )
        result = checker.run()
        assert "out" in result.output
        assert "err" in result.output

    def test_parser_extracts_diagnostics(self) -> None:
        def parse_output(output: str, code: int) -> tuple[Diagnostic, ...]:
            return (Diagnostic(message=f"Parsed: {output.strip()}"),)

        checker = SubprocessChecker(
            name="parsed",
            description="Uses parser",
            command=["echo", "test message"],
            parser=parse_output,
        )
        result = checker.run()
        assert len(result.diagnostics) == 1
        assert "Parsed: test message" in result.diagnostics[0].message

    def test_timeout_handling(self) -> None:
        checker = SubprocessChecker(
            name="slow",
            description="Times out",
            command=["sleep", "10"],
            timeout=1,
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_command_not_found(self) -> None:
        checker = SubprocessChecker(
            name="missing",
            description="Command not found",
            command=["nonexistent_command_12345"],
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("Command not found" in d.message for d in result.diagnostics)

    def test_custom_environment(self) -> None:
        checker = SubprocessChecker(
            name="env",
            description="Custom env",
            command=["bash", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "custom_value"},
        )
        result = checker.run()
        assert "custom_value" in result.output

    def test_name_and_description_properties(self) -> None:
        checker = SubprocessChecker(
            name="my-checker",
            description="My checker description",
            command=["true"],
        )
        assert checker.name == "my-checker"
        assert checker.description == "My checker description"

    def test_output_stripped(self) -> None:
        checker = SubprocessChecker(
            name="whitespace",
            description="Test whitespace",
            command=["echo", "  hello  "],
        )
        result = checker.run()
        # Output should be stripped
        assert result.output == "hello"


class TestIsCiEnvironment:
    """Tests for is_ci_environment detection."""

    def test_github_actions_detected(self) -> None:
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            assert is_ci_environment() is True

    def test_ci_env_detected(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            assert is_ci_environment() is True

    def test_local_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            assert is_ci_environment() is False

    def test_non_true_values_ignored(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "false"}, clear=True):
            assert is_ci_environment() is False


class TestAutoFormatChecker:
    """Tests for AutoFormatChecker."""

    def test_check_only_in_ci(self) -> None:
        """In CI, should only check without fixing."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],  # Always succeeds
            fix_command=["echo", "should not run"],
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "passed"

    def test_check_only_fails_in_ci(self) -> None:
        """In CI, failing check should fail the result."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],  # Always fails
            fix_command=["echo", "should not run"],
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"

    def test_autofix_locally_when_check_fails(self) -> None:
        """Locally, should apply fix when check fails."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],  # Check fails
            fix_command=["true"],  # Fix succeeds
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        # Should pass because fix was applied
        assert result.status == "passed"
        # Should have info diagnostic about fixing
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1

    def test_no_autofix_when_check_passes(self) -> None:
        """Locally, should not fix when check passes."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],  # Check passes
            fix_command=["echo", "should not run"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_timeout_handling(self) -> None:
        """Timeout should be reported as failure."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["sleep", "10"],
            fix_command=["true"],
            timeout=1,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_parse_fixed_files(self) -> None:
        """Test parsing of reformatted file paths."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        output = "src/foo.py\nsrc/bar.py\n1 file reformatted"
        files = checker._parse_fixed_files(output)
        assert files == ["src/foo.py", "src/bar.py"]

    def test_reports_single_file_fixed(self) -> None:
        """Should report single file fix in natural language."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],
            fix_command=["bash", "-c", "echo src/test.py"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "1 file" in info_diags[0].message
        assert "src/test.py" in info_diags[0].message

    def test_reports_multiple_files_fixed(self) -> None:
        """Should report multiple files fix in natural language."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],
            fix_command=["bash", "-c", "echo -e 'a.py\nb.py\nc.py'"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "3 files" in info_diags[0].message

    def test_reports_many_files_truncated(self) -> None:
        """Should truncate file list when more than 5 files are fixed."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],
            fix_command=[
                "bash",
                "-c",
                "echo -e 'a.py\nb.py\nc.py\nd.py\ne.py\nf.py\ng.py'",
            ],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "7 files" in info_diags[0].message
        assert "and 2 more" in info_diags[0].message

    def test_stderr_captured_in_ci(self) -> None:
        """In CI, stderr should be captured in the output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "echo err >&2; false"],
            fix_command=["true"],
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert "err" in result.output

    def test_timeout_during_autofix(self) -> None:
        """Timeout during autofix should be reported as failure."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],
            fix_command=["sleep", "10"],
            timeout=1,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_fallback_message_when_no_files_parsed(self) -> None:
        """Should use generic message when no .py files in output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],
            fix_command=["bash", "-c", "echo 'reformatting complete'"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "automatically fixed" in info_diags[0].message.lower()

    def test_fix_command_failure_reports_error(self) -> None:
        """Should fail when fix command exits non-zero."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["false"],  # Check fails, triggers fix
            fix_command=["bash", "-c", "echo 'error message' >&2; exit 1"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Auto-fix command failed" in result.diagnostics[0].message
        assert "error message" in result.output
