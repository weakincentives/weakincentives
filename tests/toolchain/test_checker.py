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

from toolchain.checker import (
    AutoFormatChecker,
    SubprocessChecker,
    _no_file_list_parse,
    is_ci_environment,
)
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

    def test_autofix_locally_reports_changes_with_json(self) -> None:
        """Locally with JSON check, should report file names from JSON."""
        # JSON check returns file info, fix command runs
        json_output = '[{"filename": "src/changed.py", "message": "File would be reformatted"}]'
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
            json_check_command=["bash", "-c", f"echo '{json_output}'; exit 1"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "src/changed.py" in info_diags[0].message

    def test_no_changes_needed_locally_with_json(self) -> None:
        """Locally with JSON check, no diagnostics when nothing needs formatting."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
            json_check_command=["true"],  # Exit 0 means nothing needs formatting
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_autofix_locally_fallback_without_json(self) -> None:
        """Locally without JSON check, should fall back to count parsing."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo '2 files reformatted'"],
            # No json_check_command
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "2 files" in info_diags[0].message

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

    def test_parse_json_output(self) -> None:
        """Test parsing of JSON formatted output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        json_output = '[{"filename": "src/foo.py"}, {"filename": "src/bar.py"}]'
        files = checker._parse_json_output(json_output)
        assert files == ["src/bar.py", "src/foo.py"]  # Sorted

    def test_parse_json_output_invalid(self) -> None:
        """Test parsing invalid JSON returns empty list."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        assert checker._parse_json_output("not json") == []
        assert checker._parse_json_output("") == []

    def test_parse_json_output_not_list(self) -> None:
        """Test parsing valid JSON that's not a list returns empty list."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        assert checker._parse_json_output('{"filename": "test.py"}') == []
        assert checker._parse_json_output('"just a string"') == []

    def test_parse_json_output_deduplicates(self) -> None:
        """Test that duplicate filenames are deduplicated."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        # Same file can have multiple issues
        json_output = '[{"filename": "a.py"}, {"filename": "a.py"}, {"filename": "b.py"}]'
        files = checker._parse_json_output(json_output)
        assert files == ["a.py", "b.py"]

    def test_format_file_message_single(self) -> None:
        """Test message formatting for single file."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        msg = checker._format_file_message(["src/test.py"])
        assert "1 file" in msg
        assert "src/test.py" in msg

    def test_format_file_message_multiple(self) -> None:
        """Test message formatting for multiple files."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        msg = checker._format_file_message(["a.py", "b.py", "c.py"])
        assert "3 files" in msg

    def test_format_file_message_many_files(self) -> None:
        """Test message formatting shows all files."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        files = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py"]
        msg = checker._format_file_message(files)
        assert "7 files" in msg
        # All files should be listed
        for f in files:
            assert f in msg

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
            check_command=["true"],
            fix_command=["sleep", "10"],
            timeout=1,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_no_files_in_output_means_nothing_changed(self) -> None:
        """When no count in output, nothing needed formatting."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo 'All files already formatted'"],
            # No json_check_command - falls back to count parsing
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_parse_reformat_count_singular(self) -> None:
        """Should parse '1 file reformatted' from output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        assert checker._parse_reformat_count("1 file reformatted") == 1

    def test_parse_reformat_count_plural(self) -> None:
        """Should parse 'N files reformatted' from output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        assert checker._parse_reformat_count("5 files reformatted") == 5

    def test_parse_reformat_count_with_other_text(self) -> None:
        """Should parse count even with other text in output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
        )
        output = "warning: some warning\n3 files reformatted\n"
        assert checker._parse_reformat_count(output) == 3

    def test_fallback_count_plural(self) -> None:
        """Without JSON, should report count from fix output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo '2 files reformatted'"],
            # No json_check_command
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "2 files" in info_diags[0].message

    def test_fallback_count_singular(self) -> None:
        """Without JSON, should report singular count from fix output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo '1 file reformatted'"],
            # No json_check_command
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "1 file" in info_diags[0].message
        assert "files" not in info_diags[0].message

    def test_fix_command_failure_reports_error(self) -> None:
        """Should fail when fix command exits non-zero with stderr."""
        # JSON check says files need formatting, but fix fails
        json_output = '[{"filename": "test.py"}]'
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo 'error message' >&2; exit 1"],
            json_check_command=["bash", "-c", f"echo '{json_output}'; exit 1"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Auto-fix command failed" in result.diagnostics[0].message
        assert "error message" in result.output

    def test_fix_command_failure_stdout_only(self) -> None:
        """Should fail when fix command exits non-zero with only stdout."""
        json_output = '[{"filename": "test.py"}]'
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo 'stdout error'; exit 1"],
            json_check_command=["bash", "-c", f"echo '{json_output}'; exit 1"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Auto-fix command failed" in result.diagnostics[0].message
        assert "stdout error" in result.output

    def test_command_not_found_in_ci(self) -> None:
        """In CI, should handle missing command gracefully."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["nonexistent_command_12345"],
            fix_command=["true"],
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Command not found" in d.message for d in result.diagnostics)
        assert any("uv sync" in d.message for d in result.diagnostics)

    def test_command_not_found_locally(self) -> None:
        """Locally, should handle missing fix command gracefully."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["nonexistent_command_12345"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Command not found" in d.message for d in result.diagnostics)
        assert any("uv sync" in d.message for d in result.diagnostics)

    def test_file_list_parser_extracts_files_locally(self) -> None:
        """Locally with file_list_parser, should report file names from text output."""

        def parse_files(output: str) -> list[str]:
            """Parse files from 'Error: File "..." is not formatted.' pattern."""
            import re

            return sorted(
                m.group(1)
                for m in re.finditer(r'Error: File "([^"]+)" is not formatted\.', output)
            )

        check_output = (
            'Error: File "docs/README.md" is not formatted.\n'
            'Error: File "specs/DESIGN.md" is not formatted.'
        )
        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            check_command=["bash", "-c", f"echo '{check_output}'; exit 1"],
            fix_command=["true"],
            file_list_parser=parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "docs/README.md" in info_diags[0].message
        assert "specs/DESIGN.md" in info_diags[0].message

    def test_file_list_parser_no_changes_needed(self) -> None:
        """Locally with file_list_parser, no diagnostics when nothing needs formatting."""

        def parse_files(_output: str) -> list[str]:
            return []

        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            check_command=["true"],  # Exit 0 means nothing needs formatting
            fix_command=["true"],
            file_list_parser=parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_file_list_parser_check_only_in_ci(self) -> None:
        """In CI, file_list_parser should not be used - check-only mode."""

        def parse_files(_output: str) -> list[str]:
            return ["should_not_see_this.md"]

        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            check_command=["true"],
            fix_command=["echo", "should not run"],
            file_list_parser=parse_files,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        # No info diagnostics in CI mode
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 0

    def test_file_list_parser_single_file(self) -> None:
        """Locally with single file, message should be singular."""

        def parse_files(_output: str) -> list[str]:
            return ["single.md"]

        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["true"],
            file_list_parser=parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "1 file" in info_diags[0].message
        assert "single.md" in info_diags[0].message

    def test_file_list_parser_stderr_only(self) -> None:
        """Locally with file_list_parser, should handle stderr-only output."""

        def parse_files(output: str) -> list[str]:
            """Parse files from output (including stderr)."""
            if "error.md" in output:
                return ["error.md"]
            return []

        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            # Output goes to stderr only, not stdout
            check_command=["bash", "-c", "echo 'Error: error.md needs formatting' >&2; exit 1"],
            fix_command=["true"],
            file_list_parser=parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "error.md" in info_diags[0].message

    def test_default_file_list_parser_not_used(self) -> None:
        """Without custom file_list_parser, should fall back to count parsing."""
        # This tests the default _no_file_list_parse path where no parser is set
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["bash", "-c", "echo '1 file reformatted'"],
            # No json_check_command and no custom file_list_parser
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "1 file" in info_diags[0].message


class TestNoFileListParse:
    """Tests for _no_file_list_parse helper."""

    def test_returns_empty_list(self) -> None:
        """Default file list parser should always return empty list."""
        assert _no_file_list_parse("any output") == []
        assert _no_file_list_parse("") == []
        assert _no_file_list_parse("Error: File 'test.md' is not formatted.") == []
