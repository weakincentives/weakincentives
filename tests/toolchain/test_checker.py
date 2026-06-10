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
import re
from unittest import mock

from toolchain.checker import (
    AutoFormatChecker,
    SubprocessChecker,
    is_ci_environment,
    merge_output,
    output_tail,
    parser_drift_warning,
)
from toolchain.result import Diagnostic, Location


def _parse_files(output: str) -> list[str]:
    """Parse files from 'Error: File "..." is not formatted.' pattern."""
    return sorted(
        m.group(1)
        for m in re.finditer(r'Error: File "([^"]+)" is not formatted\.', output)
    )


def _no_files(_output: str) -> list[str]:
    return []


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
        # No parser registered: no drift warning, raw output is the fallback
        assert result.diagnostics == ()

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

    def test_parser_drift_warning_on_unparseable_failure(self) -> None:
        """A failing command whose parser extracts nothing gets a drift warning."""

        def parse_nothing(_output: str, _code: int) -> tuple[Diagnostic, ...]:
            return ()

        checker = SubprocessChecker(
            name="drifted",
            description="Parser no longer matches",
            command=["bash", "-c", "echo 'new output format'; exit 1"],
            parser=parse_nothing,
        )
        result = checker.run()
        assert result.status == "failed"
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "warning"
        assert "may have drifted" in result.diagnostics[0].message
        assert "new output format" in result.diagnostics[0].message

    def test_no_drift_warning_when_parser_extracts_diagnostics(self) -> None:
        def parse_one(_output: str, _code: int) -> tuple[Diagnostic, ...]:
            return (Diagnostic(message="real diagnostic"),)

        checker = SubprocessChecker(
            name="parsed",
            description="Parser works",
            command=["bash", "-c", "echo broken; exit 1"],
            parser=parse_one,
        )
        result = checker.run()
        assert result.status == "failed"
        assert len(result.diagnostics) == 1
        assert "drifted" not in result.diagnostics[0].message

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

    def test_command_stored_in_result(self) -> None:
        """Command args are stored in result for reproduction hints (all exit paths)."""
        # Passing command
        checker = SubprocessChecker(
            name="echo", description="Echo", command=["echo", "hi"]
        )
        assert checker.run().command == ("echo", "hi")

        # Failing command
        checker = SubprocessChecker(
            name="false", description="Fails", command=["false"]
        )
        assert checker.run().command == ("false",)

        # Timeout
        checker = SubprocessChecker(
            name="slow", description="Slow", command=["sleep", "10"], timeout=1
        )
        assert checker.run().command == ("sleep", "10")

        # FileNotFoundError
        checker = SubprocessChecker(
            name="missing", description="Missing", command=["nonexistent_xyz"]
        )
        assert checker.run().command == ("nonexistent_xyz",)


class TestMergeOutput:
    """Tests for merge_output."""

    def test_both_streams(self) -> None:
        assert merge_output("out", "err") == "out\nerr"

    def test_stdout_only(self) -> None:
        assert merge_output("out", "") == "out"

    def test_stderr_only(self) -> None:
        assert merge_output("", "err") == "err"

    def test_neither(self) -> None:
        assert merge_output("", "") == ""


class TestOutputTail:
    """Tests for output_tail."""

    def test_short_output_returned_whole(self) -> None:
        assert output_tail("a\nb\nc") == "a\nb\nc"

    def test_long_output_truncated_to_tail(self) -> None:
        output = "\n".join(f"line {i}" for i in range(30))
        tail = output_tail(output, max_lines=5)
        assert tail == "line 25\nline 26\nline 27\nline 28\nline 29"


class TestParserDriftWarning:
    """Tests for parser_drift_warning."""

    def test_warning_includes_name_and_tail(self) -> None:
        diagnostic = parser_drift_warning("lint", "some raw output")
        assert diagnostic.severity == "warning"
        assert "lint:" in diagnostic.message
        assert "may have drifted" in diagnostic.message
        assert "some raw output" in diagnostic.message


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
            file_list_parser=_no_files,
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
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"

    def test_ci_failure_with_parser_diagnostics(self) -> None:
        """In CI, parser diagnostics are attached to the failure."""

        def parse_diag(_output: str, _code: int) -> tuple[Diagnostic, ...]:
            return (
                Diagnostic(
                    message="File needs formatting", location=Location(file="a.py")
                ),
            )

        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "echo 'Would reformat: a.py'; exit 1"],
            fix_command=["true"],
            file_list_parser=_no_files,
            parser=parse_diag,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("needs formatting" in d.message for d in result.diagnostics)

    def test_ci_failure_drift_warning_when_parser_extracts_nothing(self) -> None:
        """In CI, an unparseable failure gets a drift warning."""

        def parse_nothing(_output: str, _code: int) -> tuple[Diagnostic, ...]:
            return ()

        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "echo 'mystery output'; exit 1"],
            fix_command=["true"],
            file_list_parser=_no_files,
            parser=parse_nothing,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "warning"
        assert "may have drifted" in result.diagnostics[0].message

    def test_autofix_locally_reports_files(self) -> None:
        """Locally, should report file names parsed from check output."""
        check_output = (
            'Error: File "docs/README.md" is not formatted.\n'
            'Error: File "specs/DESIGN.md" is not formatted.'
        )
        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            check_command=["bash", "-c", f"echo '{check_output}'; exit 1"],
            fix_command=["true"],
            file_list_parser=_parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "docs/README.md" in info_diags[0].message
        assert "specs/DESIGN.md" in info_diags[0].message

    def test_no_changes_needed_locally(self) -> None:
        """Locally, no diagnostics when nothing needs formatting."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],  # Exit 0 means nothing needs formatting
            fix_command=["echo", "should not run"],
            file_list_parser=_parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_autofix_drift_warning_when_file_list_empty(self) -> None:
        """Locally, a failing check with no parsed files yields a drift warning."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "echo 'unrecognized format'; exit 1"],
            fix_command=["true"],
            file_list_parser=_parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        # Fix ran successfully, so the check passes, but the drift is flagged
        assert result.status == "passed"
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "warning"
        assert "may have drifted" in result.diagnostics[0].message
        assert "unrecognized format" in result.diagnostics[0].message

    def test_timeout_handling(self) -> None:
        """Timeout should be reported as failure."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["sleep", "10"],
            fix_command=["true"],
            file_list_parser=_no_files,
            timeout=1,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_format_file_message_single(self) -> None:
        """Test message formatting for single file."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
            file_list_parser=_no_files,
        )
        msg = checker._format_file_message(["src/test.py"])
        assert "1 file" in msg
        assert "src/test.py" in msg
        # The caller must learn the working tree changed
        assert "uncommitted changes" in msg

    def test_format_file_message_multiple(self) -> None:
        """Test message formatting for multiple files."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["true"],
            fix_command=["true"],
            file_list_parser=_no_files,
        )
        msg = checker._format_file_message(["a.py", "b.py", "c.py"])
        assert "3 files" in msg
        for f in ("a.py", "b.py", "c.py"):
            assert f in msg
        assert "uncommitted changes" in msg

    def test_stderr_captured_in_ci(self) -> None:
        """In CI, stderr should be captured in the output."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "echo err >&2; false"],
            fix_command=["true"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            result = checker.run()
        assert "err" in result.output

    def test_stderr_parsed_for_file_list_locally(self) -> None:
        """Locally, the file list parser sees stderr output too."""

        def parse_stderr(output: str) -> list[str]:
            return ["error.md"] if "error.md" in output else []

        checker = AutoFormatChecker(
            name="markdown",
            description="Test markdown",
            # Output goes to stderr only, not stdout
            check_command=[
                "bash",
                "-c",
                "echo 'error.md needs formatting' >&2; exit 1",
            ],
            fix_command=["true"],
            file_list_parser=parse_stderr,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "passed"
        info_diags = [d for d in result.diagnostics if d.severity == "info"]
        assert len(info_diags) == 1
        assert "error.md" in info_diags[0].message

    def test_timeout_during_autofix(self) -> None:
        """Timeout during autofix should be reported as failure."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["sleep", "10"],
            file_list_parser=_no_files,
            timeout=1,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)
        # The command that timed out is recorded for reproduction
        assert result.command == ("sleep", "10")

    def test_timeout_during_local_check(self) -> None:
        """Timeout during the local check command reports that command."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["sleep", "10"],
            fix_command=["true"],
            file_list_parser=_no_files,
            timeout=1,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert result.command == ("sleep", "10")

    def test_fix_command_failure_reports_error(self) -> None:
        """Should fail when fix command exits non-zero with stderr."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=[
                "bash",
                "-c",
                "echo 'Error: File \"test.py\" is not formatted.'; exit 1",
            ],
            fix_command=["bash", "-c", "echo 'error message' >&2; exit 1"],
            file_list_parser=_parse_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Auto-fix command failed" in result.diagnostics[0].message
        assert "error message" in result.output

    def test_fix_command_failure_stdout_only(self) -> None:
        """Should fail when fix command exits non-zero with only stdout."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["bash", "-c", "echo 'stdout error'; exit 1"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Auto-fix command failed" in result.diagnostics[0].message
        assert "stdout error" in result.output

    def test_fix_failure_surfaces_remaining_issues_via_parser(self) -> None:
        """Unfixable issues left after the fix command surface as diagnostics.

        Tools like biome apply safe fixes but still exit non-zero when issues
        without an auto-fix remain; those must be parsed, not dumped raw.
        """

        def parse_remaining(output: str, code: int) -> tuple[Diagnostic, ...]:
            if code != 0 and "unfixable" in output:
                return (
                    Diagnostic(
                        message="Unfixable issue",
                        location=Location(file="a.js", line=3),
                    ),
                )
            return ()

        checker = AutoFormatChecker(
            name="biome",
            description="Test biome",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["bash", "-c", "echo 'unfixable: a.js'; exit 1"],
            file_list_parser=_no_files,
            parser=parse_remaining,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert "Unfixable issue" in result.diagnostics[0].message
        assert "Auto-fix command failed" not in result.diagnostics[0].message

    def test_command_not_found_in_ci(self) -> None:
        """In CI, should handle missing command gracefully."""
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["nonexistent_command_12345"],
            fix_command=["true"],
            file_list_parser=_no_files,
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
            check_command=["bash", "-c", "exit 1"],
            fix_command=["nonexistent_command_12345"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = checker.run()
        assert result.status == "failed"
        assert any("Command not found" in d.message for d in result.diagnostics)
        assert any("uv sync" in d.message for d in result.diagnostics)

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

    def test_command_stored_in_result(self) -> None:
        """Command is stored in result for all exit paths (CI and local)."""
        # CI: check_command stored on pass
        checker = AutoFormatChecker(
            name="format",
            description="Test format",
            check_command=["uv", "run", "ruff", "format", "--check", "."],
            fix_command=["uv", "run", "ruff", "format", "."],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            assert checker.run().command == (
                "uv",
                "run",
                "ruff",
                "format",
                "--check",
                ".",
            )

        # CI: check_command stored on timeout
        checker = AutoFormatChecker(
            name="f",
            description="f",
            check_command=["sleep", "10"],
            fix_command=["true"],
            file_list_parser=_no_files,
            timeout=1,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            assert checker.run().command == ("sleep", "10")

        # CI: check_command stored on FileNotFoundError
        checker = AutoFormatChecker(
            name="f",
            description="f",
            check_command=["nonexistent_cmd_abc"],
            fix_command=["true"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=True):
            assert checker.run().command == ("nonexistent_cmd_abc",)

        # Local: fix_command stored on fix failure
        checker = AutoFormatChecker(
            name="f",
            description="f",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["bash", "-c", "exit 1"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            assert checker.run().command == ("bash", "-c", "exit 1")

        # Local: fix_command stored on FileNotFoundError
        checker = AutoFormatChecker(
            name="f",
            description="f",
            check_command=["bash", "-c", "exit 1"],
            fix_command=["nonexistent_cmd_xyz"],
            file_list_parser=_no_files,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            assert checker.run().command == ("nonexistent_cmd_xyz",)
