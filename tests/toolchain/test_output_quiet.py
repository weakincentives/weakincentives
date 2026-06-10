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

"""Tests for the QuietFormatter output formatter."""

from __future__ import annotations

import io
from pathlib import Path

from toolchain.output import QuietFormatter
from toolchain.result import CheckResult, Diagnostic, Location, Report


def _make_passing_report() -> Report:
    return Report(
        results=(
            CheckResult(name="lint", status="passed", duration_ms=1200),
            CheckResult(name="test", status="passed", duration_ms=45000),
        ),
        total_duration_ms=46200,
    )


def _make_failing_report() -> Report:
    return Report(
        results=(
            CheckResult(name="lint", status="passed", duration_ms=1200),
            CheckResult(
                name="test",
                status="failed",
                duration_ms=45000,
                diagnostics=(
                    Diagnostic(
                        message="AssertionError: assert 1 == 2",
                        location=Location(file="tests/test_foo.py", line=42),
                    ),
                    Diagnostic(
                        message="AssertionError: assert True is False",
                        location=Location(file="tests/test_bar.py", line=17),
                    ),
                ),
            ),
        ),
        total_duration_ms=46200,
    )


class TestQuietFormatter:
    """Tests for QuietFormatter."""

    def test_passing_report_empty_output(self) -> None:
        formatter = QuietFormatter(color=False)
        output = formatter.format(_make_passing_report())
        assert output == ""

    def test_failing_report_shows_failures(self) -> None:
        formatter = QuietFormatter(color=False)
        output = formatter.format(_make_failing_report())
        assert "test" in output
        assert "tests/test_foo.py:42" in output
        assert "lint" not in output  # Passing check not shown

    def test_failing_report_with_color(self) -> None:
        formatter = QuietFormatter(color=True)
        output = formatter.format(_make_failing_report())
        assert "\033[31m" in output  # Red color code

    def test_truncates_diagnostics(self, tmp_path: Path) -> None:
        diagnostics = tuple(Diagnostic(message=f"Error {i}") for i in range(15))
        report = Report(
            results=(
                CheckResult(
                    name="lint",
                    status="failed",
                    duration_ms=100,
                    diagnostics=diagnostics,
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=False, log_dir=tmp_path)
        output = formatter.format(report)
        assert "... and 5 more" in output
        # Full version saved, path and re-run hint included
        log_file = tmp_path / "lint.log"
        assert f"full report: {log_file}" in output
        assert "Re-run just this check: uv run python check.py lint" in output
        saved = log_file.read_text(encoding="utf-8")
        assert "Error 14" in saved  # beyond the display cap

    def test_reproduction_hint_not_shown_when_no_truncation(self) -> None:
        report = Report(
            results=(
                CheckResult(
                    name="lint",
                    status="failed",
                    duration_ms=100,
                    diagnostics=(Diagnostic(message="Error 1"),),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "Run: python check.py" not in output

    def test_auto_color_detection(self) -> None:
        stream = io.StringIO()
        formatter = QuietFormatter(stream=stream)
        assert formatter._use_color() is False

    def test_info_diagnostics_shown_for_passed_checks(self) -> None:
        """Info diagnostics should be shown even in quiet mode for passed checks."""
        report = Report(
            results=(
                CheckResult(
                    name="format",
                    status="passed",
                    duration_ms=100,
                    diagnostics=(
                        Diagnostic(
                            message="Automatically reformatted 2 files",
                            severity="info",
                        ),
                    ),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "Automatically reformatted" in output

    def test_info_diagnostics_with_color(self) -> None:
        """Info diagnostics should use cyan color in quiet mode."""
        report = Report(
            results=(
                CheckResult(
                    name="format",
                    status="passed",
                    duration_ms=100,
                    diagnostics=(Diagnostic(message="Auto-fixed", severity="info"),),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=True)
        output = formatter.format(report)
        assert "\033[36m" in output  # Cyan color code

    def test_warning_diagnostics_shown_for_passed_checks(self) -> None:
        """Warning diagnostics should be shown in quiet mode for passed checks."""
        report = Report(
            results=(
                CheckResult(
                    name="code-length",
                    status="passed",
                    duration_ms=50,
                    diagnostics=(
                        Diagnostic(
                            message="File has 700 lines (max 620)",
                            location=Location(file="src/big.py"),
                            severity="warning",
                        ),
                    ),
                ),
            ),
            total_duration_ms=50,
        )
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "700 lines" in output

    def test_warning_diagnostics_with_color(self) -> None:
        """Warning diagnostics should use yellow color in quiet mode."""
        report = Report(
            results=(
                CheckResult(
                    name="code-length",
                    status="passed",
                    duration_ms=50,
                    diagnostics=(
                        Diagnostic(
                            message="File has 700 lines (max 620)",
                            location=Location(file="src/big.py"),
                            severity="warning",
                        ),
                    ),
                ),
            ),
            total_duration_ms=50,
        )
        formatter = QuietFormatter(color=True)
        output = formatter.format(report)
        assert "\033[33m" in output  # Yellow color code

    def test_mixed_severity_diagnostics_in_passed_check(self) -> None:
        """Only info and warning diagnostics are shown for passed checks."""
        report = Report(
            results=(
                CheckResult(
                    name="code-length",
                    status="passed",
                    duration_ms=50,
                    diagnostics=(
                        Diagnostic(message="Info msg", severity="info"),
                        Diagnostic(message="Warn msg", severity="warning"),
                        Diagnostic(message="Err msg", severity="error"),
                    ),
                ),
            ),
            total_duration_ms=50,
        )
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "Info msg" in output
        assert "Warn msg" in output
        assert "Err msg" not in output

    def test_raw_output_surfaced_when_no_diagnostics(self) -> None:
        """Raw output and reproduce command shown when no structured diagnostics.

        Same guarantee as ConsoleFormatter: root cause is immediately visible,
        structured diagnostics take priority, and empty output is handled safely.
        """
        formatter = QuietFormatter(color=False)

        # Raw output + command shown
        result = CheckResult(
            name="lint",
            status="failed",
            duration_ms=100,
            diagnostics=(),
            output="crash: null\nsecond line",
            command=("uv", "run", "ruff", "."),
        )
        out = formatter.format(Report(results=(result,), total_duration_ms=100))
        assert "crash: null" in out
        assert "Reproduce: uv run ruff ." in out

        # No reproduce when diagnostics present
        result2 = CheckResult(
            name="lint",
            status="failed",
            duration_ms=100,
            diagnostics=(Diagnostic(message="Type error"),),
            command=("uv", "run", "ruff", "."),
        )
        assert "Reproduce:" not in formatter.format(
            Report(results=(result2,), total_duration_ms=100)
        )

        # No reproduce when output empty
        result3 = CheckResult(
            name="lint", status="failed", duration_ms=100, diagnostics=(), output=""
        )
        out3 = formatter.format(Report(results=(result3,), total_duration_ms=100))
        assert "✗ lint" in out3
        assert "Reproduce:" not in out3

    def test_raw_output_truncated_at_30_lines(self, tmp_path: Path) -> None:
        """Raw output is truncated at 30 lines; the full version is saved."""
        long_output = "\n".join(f"line {i}" for i in range(40))
        report = Report(
            results=(
                CheckResult(
                    name="test",
                    status="failed",
                    duration_ms=100,
                    diagnostics=(),
                    output=long_output,
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=False, log_dir=tmp_path)
        output = formatter.format(report)
        assert "line 0" in output
        assert "line 29" in output
        assert "line 30\n" not in output
        assert "10 more output lines" in output
        # Full version saved, path and re-run hint included
        log_file = tmp_path / "test.log"
        assert f"full report: {log_file}" in output
        assert "Re-run just this check: uv run python check.py test" in output
        saved = log_file.read_text(encoding="utf-8")
        assert "line 39" in saved  # beyond the display cap

    def test_multiline_diagnostics_indent_in_quiet_mode(self) -> None:
        report = Report(
            results=(
                CheckResult(
                    name="lint",
                    status="failed",
                    duration_ms=50,
                    diagnostics=(Diagnostic(message="first line\nsecond line"),),
                ),
                CheckResult(
                    name="format",
                    status="passed",
                    duration_ms=50,
                    diagnostics=(
                        Diagnostic(message="info first\ninfo second", severity="info"),
                        Diagnostic(
                            message="warn first\nwarn second", severity="warning"
                        ),
                    ),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "  first line\n    second line" in output
        assert "info first\n  info second" in output
        assert "warn first\n  warn second" in output
