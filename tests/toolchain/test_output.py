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

"""Tests for toolchain output formatters."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

from toolchain.output import ConsoleFormatter, JSONFormatter, QuietFormatter, _supports_color

if TYPE_CHECKING:
    pass
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


def _make_skipped_report() -> Report:
    return Report(
        results=(
            CheckResult(name="lint", status="passed", duration_ms=1200),
            CheckResult(name="test", status="skipped", duration_ms=0),
        ),
        total_duration_ms=1200,
    )


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_passing_report_no_color(self) -> None:
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(_make_passing_report())
        assert "lint" in output
        assert "test" in output
        assert "All checks passed" in output
        assert "46.2s" in output

    def test_passing_report_with_color(self) -> None:
        formatter = ConsoleFormatter(color=True)
        output = formatter.format(_make_passing_report())
        assert "\033[32m" in output  # Green color code

    def test_failing_report_no_color(self) -> None:
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(_make_failing_report())
        assert "lint" in output
        assert "test" in output
        assert "1 failed" in output
        assert "1 passed" in output
        assert "tests/test_foo.py:42" in output

    def test_failing_report_with_color(self) -> None:
        formatter = ConsoleFormatter(color=True)
        output = formatter.format(_make_failing_report())
        assert "\033[31m" in output  # Red color code

    def test_skipped_report_no_color(self) -> None:
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(_make_skipped_report())
        assert "skipped" in output
        assert "All checks passed" in output  # Skipped counts as passed

    def test_skipped_report_with_color(self) -> None:
        formatter = ConsoleFormatter(color=True)
        output = formatter.format(_make_skipped_report())
        assert "\033[33m" in output  # Yellow color code for skipped

    def test_verbose_shows_output(self) -> None:
        report = Report(
            results=(
                CheckResult(
                    name="test",
                    status="failed",
                    duration_ms=100,
                    diagnostics=(),
                    output="Some raw\noutput here",
                ),
            ),
            total_duration_ms=100,
        )
        formatter = ConsoleFormatter(verbose=True, color=False)
        output = formatter.format(report)
        assert "Some raw" in output
        assert "output here" in output

    def test_max_diagnostics_truncation(self) -> None:
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
        formatter = ConsoleFormatter(color=False, max_diagnostics=10)
        output = formatter.format(report)
        assert "... and 5 more" in output
        assert "Run: python check.py lint -v" in output

    def test_checker_hint_shown_on_failure(self) -> None:
        report = Report(
            results=(
                CheckResult(
                    name="typecheck",
                    status="failed",
                    duration_ms=5200,
                    diagnostics=(Diagnostic(message="Type error"),),
                ),
            ),
            total_duration_ms=5200,
        )
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(report)
        # Should show checker hint in parentheses
        assert "typecheck (ty + pyright)" in output

    def test_checker_hint_not_shown_for_unknown_checker(self) -> None:
        report = Report(
            results=(
                CheckResult(
                    name="unknown_checker",
                    status="failed",
                    duration_ms=100,
                    diagnostics=(Diagnostic(message="Error"),),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(report)
        # Should show just the name without extra parentheses
        assert "unknown_checker" in output
        assert "unknown_checker ()" not in output

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
        formatter = ConsoleFormatter(color=False, max_diagnostics=10)
        output = formatter.format(report)
        assert "Run: python check.py" not in output

    def test_duration_formatting_milliseconds(self) -> None:
        formatter = ConsoleFormatter(color=False)
        assert formatter._format_duration(500) == "500ms"

    def test_duration_formatting_seconds(self) -> None:
        formatter = ConsoleFormatter(color=False)
        assert formatter._format_duration(2500) == "2.5s"

    def test_duration_formatting_minutes(self) -> None:
        formatter = ConsoleFormatter(color=False)
        assert formatter._format_duration(90000) == "1m30s"

    def test_auto_color_detection_tty(self) -> None:
        # Non-tty stream should not use color
        stream = io.StringIO()
        formatter = ConsoleFormatter(stream=stream)
        assert formatter._use_color() is False

    def test_explicit_color_overrides_detection(self) -> None:
        stream = io.StringIO()
        formatter = ConsoleFormatter(color=True, stream=stream)
        assert formatter._use_color() is True

    def test_info_diagnostics_shown_for_passed_checks(self) -> None:
        """Info diagnostics (like auto-format messages) should be shown even for passed checks."""
        report = Report(
            results=(
                CheckResult(
                    name="format",
                    status="passed",
                    duration_ms=100,
                    diagnostics=(
                        Diagnostic(
                            message="Automatically reformatted 2 files: foo.py, bar.py",
                            severity="info",
                        ),
                    ),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = ConsoleFormatter(color=False)
        output = formatter.format(report)
        assert "Automatically reformatted" in output
        assert "foo.py" in output

    def test_info_diagnostics_shown_with_color(self) -> None:
        """Info diagnostics should use cyan color."""
        report = Report(
            results=(
                CheckResult(
                    name="format",
                    status="passed",
                    duration_ms=100,
                    diagnostics=(
                        Diagnostic(message="Auto-fixed", severity="info"),
                    ),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = ConsoleFormatter(color=True)
        output = formatter.format(report)
        assert "\033[36m" in output  # Cyan color code


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_passing_report_structure(self) -> None:
        formatter = JSONFormatter()
        output = formatter.format(_make_passing_report())
        data = json.loads(output)

        assert data["passed"] is True
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0
        assert data["summary"]["skipped"] == 0
        assert data["summary"]["duration_ms"] == 46200
        assert len(data["results"]) == 2

    def test_failing_report_structure(self) -> None:
        formatter = JSONFormatter()
        output = formatter.format(_make_failing_report())
        data = json.loads(output)

        assert data["passed"] is False
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1

        # Check diagnostics are included
        test_result = next(r for r in data["results"] if r["name"] == "test")
        assert len(test_result["diagnostics"]) == 2
        assert test_result["diagnostics"][0]["location"]["file"] == "tests/test_foo.py"
        assert test_result["diagnostics"][0]["location"]["line"] == 42

    def test_indent_option(self) -> None:
        formatter = JSONFormatter(indent=None)
        output = formatter.format(_make_passing_report())
        # No indent means single line
        assert "\n" not in output.strip()

    def test_valid_json(self) -> None:
        formatter = JSONFormatter()
        output = formatter.format(_make_failing_report())
        # Should not raise
        json.loads(output)


class TestSupportsColor:
    """Tests for _supports_color helper."""

    def test_stream_without_isatty(self) -> None:
        # Object without isatty method
        class NoIsatty:
            pass

        assert _supports_color(NoIsatty()) is False  # type: ignore[arg-type]

    def test_no_color_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock a TTY stream
        class MockTTY:
            def isatty(self) -> bool:
                return True

        monkeypatch.setenv("NO_COLOR", "1")
        assert _supports_color(MockTTY()) is False  # type: ignore[arg-type]

    def test_tty_without_no_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class MockTTY:
            def isatty(self) -> bool:
                return True

        monkeypatch.delenv("NO_COLOR", raising=False)
        assert _supports_color(MockTTY()) is True  # type: ignore[arg-type]


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

    def test_truncates_diagnostics(self) -> None:
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
        formatter = QuietFormatter(color=False)
        output = formatter.format(report)
        assert "... and 5 more" in output
        assert "Run: python check.py lint -v" in output

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
                    diagnostics=(
                        Diagnostic(message="Auto-fixed", severity="info"),
                    ),
                ),
            ),
            total_duration_ms=100,
        )
        formatter = QuietFormatter(color=True)
        output = formatter.format(report)
        assert "\033[36m" in output  # Cyan color code
