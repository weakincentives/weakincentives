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

"""Tests for toolchain result types."""

from __future__ import annotations

import pytest

from toolchain.result import CheckResult, Diagnostic, Location, Report


class TestLocation:
    """Tests for Location dataclass."""

    def test_str_file_only(self) -> None:
        loc = Location(file="src/foo.py")
        assert str(loc) == "src/foo.py"

    def test_str_file_and_line(self) -> None:
        loc = Location(file="src/foo.py", line=42)
        assert str(loc) == "src/foo.py:42"

    def test_str_file_line_column(self) -> None:
        loc = Location(file="src/foo.py", line=42, column=10)
        assert str(loc) == "src/foo.py:42:10"

    def test_str_with_end_line_no_columns(self) -> None:
        loc = Location(file="src/foo.py", line=10, end_line=12)
        assert str(loc) == "src/foo.py:10-12"

    def test_str_with_end_line_and_columns(self) -> None:
        loc = Location(file="src/foo.py", line=10, column=5, end_line=12, end_column=3)
        assert str(loc) == "src/foo.py:10:5-12:3"

    def test_str_end_line_without_end_column_ignored(self) -> None:
        # If we have column but only end_line (no end_column), end position is ignored
        loc = Location(file="src/foo.py", line=10, column=5, end_line=12)
        assert str(loc) == "src/foo.py:10:5"

    def test_str_end_column_without_end_line_ignored(self) -> None:
        # If we have column and end_column but no end_line, end position is ignored
        loc = Location(file="src/foo.py", line=10, column=5, end_column=3)
        assert str(loc) == "src/foo.py:10:5"

    def test_frozen(self) -> None:
        loc = Location(file="src/foo.py", line=42, column=10)
        with pytest.raises(AttributeError):
            loc.file = "other.py"  # type: ignore[misc]


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_str_message_only(self) -> None:
        diag = Diagnostic(message="Something went wrong")
        assert str(diag) == "Something went wrong"

    def test_str_with_location(self) -> None:
        diag = Diagnostic(
            message="Type error",
            location=Location(file="src/foo.py", line=42),
        )
        assert str(diag) == "src/foo.py:42: Type error"

    def test_default_severity(self) -> None:
        diag = Diagnostic(message="Error")
        assert diag.severity == "error"

    def test_warning_severity(self) -> None:
        diag = Diagnostic(message="Warning", severity="warning")
        assert diag.severity == "warning"

    def test_info_severity(self) -> None:
        diag = Diagnostic(message="Info", severity="info")
        assert diag.severity == "info"

    def test_frozen(self) -> None:
        diag = Diagnostic(message="Error")
        with pytest.raises(AttributeError):
            diag.message = "Other"  # type: ignore[misc]


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_passed(self) -> None:
        result = CheckResult(name="lint", status="passed", duration_ms=100)
        assert result.passed is True
        assert result.failed is False

    def test_failed(self) -> None:
        result = CheckResult(name="lint", status="failed", duration_ms=100)
        assert result.passed is False
        assert result.failed is True

    def test_skipped(self) -> None:
        result = CheckResult(name="lint", status="skipped", duration_ms=0)
        assert result.passed is False
        assert result.failed is False

    def test_error_count(self) -> None:
        result = CheckResult(
            name="lint",
            status="failed",
            duration_ms=100,
            diagnostics=(
                Diagnostic(message="Error 1", severity="error"),
                Diagnostic(message="Warning 1", severity="warning"),
                Diagnostic(message="Error 2", severity="error"),
            ),
        )
        assert result.error_count == 2

    def test_warning_count(self) -> None:
        result = CheckResult(
            name="lint",
            status="failed",
            duration_ms=100,
            diagnostics=(
                Diagnostic(message="Error 1", severity="error"),
                Diagnostic(message="Warning 1", severity="warning"),
                Diagnostic(message="Warning 2", severity="warning"),
            ),
        )
        assert result.warning_count == 2

    def test_default_diagnostics_empty(self) -> None:
        result = CheckResult(name="lint", status="passed", duration_ms=100)
        assert result.diagnostics == ()

    def test_default_output_empty(self) -> None:
        result = CheckResult(name="lint", status="passed", duration_ms=100)
        assert result.output == ""

    def test_frozen(self) -> None:
        result = CheckResult(name="lint", status="passed", duration_ms=100)
        with pytest.raises(AttributeError):
            result.name = "other"  # type: ignore[misc]


class TestReport:
    """Tests for Report dataclass."""

    def test_passed_all_pass(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="passed", duration_ms=100),
                CheckResult(name="test", status="passed", duration_ms=200),
            ),
            total_duration_ms=300,
        )
        assert report.passed is True

    def test_passed_with_skipped(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="passed", duration_ms=100),
                CheckResult(name="test", status="skipped", duration_ms=0),
            ),
            total_duration_ms=100,
        )
        assert report.passed is True

    def test_failed_with_one_failure(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="passed", duration_ms=100),
                CheckResult(name="test", status="failed", duration_ms=200),
            ),
            total_duration_ms=300,
        )
        assert report.passed is False

    def test_failed_results(self) -> None:
        lint = CheckResult(name="lint", status="passed", duration_ms=100)
        test = CheckResult(name="test", status="failed", duration_ms=200)
        typecheck = CheckResult(name="typecheck", status="failed", duration_ms=300)
        report = Report(results=(lint, test, typecheck), total_duration_ms=600)
        assert report.failed_results == (test, typecheck)

    def test_passed_count(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="passed", duration_ms=100),
                CheckResult(name="test", status="failed", duration_ms=200),
                CheckResult(name="typecheck", status="passed", duration_ms=300),
            ),
            total_duration_ms=600,
        )
        assert report.passed_count == 2

    def test_failed_count(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="passed", duration_ms=100),
                CheckResult(name="test", status="failed", duration_ms=200),
                CheckResult(name="typecheck", status="failed", duration_ms=300),
            ),
            total_duration_ms=600,
        )
        assert report.failed_count == 2

    def test_skipped_count(self) -> None:
        report = Report(
            results=(
                CheckResult(name="lint", status="skipped", duration_ms=0),
                CheckResult(name="test", status="skipped", duration_ms=0),
                CheckResult(name="typecheck", status="passed", duration_ms=300),
            ),
            total_duration_ms=300,
        )
        assert report.skipped_count == 2

    def test_frozen(self) -> None:
        report = Report(results=(), total_duration_ms=0)
        with pytest.raises(AttributeError):
            report.total_duration_ms = 100  # type: ignore[misc]
