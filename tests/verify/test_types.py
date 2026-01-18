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

"""Tests for verify types."""

from __future__ import annotations

from pathlib import Path

from weakincentives.verify._types import (
    CheckContext,
    CheckResult,
    Finding,
    RunConfig,
    Severity,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self) -> None:
        """Severity enum has expected values."""
        assert Severity.ERROR.name == "ERROR"
        assert Severity.WARNING.name == "WARNING"
        assert Severity.INFO.name == "INFO"


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_minimal(self) -> None:
        """Finding with required fields only."""
        finding = Finding(
            checker="test.checker",
            severity=Severity.ERROR,
            message="Something went wrong",
        )
        assert finding.checker == "test.checker"
        assert finding.severity == Severity.ERROR
        assert finding.message == "Something went wrong"
        assert finding.file is None
        assert finding.line is None
        assert finding.suggestion is None

    def test_finding_full(self, tmp_path: Path) -> None:
        """Finding with all fields."""
        test_file = tmp_path / "test.py"
        finding = Finding(
            checker="test.checker",
            severity=Severity.WARNING,
            message="Consider refactoring",
            file=test_file,
            line=42,
            suggestion="Use a context manager",
        )
        assert finding.file == test_file
        assert finding.line == 42
        assert finding.suggestion == "Use a context manager"

    def test_finding_format_minimal(self) -> None:
        """Format a minimal finding."""
        finding = Finding(
            checker="test.checker",
            severity=Severity.ERROR,
            message="Bad code",
        )
        formatted = finding.format()
        assert "[test.checker]" in formatted
        assert "error:" in formatted
        assert "Bad code" in formatted

    def test_finding_format_with_location(self, tmp_path: Path) -> None:
        """Format a finding with file location."""
        test_file = tmp_path / "test.py"
        finding = Finding(
            checker="arch.layers",
            severity=Severity.WARNING,
            message="Layer violation",
            file=test_file,
            line=10,
        )
        formatted = finding.format()
        assert "test.py:10" in formatted or str(test_file) in formatted

    def test_finding_format_with_suggestion(self) -> None:
        """Format a finding with suggestion."""
        finding = Finding(
            checker="test.checker",
            severity=Severity.ERROR,
            message="Issue found",
            suggestion="Try this instead",
        )
        formatted = finding.format()
        assert "suggestion:" in formatted
        assert "Try this instead" in formatted

    def test_finding_format_hide_checker(self) -> None:
        """Format without showing checker name."""
        finding = Finding(
            checker="test.checker",
            severity=Severity.ERROR,
            message="Error message",
        )
        formatted = finding.format(show_checker=False)
        assert "[test.checker]" not in formatted
        assert "Error message" in formatted


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_result_passed_no_findings(self) -> None:
        """Result with no findings is passed."""
        result = CheckResult(
            checker="test.checker",
            findings=(),
            duration_ms=100,
        )
        assert result.passed is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_result_passed_only_warnings(self) -> None:
        """Result with only warnings is passed."""
        result = CheckResult(
            checker="test.checker",
            findings=(
                Finding(
                    checker="test.checker",
                    severity=Severity.WARNING,
                    message="Warning",
                ),
            ),
            duration_ms=50,
        )
        assert result.passed is True
        assert result.error_count == 0
        assert result.warning_count == 1

    def test_result_failed_with_errors(self) -> None:
        """Result with errors is failed."""
        result = CheckResult(
            checker="test.checker",
            findings=(
                Finding(
                    checker="test.checker",
                    severity=Severity.ERROR,
                    message="Error",
                ),
                Finding(
                    checker="test.checker",
                    severity=Severity.WARNING,
                    message="Warning",
                ),
            ),
            duration_ms=200,
        )
        assert result.passed is False
        assert result.error_count == 1
        assert result.warning_count == 1


class TestCheckContext:
    """Tests for CheckContext dataclass."""

    def test_from_project_root(self, tmp_path: Path) -> None:
        """Create context from project root."""
        ctx = CheckContext.from_project_root(tmp_path)
        assert ctx.project_root == tmp_path.resolve()
        assert ctx.src_dir == (tmp_path / "src").resolve()
        assert ctx.quiet is False
        assert ctx.fix is False

    def test_from_project_root_with_options(self, tmp_path: Path) -> None:
        """Create context with options."""
        ctx = CheckContext.from_project_root(tmp_path, quiet=True, fix=True)
        assert ctx.quiet is True
        assert ctx.fix is True


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config allows all checkers."""
        config = RunConfig()
        assert config.max_failures is None
        assert config.max_parallel is None
        assert config.categories is None
        assert config.checkers is None

    def test_should_run_no_filter(self) -> None:
        """No filters means all checkers run."""

        class FakeChecker:
            name = "test"
            category = "testing"

        config = RunConfig()
        assert config.should_run(FakeChecker()) is True

    def test_should_run_category_filter(self) -> None:
        """Category filter restricts checkers."""

        class ArchChecker:
            name = "layers"
            category = "architecture"

        class DocChecker:
            name = "links"
            category = "documentation"

        config = RunConfig(categories=frozenset({"architecture"}))
        assert config.should_run(ArchChecker()) is True
        assert config.should_run(DocChecker()) is False

    def test_should_run_checker_filter(self) -> None:
        """Checker name filter restricts checkers."""

        class Checker1:
            name = "checker1"
            category = "test"

        class Checker2:
            name = "checker2"
            category = "test"

        config = RunConfig(checkers=frozenset({"checker1"}))
        assert config.should_run(Checker1()) is True
        assert config.should_run(Checker2()) is False
