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

"""Tests for output formatting."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from weakincentives.verify._output import Output, OutputConfig
from weakincentives.verify._types import CheckResult, Finding, Severity


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_config(self) -> None:
        """Default config values."""
        config = OutputConfig()
        assert config.quiet is False
        assert config.color is None
        assert config.json_output is False

    def test_use_color_explicit(self) -> None:
        """Explicit color setting."""
        assert OutputConfig(color=True).use_color is True
        assert OutputConfig(color=False).use_color is False


class TestOutput:
    """Tests for Output class."""

    def test_checker_success_quiet(self) -> None:
        """Success output is suppressed in quiet mode."""
        stdout = StringIO()
        output = Output(OutputConfig(quiet=True), stdout=stdout)
        result = CheckResult(checker="test", findings=(), duration_ms=100)

        output.checker_success(result)

        assert stdout.getvalue() == ""

    def test_checker_success_normal(self) -> None:
        """Success output shows in normal mode."""
        stdout = StringIO()
        output = Output(OutputConfig(quiet=False, color=False), stdout=stdout)
        result = CheckResult(checker="test.checker", findings=(), duration_ms=50)

        output.checker_success(result)

        out = stdout.getvalue()
        assert "PASS" in out
        assert "test.checker" in out
        assert "50ms" in out

    def test_checker_failure(self) -> None:
        """Failure output shows findings."""
        stderr = StringIO()
        output = Output(OutputConfig(color=False), stderr=stderr)
        result = CheckResult(
            checker="test.checker",
            findings=(
                Finding(
                    checker="test.checker",
                    severity=Severity.ERROR,
                    message="Something went wrong",
                ),
            ),
            duration_ms=100,
        )

        output.checker_failure(result)

        out = stderr.getvalue()
        assert "FAIL" in out
        assert "test.checker" in out
        assert "Something went wrong" in out

    def test_finding_with_location(self, tmp_path: Path) -> None:
        """Finding output includes file location."""
        stderr = StringIO()
        output = Output(OutputConfig(color=False), stderr=stderr)
        test_file = tmp_path / "test.py"
        finding = Finding(
            checker="test.checker",
            severity=Severity.WARNING,
            message="Warning message",
            file=test_file,
            line=42,
        )

        output.finding(finding)

        out = stderr.getvalue()
        assert "warning" in out
        assert "42" in out
        assert "Warning message" in out

    def test_summary_all_passed(self) -> None:
        """Summary for all passed checks."""
        stdout = StringIO()
        output = Output(OutputConfig(color=False), stdout=stdout)
        results = [
            CheckResult(checker="check1", findings=(), duration_ms=50),
            CheckResult(checker="check2", findings=(), duration_ms=30),
        ]

        output.summary(results)

        out = stdout.getvalue()
        assert "All checks passed" in out
        assert "2/2" in out

    def test_summary_some_failed(self) -> None:
        """Summary for some failed checks."""
        stdout = StringIO()
        output = Output(OutputConfig(color=False), stdout=stdout)
        results = [
            CheckResult(checker="check1", findings=(), duration_ms=50),
            CheckResult(
                checker="check2",
                findings=(
                    Finding(
                        checker="check2",
                        severity=Severity.ERROR,
                        message="Error",
                    ),
                ),
                duration_ms=30,
            ),
        ]

        output.summary(results)

        out = stdout.getvalue()
        assert "failed" in out
        assert "1/2" in out

    def test_json_output(self) -> None:
        """JSON output mode."""
        stdout = StringIO()
        output = Output(OutputConfig(json_output=True), stdout=stdout)
        results = [
            CheckResult(
                checker="test.checker",
                findings=(
                    Finding(
                        checker="test.checker",
                        severity=Severity.ERROR,
                        message="Error found",
                    ),
                ),
                duration_ms=100,
            ),
        ]

        output.summary(results)

        out = stdout.getvalue()
        data = json.loads(out)
        assert data["total"] == 1
        assert data["failed"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["checker"] == "test.checker"

    def test_error_message(self) -> None:
        """Error message output."""
        stderr = StringIO()
        output = Output(OutputConfig(color=False), stderr=stderr)

        output.error("Something bad happened")

        assert "error" in stderr.getvalue()
        assert "Something bad happened" in stderr.getvalue()

    def test_info_message_quiet(self) -> None:
        """Info suppressed in quiet mode."""
        stdout = StringIO()
        output = Output(OutputConfig(quiet=True), stdout=stdout)

        output.info("Some info")

        assert stdout.getvalue() == ""

    def test_info_message_normal(self) -> None:
        """Info shown in normal mode."""
        stdout = StringIO()
        output = Output(OutputConfig(quiet=False), stdout=stdout)

        output.info("Some info")

        assert "Some info" in stdout.getvalue()
