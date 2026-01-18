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

"""Test execution verification checkers.

These checkers run tests:
- Pytest: Unit test execution with coverage
"""

from __future__ import annotations

import re
import time

from weakincentives.verify._subprocess import run_python_module
from weakincentives.verify._types import CheckContext, CheckResult, Finding, Severity


class PytestChecker:
    """Checker for running pytest tests.

    Runs pytest with coverage enforcement.
    """

    def __init__(
        self,
        *,
        coverage_threshold: int = 100,
        max_failures: int = 1,
    ) -> None:
        """Initialize the checker.

        Args:
            coverage_threshold: Minimum required coverage percentage.
            max_failures: Stop after this many test failures.
        """
        self._coverage_threshold = coverage_threshold
        self._max_failures = max_failures

    @property
    def name(self) -> str:
        return "pytest"

    @property
    def category(self) -> str:
        return "tests"

    @property
    def description(self) -> str:
        return "Run tests with coverage enforcement"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        tests_dir = ctx.project_root / "tests"
        if not tests_dir.exists():
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message="Tests directory not found",
                    ),
                ),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        args = [
            "--strict-config",
            "--strict-markers",
            f"--maxfail={self._max_failures}",
            f"--cov-fail-under={self._coverage_threshold}",
            "-q",
            "--no-header",
            "--cov-report=",
            str(tests_dir),
        ]

        result = run_python_module(
            "pytest",
            args,
            cwd=ctx.project_root,
            timeout_seconds=300.0,  # Tests can take a while
        )

        if not result.success:
            output = result.output.strip()

            # Extract test failures
            failure_pattern = re.compile(r"^FAILED\s+(.+)$", re.MULTILINE)
            for match in failure_pattern.finditer(output):
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=f"Test failed: {match.group(1)}",
                    )
                )

            # Check for coverage failures
            if "FAIL Required test coverage" in output:
                coverage_match = re.search(
                    r"coverage (?:of )?(\d+(?:\.\d+)?%)", output
                )
                if coverage_match:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=(
                                f"Coverage {coverage_match.group(1)} is below "
                                f"required {self._coverage_threshold}%"
                            ),
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=(
                                f"Coverage is below required {self._coverage_threshold}%"
                            ),
                        )
                    )

            # Check for collection errors
            if "error" in output.lower() and "collected" not in output.lower():
                error_lines = [
                    line
                    for line in output.splitlines()
                    if "error" in line.lower() or "Error" in line
                ]
                for line in error_lines[:5]:  # Limit to first 5
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=line.strip(),
                        )
                    )

            # If we couldn't parse specific failures, add a generic one
            if not findings:
                # Try to extract the summary line
                summary = self._extract_summary(output)
                if summary:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=f"Tests failed: {summary}",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message="Tests failed (see output for details)",
                        )
                    )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

    def _extract_summary(self, output: str) -> str | None:
        """Extract the pytest summary line from output."""
        lines = output.splitlines()
        for line in reversed(lines):
            stripped = line.strip()
            lower = stripped.lower()
            if " passed" in lower or " failed" in lower or " deselected" in lower:
                return stripped
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.lower().startswith("collected "):
                return stripped
        return None
