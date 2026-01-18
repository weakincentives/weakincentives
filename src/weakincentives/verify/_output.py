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

"""Unified output formatting for the verification toolbox."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

from weakincentives.verify._types import CheckResult, Finding, Severity

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output configuration.

    Attributes:
        quiet: Suppress success messages.
        color: Use ANSI colors (auto-detected if not specified).
        json_output: Output as JSON for machine consumption.
    """

    quiet: bool = False
    color: bool | None = None
    json_output: bool = False

    @property
    def use_color(self) -> bool:
        """Whether to use color output."""
        if self.color is not None:
            return self.color
        # Auto-detect: use color if stdout is a TTY
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class Output:
    """Unified output handler for verification results."""

    def __init__(
        self,
        config: OutputConfig | None = None,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        """Initialize the output handler.

        Args:
            config: Output configuration. Defaults to default config.
            stdout: Output stream for normal messages. Defaults to sys.stdout.
            stderr: Output stream for errors. Defaults to sys.stderr.
        """
        self._config = config or OutputConfig()
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def _colorize(self, text: str, color_code: str) -> str:
        """Apply ANSI color code if colors are enabled."""
        if not self._config.use_color:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _severity_color(self, severity: Severity) -> str:
        """Get the ANSI color code for a severity level."""
        if severity == Severity.ERROR:
            return "31"  # Red
        if severity == Severity.WARNING:
            return "33"  # Yellow
        return "36"  # Cyan for INFO

    def checker_start(self, checker_name: str) -> None:
        """Report that a checker is starting."""
        if self._config.quiet or self._config.json_output:
            return
        # Don't print anything on start - wait for result

    def checker_success(self, result: CheckResult) -> None:
        """Report checker success."""
        if self._config.json_output:
            return  # JSON output is batched at the end
        if self._config.quiet:
            return
        checkmark = self._colorize("PASS", "32")
        print(f"{checkmark} {result.checker} ({result.duration_ms}ms)", file=self._stdout)

    def checker_failure(self, result: CheckResult) -> None:
        """Report checker failure."""
        if self._config.json_output:
            return  # JSON output is batched at the end

        x_mark = self._colorize("FAIL", "31")
        print(f"{x_mark} {result.checker} ({result.duration_ms}ms)", file=self._stderr)

        for finding in result.findings:
            self.finding(finding, show_checker=False)

    def finding(self, finding: Finding, *, show_checker: bool = True) -> None:
        """Report a single finding."""
        if self._config.json_output:
            return

        severity_str = finding.severity.name.lower()
        colored_severity = self._colorize(
            severity_str, self._severity_color(finding.severity)
        )

        parts: list[str] = ["  "]
        if show_checker:
            parts.append(f"[{finding.checker}] ")

        parts.append(f"{colored_severity}: ")

        if finding.file is not None:
            location = str(finding.file)
            if finding.line is not None:
                location = f"{location}:{finding.line}"
            parts.append(f"{location}: ")

        parts.append(finding.message)

        print("".join(parts), file=self._stderr)

        if finding.suggestion:
            print(f"    suggestion: {finding.suggestion}", file=self._stderr)

    def summary(self, results: Sequence[CheckResult]) -> None:
        """Report final summary of all results."""
        if self._config.json_output:
            self._json_summary(results)
            return

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        total_errors = sum(r.error_count for r in results)
        total_warnings = sum(r.warning_count for r in results)
        total_duration_ms = sum(r.duration_ms for r in results)

        print(file=self._stdout)

        if failed == 0:
            status = self._colorize("All checks passed", "32")
        else:
            status = self._colorize(f"{failed} check(s) failed", "31")

        print(
            f"{status}: {passed}/{total} passed, "
            f"{total_errors} error(s), {total_warnings} warning(s) "
            f"in {total_duration_ms}ms",
            file=self._stdout,
        )

    def _json_summary(self, results: Sequence[CheckResult]) -> None:
        """Output results as JSON."""
        output = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "total_errors": sum(r.error_count for r in results),
            "total_warnings": sum(r.warning_count for r in results),
            "total_duration_ms": sum(r.duration_ms for r in results),
            "results": [
                {
                    "checker": r.checker,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error_count": r.error_count,
                    "warning_count": r.warning_count,
                    "findings": [
                        {
                            "severity": f.severity.name.lower(),
                            "message": f.message,
                            "file": str(f.file) if f.file else None,
                            "line": f.line,
                            "suggestion": f.suggestion,
                        }
                        for f in r.findings
                    ],
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2), file=self._stdout)

    def error(self, message: str) -> None:
        """Report a general error message."""
        if self._config.json_output:
            print(json.dumps({"error": message}), file=self._stderr)
        else:
            colored = self._colorize("error", "31")
            print(f"{colored}: {message}", file=self._stderr)

    def info(self, message: str) -> None:
        """Report an informational message."""
        if self._config.quiet or self._config.json_output:
            return
        print(message, file=self._stdout)
