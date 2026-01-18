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

"""Output formatting for verification results.

Provides formatters for rendering Reports to different outputs:
- ConsoleFormatter: Pretty terminal output with colors
- JSONFormatter: Machine-readable JSON output
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import IO, Protocol

from .result import CheckResult, Report


class Formatter(Protocol):
    """Protocol for result formatters."""

    def format(self, report: Report) -> str:
        """Format a report as a string."""
        ...


def _supports_color(stream: IO[str]) -> bool:
    """Check if the stream supports ANSI colors."""
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    # Check for NO_COLOR environment variable
    import os

    if os.environ.get("NO_COLOR"):
        return False
    return True


@dataclass
class ConsoleFormatter:
    """Formats reports for terminal output.

    Features:
    - Colored output when supported
    - Concise pass/fail indicators
    - Detailed diagnostics for failures
    - Summary line with counts and timing
    """

    verbose: bool = False
    color: bool | None = None  # None = auto-detect
    stream: IO[str] | None = None
    max_diagnostics: int = 10  # Max diagnostics to show per checker

    def _use_color(self) -> bool:
        if self.color is not None:
            return self.color
        stream = self.stream or sys.stdout
        return _supports_color(stream)

    def _format_duration(self, ms: int) -> str:
        """Format duration in human-readable form."""
        if ms < 1000:
            return f"{ms}ms"
        seconds = ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"

    def _format_result(self, result: CheckResult) -> list[str]:
        """Format a single check result."""
        lines = []
        use_color = self._use_color()

        # Status indicator and name
        duration = self._format_duration(result.duration_ms)
        if result.passed:
            mark = "\033[32m✓\033[0m" if use_color else "✓"
            lines.append(f"{mark} {result.name:<20} ({duration})")
        elif result.status == "skipped":
            mark = "\033[33m○\033[0m" if use_color else "○"
            lines.append(f"{mark} {result.name:<20} (skipped)")
        else:
            mark = "\033[31m✗\033[0m" if use_color else "✗"
            lines.append(f"{mark} {result.name:<20} ({duration})")

            # Show diagnostics for failures
            if result.diagnostics:
                shown = result.diagnostics[: self.max_diagnostics]
                for diag in shown:
                    prefix = "  "
                    if use_color:
                        lines.append(f"{prefix}\033[90m{diag}\033[0m")
                    else:
                        lines.append(f"{prefix}{diag}")

                remaining = len(result.diagnostics) - len(shown)
                if remaining > 0:
                    lines.append(f"  ... and {remaining} more")
            elif self.verbose and result.output:  # pragma: no cover
                # Show raw output if no structured diagnostics
                for line in result.output.split("\n")[:20]:
                    lines.append(f"  {line}")

        return lines

    def format(self, report: Report) -> str:
        """Format the complete report."""
        lines = []
        use_color = self._use_color()

        # Individual results
        for result in report.results:
            lines.extend(self._format_result(result))

        # Summary line
        lines.append("")
        total = self._format_duration(report.total_duration_ms)

        if report.passed:
            if use_color:
                summary = f"\033[32m✓ All checks passed\033[0m"
            else:
                summary = "✓ All checks passed"
        else:
            failed = report.failed_count
            passed = report.passed_count
            if use_color:
                summary = f"\033[31m✗ {failed} failed\033[0m, {passed} passed"
            else:
                summary = f"✗ {failed} failed, {passed} passed"

        lines.append(f"{summary} ({total})")

        return "\n".join(lines)


@dataclass
class JSONFormatter:
    """Formats reports as JSON for machine consumption."""

    indent: int | None = 2

    def format(self, report: Report) -> str:
        """Format the report as JSON."""
        data = {
            "passed": report.passed,
            "summary": {
                "passed": report.passed_count,
                "failed": report.failed_count,
                "skipped": report.skipped_count,
                "duration_ms": report.total_duration_ms,
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "diagnostics": [asdict(d) for d in r.diagnostics],
                }
                for r in report.results
            ],
        }
        return json.dumps(data, indent=self.indent)


@dataclass
class QuietFormatter:
    """Minimal output - only shows failures."""

    color: bool | None = None
    stream: IO[str] | None = None

    def _use_color(self) -> bool:
        if self.color is not None:
            return self.color
        stream = self.stream or sys.stdout
        return _supports_color(stream)

    def format(self, report: Report) -> str:
        """Format only failures, or a brief success message."""
        if report.passed:
            return ""

        lines = []
        use_color = self._use_color()

        for result in report.failed_results:
            if use_color:
                lines.append(f"\033[31m✗ {result.name}\033[0m")
            else:
                lines.append(f"✗ {result.name}")

            for diag in result.diagnostics[:10]:
                lines.append(f"  {diag}")

            if len(result.diagnostics) > 10:
                lines.append(f"  ... and {len(result.diagnostics) - 10} more")

        return "\n".join(lines)
