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
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, ClassVar, Protocol

from .result import CheckResult, Report


class Formatter(Protocol):
    """Protocol for result formatters."""

    def format(self, report: Report) -> str:
        """Format a report as a string."""
        ...


# Full untruncated reports for truncated results land here, one file per
# checker, cleared at the start of every run.
FULL_REPORT_DIR = Path(".check-logs")


def clear_full_reports(log_dir: Path = FULL_REPORT_DIR) -> None:
    """Remove full-report logs from previous runs.

    Called at the start of each run so a stale report never describes an
    earlier failure.
    """
    if not log_dir.is_dir():
        return
    for log_file in log_dir.glob("*.log"):
        log_file.unlink(missing_ok=True)


def _rerun_hint(name: str) -> str:
    """Command to re-run just one checker for quick feedback."""
    return f"Re-run just this check: uv run python check.py {name}"


def _write_full_report(result: CheckResult, log_dir: Path) -> str | None:
    """Write the complete, untruncated result to ``<log_dir>/<name>.log``.

    Returns the path for inclusion in truncation hints, or None when the
    file cannot be written (the truncated display output stays the only
    record).
    """
    lines = [f"check: {result.name}", f"status: {result.status}"]
    if result.command:
        lines.append(f"reproduce: {shlex.join(result.command)}")
    lines.append(_rerun_hint(result.name))
    lines.append("")
    lines.append(f"diagnostics ({len(result.diagnostics)}):")
    lines.extend(str(diag) for diag in result.diagnostics)
    if result.output:
        lines.extend(["", "raw output:", result.output])
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{result.name}.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return None
    return str(path)


def _truncation_note(result: CheckResult, log_dir: Path) -> str:
    """Suffix pointing at the full report, or empty when it can't be saved."""
    path = _write_full_report(result, log_dir)
    return f" (full report: {path})" if path else ""


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


def _indent_block(text: str, first: str, rest: str) -> str:
    """Indent a possibly multi-line string.

    The first line gets the *first* prefix; continuation lines get *rest* so
    they stay visually attached to their diagnostic and never collide with
    the `file:line: message` grammar of the following entry.
    """
    lines = text.split("\n")
    indented = [f"{first}{lines[0]}"]
    indented.extend(f"{rest}{line}" for line in lines[1:])
    return "\n".join(indented)


@dataclass
class ConsoleFormatter:
    """Formats reports for terminal output.

    Features:
    - Colored output when supported
    - Concise pass/fail indicators
    - Detailed diagnostics for failures
    - Summary line with counts and timing
    """

    # Mapping of checker names to short descriptions for display
    CHECKER_HINTS: ClassVar[dict[str, str]] = {
        "format": "ruff format",
        "lint": "ruff",
        "typecheck": "ty + pyright",
        "test": "pytest",
        "biome": "biome",
        "bun-test": "bun test",
        "bandit": "bandit",
        "deptry": "deptry",
        "pip-audit": "pip-audit",
        "markdown": "mdformat",
        "architecture": "core/contrib",
        "dead-code": "vulture",
        "docs": "examples, links",
    }

    verbose: bool = False
    color: bool | None = None  # None = auto-detect
    stream: IO[str] | None = None
    max_diagnostics: int = 100  # Max diagnostics to show per checker
    log_dir: Path = FULL_REPORT_DIR  # Where full reports land on truncation

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

            # Show info diagnostics (e.g., auto-format messages)
            info_diags = [d for d in result.diagnostics if d.severity == "info"]
            for diag in info_diags:
                text = _indent_block(diag.message, "  ", "    ")
                if use_color:
                    lines.append(f"\033[36m{text}\033[0m")
                else:
                    lines.append(text)

            # Show warning diagnostics (e.g., code length warnings)
            warn_diags = [d for d in result.diagnostics if d.severity == "warning"]
            if warn_diags:
                shown = warn_diags[: self.max_diagnostics]
                for diag in shown:
                    text = _indent_block(str(diag), "  ", "    ")
                    if use_color:
                        lines.append(f"\033[33m{text}\033[0m")
                    else:
                        lines.append(text)
                remaining = len(warn_diags) - len(shown)
                if remaining > 0:
                    note = _truncation_note(result, self.log_dir)
                    lines.append(f"  ... and {remaining} more warnings{note}")
        elif result.status == "skipped":
            mark = "\033[33m○\033[0m" if use_color else "○"
            lines.append(f"{mark} {result.name:<20} (skipped)")
        else:
            mark = "\033[31m✗\033[0m" if use_color else "✗"
            hint = self.CHECKER_HINTS.get(result.name, "")
            hint_text = f" ({hint})" if hint else ""
            name_with_hint = f"{result.name}{hint_text}"
            lines.append(f"{mark} {name_with_hint:<32} ({duration})")

            # Show diagnostics for failures
            if result.diagnostics:
                shown = result.diagnostics[: self.max_diagnostics]
                for diag in shown:
                    text = _indent_block(str(diag), "  ", "    ")
                    if use_color:
                        lines.append(f"\033[90m{text}\033[0m")
                    else:
                        lines.append(text)

                remaining = len(result.diagnostics) - len(shown)
                if remaining > 0:
                    note = _truncation_note(result, self.log_dir)
                    lines.append(f"  ... and {remaining} more{note}")
                    lines.append(f"  {_rerun_hint(result.name)}")
            elif result.output:
                # No structured diagnostics: always show raw output so the
                # root cause is immediately visible without re-running with -v.
                output_lines = result.output.split("\n")
                for line in output_lines[:50]:
                    lines.append(f"  {line}")
                if len(output_lines) > 50:
                    note = _truncation_note(result, self.log_dir)
                    lines.append(
                        f"  ... and {len(output_lines) - 50} more output lines{note}"
                    )
                    lines.append(f"  {_rerun_hint(result.name)}")
                if result.command:
                    cmd_str = shlex.join(result.command)
                    lines.append(f"  Reproduce: {cmd_str}")

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
    log_dir: Path = FULL_REPORT_DIR  # Where full reports land on truncation

    def _use_color(self) -> bool:
        if self.color is not None:
            return self.color
        stream = self.stream or sys.stdout
        return _supports_color(stream)

    def format(self, report: Report) -> str:
        """Format only failures, or info messages for auto-fixes."""
        lines = []
        use_color = self._use_color()

        # Show info and warning diagnostics for passed checks
        for result in report.results:
            if result.passed:
                for diag in result.diagnostics:
                    if diag.severity == "info":
                        text = _indent_block(diag.message, "", "  ")
                        if use_color:
                            lines.append(f"\033[36m{text}\033[0m")
                        else:
                            lines.append(text)
                    elif diag.severity == "warning":
                        text = _indent_block(str(diag), "", "  ")
                        if use_color:
                            lines.append(f"\033[33m{text}\033[0m")
                        else:
                            lines.append(text)

        if report.passed:
            return "\n".join(lines) if lines else ""

        for result in report.failed_results:
            if use_color:
                lines.append(f"\033[31m✗ {result.name}\033[0m")
            else:
                lines.append(f"✗ {result.name}")

            if result.diagnostics:
                for diag in result.diagnostics[:10]:
                    lines.append(_indent_block(str(diag), "  ", "    "))

                if len(result.diagnostics) > 10:
                    note = _truncation_note(result, self.log_dir)
                    lines.append(f"  ... and {len(result.diagnostics) - 10} more{note}")
                    lines.append(f"  {_rerun_hint(result.name)}")
            elif result.output:
                # No structured diagnostics: always show raw output so the
                # root cause is immediately visible without re-running with -v.
                output_lines = result.output.split("\n")
                for line in output_lines[:30]:
                    lines.append(f"  {line}")
                if len(output_lines) > 30:
                    note = _truncation_note(result, self.log_dir)
                    lines.append(
                        f"  ... and {len(output_lines) - 30} more output lines{note}"
                    )
                    lines.append(f"  {_rerun_hint(result.name)}")
                if result.command:
                    cmd_str = shlex.join(result.command)
                    lines.append(f"  Reproduce: {cmd_str}")

        return "\n".join(lines)
