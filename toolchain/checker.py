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

"""Checker protocol and base implementations.

A Checker is anything that can verify an aspect of the codebase and return
a CheckResult. This module provides:

- Checker: Protocol defining the checker interface
- SubprocessChecker: Base class for checkers that run shell commands
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

from .result import CheckResult, Diagnostic

if TYPE_CHECKING:
    pass


class Checker(Protocol):
    """Protocol for verification checkers."""

    @property
    def name(self) -> str:
        """Short identifier for the checker (e.g., 'lint', 'test')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this checker verifies."""
        ...

    def run(self) -> CheckResult:
        """Execute the check and return the result."""
        ...


# Type alias for functions that parse command output into diagnostics
DiagnosticParser = Callable[[str, int], tuple[Diagnostic, ...]]


def is_ci_environment() -> bool:
    """Detect if running in a CI environment (GitHub Actions, etc.)."""
    import os

    # GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return True
    # Generic CI variable (used by many CI systems)
    if os.environ.get("CI") == "true":
        return True
    return False


def _no_parse(_output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Default parser that produces no diagnostics."""
    return ()


def merge_output(stdout: str, stderr: str) -> str:
    """Combine captured stdout and stderr into one block."""
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


_DRIFT_TAIL_LINES = 15


def output_tail(output: str, max_lines: int = _DRIFT_TAIL_LINES) -> str:
    """Return the last *max_lines* lines of output."""
    return "\n".join(output.strip().splitlines()[-max_lines:])


def parser_drift_warning(checker_name: str, output: str) -> Diagnostic:
    """Build a warning for a failing tool whose output could not be parsed.

    A failure with zero structured diagnostics usually means the tool's
    output format changed and the parser silently stopped matching. Surface
    the raw tail so the failure stays actionable while flagging that the
    toolchain itself needs attention.
    """
    message = (
        f"{checker_name}: command failed but the parser extracted no "
        "structured diagnostics - the tool's output format may have drifted "
        "from what the parser expects.\n"
        "Raw output tail:\n"
        f"{output_tail(output)}"
    )
    return Diagnostic(message=message, severity="warning")


@dataclass
class SubprocessChecker:
    """Base checker that runs a shell command.

    This handles the common pattern of:
    1. Run a command
    2. Check exit code for pass/fail
    3. Parse output for diagnostics
    """

    name: str
    description: str
    command: list[str]
    parser: DiagnosticParser = _no_parse
    timeout: int = 300  # 5 minutes default
    env: dict[str, str] = field(default_factory=dict)

    def run(self) -> CheckResult:
        """Run the command and return the result."""
        import os

        start = time.monotonic()
        try:
            # Merge environment
            run_env = {**os.environ, **self.env} if self.env else None

            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=run_env,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            output = merge_output(result.stdout, result.stderr)

            diagnostics = self.parser(output, result.returncode)
            if result.returncode != 0 and not diagnostics and self.parser is not _no_parse:
                diagnostics = (parser_drift_warning(self.name, output),)

            return CheckResult(
                name=self.name,
                status="passed" if result.returncode == 0 else "failed",
                duration_ms=duration_ms,
                diagnostics=diagnostics,
                output=output.strip(),
                command=tuple(self.command),
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            cmd_str = " ".join(self.command)
            msg = (
                f"Timed out after {self.timeout}s\n"
                f"Command: {cmd_str}\n"
                f"Fix: Increase timeout or investigate hanging process\n"
                f"Run manually: {cmd_str}"
            )
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(msg),),
                output="",
                command=tuple(self.command),
            )
        except FileNotFoundError as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            cmd_str = " ".join(self.command)
            msg = (
                f"Command not found: {e.filename}\n"
                f"Attempted: {cmd_str}\n"
                f"Fix: Ensure dependencies are installed\n"
                f"Run: uv sync && ./install-hooks.sh"
            )
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(msg),),
                output="",
                command=tuple(self.command),
            )


# Type alias for functions that parse check output into file paths
FileListParser = Callable[[str], list[str]]


@dataclass
class AutoFormatChecker:
    """A checker that auto-fixes formatting locally but only checks in CI.

    In local environments: runs the formatter to apply fixes, reports changes.
    In CI environments: runs the formatter in check mode, fails if changes needed.

    The check command's text output is parsed with file_list_parser to report
    which files needed formatting.
    """

    name: str
    description: str
    check_command: list[str]
    fix_command: list[str]
    file_list_parser: FileListParser
    parser: DiagnosticParser = _no_parse
    timeout: int = 300

    def run(self) -> CheckResult:
        """Run the check, auto-fixing if not in CI."""
        start = time.monotonic()

        if is_ci_environment():
            return self._run_check_only(start)
        return self._run_with_autofix(start)

    def _run_check_only(self, start: float) -> CheckResult:
        """Run in check-only mode (CI behavior)."""
        try:
            result = subprocess.run(
                self.check_command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            output = merge_output(result.stdout, result.stderr)

            diagnostics = self.parser(output, result.returncode)
            if result.returncode != 0 and not diagnostics and self.parser is not _no_parse:
                diagnostics = (parser_drift_warning(self.name, output),)

            return CheckResult(
                name=self.name,
                status="passed" if result.returncode == 0 else "failed",
                duration_ms=duration_ms,
                diagnostics=diagnostics,
                output=output.strip(),
                command=tuple(self.check_command),
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Timed out after {self.timeout}s"),),
                output="",
                command=tuple(self.check_command),
            )
        except FileNotFoundError as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            cmd_str = " ".join(self.check_command)
            msg = (
                f"Command not found: {e.filename}\n"
                f"Attempted: {cmd_str}\n"
                f"Fix: Ensure dependencies are installed\n"
                f"Run: uv sync && ./install-hooks.sh"
            )
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(msg),),
                output="",
                command=tuple(self.check_command),
            )

    def _run_with_autofix(self, start: float) -> CheckResult:
        """Run with auto-fix and report changes (local behavior).

        Runs the check command to learn which files need formatting, then the
        fix command to apply changes, reporting the affected files.
        """
        try:
            check_result = subprocess.run(
                self.check_command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if check_result.returncode == 0:
                duration_ms = int((time.monotonic() - start) * 1000)
                return CheckResult(
                    name=self.name,
                    status="passed",
                    duration_ms=duration_ms,
                    diagnostics=(),
                    output="",
                )

            check_output = merge_output(check_result.stdout, check_result.stderr)
            files_to_format = self.file_list_parser(check_output)

            fix_result = subprocess.run(
                self.fix_command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            # Fix command failed: either a hard error (syntax error, crash) or
            # issues remain that have no auto-fix. Parse its output so the
            # remaining issues surface as structured diagnostics.
            if fix_result.returncode != 0:
                output = merge_output(fix_result.stdout, fix_result.stderr)
                diagnostics = self.parser(output, fix_result.returncode)
                if not diagnostics:
                    diagnostics = (Diagnostic("Auto-fix command failed"),)
                return CheckResult(
                    name=self.name,
                    status="failed",
                    duration_ms=duration_ms,
                    diagnostics=diagnostics,
                    output=output.strip(),
                    command=tuple(self.fix_command),
                )

            if files_to_format:
                message = self._format_file_message(files_to_format)
                return CheckResult(
                    name=self.name,
                    status="passed",
                    duration_ms=duration_ms,
                    diagnostics=(Diagnostic(message=message, severity="info"),),
                    output=fix_result.stdout.strip(),
                )

            # Check reported issues but the file list parser matched nothing:
            # the fix was applied, but the parser likely drifted.
            return CheckResult(
                name=self.name,
                status="passed",
                duration_ms=duration_ms,
                diagnostics=(parser_drift_warning(self.name, check_output),),
                output=fix_result.stdout.strip(),
            )

        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            timed_out_command = tuple(str(part) for part in e.cmd)
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Timed out after {self.timeout}s"),),
                output="",
                command=timed_out_command,
            )
        except FileNotFoundError as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            cmd_str = " ".join(self.fix_command)
            msg = (
                f"Command not found: {e.filename}\n"
                f"Attempted: {cmd_str}\n"
                f"Fix: Ensure dependencies are installed\n"
                f"Run: uv sync && ./install-hooks.sh"
            )
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(msg),),
                output="",
                command=tuple(self.fix_command),
            )

    def _format_file_message(self, files: list[str]) -> str:
        """Format a human-readable message about reformatted files."""
        if len(files) == 1:
            summary = f"Automatically reformatted 1 file: {files[0]}"
        else:
            file_list = ", ".join(files)
            summary = f"Automatically reformatted {len(files)} files: {file_list}"
        return (
            f"{summary}\n"
            "These fixes are new uncommitted changes in the working tree - "
            "include them in your commit."
        )
