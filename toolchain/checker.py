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

            output = result.stdout
            if result.stderr:
                output = f"{output}\n{result.stderr}" if output else result.stderr

            diagnostics = self.parser(output, result.returncode)

            return CheckResult(
                name=self.name,
                status="passed" if result.returncode == 0 else "failed",
                duration_ms=duration_ms,
                diagnostics=diagnostics,
                output=output.strip(),
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
            )


@dataclass
class AutoFormatChecker:
    """A checker that auto-fixes formatting locally but only checks in CI.

    In local environments: runs the formatter to apply fixes, reports changes.
    In CI environments: runs the formatter in check mode, fails if changes needed.

    Uses JSON output format internally for reliable file path extraction.
    """

    name: str
    description: str
    check_command: list[str]
    fix_command: list[str]
    json_check_command: list[str] | None = None  # For JSON output parsing
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

            output = result.stdout
            if result.stderr:
                output = f"{output}\n{result.stderr}" if output else result.stderr

            diagnostics = self.parser(output, result.returncode)

            return CheckResult(
                name=self.name,
                status="passed" if result.returncode == 0 else "failed",
                duration_ms=duration_ms,
                diagnostics=diagnostics,
                output=output.strip(),
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Timed out after {self.timeout}s"),),
                output="",
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
            )

    def _run_with_autofix(self, start: float) -> CheckResult:
        """Run with auto-fix and report changes (local behavior).

        Uses JSON check command (if available) to get precise file list,
        then runs fix command to apply changes.
        """
        try:
            # First, check what files need formatting using JSON output
            files_to_format: list[str] = []
            if self.json_check_command:
                check_result = subprocess.run(
                    self.json_check_command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                # Parse JSON output to get file list (non-zero exit means files need formatting)
                if check_result.returncode != 0:
                    files_to_format = self._parse_json_output(check_result.stdout)

                # If check passed, nothing needs formatting
                if check_result.returncode == 0:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    return CheckResult(
                        name=self.name,
                        status="passed",
                        duration_ms=duration_ms,
                        diagnostics=(),
                        output="",
                    )

            # Run fix command to apply formatting
            fix_result = subprocess.run(
                self.fix_command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            # Check if fix command failed (e.g., syntax error, internal crash)
            if fix_result.returncode != 0:
                output = fix_result.stdout
                if fix_result.stderr:
                    output = f"{output}\n{fix_result.stderr}" if output else fix_result.stderr
                return CheckResult(
                    name=self.name,
                    status="failed",
                    duration_ms=duration_ms,
                    diagnostics=(Diagnostic("Auto-fix command failed"),),
                    output=output.strip(),
                )

            # Report which files were reformatted
            if files_to_format:
                message = self._format_file_message(files_to_format)
                return CheckResult(
                    name=self.name,
                    status="passed",
                    duration_ms=duration_ms,
                    diagnostics=(
                        Diagnostic(message=message, severity="info"),
                    ),
                    output=fix_result.stdout.strip(),
                )

            # No JSON check command - fall back to parsing fix output
            reformat_count = self._parse_reformat_count(fix_result.stdout)
            if reformat_count > 0:
                if reformat_count == 1:
                    message = "Automatically reformatted 1 file"
                else:
                    message = f"Automatically reformatted {reformat_count} files"

                return CheckResult(
                    name=self.name,
                    status="passed",
                    duration_ms=duration_ms,
                    diagnostics=(
                        Diagnostic(message=message, severity="info"),
                    ),
                    output=fix_result.stdout.strip(),
                )

            # No files reformatted - everything was already formatted
            return CheckResult(
                name=self.name,
                status="passed",
                duration_ms=duration_ms,
                diagnostics=(),
                output="",
            )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Timed out after {self.timeout}s"),),
                output="",
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
            )

    def _parse_json_output(self, output: str) -> list[str]:
        """Parse file paths from ruff JSON output.

        Expected format is a JSON array of objects with 'filename' field.
        """
        import json

        try:
            data = json.loads(output)
            if isinstance(data, list):
                # Extract unique filenames
                filenames = {item.get("filename") for item in data if isinstance(item, dict)}
                return sorted(f for f in filenames if f)
        except json.JSONDecodeError:
            pass
        return []

    def _format_file_message(self, files: list[str]) -> str:
        """Format a human-readable message about reformatted files."""
        if len(files) == 1:
            return f"Automatically reformatted 1 file: {files[0]}"
        file_list = ", ".join(files)
        return f"Automatically reformatted {len(files)} files: {file_list}"

    def _parse_reformat_count(self, output: str) -> int:
        """Parse the number of reformatted files from ruff text output.

        Fallback for when JSON output is not available.
        Looks for patterns like "1 file reformatted" or "3 files reformatted".
        """
        import re

        for line in output.strip().split("\n"):
            match = re.search(r"(\d+)\s+files?\s+reformatted", line)
            if match:
                return int(match.group(1))
        return 0
