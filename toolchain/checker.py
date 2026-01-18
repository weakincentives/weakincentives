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
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Timed out after {self.timeout}s"),),
                output="",
            )
        except FileNotFoundError as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=duration_ms,
                diagnostics=(Diagnostic(f"Command not found: {e.filename}"),),
                output="",
            )
