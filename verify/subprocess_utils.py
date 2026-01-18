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

"""Subprocess execution utilities for the verification toolbox."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - subprocess use is intentional for verification tools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """Result of subprocess execution.

    Attributes:
        returncode: The exit code of the process.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_ms: How long the process took in milliseconds.
    """

    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

    @property
    def success(self) -> bool:
        """Check if the process exited successfully."""
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts)


def run_tool(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 120.0,
    capture: bool = True,
) -> SubprocessResult:
    """Run an external tool with consistent handling.

    Args:
        cmd: The command and arguments to run.
        cwd: Working directory for the command.
        env: Additional environment variables (merged with current env).
        timeout_seconds: Maximum time to wait for the command.
        capture: Whether to capture stdout/stderr.

    Returns:
        A SubprocessResult with the command's output and exit code.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
        FileNotFoundError: If the command is not found.
    """
    # Merge environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start_time = time.monotonic()

    try:
        result = subprocess.run(  # nosec B603 - cmd from trusted internal callers
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=capture,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return SubprocessResult(
            returncode=124,  # Standard timeout exit code
            stdout="",
            stderr=f"Command timed out after {timeout_seconds}s: {' '.join(cmd)}",
            duration_ms=duration_ms,
        )
    except FileNotFoundError as e:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return SubprocessResult(
            returncode=127,  # Standard "command not found" exit code
            stdout="",
            stderr=f"Command not found: {e.filename}",
            duration_ms=duration_ms,
        )

    duration_ms = int((time.monotonic() - start_time) * 1000)

    return SubprocessResult(
        returncode=result.returncode,
        stdout=result.stdout if capture else "",
        stderr=result.stderr if capture else "",
        duration_ms=duration_ms,
    )


def run_python_module(
    module: str,
    args: Sequence[str] = (),
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 120.0,
) -> SubprocessResult:
    """Run a Python module as a subprocess.

    Args:
        module: The module name (e.g., "bandit").
        args: Arguments to pass to the module.
        cwd: Working directory.
        env: Additional environment variables.
        timeout_seconds: Maximum time to wait.

    Returns:
        A SubprocessResult with the module's output and exit code.
    """
    cmd = [sys.executable, "-m", module, *args]
    return run_tool(cmd, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
