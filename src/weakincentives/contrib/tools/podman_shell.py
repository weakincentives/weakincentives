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

"""Podman shell execution tool suite.

This module provides the shell execution handler that runs commands inside
a Podman container. It includes validation and normalization for shell
parameters.

Example usage::

    # Typically used internally by PodmanSandboxSection
    suite = PodmanShellSuite(section=podman_section)
    result = suite.run_shell(params, context=context)
"""

from __future__ import annotations

import math
import posixpath
import subprocess  # nosec: B404
import time
from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Final, Protocol

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.tool import ToolContext, ToolResult
from ._context import ensure_context_uses_session

if TYPE_CHECKING:
    from ...runtime.session import Session

_MAX_STDIO_CHARS: Final[int] = 32 * 1024
_MAX_COMMAND_LENGTH: Final[int] = 4_096
_MAX_ENV_LENGTH: Final[int] = 512
_MAX_ENV_VARS: Final[int] = 64
_MAX_TIMEOUT: Final[float] = 120.0
_MIN_TIMEOUT: Final[float] = 1.0
_DEFAULT_TIMEOUT: Final[float] = 30.0
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_PATH_SEGMENT: Final[int] = 80
_DEFAULT_WORKDIR: Final[str] = "/workspace"
_ASCII: Final[str] = "ascii"
_CAPTURE_DISABLED: Final[str] = "capture disabled"


@FrozenDataclass()
class PodmanShellParams:
    """Parameter payload accepted by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    stdin: str | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT
    capture_output: bool = True


@FrozenDataclass()
class PodmanShellResult:
    """Structured command summary returned by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

    def render(self) -> str:
        command_str = " ".join(self.command)
        lines = [
            "Shell command result:",
            f"Command: {command_str}",
            f"CWD: {self.cwd}",
            f"Exit code: {self.exit_code}",
            f"Timed out: {self.timed_out}",
            f"Duration: {self.duration_ms} ms",
            "STDOUT:",
            self.stdout or "<empty>",
            "STDERR:",
            self.stderr or "<empty>",
        ]
        return "\n".join(lines)


@FrozenDataclass()
class ShellExecConfig:
    """Configuration for shell command execution."""

    command: Sequence[str]
    stdin: str | None = None
    cwd: str | None = None
    environment: Mapping[str, str] | None = None
    timeout: float | None = None
    capture_output: bool = True


class PodmanShellSectionProtocol(Protocol):
    """Protocol for PodmanSandboxSection to avoid circular imports."""

    @property
    def session(self) -> Session: ...

    def ensure_workspace(self) -> object: ...

    def run_cli_exec(
        self,
        *,
        config: ShellExecConfig,
    ) -> subprocess.CompletedProcess[str]: ...

    def touch_workspace(self) -> None: ...


def ensure_ascii(value: str, *, field: str) -> str:
    """Validate that a string contains only ASCII characters.

    Args:
        value: String to validate.
        field: Field name for error message.

    Returns:
        The validated string.

    Raises:
        ToolValidationError: If string contains non-ASCII characters.
    """
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:
        raise ToolValidationError(f"{field} must be ASCII.") from error
    return value


def normalize_command(command: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and validate shell command.

    Args:
        command: Command tuple to validate.

    Returns:
        Normalized command tuple.

    Raises:
        ToolValidationError: If command is empty or exceeds limits.
    """
    if not command:
        raise ToolValidationError("command must contain at least one entry.")
    total_length = 0
    normalized: list[str] = []
    for index, entry in enumerate(command):
        if not entry:
            raise ToolValidationError(f"command[{index}] must not be empty.")
        normalized_entry = ensure_ascii(entry, field="command")
        total_length += len(normalized_entry)
        if total_length > _MAX_COMMAND_LENGTH:
            raise ToolValidationError("command is too long (limit 4,096 characters).")
        normalized.append(normalized_entry)
    return tuple(normalized)


def normalize_env(env: Mapping[str, str]) -> dict[str, str]:
    """Normalize and validate environment variables.

    Args:
        env: Environment mapping to validate.

    Returns:
        Normalized environment dictionary.

    Raises:
        ToolValidationError: If env exceeds limits or contains invalid values.
    """
    if len(env) > _MAX_ENV_VARS:
        raise ToolValidationError("env contains too many entries (max 64).")
    normalized: dict[str, str] = {}
    for key, value in env.items():
        normalized_key = ensure_ascii(key, field="env key").upper()
        if not normalized_key:
            raise ToolValidationError("env keys must not be empty.")
        if len(normalized_key) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"env key {normalized_key!r} is longer than {_MAX_PATH_SEGMENT} characters."
            )
        normalized_value = ensure_ascii(value, field="env value")
        if len(normalized_value) > _MAX_ENV_LENGTH:
            raise ToolValidationError(
                f"env value for {normalized_key!r} exceeds {_MAX_ENV_LENGTH} characters."
            )
        normalized[normalized_key] = normalized_value
    return normalized


def normalize_timeout(timeout_seconds: float) -> float:
    """Clamp timeout to allowed range.

    Args:
        timeout_seconds: Requested timeout in seconds.

    Returns:
        Timeout clamped to [1, 120] seconds.

    Raises:
        ToolValidationError: If timeout is NaN.
    """
    if math.isnan(timeout_seconds):
        raise ToolValidationError("timeout_seconds must be a real number.")
    return max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, timeout_seconds))


def normalize_cwd(path: str | None, workdir: str = _DEFAULT_WORKDIR) -> str:
    """Normalize and validate working directory path.

    Args:
        path: Relative path or None for default.
        workdir: Base working directory.

    Returns:
        Normalized absolute path within workspace.

    Raises:
        ToolValidationError: If path is absolute or contains invalid segments.
    """
    if path is None or path == "":
        return workdir
    stripped = path.strip()
    if not stripped:
        return workdir
    if stripped.startswith("/"):
        raise ToolValidationError("cwd must be relative to /workspace.")
    parts = [segment for segment in stripped.split("/") if segment]
    if len(parts) > _MAX_PATH_DEPTH:
        raise ToolValidationError("cwd exceeds maximum depth of 16 segments.")
    normalized_segments: list[str] = []
    for segment in parts:
        if segment in {".", ".."}:
            raise ToolValidationError("cwd must not contain '.' or '..' segments.")
        if len(segment) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"cwd segment {segment!r} exceeds {_MAX_PATH_SEGMENT} characters."
            )
        normalized_segment = ensure_ascii(segment, field="cwd")
        normalized_segments.append(normalized_segment)
    return posixpath.join(workdir, *normalized_segments)


def truncate_stream(value: str) -> str:
    """Truncate output stream to maximum length.

    Args:
        value: Stream content to truncate.

    Returns:
        Truncated string with [truncated] suffix if needed.
    """
    if len(value) <= _MAX_STDIO_CHARS:
        return value
    truncated = value[: _MAX_STDIO_CHARS - len("[truncated]")]
    return f"{truncated}[truncated]"


class PodmanShellSuite:
    """Handler collection for shell execution in Podman containers.

    Provides the ``shell_execute`` tool handler that runs commands
    inside the Podman workspace.
    """

    def __init__(self, *, section: PodmanShellSectionProtocol) -> None:
        """Initialize the shell suite.

        Args:
            section: The PodmanSandboxSection instance that owns this suite.
        """
        super().__init__()
        self._section = section

    def run_shell(
        self, params: PodmanShellParams, *, context: ToolContext
    ) -> ToolResult[PodmanShellResult]:
        """Execute a shell command in the Podman container.

        Args:
            params: Shell execution parameters.
            context: Tool execution context.

        Returns:
            ToolResult containing the command outcome.
        """
        ensure_context_uses_session(context=context, session=self._section.session)
        command = normalize_command(params.command)
        cwd = normalize_cwd(params.cwd)
        env_overrides = normalize_env(params.env)
        timeout_seconds = normalize_timeout(params.timeout_seconds)
        if params.stdin:
            _ = ensure_ascii(params.stdin, field="stdin")
        _ = self._section.ensure_workspace()

        return self._run_shell_via_cli(
            params=params,
            command=command,
            cwd=cwd,
            environment=env_overrides,
            timeout_seconds=timeout_seconds,
        )

    def _run_shell_via_cli(
        self,
        *,
        params: PodmanShellParams,
        command: tuple[str, ...],
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ToolResult[PodmanShellResult]:
        """Execute shell command via CLI.

        Args:
            params: Original shell parameters.
            command: Normalized command tuple.
            cwd: Normalized working directory.
            environment: Normalized environment.
            timeout_seconds: Normalized timeout.

        Returns:
            ToolResult containing the command outcome.
        """
        exec_cmd = list(command)
        start = time.perf_counter()
        try:
            completed = self._section.run_cli_exec(
                config=ShellExecConfig(
                    command=exec_cmd,
                    stdin=params.stdin if params.stdin else None,
                    cwd=cwd,
                    environment=environment,
                    timeout=timeout_seconds,
                    capture_output=params.capture_output,
                )
            )
            timed_out = False
            exit_code = completed.returncode
            stdout_text = completed.stdout
            stderr_text = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_code = 124
            stdout_text = str(error.stdout or "")
            stderr_text = str(error.stderr or "")
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute commands over SSH connections."
            ) from error
        duration_ms = int((time.perf_counter() - start) * 1_000)
        self._section.touch_workspace()
        stdout_text_clean = str(stdout_text or "").rstrip()
        stderr_text_clean = str(stderr_text or "").rstrip()
        if not params.capture_output:
            stdout_text_final = _CAPTURE_DISABLED
            stderr_text_final = _CAPTURE_DISABLED
        else:
            stdout_text_final = truncate_stream(stdout_text_clean)
            stderr_text_final = truncate_stream(stderr_text_clean)
        result = PodmanShellResult(
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=stdout_text_final,
            stderr=stderr_text_final,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
        message = f"`shell_execute` exited with {exit_code}."
        if timed_out:
            message = "`shell_execute` exceeded the configured timeout."
        return ToolResult(message=message, value=result)


__all__ = [
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanShellSectionProtocol",
    "PodmanShellSuite",
    "ShellExecConfig",
    "ensure_ascii",
    "normalize_command",
    "normalize_cwd",
    "normalize_env",
    "normalize_timeout",
    "truncate_stream",
]
