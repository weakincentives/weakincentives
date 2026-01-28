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

"""Podman-backed Python evaluation tool suite.

This module provides the evaluation tool handler that runs Python code
inside a Podman container via ``python3 -c``. Unlike the asteval-based
evaluator, this executes unrestricted Python in an isolated container.

Example usage::

    # Typically used internally by PodmanSandboxSection
    suite = PodmanEvalSuite(section=podman_section)
    result = suite.evaluate_python(params, context=context)
"""

from __future__ import annotations

import subprocess  # nosec: B404
from typing import TYPE_CHECKING, Final, Protocol

from ...errors import ToolValidationError
from ...prompt.tool import ToolContext, ToolResult
from ._context import ensure_context_uses_session
from .asteval import EvalParams, EvalResult

if TYPE_CHECKING:
    from ...runtime.session import Session

_EVAL_TIMEOUT_SECONDS: Final[float] = 5.0
_EVAL_MAX_STREAM_LENGTH: Final[int] = 4_096
_LOWEST_PRINTABLE_CODEPOINT: Final[int] = 32
_ALLOWED_CONTROL_CHARACTERS: Final[tuple[str, str]] = ("\n", "\t")


class PodmanSectionProtocol(Protocol):
    """Protocol for PodmanSandboxSection to avoid circular imports."""

    @property
    def session(self) -> Session: ...

    def ensure_workspace(self) -> object: ...

    def run_python_script(
        self,
        *,
        script: str,
        args: tuple[str, ...],
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...

    def touch_workspace(self) -> None: ...


def truncate_eval_stream(value: str) -> str:
    """Truncate evaluation output stream to the maximum length.

    Args:
        value: Stream content to truncate.

    Returns:
        Truncated string with ``...`` suffix if truncation occurred.
    """
    if len(value) <= _EVAL_MAX_STREAM_LENGTH:
        return value
    suffix = "..."
    keep = _EVAL_MAX_STREAM_LENGTH - len(suffix)
    return f"{value[:keep]}{suffix}"


def normalize_podman_eval_code(code: str) -> str:
    """Validate and normalize Python code for Podman evaluation.

    Ensures code contains only printable ASCII characters plus
    allowed control characters (newline, tab).

    Args:
        code: Python code to validate.

    Returns:
        The validated code string.

    Raises:
        ToolValidationError: If code contains unsupported control characters.
    """
    for char in code:
        code_point = ord(char)
        if (
            code_point < _LOWEST_PRINTABLE_CODEPOINT
            and char not in _ALLOWED_CONTROL_CHARACTERS
        ):
            raise ToolValidationError("Code contains unsupported control characters.")
    return code


class PodmanEvalSuite:
    """Tool suite for Python evaluation in Podman containers.

    Provides the ``evaluate_python`` tool handler that executes Python
    code via ``python3 -c`` inside the Podman workspace.
    """

    def __init__(self, *, section: PodmanSectionProtocol) -> None:
        """Initialize the eval suite.

        Args:
            section: The PodmanSandboxSection instance that owns this suite.
        """
        super().__init__()
        self._section = section

    def evaluate_python(
        self, params: EvalParams, *, context: ToolContext
    ) -> ToolResult[EvalResult]:
        """Execute Python code in the Podman container.

        Runs the provided code via ``python3 -c`` with a 5-second timeout.
        Captures stdout and stderr, returning them in the result.

        Args:
            params: Evaluation parameters containing the code to run.
            context: Tool execution context.

        Returns:
            ToolResult containing the evaluation outcome.

        Raises:
            ToolValidationError: If reads, writes, or globals are specified
                (not supported in Podman evaluation).
        """
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        self._ensure_passthrough_payload_is_empty(params)
        code = normalize_podman_eval_code(params.code)
        _ = self._section.ensure_workspace()
        try:
            completed = self._section.run_python_script(
                script=code,
                args=(),
                timeout=_EVAL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return self._timeout_result()
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Podman CLI is required to execute evaluation commands."
            ) from error

        stdout = truncate_eval_stream(str(completed.stdout or ""))
        stderr = truncate_eval_stream(str(completed.stderr or ""))
        success = completed.returncode == 0
        if success:
            message = f"Evaluation succeeded (exit code {completed.returncode})."
        else:
            message = f"Evaluation failed (exit code {completed.returncode})."
        result = EvalResult(
            value_repr=None,
            stdout=stdout,
            stderr=stderr,
            globals={},
            reads=(),
            writes=(),
        )
        self._section.touch_workspace()
        return ToolResult(message=message, value=result, success=success)

    @staticmethod
    def _ensure_passthrough_payload_is_empty(params: EvalParams) -> None:
        """Validate that passthrough parameters are not used.

        Podman evaluation doesn't support reads, writes, or globals
        as the asteval-based evaluator does.
        """
        if params.reads:
            raise ToolValidationError(
                "Podman evaluate_python reads are not supported; access the workspace directly."
            )
        if params.writes:
            raise ToolValidationError(
                "Podman evaluate_python writes are not supported; use the write_file tool."
            )
        if params.globals:
            raise ToolValidationError(
                "Podman evaluate_python globals are not supported."
            )

    @staticmethod
    def _timeout_result() -> ToolResult[EvalResult]:
        """Generate a timeout result for timed-out evaluations."""
        result = EvalResult(
            value_repr=None,
            stdout="",
            stderr="Execution timed out.",
            globals={},
            reads=(),
            writes=(),
        )
        return ToolResult(message="Evaluation timed out.", value=result, success=False)


__all__ = [
    "PodmanEvalSuite",
    "PodmanSectionProtocol",
    "normalize_podman_eval_code",
    "truncate_eval_stream",
]
