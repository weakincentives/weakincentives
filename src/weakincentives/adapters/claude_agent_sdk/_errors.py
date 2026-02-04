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

"""Error normalization for Claude Agent SDK exceptions.

This module uses SDK native exception types for type-safe error handling.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)

from ...runtime.logging import StructuredLogger, get_logger
from ..core import PromptEvaluationError

if TYPE_CHECKING:
    from ..throttle import ThrottleError, ThrottleKind

__all__ = [
    "normalize_sdk_error",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk.errors"}
)


def _create_throttle_error(
    message: str,
    prompt_name: str,
    kind: ThrottleKind,
    retry_after: timedelta | None = None,
) -> ThrottleError:
    """Create a ThrottleError with proper details."""
    from ..throttle import ThrottleDetails, ThrottleError

    details = ThrottleDetails(
        kind=kind,
        retry_after=retry_after,
        attempts=1,
        retry_safe=True,
        provider_payload=None,
    )
    return ThrottleError(
        message=message,
        prompt_name=prompt_name,
        phase="request",
        details=details,
    )


def normalize_sdk_error(  # noqa: C901, PLR0911, PLR0912 - complexity needed for comprehensive error handling
    error: Exception,
    prompt_name: str,
    *,
    stderr_output: str | None = None,
) -> PromptEvaluationError:
    """Convert SDK exceptions to weakincentives error types.

    Args:
        error: The exception from the Claude Agent SDK.
        prompt_name: Name of the prompt being evaluated.
        stderr_output: Captured stderr output from the SDK process, if any.
            This is particularly useful for debugging process failures.

    Returns:
        A normalized PromptEvaluationError or subclass.
    """
    error_type = type(error).__name__
    error_module = type(error).__module__

    logger.debug(
        "claude_agent_sdk.error.normalizing",
        event="error.normalizing",
        context={
            "error_type": error_type,
            "error_module": error_module,
            "error_message": str(error),
            "has_stderr_output": stderr_output is not None,
            "stderr_preview": stderr_output[:1000] if stderr_output else None,
        },
    )

    # Handle ExceptionGroup from TaskGroup cleanup (Python 3.11+)
    if error_type in {"ExceptionGroup", "BaseExceptionGroup"}:
        # Extract information about the nested exceptions
        sub_exceptions: list[dict[str, str]] = []
        if hasattr(error, "exceptions"):
            # Access exceptions attribute (available on ExceptionGroup)
            exceptions = getattr(error, "exceptions", [])
            sub_exceptions = [
                {
                    "type": type(sub_error).__name__,
                    "message": str(sub_error),
                }
                for sub_error in exceptions
            ]

        logger.debug(
            "claude_agent_sdk.error.exception_group",
            event="error.exception_group",
            context={
                "prompt_name": prompt_name,
                "sub_exception_count": len(sub_exceptions),
                "sub_exceptions": sub_exceptions,
                "stderr_output": stderr_output,
            },
        )

        # Check if all sub-exceptions are CLIConnectionError during cleanup
        all_cli_connection_errors = all(
            exc["type"] == "CLIConnectionError"
            and "not ready for writing" in exc["message"]
            for exc in sub_exceptions
        )

        if all_cli_connection_errors:
            # This is the TaskGroup cleanup race condition
            return PromptEvaluationError(
                message=(
                    f"SDK cleanup error: Transport closed while processing "
                    f"{len(sub_exceptions)} pending control requests. "
                    f"This may occur when sub-agents are interrupted during "
                    f"structured output validation."
                ),
                prompt_name=prompt_name,
                phase="response",
                provider_payload={
                    "error_type": "TaskGroupCleanupError",
                    "sub_exception_count": len(sub_exceptions),
                    "stderr": stderr_output,
                }
                if stderr_output
                else None,
            )

        # Generic ExceptionGroup handling
        return PromptEvaluationError(
            message=str(error),
            prompt_name=prompt_name,
            phase="request",
            provider_payload={
                "sub_exceptions": sub_exceptions,
                "stderr": stderr_output,
            }
            if sub_exceptions or stderr_output
            else None,
        )

    # Use isinstance checks with SDK native exception types for type safety
    if isinstance(error, CLINotFoundError):
        logger.debug(
            "claude_agent_sdk.error.cli_not_found",
            event="error.cli_not_found",
            context={"prompt_name": prompt_name},
        )
        return PromptEvaluationError(
            message=(
                "Claude Code CLI not found. Install: "
                "npm install -g @anthropic-ai/claude-code"
            ),
            prompt_name=prompt_name,
            phase="request",
        )

    if isinstance(error, CLIConnectionError):
        logger.debug(
            "claude_agent_sdk.error.cli_connection_error",
            event="error.cli_connection_error",
            context={
                "prompt_name": prompt_name,
                "stderr_output": stderr_output,
            },
        )
        return _create_throttle_error(
            message=str(error),
            prompt_name=prompt_name,
            kind="timeout",
            retry_after=None,
        )

    if isinstance(error, ProcessError):
        provider_payload: dict[str, Any] = {}
        if hasattr(error, "exit_code"):  # pragma: no branch - SDK always has exit_code
            provider_payload["exit_code"] = error.exit_code
        # Include stderr from both error attribute and captured output
        error_stderr = getattr(error, "stderr", None)
        if error_stderr:
            provider_payload["stderr"] = error_stderr
        if stderr_output:
            # Prefer captured stderr as it may be more complete
            provider_payload["stderr_captured"] = stderr_output
            if not error_stderr:
                provider_payload["stderr"] = stderr_output

        logger.debug(
            "claude_agent_sdk.error.process_error",
            event="error.process_error",
            context={
                "prompt_name": prompt_name,
                "exit_code": provider_payload.get("exit_code"),
                "stderr": provider_payload.get("stderr"),
                "stderr_captured": provider_payload.get("stderr_captured"),
            },
        )

        return PromptEvaluationError(
            message=f"Claude Code process failed: {error}",
            prompt_name=prompt_name,
            phase="request",
            provider_payload=provider_payload if provider_payload else None,
        )

    if isinstance(error, CLIJSONDecodeError):
        logger.debug(
            "claude_agent_sdk.error.json_decode_error",
            event="error.json_decode_error",
            context={
                "prompt_name": prompt_name,
                "stderr_output": stderr_output,
            },
        )
        provider_payload_json: dict[str, Any] = {}
        if stderr_output:
            provider_payload_json["stderr"] = stderr_output
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase="response",
            provider_payload=provider_payload_json if provider_payload_json else None,
        )

    # MaxTurnsExceededError is checked by name since it may not be exported
    if error_type == "MaxTurnsExceededError":
        logger.debug(
            "claude_agent_sdk.error.max_turns_exceeded",
            event="error.max_turns_exceeded",
            context={"prompt_name": prompt_name},
        )
        return PromptEvaluationError(
            message=f"SDK exceeded maximum turns: {error}",
            prompt_name=prompt_name,
            phase="response",
        )

    # Handle unknown SDK errors or general errors
    is_sdk_error = "claude_agent_sdk" in error_module or "claude_code" in error_module
    message = f"Claude Agent SDK error: {error}" if is_sdk_error else str(error)

    logger.debug(
        "claude_agent_sdk.error.unknown",
        event="error.unknown",
        context={
            "prompt_name": prompt_name,
            "is_sdk_error": is_sdk_error,
            "error_type": error_type,
            "stderr_output": stderr_output,
        },
    )

    # Include stderr in unknown errors as well
    unknown_payload: dict[str, Any] = {}
    if stderr_output:
        unknown_payload["stderr"] = stderr_output

    return PromptEvaluationError(
        message=message,
        prompt_name=prompt_name,
        phase="request",
        provider_payload=unknown_payload if unknown_payload else None,
    )
