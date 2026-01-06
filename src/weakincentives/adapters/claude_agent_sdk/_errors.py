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

"""Error normalization for Claude Agent SDK exceptions."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

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


def normalize_sdk_error(  # noqa: C901 - complexity needed for comprehensive debug logging
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

    if error_type == "CLINotFoundError":
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

    if error_type == "CLIConnectionError":
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

    if error_type == "ProcessError":
        provider_payload: dict[str, Any] = {}
        if hasattr(error, "exit_code"):
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

    if error_type == "CLIJSONDecodeError":
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
