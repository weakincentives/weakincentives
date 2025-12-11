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

"""Error normalization for Claude Agent SDK adapter."""

from __future__ import annotations

from typing import Any

from ..core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
)
from ..shared import ThrottleDetails, ThrottleError

__all__ = [
    "normalize_sdk_error",
]


def normalize_sdk_error(
    error: Exception,
    *,
    prompt_name: str,
) -> PromptEvaluationError:
    """Convert SDK exceptions to weakincentives error types.

    Args:
        error: The exception raised by the SDK.
        prompt_name: Name of the prompt being evaluated.

    Returns:
        A PromptEvaluationError or subclass appropriate for the error type.
    """
    error_type = type(error).__name__

    if error_type == "CLINotFoundError":
        return PromptEvaluationError(
            message=(
                "Claude Code CLI not found. "
                "Install: npm install -g @anthropic-ai/claude-code"
            ),
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    if error_type == "CLIConnectionError":
        return ThrottleError(
            message=str(error),
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=ThrottleDetails(
                kind="timeout",
                retry_after=None,
                attempts=1,
                retry_safe=True,
                provider_payload=None,
            ),
        )

    if error_type == "ProcessError":
        # Extract exit_code and stderr if available
        exit_code: int | None = getattr(error, "exit_code", None)
        stderr: str | None = getattr(error, "stderr", None)
        provider_payload: dict[str, Any] = {}
        if exit_code is not None:
            provider_payload["exit_code"] = exit_code
        if stderr is not None:
            provider_payload["stderr"] = stderr

        return PromptEvaluationError(
            message=f"Claude Code process failed: {stderr or str(error)}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=provider_payload if provider_payload else None,
        )

    if error_type == "CLIJSONDecodeError":
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

    # Default case for unknown exceptions
    return PromptEvaluationError(
        message=str(error),
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
    )
