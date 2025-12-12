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

from ..core import PromptEvaluationError

if TYPE_CHECKING:
    from ..shared import ThrottleError, ThrottleKind

__all__ = [
    "normalize_sdk_error",
]


def _create_throttle_error(
    message: str,
    prompt_name: str,
    kind: ThrottleKind,
    retry_after: timedelta | None = None,
) -> ThrottleError:
    """Create a ThrottleError with proper details."""
    from ..shared import ThrottleDetails, ThrottleError

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


def normalize_sdk_error(
    error: Exception,
    prompt_name: str,
) -> PromptEvaluationError:
    """Convert SDK exceptions to weakincentives error types.

    Args:
        error: The exception from the Claude Agent SDK.
        prompt_name: Name of the prompt being evaluated.

    Returns:
        A normalized PromptEvaluationError or subclass.
    """
    error_type = type(error).__name__
    error_module = type(error).__module__

    if error_type == "CLINotFoundError":
        return PromptEvaluationError(
            message=(
                "Claude Code CLI not found. Install: "
                "npm install -g @anthropic-ai/claude-code"
            ),
            prompt_name=prompt_name,
            phase="request",
        )

    if error_type == "CLIConnectionError":
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
        if hasattr(error, "stderr"):
            provider_payload["stderr"] = error.stderr

        return PromptEvaluationError(
            message=f"Claude Code process failed: {error}",
            prompt_name=prompt_name,
            phase="request",
            provider_payload=provider_payload if provider_payload else None,
        )

    if error_type == "CLIJSONDecodeError":
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase="response",
        )

    if error_type == "MaxTurnsExceededError":
        return PromptEvaluationError(
            message=f"SDK exceeded maximum turns: {error}",
            prompt_name=prompt_name,
            phase="response",
        )

    # Handle unknown SDK errors or general errors
    is_sdk_error = "claude_agent_sdk" in error_module or "claude_code" in error_module
    message = f"Claude Agent SDK error: {error}" if is_sdk_error else str(error)
    return PromptEvaluationError(
        message=message,
        prompt_name=prompt_name,
        phase="request",
    )
