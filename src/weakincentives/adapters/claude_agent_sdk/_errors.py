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

"""Error normalization helpers for the Claude Agent SDK adapter."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)

from ..core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
)
from ..shared import ThrottleDetails, ThrottleError

_ERROR_CLI_NOT_FOUND = (
    "Claude Code CLI not found. Install it with `npm install -g @anthropic-ai/claude-code` "
    "or configure CLAUDE_CODE_ENTRYPOINT."
)


def normalize_sdk_error(error: Exception, *, prompt_name: str) -> PromptEvaluationError:
    """Convert SDK exceptions into adapter-friendly errors."""

    if isinstance(error, CLINotFoundError):
        return PromptEvaluationError(
            message=_ERROR_CLI_NOT_FOUND,
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    if isinstance(error, CLIConnectionError):
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

    if isinstance(error, ProcessError):
        provider_payload: dict[str, Any] | None = None
        if getattr(error, "stderr", None):
            provider_payload = {"stderr": error.stderr}
        return PromptEvaluationError(
            message=f"Claude Code process failed: {error}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=provider_payload,
        )

    if isinstance(error, CLIJSONDecodeError):
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

    return PromptEvaluationError(
        message=str(error),
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
    )
