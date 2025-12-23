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

"""Adapter-specific exceptions for provider integrations."""

from __future__ import annotations

from typing import Any

from .core import PromptEvaluationError, PromptEvaluationPhase


class RateLimitError(PromptEvaluationError):
    """Provider rate limit exceeded.

    This exception is raised when a provider returns a rate limit error.
    Unlike ThrottleError which includes full retry policy details, this
    exception provides a simpler interface for rate limit scenarios.
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        retry_after: float | None = None,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=provider_payload,
        )
        self.retry_after = retry_after


class OpenAIError(PromptEvaluationError):
    """OpenAI-specific errors with API error codes.

    This exception is raised for errors specific to OpenAI's API,
    providing access to the error code returned by the service.
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        error_code: str | None = None,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=provider_payload,
        )
        self.error_code = error_code


class LiteLLMError(PromptEvaluationError):
    """LiteLLM-specific errors with provider details.

    This exception is raised for errors specific to LiteLLM,
    which may wrap errors from various underlying providers.
    """

    def __init__(  # noqa: PLR0913
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        error_code: str | None = None,
        original_provider: str | None = None,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=provider_payload,
        )
        self.error_code = error_code
        self.original_provider = original_provider


class ClaudeAgentSDKError(PromptEvaluationError):
    """Claude Agent SDK-specific errors.

    This exception is raised for errors specific to the Claude Agent SDK,
    providing access to SDK-specific error details.
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        error_type: str | None = None,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=provider_payload,
        )
        self.error_type = error_type


__all__ = [
    "ClaudeAgentSDKError",
    "LiteLLMError",
    "OpenAIError",
    "RateLimitError",
]
