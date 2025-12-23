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

"""Tests for adapter-specific exceptions."""

from __future__ import annotations

import pytest

from weakincentives.adapters import (
    ClaudeAgentSDKError,
    LiteLLMError,
    OpenAIError,
    PromptEvaluationError,
    RateLimitError,
)
from weakincentives.adapters.core import PROMPT_EVALUATION_PHASE_REQUEST


class TestRateLimitError:
    def test_inherits_from_prompt_evaluation_error(self) -> None:
        error = RateLimitError(
            "Rate limit exceeded",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            retry_after=30.0,
        )

        assert isinstance(error, PromptEvaluationError)
        assert isinstance(error, RuntimeError)

    def test_stores_retry_after(self) -> None:
        error = RateLimitError(
            "Rate limit exceeded",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            retry_after=42.5,
        )

        assert error.retry_after == 42.5
        assert error.prompt_name == "test_prompt"
        assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST

    def test_retry_after_defaults_to_none(self) -> None:
        error = RateLimitError(
            "Rate limit exceeded",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

        assert error.retry_after is None

    def test_stores_provider_payload(self) -> None:
        payload = {"status": 429, "message": "Too many requests"}
        error = RateLimitError(
            "Rate limit exceeded",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            retry_after=30.0,
            provider_payload=payload,
        )

        assert error.provider_payload == payload


class TestOpenAIError:
    def test_inherits_from_prompt_evaluation_error(self) -> None:
        error = OpenAIError(
            "OpenAI request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_code="invalid_api_key",
        )

        assert isinstance(error, PromptEvaluationError)
        assert isinstance(error, RuntimeError)

    def test_stores_error_code(self) -> None:
        error = OpenAIError(
            "OpenAI request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_code="context_length_exceeded",
        )

        assert error.error_code == "context_length_exceeded"
        assert error.prompt_name == "test_prompt"
        assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST

    def test_error_code_defaults_to_none(self) -> None:
        error = OpenAIError(
            "OpenAI request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

        assert error.error_code is None

    def test_stores_provider_payload(self) -> None:
        payload = {"error": {"type": "server_error", "code": 500}}
        error = OpenAIError(
            "OpenAI request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_code="server_error",
            provider_payload=payload,
        )

        assert error.provider_payload == payload


class TestLiteLLMError:
    def test_inherits_from_prompt_evaluation_error(self) -> None:
        error = LiteLLMError(
            "LiteLLM request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_code="invalid_model",
        )

        assert isinstance(error, PromptEvaluationError)
        assert isinstance(error, RuntimeError)

    def test_stores_error_code(self) -> None:
        error = LiteLLMError(
            "LiteLLM request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_code="model_not_found",
        )

        assert error.error_code == "model_not_found"
        assert error.prompt_name == "test_prompt"
        assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST

    def test_stores_original_provider(self) -> None:
        error = LiteLLMError(
            "LiteLLM request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            original_provider="anthropic",
        )

        assert error.original_provider == "anthropic"

    def test_defaults_to_none(self) -> None:
        error = LiteLLMError(
            "LiteLLM request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

        assert error.error_code is None
        assert error.original_provider is None

    def test_stores_provider_payload(self) -> None:
        payload = {"provider": "openai", "status_code": 400}
        error = LiteLLMError(
            "LiteLLM request failed",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=payload,
        )

        assert error.provider_payload == payload


class TestClaudeAgentSDKError:
    def test_inherits_from_prompt_evaluation_error(self) -> None:
        error = ClaudeAgentSDKError(
            "Claude Agent SDK error",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_type="CLINotFoundError",
        )

        assert isinstance(error, PromptEvaluationError)
        assert isinstance(error, RuntimeError)

    def test_stores_error_type(self) -> None:
        error = ClaudeAgentSDKError(
            "Claude Agent SDK error",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_type="ProcessError",
        )

        assert error.error_type == "ProcessError"
        assert error.prompt_name == "test_prompt"
        assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST

    def test_error_type_defaults_to_none(self) -> None:
        error = ClaudeAgentSDKError(
            "Claude Agent SDK error",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

        assert error.error_type is None

    def test_stores_provider_payload(self) -> None:
        payload = {"exit_code": 1, "stderr": "Command failed"}
        error = ClaudeAgentSDKError(
            "Claude Agent SDK error",
            prompt_name="test_prompt",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            error_type="ProcessError",
            provider_payload=payload,
        )

        assert error.provider_payload == payload


class TestExceptionHierarchy:
    def test_all_exceptions_catchable_as_prompt_evaluation_error(self) -> None:
        exceptions = [
            RateLimitError(
                "Rate limit",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            OpenAIError(
                "OpenAI error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            LiteLLMError(
                "LiteLLM error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            ClaudeAgentSDKError(
                "Claude error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
        ]

        for exc in exceptions:
            with pytest.raises(PromptEvaluationError):
                raise exc

    def test_all_exceptions_catchable_as_runtime_error(self) -> None:
        exceptions = [
            RateLimitError(
                "Rate limit",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            OpenAIError(
                "OpenAI error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            LiteLLMError(
                "LiteLLM error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
            ClaudeAgentSDKError(
                "Claude error",
                prompt_name="p",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ),
        ]

        for exc in exceptions:
            with pytest.raises(RuntimeError):
                raise exc
