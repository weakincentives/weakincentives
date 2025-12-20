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

"""Tests for adapter configuration dataclasses."""

from __future__ import annotations

import pytest

from weakincentives.adapters.config import (
    LiteLLMClientConfig,
    LiteLLMModelConfig,
    LLMConfig,
    OpenAIClientConfig,
    OpenAIModelConfig,
    merge_config_params,
)

# LLMConfig tests


def test_llm_config_to_request_params_all_none() -> None:
    config = LLMConfig()
    params = config.to_request_params()
    assert params == {}


def test_llm_config_to_request_params_all_set() -> None:
    config = LLMConfig(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        presence_penalty=0.5,
        frequency_penalty=0.3,
        stop=("STOP",),
        seed=42,
    )
    params = config.to_request_params()
    assert params == {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "stop": ["STOP"],
        "seed": 42,
    }


def test_llm_config_to_request_params_partial() -> None:
    config = LLMConfig(temperature=0.5, max_tokens=200)
    params = config.to_request_params()
    assert params == {"temperature": 0.5, "max_tokens": 200}


# OpenAIClientConfig tests


def test_openai_client_config_to_client_kwargs_all_none() -> None:
    config = OpenAIClientConfig()
    kwargs = config.to_client_kwargs()
    assert kwargs == {}


def test_openai_client_config_to_client_kwargs_all_set() -> None:
    config = OpenAIClientConfig(
        api_key="sk-test",
        base_url="https://api.example.com",
        organization="org-123",
        timeout=30.0,
        max_retries=3,
    )
    kwargs = config.to_client_kwargs()
    assert kwargs == {
        "api_key": "sk-test",
        "base_url": "https://api.example.com",
        "organization": "org-123",
        "timeout": 30.0,
        "max_retries": 3,
    }


def test_openai_client_config_to_client_kwargs_partial() -> None:
    config = OpenAIClientConfig(api_key="sk-test", timeout=60.0)
    kwargs = config.to_client_kwargs()
    assert kwargs == {"api_key": "sk-test", "timeout": 60.0}


# OpenAIModelConfig tests


def test_openai_model_config_to_request_params_defaults() -> None:
    config = OpenAIModelConfig()
    params = config.to_request_params()
    assert params == {"parallel_tool_calls": True}


def test_openai_model_config_to_request_params_base_fields() -> None:
    config = OpenAIModelConfig(temperature=0.5, max_tokens=100)
    params = config.to_request_params()
    # Responses API uses max_output_tokens instead of max_tokens
    assert params == {
        "temperature": 0.5,
        "max_output_tokens": 100,
        "parallel_tool_calls": True,
    }


def test_openai_model_config_to_request_params_openai_specific() -> None:
    config = OpenAIModelConfig(
        logprobs=True,
        top_logprobs=5,
        parallel_tool_calls=False,
        store=True,
        user="user-123",
    )
    params = config.to_request_params()
    assert params == {
        "logprobs": True,
        "top_logprobs": 5,
        "parallel_tool_calls": False,
        "store": True,
        "user": "user-123",
    }


def test_openai_model_config_to_request_params_all_set() -> None:
    config = OpenAIModelConfig(
        temperature=0.7,
        max_tokens=200,
        top_p=0.9,
        top_logprobs=3,
        user="user-456",
    )
    params = config.to_request_params()
    # Responses API uses max_output_tokens instead of max_tokens
    assert params == {
        "temperature": 0.7,
        "max_output_tokens": 200,
        "top_p": 0.9,
        "top_logprobs": 3,
        "parallel_tool_calls": True,
        "user": "user-456",
    }


def test_openai_model_config_rejects_unsupported_fields() -> None:
    expected_message = "Unsupported OpenAI Responses parameters: seed"
    with pytest.raises(ValueError, match=expected_message):
        OpenAIModelConfig(seed=123)

    with pytest.raises(ValueError, match="stop"):
        OpenAIModelConfig(stop=("STOP",))

    with pytest.raises(ValueError, match="presence_penalty"):
        OpenAIModelConfig(presence_penalty=0.5)

    with pytest.raises(ValueError, match="frequency_penalty"):
        OpenAIModelConfig(frequency_penalty=0.1)


def test_openai_model_config_with_none_parallel_tool_calls() -> None:
    """Test that parallel_tool_calls=None skips adding it to params."""
    # Create config and set parallel_tool_calls to None to test the skip branch
    config = OpenAIModelConfig(temperature=0.5)
    # Use object.__setattr__ to bypass frozen dataclass restriction
    object.__setattr__(config, "parallel_tool_calls", None)
    params = config.to_request_params()
    # When parallel_tool_calls is None, it should not be in params
    assert "parallel_tool_calls" not in params
    assert params == {"temperature": 0.5}


# LiteLLMClientConfig tests


def test_litellm_client_config_to_completion_kwargs_all_none() -> None:
    config = LiteLLMClientConfig()
    kwargs = config.to_completion_kwargs()
    assert kwargs == {}


def test_litellm_client_config_to_completion_kwargs_all_set() -> None:
    config = LiteLLMClientConfig(
        api_key="test-key",
        api_base="https://api.example.com",
        timeout=30.0,
        num_retries=3,
    )
    kwargs = config.to_completion_kwargs()
    assert kwargs == {
        "api_key": "test-key",
        "api_base": "https://api.example.com",
        "timeout": 30.0,
        "num_retries": 3,
    }


def test_litellm_client_config_to_completion_kwargs_partial() -> None:
    config = LiteLLMClientConfig(api_key="test-key", timeout=60.0)
    kwargs = config.to_completion_kwargs()
    assert kwargs == {"api_key": "test-key", "timeout": 60.0}


# LiteLLMModelConfig tests


def test_litellm_model_config_to_request_params_all_none() -> None:
    config = LiteLLMModelConfig()
    params = config.to_request_params()
    assert params == {}


def test_litellm_model_config_to_request_params_base_fields() -> None:
    config = LiteLLMModelConfig(temperature=0.5, max_tokens=100)
    params = config.to_request_params()
    assert params == {"temperature": 0.5, "max_tokens": 100}


def test_litellm_model_config_to_request_params_litellm_specific() -> None:
    config = LiteLLMModelConfig(n=3, user="user-123")
    params = config.to_request_params()
    assert params == {"n": 3, "user": "user-123"}


def test_litellm_model_config_to_request_params_all_set() -> None:
    config = LiteLLMModelConfig(
        temperature=0.7,
        max_tokens=200,
        n=2,
        user="user-456",
    )
    params = config.to_request_params()
    assert params == {
        "temperature": 0.7,
        "max_tokens": 200,
        "n": 2,
        "user": "user-456",
    }


# merge_config_params tests


def test_merge_config_params_with_none_config() -> None:
    base = {"model": "gpt-4", "messages": []}
    result = merge_config_params(base, None)
    assert result == {"model": "gpt-4", "messages": []}


def test_merge_config_params_with_config() -> None:
    base = {"model": "gpt-4", "messages": []}
    config = LLMConfig(temperature=0.5, max_tokens=100)
    result = merge_config_params(base, config)
    assert result == {
        "model": "gpt-4",
        "messages": [],
        "temperature": 0.5,
        "max_tokens": 100,
    }


def test_merge_config_params_config_overrides_base() -> None:
    base = {"model": "gpt-4", "temperature": 1.0}
    config = LLMConfig(temperature=0.5)
    result = merge_config_params(base, config)
    assert result == {"model": "gpt-4", "temperature": 0.5}
