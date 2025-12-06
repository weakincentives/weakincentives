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

"""Typed configuration objects for LLM adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from ..dataclasses import FrozenDataclass

__all__ = [
    "AnthropicClientConfig",
    "AnthropicModelConfig",
    "LLMConfig",
    "LiteLLMClientConfig",
    "LiteLLMModelConfig",
    "OpenAIClientConfig",
    "OpenAIModelConfig",
]


@FrozenDataclass()
class LLMConfig:
    """Base configuration for LLM model parameters.

    These parameters are common across most LLM providers and control
    the generation behavior of the model.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Higher values increase
            randomness. None uses the provider default.
        max_tokens: Maximum tokens to generate. None uses the provider default.
        top_p: Nucleus sampling probability mass (0.0-1.0). None uses the
            provider default.
        presence_penalty: Penalty for token presence (-2.0-2.0). Positive values
            discourage repetition. None uses the provider default.
        frequency_penalty: Penalty for token frequency (-2.0-2.0). Positive
            values discourage frequent tokens. None uses the provider default.
        stop: Sequences where the model stops generating. None means no
            stop sequences.
        seed: Random seed for deterministic sampling. None means non-deterministic.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None

    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to request parameters."""
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.stop is not None:
            params["stop"] = list(self.stop)
        if self.seed is not None:
            params["seed"] = self.seed
        return params


@FrozenDataclass()
class OpenAIClientConfig:
    """Configuration for OpenAI client instantiation.

    These parameters are passed to the OpenAI client constructor.

    Attributes:
        api_key: OpenAI API key. None uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for API requests. None uses the default OpenAI endpoint.
        organization: OpenAI organization ID. None uses no organization header.
        timeout: Request timeout in seconds. None uses the client default.
        max_retries: Maximum number of retries. None uses the client default.
    """

    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout: float | None = None
    max_retries: int | None = None

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to client constructor kwargs."""
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.organization is not None:
            kwargs["organization"] = self.organization
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs


@FrozenDataclass()
class OpenAIModelConfig(LLMConfig):
    """OpenAI-specific model configuration.

    Extends LLMConfig with parameters specific to OpenAI's API.

    Attributes:
        logprobs: Whether to return log probabilities. None uses the provider default.
        top_logprobs: Number of top log probabilities to return (0-20). Requires
            logprobs=True. None uses the provider default.
        parallel_tool_calls: Whether to allow parallel tool calls. Defaults to True.
        store: Whether to store the conversation for fine-tuning. None uses the
            provider default.
        user: Unique identifier for the end-user. None omits the field.

    Notes:
        The OpenAI Responses API does not accept ``seed``, ``stop``,
        ``presence_penalty``, or ``frequency_penalty``. If any of these fields
        are provided, ``OpenAIModelConfig`` raises ``ValueError`` so callers fail
        fast instead of issuing an invalid request.
    """

    logprobs: bool | None = None
    top_logprobs: int | None = None
    parallel_tool_calls: bool | None = True
    store: bool | None = None
    user: str | None = None

    def __post_init__(self) -> None:
        unsupported: dict[str, object | None] = {
            "seed": self.seed,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        set_unsupported = [
            key for key, value in unsupported.items() if value is not None
        ]
        if set_unsupported:
            raise ValueError(
                "Unsupported OpenAI Responses parameters: "
                + ", ".join(sorted(set_unsupported))
                + ". Remove them from OpenAIModelConfig."
            )

    @override
    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to request parameters.

        The Responses API uses ``max_output_tokens`` instead of ``max_tokens``,
        so this override renames the key accordingly.
        """
        params: dict[str, Any] = {}

        # Supported core fields
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p

        # OpenAI Responses-specific fields
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            params["top_logprobs"] = self.top_logprobs
        if self.parallel_tool_calls is not None:
            params["parallel_tool_calls"] = self.parallel_tool_calls
        if self.store is not None:
            params["store"] = self.store
        if self.user is not None:
            params["user"] = self.user

        # The Responses API currently does not support ``seed``, ``stop``,
        # ``presence_penalty``, or ``frequency_penalty``. They are intentionally
        # omitted to keep request payloads aligned with the SDK surface.

        return params


@FrozenDataclass()
class LiteLLMClientConfig:
    """Configuration for LiteLLM client instantiation.

    These parameters are merged into LiteLLM completion calls.

    Attributes:
        api_key: API key for the underlying provider. None uses environment variables.
        api_base: Base URL for API requests. None uses the provider default.
        timeout: Request timeout in seconds. None uses the client default.
        num_retries: Number of retries on failure. None uses the client default.
    """

    api_key: str | None = None
    api_base: str | None = None
    timeout: float | None = None
    num_retries: int | None = None

    def to_completion_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to completion kwargs."""
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.num_retries is not None:
            kwargs["num_retries"] = self.num_retries
        return kwargs


@FrozenDataclass()
class LiteLLMModelConfig(LLMConfig):
    """LiteLLM-specific model configuration.

    Extends LLMConfig with parameters that LiteLLM passes through to
    the underlying provider.

    Attributes:
        n: Number of completions to generate. None uses the provider default.
        user: Unique identifier for the end-user. None omits the field.
    """

    n: int | None = None
    user: str | None = None

    @override
    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to request parameters."""
        params = LLMConfig.to_request_params(self)
        if self.n is not None:
            params["n"] = self.n
        if self.user is not None:
            params["user"] = self.user
        return params


@FrozenDataclass()
class AnthropicClientConfig:
    """Configuration for Anthropic client instantiation.

    Attributes:
        api_key: Anthropic API key. None uses the ANTHROPIC_API_KEY environment variable.
        base_url: Base URL for API requests. None uses the default Anthropic endpoint.
        timeout: Request timeout in seconds. None uses the client default.
        max_retries: Maximum number of retries. None uses the client default.
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to client constructor kwargs."""
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs


@FrozenDataclass()
class AnthropicModelConfig(LLMConfig):
    """Anthropic-specific model configuration.

    Extends LLMConfig with parameters specific to Anthropic's API.

    Attributes:
        top_k: Sample from the top K most likely tokens. None uses the provider default.
        metadata: Optional metadata to include with the request.

    Notes:
        Anthropic does not support ``presence_penalty``, ``frequency_penalty``, or ``seed``.
        If any of these fields are provided, ``AnthropicModelConfig`` raises ``ValueError``
        so callers fail fast instead of issuing an invalid request.
    """

    top_k: int | None = None
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        unsupported: dict[str, object | None] = {
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        set_unsupported = [
            key for key, value in unsupported.items() if value is not None
        ]
        if set_unsupported:
            raise ValueError(
                "Unsupported Anthropic parameters: "
                + ", ".join(sorted(set_unsupported))
                + ". Remove them from AnthropicModelConfig."
            )

    @override
    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to request parameters."""
        params: dict[str, Any] = {}

        # Supported core fields
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop_sequences"] = list(self.stop)

        # Anthropic-specific fields
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.metadata is not None:
            params["metadata"] = dict(self.metadata)

        return params


def merge_config_params(
    base: Mapping[str, Any],
    config: LLMConfig | None,
) -> dict[str, Any]:
    """Merge config parameters into a base request payload.

    Parameters from config override those in base when both are present.
    """
    result = dict(base)
    if config is not None:
        result.update(config.to_request_params())
    return result
