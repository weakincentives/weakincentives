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
from typing import Any

from ..dataclasses import FrozenDataclass

__all__ = [
    "LLMConfig",
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
