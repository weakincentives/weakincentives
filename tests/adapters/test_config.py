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

from weakincentives.adapters.config import (
    LLMConfig,
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
