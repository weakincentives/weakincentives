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

"""Smoke tests to verify the test harness is wired correctly."""

from typing import Any, cast

import pytest

import weakincentives
from weakincentives import adapters, prompt, runtime, tools
from weakincentives.api import Prompt


def test_example() -> None:
    assert 1 + 1 == 2


def test_package_dir_lists_public_symbols() -> None:
    symbols = weakincentives.__dir__()

    assert symbols == sorted(symbols)
    assert "api" in symbols
    assert "prompt" in symbols
    assert "tools" in symbols


def test_package_does_not_forward_api_attributes() -> None:
    with pytest.raises(AttributeError):
        _ = cast(Any, weakincentives).Prompt


def test_adapters_dir_lists_public_symbols() -> None:
    symbols = adapters.__dir__()

    assert symbols == sorted(symbols)
    for symbol in ("PromptEvaluationError", "PromptResponse", "ProviderAdapter"):
        assert hasattr(adapters, symbol)
        assert getattr(adapters, symbol) is getattr(adapters.api, symbol)


def test_namespace_forwarding() -> None:
    runtime_session = cast(Any, runtime.api).Session
    tools_plan = cast(Any, tools.api).Plan
    adapters_response = cast(Any, adapters.api).PromptResponse

    prompt_api = cast(Any, prompt).api

    assert Prompt is prompt_api.Prompt

    assert cast(Any, runtime).Session is runtime_session
    assert cast(Any, tools).Plan is tools_plan
    assert cast(Any, adapters).PromptResponse is adapters_response


def test_runtime_exports_match_api() -> None:
    assert cast(Any, runtime).Session is cast(Any, runtime.api).Session


def test_adapters_exports_match_api() -> None:
    assert cast(Any, adapters).PromptResponse is cast(Any, adapters.api).PromptResponse


def test_api_dir_helpers() -> None:
    assert "Prompt" in weakincentives.api.__dir__()
    assert "PromptResponse" in adapters.api.__dir__()
    assert "api" in prompt.__dir__()
    assert "api" in tools.__dir__()
