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

from weakincentives import adapters, prompt, runtime
from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.prompt import Prompt
from weakincentives.runtime.logging import configure_logging


def test_example() -> None:
    assert 1 + 1 == 2


def test_package_dir_lists_public_symbols() -> None:
    symbols = runtime.__dir__()

    assert symbols == sorted(symbols)
    for symbol in ("configure_logging", "Session", "api"):
        assert symbol in symbols


def test_package_does_not_forward_api_attributes() -> None:
    assert adapters.PromptResponse is PromptResponse
    assert runtime.configure_logging is configure_logging
    assert prompt.Prompt is Prompt


def test_adapters_dir_lists_public_symbols() -> None:
    symbols = adapters.__dir__()

    assert symbols == sorted(symbols)
    for symbol in ("PromptEvaluationError", "PromptResponse", "ProviderAdapter"):
        assert hasattr(adapters, symbol)
        assert getattr(adapters, symbol) is getattr(adapters.api, symbol)


def test_namespace_forwarding() -> None:
    assert prompt.Prompt is Prompt
    assert adapters.PromptResponse is PromptResponse
    assert runtime.configure_logging is configure_logging
