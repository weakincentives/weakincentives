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

import weakincentives
from weakincentives import adapters


def test_example() -> None:
    assert 1 + 1 == 2


def test_package_dir_lists_public_symbols() -> None:
    symbols = weakincentives.__dir__()

    assert "Prompt" in symbols
    assert "tools" in symbols


def test_adapters_dir_lists_public_symbols() -> None:
    symbols = adapters.__dir__()

    assert "PromptEvaluationError" in symbols
    assert "PromptResponse" in symbols
    assert "ProviderAdapter" in symbols
    assert "OpenAIAdapter" not in symbols
    assert "LiteLLMAdapter" not in symbols
