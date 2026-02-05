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
from weakincentives import Prompt, adapters, contrib, prompt, runtime


def test_example() -> None:
    assert 1 + 1 == 2


def test_package_dir_lists_public_symbols() -> None:
    symbols = weakincentives.__dir__()

    assert symbols == sorted(symbols)
    assert "Prompt" in symbols
    assert "prompt" in symbols
    assert "contrib" in symbols


def test_package_exposes_public_attributes() -> None:
    assert weakincentives.Prompt is prompt.Prompt is Prompt


def test_adapters_dir_lists_public_symbols() -> None:
    symbols = adapters.__dir__()

    assert symbols == sorted(symbols)
    for symbol in ("PromptEvaluationError", "PromptResponse", "ProviderAdapter"):
        assert hasattr(adapters, symbol)
        assert getattr(adapters, symbol) is getattr(adapters, symbol)


def test_public_namespaces_resolve_symbols() -> None:
    assert runtime.Session is runtime.Session
    assert contrib.tools.WorkspaceDigestSection is contrib.tools.WorkspaceDigestSection
    assert prompt.MarkdownSection is prompt.MarkdownSection


def test_submodule_dir_lists_exports() -> None:
    prompt_symbols = prompt.__dir__()
    tools_symbols = contrib.tools.__dir__()

    assert prompt_symbols == sorted(prompt_symbols)
    assert "Prompt" in prompt_symbols
    assert tools_symbols == sorted(tools_symbols)
    assert "WorkspaceDigestSection" in tools_symbols
