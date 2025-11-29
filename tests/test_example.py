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
import weakincentives.prompt.tool as prompt_tool
import weakincentives.runtime.logging as runtime_logging
import weakincentives.tools.errors as tool_errors
from weakincentives import adapters, prompt, runtime, tools
from weakincentives.prompt import Prompt


def test_example() -> None:
    assert 1 + 1 == 2


def test_package_dir_lists_public_symbols() -> None:
    symbols = weakincentives.__dir__()

    assert symbols == sorted(symbols)
    assert "api" not in symbols
    assert {"prompt", "tools", "runtime", "adapters"}.issubset(symbols)


def test_package_does_not_forward_api_attributes() -> None:
    assert not hasattr(weakincentives, "Prompt")


def test_adapters_dir_lists_public_symbols() -> None:
    symbols = adapters.__dir__()

    assert symbols == sorted(symbols)
    assert {"_names", "core", "litellm", "openai", "shared"}.issubset(symbols)


def test_namespace_forwarding() -> None:
    assert Prompt is prompt.Prompt
    assert prompt.ToolHandler is prompt_tool.ToolHandler

    assert runtime.configure_logging is runtime_logging.configure_logging
    assert runtime.StructuredLogger is runtime_logging.StructuredLogger

    assert tools.DeadlineExceededError is tool_errors.DeadlineExceededError
    assert tools.ToolValidationError is tool_errors.ToolValidationError
