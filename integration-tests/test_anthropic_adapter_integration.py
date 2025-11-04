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

"""Integration tests for the Anthropic adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters.anthropic import AnthropicAdapter
from weakincentives.events import NullEventBus
from weakincentives.prompt import MarkdownSection, Prompt

pytest.importorskip("anthropic")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in os.environ,
        reason="ANTHROPIC_API_KEY not set; skipping Anthropic integration tests.",
    ),
    pytest.mark.timeout(10),
]

_MODEL_ENV_VAR = "ANTHROPIC_TEST_MODEL"
_DEFAULT_MODEL = "claude-sonnet-4-5"
_PROMPT_NS = "integration/anthropic"


@dataclass(slots=True)
class GreetingParams:
    """Prompt parameters for a greeting scenario."""

    audience: str


@pytest.fixture(scope="module")
def anthropic_model() -> str:
    """Return the Anthropic model used for integration tests."""

    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def adapter(anthropic_model: str) -> AnthropicAdapter:
    """Create an Anthropic adapter instance for basic evaluations."""

    return AnthropicAdapter(model=anthropic_model)


def _build_greeting_prompt() -> Prompt[object]:
    greeting_section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}."
        ),
        key="greeting",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-greeting",
        name="anthropic_greeting",
        sections=[greeting_section],
    )


def test_anthropic_adapter_returns_text(adapter: AnthropicAdapter) -> None:
    prompt = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")

    response = adapter.evaluate(prompt, params, parse_output=False, bus=NullEventBus())

    assert response.prompt_name == "anthropic_greeting"
    assert response.text is not None
    assert response.text.strip()
    assert response.tool_results == ()
