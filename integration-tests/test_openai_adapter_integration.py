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

"""Integration tests for the OpenAI adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters import OpenAIAdapter
from weakincentives.events import NullEventBus
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult

pytest.importorskip("openai")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set; skipping OpenAI integration tests.",
    ),
    pytest.mark.timeout(10),
]

_MODEL_ENV_VAR = "OPENAI_TEST_MODEL"
_DEFAULT_MODEL = "gpt-4.1"


@pytest.fixture(scope="module")
def openai_model() -> str:
    """Return the model name used for integration tests."""

    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def adapter(openai_model: str) -> OpenAIAdapter:
    """Create an OpenAI adapter instance for basic evaluations."""

    return OpenAIAdapter(model=openai_model)


@dataclass(slots=True)
class GreetingParams:
    """Prompt parameters for a greeting scenario."""

    audience: str


@dataclass(slots=True)
class TransformRequest:
    """Input sent to the uppercase helper tool."""

    text: str


@dataclass(slots=True)
class TransformResult:
    """Payload emitted by the uppercase helper tool."""

    text: str


@dataclass(slots=True)
class ReviewParams:
    """Prompt parameters for structured output verification."""

    text: str


@dataclass(slots=True)
class ReviewAnalysis:
    """Structured payload expected from the provider."""

    summary: str
    sentiment: str


def _build_greeting_prompt() -> Prompt:
    greeting_section = TextSection[GreetingParams](
        title="Greeting",
        body=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}."
        ),
    )
    return Prompt(name="greeting", sections=[greeting_section])


def _build_uppercase_tool() -> Tool[TransformRequest, TransformResult]:
    def uppercase_tool(params: TransformRequest) -> ToolResult[TransformResult]:
        transformed = params.text.upper()
        message = f"Transformed '{params.text}' to uppercase."
        return ToolResult(message=message, payload=TransformResult(text=transformed))

    return Tool[TransformRequest, TransformResult](
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=uppercase_tool,
    )


def _build_tool_prompt(
    tool: Tool[TransformRequest, TransformResult],
) -> Prompt:
    instruction_section = TextSection[TransformRequest](
        title="Instruction",
        body=(
            "You must call the `uppercase_text` tool exactly once using the "
            'payload {"text": "${text}"}. After the tool response is '
            "observed, reply to the user summarizing the uppercase text."
        ),
        tools=(tool,),
    )
    return Prompt(name="uppercase_workflow", sections=[instruction_section])


def _build_structured_prompt() -> Prompt[ReviewAnalysis]:
    analysis_section = TextSection[ReviewParams](
        title="Analysis Task",
        body=(
            "Review the provided passage and produce a concise summary and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Use only the available response schema and keep strings short."
        ),
    )
    return Prompt[ReviewAnalysis](
        name="structured_review",
        sections=[analysis_section],
    )


def test_openai_adapter_returns_text(adapter: OpenAIAdapter) -> None:
    prompt = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")

    response = adapter.evaluate(prompt, params, parse_output=False, bus=NullEventBus())

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    assert response.tool_results == ()


def test_openai_adapter_processes_tool_invocation(openai_model: str) -> None:
    tool = _build_uppercase_tool()
    prompt = _build_tool_prompt(tool)
    params = TransformRequest(text="integration tests")
    adapter = OpenAIAdapter(
        model=openai_model,
        tool_choice={"type": "function", "function": {"name": tool.name}},
    )

    response = adapter.evaluate(prompt, params, parse_output=False, bus=NullEventBus())

    assert response.prompt_name == "uppercase_workflow"
    assert response.tool_results, "Expected at least one tool invocation."

    first_call = response.tool_results[0]
    assert first_call.name == tool.name
    assert first_call.params.text
    assert first_call.result.payload.text == first_call.params.text.upper()
    assert first_call.result.message

    assert response.text is not None and response.text.strip()
    assert params.text.upper() in response.text


def test_openai_adapter_parses_structured_output(adapter: OpenAIAdapter) -> None:
    prompt = _build_structured_prompt()
    sample = ReviewParams(
        text=(
            "The new release shipped important bug fixes and improved the onboarding flow."
            " Early adopters report smoother setup, though some mention learning curves."
        ),
    )

    response = adapter.evaluate(prompt, sample, bus=NullEventBus())

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    assert response.text is None
