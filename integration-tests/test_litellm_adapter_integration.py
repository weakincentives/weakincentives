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

"""Integration tests for the LiteLLM adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters import LiteLLMAdapter
from weakincentives.events import NullEventBus
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult

pytest.importorskip("litellm")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "LITELLM_API_KEY" not in os.environ,
        reason="LITELLM_API_KEY not set; skipping LiteLLM integration tests.",
    ),
    pytest.mark.timeout(10),
]

_MODEL_ENV_VAR = "LITELLM_TEST_MODEL"
_DEFAULT_MODEL = "gpt-4o-mini"
_ADDITIONAL_KWARGS_ENV = "LITELLM_COMPLETION_KWARGS"


@pytest.fixture(scope="module")
def litellm_model() -> str:
    """Return the model name used for LiteLLM integration tests."""

    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def completion_kwargs() -> dict[str, object]:
    """Build the completion kwargs passed to LiteLLM."""

    api_key = os.environ["LITELLM_API_KEY"]
    kwargs: dict[str, object] = {"api_key": api_key}

    base_url = os.environ.get("LITELLM_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url

    extra_kwargs = os.environ.get(_ADDITIONAL_KWARGS_ENV)
    if extra_kwargs:
        payload = json.loads(extra_kwargs)
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise TypeError(f"{_ADDITIONAL_KWARGS_ENV} must contain a JSON object.")
        kwargs.update(payload)

    return kwargs


@pytest.fixture(scope="module")
def adapter(litellm_model: str, completion_kwargs: dict[str, object]) -> LiteLLMAdapter:
    """Create a LiteLLM adapter instance for basic evaluations."""

    return LiteLLMAdapter(
        model=litellm_model,
        completion_kwargs=completion_kwargs,
    )


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
    return Prompt(
        key="litellm-integration-greeting",
        name="greeting",
        sections=[greeting_section],
    )


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
    return Prompt(
        key="litellm-integration-uppercase",
        name="uppercase_workflow",
        sections=[instruction_section],
    )


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
        key="litellm-integration-structured",
        name="structured_review",
        sections=[analysis_section],
    )


def test_litellm_adapter_returns_text(adapter: LiteLLMAdapter) -> None:
    prompt = _build_greeting_prompt()
    params = GreetingParams(audience="LiteLLM integration tests")

    response = adapter.evaluate(prompt, params, parse_output=False, bus=NullEventBus())

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    assert response.tool_results == ()


def test_litellm_adapter_executes_tools(adapter: LiteLLMAdapter) -> None:
    tool = _build_uppercase_tool()
    prompt = _build_tool_prompt(tool)
    params = TransformRequest(text="LiteLLM integration")

    response = adapter.evaluate(prompt, params, bus=NullEventBus())

    assert response.text is not None
    assert "LITELLM INTEGRATION" in response.text.upper()
    assert response.tool_results
    record = response.tool_results[0]
    assert record.name == "uppercase_text"
    assert isinstance(record.result.payload, TransformResult)
    assert record.result.payload.text == "LITELLM INTEGRATION"


def test_litellm_adapter_parses_structured_output(adapter: LiteLLMAdapter) -> None:
    prompt = _build_structured_prompt()
    params = ReviewParams(text="Integration tests should remain deterministic.")

    response = adapter.evaluate(prompt, params, bus=NullEventBus())

    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.text is None
    assert response.output.summary
    assert response.output.sentiment
