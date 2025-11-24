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
from typing import cast

import pytest

from tests.helpers.events import NullEventBus
from weakincentives.adapters.core import SessionProtocol
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import MarkdownSection, Prompt, Tool, ToolContext, ToolResult
from weakincentives.runtime.session import Session

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
_DEFAULT_MODEL = "gpt-5.1-codex"
_PROMPT_NS = "integration/openai"


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


@dataclass(slots=True)
class ReviewFinding:
    """Structured bullet point captured in array outputs."""

    summary: str
    sentiment: str


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
        name="greeting",
        sections=[greeting_section],
    )


def _build_uppercase_tool() -> Tool[TransformRequest, TransformResult]:
    def uppercase_tool(
        params: TransformRequest, *, context: ToolContext
    ) -> ToolResult[TransformResult]:
        del context
        transformed = params.text.upper()
        message = f"Transformed '{params.text}' to uppercase."
        return ToolResult(message=message, value=TransformResult(text=transformed))

    return Tool[TransformRequest, TransformResult](
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=uppercase_tool,
    )


def _build_tool_prompt(
    tool: Tool[TransformRequest, TransformResult],
) -> Prompt[object]:
    instruction_section = MarkdownSection[TransformRequest](
        title="Instruction",
        template=(
            "You must call the `uppercase_text` tool exactly once using the "
            'payload {"text": "${text}"}. After the tool response is '
            "observed, reply to the user summarizing the uppercase text."
        ),
        tools=(tool,),
        key="instruction",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-uppercase",
        name="uppercase_workflow",
        sections=[instruction_section],
    )


def _build_structured_prompt() -> Prompt[ReviewAnalysis]:
    analysis_section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce a concise summary and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Use only the available response schema and keep strings short."
        ),
        key="analysis-task",
    )
    return Prompt[ReviewAnalysis](
        ns=_PROMPT_NS,
        key="integration-structured",
        name="structured_review",
        sections=[analysis_section],
    )


def _build_structured_list_prompt() -> Prompt[list[ReviewFinding]]:
    analysis_section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce between one and two findings.\n"
            "Each finding must include a summary and sentiment label using the schema."
        ),
        key="analysis-task",
    )
    return Prompt[list[ReviewFinding]](
        ns=_PROMPT_NS,
        key="integration-structured-list",
        name="structured_review_list",
        sections=[analysis_section],
    )


def test_openai_adapter_returns_text(adapter: OpenAIAdapter) -> None:
    prompt = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")

    bus = NullEventBus()
    session = Session(bus=bus)
    response = adapter.evaluate(
        prompt, params, parse_output=False, bus=bus, session=session
    )

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

    bus = NullEventBus()
    session = Session(bus=bus)
    response = adapter.evaluate(
        prompt, params, parse_output=False, bus=bus, session=session
    )

    assert response.prompt_name == "uppercase_workflow"
    assert response.tool_results, "Expected at least one tool invocation."

    first_call = response.tool_results[0]
    assert first_call.name == tool.name
    call_params = cast(TransformRequest, first_call.params)
    call_result = cast(TransformResult, first_call.result.value)
    assert call_params.text
    assert call_result.text == call_params.text.upper()
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

    bus = NullEventBus()
    session = Session(bus=bus)
    response = adapter.evaluate(prompt, sample, bus=bus, session=session)

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    assert response.text is None


def test_openai_adapter_parses_structured_output_without_native_schema(
    openai_model: str,
) -> None:
    prompt = _build_structured_prompt()
    sample = ReviewParams(
        text=(
            "Customers praise the simplified dashboards and clearer metrics, "
            "though a few still flag onboarding friction when importing legacy data."
        ),
    )

    custom_adapter = OpenAIAdapter(
        model=openai_model,
        use_native_response_format=False,
    )

    bus = NullEventBus()
    session = Session(bus=bus)
    response = custom_adapter.evaluate(
        prompt,
        sample,
        bus=bus,
        session=cast(SessionProtocol, session),
    )

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    assert response.text is None


def test_openai_adapter_parses_structured_output_array(adapter: OpenAIAdapter) -> None:
    prompt = _build_structured_list_prompt()
    sample = ReviewParams(
        text=(
            "Feedback mentions strong improvements to documentation and onboarding flow, "
            "but some testers highlight occasional slow responses from the support channel."
        ),
    )

    bus = NullEventBus()
    session = Session(bus=bus)
    response = adapter.evaluate(
        prompt,
        sample,
        bus=bus,
        session=cast(SessionProtocol, session),
    )

    assert response.prompt_name == "structured_review_list"
    assert response.output is not None
    assert isinstance(response.output, list)
    assert response.output, "Expected at least one structured finding."
    for finding in response.output:
        assert isinstance(finding, ReviewFinding)
        assert finding.summary
        assert finding.sentiment
