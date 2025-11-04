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

"""Integration tests for the Google Gemini adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters.google import GoogleGeminiAdapter
from weakincentives.events import NullEventBus
from weakincentives.prompt import MarkdownSection, Prompt, Tool, ToolResult

pytest.importorskip("google.genai")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "GOOGLE_API_KEY" not in os.environ,
        reason="GOOGLE_API_KEY not set; skipping Gemini integration tests.",
    ),
    pytest.mark.timeout(10),
]

_MODEL_ENV_VAR = "GOOGLE_GEMINI_TEST_MODEL"
_DEFAULT_MODEL = "gemini-2.5-flash"
_PROMPT_NS = "integration/google-gemini"


@pytest.fixture(scope="module")
def gemini_model() -> str:
    """Return the model name used for integration tests."""

    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def adapter(gemini_model: str) -> GoogleGeminiAdapter:
    """Create a Gemini adapter instance for the integration suite."""

    return GoogleGeminiAdapter(model=gemini_model)


@dataclass(slots=True)
class GreetingParams:
    audience: str


@dataclass(slots=True)
class TransformRequest:
    text: str


@dataclass(slots=True)
class TransformResult:
    text: str


@dataclass(slots=True)
class ReviewParams:
    text: str


@dataclass(slots=True)
class ReviewAnalysis:
    summary: str
    sentiment: str


@dataclass(slots=True)
class ReviewFinding:
    summary: str
    sentiment: str


def _build_greeting_prompt() -> Prompt[object]:
    section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=("You are a concise assistant. Greet ${audience} in one sentence."),
        key="greeting",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-greeting",
        name="greeting",
        sections=[section],
    )


def _build_uppercase_tool() -> Tool[TransformRequest, TransformResult]:
    def uppercase_tool(params: TransformRequest) -> ToolResult[TransformResult]:
        transformed = params.text.upper()
        message = f"Transformed '{params.text}' to uppercase."
        return ToolResult(message=message, value=TransformResult(text=transformed))

    return Tool(
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=uppercase_tool,
    )


def _build_tool_prompt(tool: Tool[TransformRequest, TransformResult]) -> Prompt[object]:
    section = MarkdownSection[TransformRequest](
        title="Instruction",
        template=(
            "You must call the `uppercase_text` tool exactly once using the payload "
            '{"text": "${text}"}. After receiving the tool response, summarise the '
            "uppercase text for the user."
        ),
        tools=(tool,),
        key="instruction",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-uppercase",
        name="uppercase_workflow",
        sections=[section],
    )


def _build_structured_prompt() -> Prompt[ReviewAnalysis]:
    section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce a short summary and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Only use the provided schema for the response."
        ),
        key="analysis-task",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-structured",
        name="structured_review",
        sections=[section],
    )


def _build_structured_list_prompt() -> Prompt[list[ReviewFinding]]:
    section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce between one and two findings.\n"
            "Each finding must include a summary and sentiment label."
        ),
        key="analysis-task",
    )
    return Prompt(
        ns=_PROMPT_NS,
        key="integration-structured-list",
        name="structured_review_list",
        sections=[section],
    )


def test_gemini_adapter_returns_text(adapter: GoogleGeminiAdapter) -> None:
    prompt = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")

    response = adapter.evaluate(prompt, params, parse_output=False, bus=NullEventBus())

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    assert response.tool_results == ()


def test_gemini_adapter_processes_tool_invocation(gemini_model: str) -> None:
    tool = _build_uppercase_tool()
    prompt = _build_tool_prompt(tool)
    adapter = GoogleGeminiAdapter(model=gemini_model)

    response = adapter.evaluate(
        prompt,
        TransformRequest(text="integration"),
        bus=NullEventBus(),
    )

    assert response.text is not None
    assert response.text.strip()
    assert len(response.tool_results) == 1


def test_gemini_adapter_emits_structured_output(adapter: GoogleGeminiAdapter) -> None:
    prompt = _build_structured_prompt()
    params = ReviewParams(text="The cafe had great coffee and friendly staff.")

    response = adapter.evaluate(prompt, params, bus=NullEventBus())

    assert response.output is not None
    assert response.text is None


def test_gemini_adapter_emits_structured_list(adapter: GoogleGeminiAdapter) -> None:
    prompt = _build_structured_list_prompt()
    params = ReviewParams(text="The cafe had great coffee and friendly staff.")

    response = adapter.evaluate(prompt, params, bus=NullEventBus())

    assert response.output is not None
    assert isinstance(response.output, list)
    assert response.text is None
