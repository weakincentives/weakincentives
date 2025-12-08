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
from typing import cast

import pytest

from weakincentives.adapters import AnthropicClientConfig, AnthropicModelConfig
from weakincentives.adapters.anthropic import AnthropicAdapter
from weakincentives.adapters.core import SessionProtocol
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events import PromptExecuted
from weakincentives.runtime.session import Session

pytest.importorskip("anthropic")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in os.environ,
        reason="ANTHROPIC_API_KEY not set; skipping Anthropic integration tests.",
    ),
    pytest.mark.timeout(30),
]
_MODEL_ENV_VAR = "ANTHROPIC_TEST_MODEL"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_PROMPT_NS = "integration/anthropic"


@pytest.fixture(scope="module")
def anthropic_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def client_config() -> AnthropicClientConfig:
    """Build a typed AnthropicClientConfig from environment variables."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    return AnthropicClientConfig(
        api_key=api_key,
        base_url=base_url,
    )


@pytest.fixture(scope="module")
def adapter(anthropic_model: str) -> AnthropicAdapter:
    """Create an Anthropic adapter instance for basic evaluations."""
    return AnthropicAdapter(model=anthropic_model)


@pytest.fixture(scope="module")
def adapter_with_typed_config(
    anthropic_model: str, client_config: AnthropicClientConfig
) -> AnthropicAdapter:
    """Create an Anthropic adapter using typed configuration objects."""
    return AnthropicAdapter(
        model=anthropic_model,
        client_config=client_config,
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


@dataclass(slots=True)
class ReviewFinding:
    """Structured bullet point captured in array outputs."""

    summary: str
    sentiment: str


def _build_greeting_prompt() -> PromptTemplate[object]:
    greeting_section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}."
        ),
        key="greeting",
    )
    return PromptTemplate(
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
) -> PromptTemplate[object]:
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
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-uppercase",
        name="uppercase_workflow",
        sections=[instruction_section],
    )


def _build_structured_prompt() -> PromptTemplate[ReviewAnalysis]:
    analysis_section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce a concise summary and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Use only the available response schema and keep strings short."
        ),
        key="analysis-task",
    )
    return PromptTemplate[ReviewAnalysis](
        ns=_PROMPT_NS,
        key="integration-structured",
        name="structured_review",
        sections=[analysis_section],
    )


def _build_structured_list_prompt() -> PromptTemplate[list[ReviewFinding]]:
    analysis_section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce between one and two findings.\n"
            "Each finding must include a summary and sentiment label using the schema."
        ),
        key="analysis-task",
    )
    return PromptTemplate[list[ReviewFinding]](
        ns=_PROMPT_NS,
        key="integration-structured-list",
        name="structured_review_list",
        sections=[analysis_section],
    )


def _make_session_with_usage_tracking() -> Session:
    return Session()


def _assert_prompt_usage(session: Session) -> None:
    event = session.query(PromptExecuted).latest()
    assert event is not None, "Expected a PromptExecuted event."
    usage = event.usage
    assert usage is not None, "Expected token usage to be recorded."
    assert usage.total_tokens is not None and usage.total_tokens > 0


def test_anthropic_adapter_returns_text(adapter: AnthropicAdapter) -> None:
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")
    prompt = Prompt(prompt_template).bind(params)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)
    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_anthropic_adapter_processes_tool_invocation(anthropic_model: str) -> None:
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="integration tests")
    adapter = AnthropicAdapter(
        model=anthropic_model,
        tool_choice={"type": "function", "function": {"name": tool.name}},
    )
    prompt = Prompt(prompt_template).bind(params)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)
    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None and response.text.strip()
    assert params.text.upper() in response.text
    _assert_prompt_usage(session)


def test_anthropic_adapter_parses_structured_output(adapter: AnthropicAdapter) -> None:
    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text=(
            "The new release shipped important bug fixes and improved the onboarding flow."
            " Early adopters report smoother setup, though some mention learning curves."
        ),
    )
    prompt = Prompt(prompt_template).bind(sample)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)
    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    assert response.text is None
    _assert_prompt_usage(session)


def test_anthropic_adapter_parses_structured_output_without_native_schema(
    anthropic_model: str,
) -> None:
    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text=(
            "Customers praise the simplified dashboards and clearer metrics, "
            "though a few still flag onboarding friction when importing legacy data."
        ),
    )
    prompt = Prompt(prompt_template).bind(sample)
    custom_adapter = AnthropicAdapter(
        model=anthropic_model,
        use_native_structured_output=False,
    )
    session = _make_session_with_usage_tracking()
    response = custom_adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )
    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    assert response.text is None
    _assert_prompt_usage(session)


def test_anthropic_adapter_parses_structured_output_array(
    adapter: AnthropicAdapter,
) -> None:
    prompt_template = _build_structured_list_prompt()
    sample = ReviewParams(
        text=(
            "Feedback mentions strong improvements to documentation and onboarding flow, "
            "but some testers highlight occasional slow responses from the support channel."
        ),
    )
    prompt = Prompt(prompt_template).bind(sample)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
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
    _assert_prompt_usage(session)


def test_anthropic_adapter_with_typed_client_config(
    adapter_with_typed_config: AnthropicAdapter,
) -> None:
    """Verify adapter instantiated with AnthropicClientConfig works correctly."""
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="typed config integration tests")
    prompt = Prompt(prompt_template).bind(params)
    session = _make_session_with_usage_tracking()
    response = adapter_with_typed_config.evaluate(
        prompt,
        session=session,
    )
    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_anthropic_adapter_with_model_config(
    anthropic_model: str, client_config: AnthropicClientConfig
) -> None:
    """Verify adapter with AnthropicModelConfig applies model parameters."""
    model_config = AnthropicModelConfig(
        temperature=0.3,
        max_tokens=150,
    )
    adapter = AnthropicAdapter(
        model=anthropic_model,
        client_config=client_config,
        model_config=model_config,
    )
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="model config tests")
    prompt = Prompt(prompt_template).bind(params)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=session,
    )
    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_anthropic_adapter_with_model_config_structured_output(
    anthropic_model: str, client_config: AnthropicClientConfig
) -> None:
    """Verify model_config works with structured output parsing."""
    model_config = AnthropicModelConfig(
        temperature=0.2,
        max_tokens=200,
    )
    adapter = AnthropicAdapter(
        model=anthropic_model,
        client_config=client_config,
        model_config=model_config,
    )
    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text="The typed configuration system provides better IDE support and validation."
    )
    prompt = Prompt(prompt_template).bind(sample)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )
    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    _assert_prompt_usage(session)


def test_anthropic_adapter_with_top_k_parameter(
    anthropic_model: str, client_config: AnthropicClientConfig
) -> None:
    """Verify Anthropic-specific top_k parameter works correctly."""
    model_config = AnthropicModelConfig(
        temperature=0.5,
        top_k=40,
        max_tokens=100,
    )
    adapter = AnthropicAdapter(
        model=anthropic_model,
        client_config=client_config,
        model_config=model_config,
    )
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="top_k parameter tests")
    prompt = Prompt(prompt_template).bind(params)
    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=session,
    )
    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)
