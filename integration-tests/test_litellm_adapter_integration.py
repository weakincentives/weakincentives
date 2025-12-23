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
from typing import cast

import pytest

from weakincentives.adapters import LiteLLMClientConfig, LiteLLMModelConfig
from weakincentives.adapters.litellm import LiteLLMAdapter, create_litellm_completion
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
from weakincentives.runtime.session.protocols import SessionProtocol

pytest.importorskip("litellm")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.litellm,
    pytest.mark.skipif(
        "LITELLM_API_KEY" not in os.environ,
        reason="LITELLM_API_KEY not set; skipping LiteLLM integration tests.",
    ),
    pytest.mark.timeout(10),
]

_MODEL_ENV_VAR = "LITELLM_TEST_MODEL"
_DEFAULT_MODEL = "gpt-4o-mini"
_ADDITIONAL_KWARGS_ENV = "LITELLM_COMPLETION_KWARGS"
_PROMPT_NS = "integration/litellm"


@pytest.fixture(scope="module")
def litellm_model() -> str:
    """Return the model name used for LiteLLM integration tests."""

    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def completion_kwargs() -> dict[str, object]:
    """Build the completion kwargs passed to LiteLLM.

    Used when extra environment-based configuration is needed beyond what
    LiteLLMClientConfig supports.
    """
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
        kwargs.update(cast(dict[str, object], payload))

    return kwargs


@pytest.fixture(scope="module")
def completion_config() -> LiteLLMClientConfig:
    """Build a typed LiteLLMClientConfig from environment variables."""
    api_key = os.environ["LITELLM_API_KEY"]
    base_url = os.environ.get("LITELLM_BASE_URL")

    return LiteLLMClientConfig(
        api_key=api_key,
        api_base=base_url,
    )


@pytest.fixture(scope="module")
def adapter(litellm_model: str, completion_kwargs: dict[str, object]) -> LiteLLMAdapter:
    """Create a LiteLLM adapter instance for basic evaluations."""

    completion = create_litellm_completion(**completion_kwargs)
    return LiteLLMAdapter(
        model=litellm_model,
        completion=completion,
    )


@pytest.fixture(scope="module")
def adapter_with_typed_config(
    litellm_model: str, completion_config: LiteLLMClientConfig
) -> LiteLLMAdapter:
    """Create a LiteLLM adapter using typed configuration objects."""
    return LiteLLMAdapter(
        model=litellm_model,
        completion_config=completion_config,
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
        key="litellm-integration-greeting",
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
        key="litellm-integration-uppercase",
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
        key="litellm-integration-structured",
        name="structured_review",
        sections=[analysis_section],
    )


def _make_session_with_usage_tracking() -> Session:
    return Session()


def _assert_prompt_usage(session: Session) -> None:
    event = session[PromptExecuted].latest()
    assert event is not None, "Expected a PromptExecuted event."
    usage = event.usage
    assert usage is not None, "Expected token usage to be recorded."
    assert usage.total_tokens is not None and usage.total_tokens > 0


def test_litellm_adapter_returns_text(adapter: LiteLLMAdapter) -> None:
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="LiteLLM integration tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_litellm_adapter_executes_tools(adapter: LiteLLMAdapter) -> None:
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="LiteLLM integration")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert response.text is not None
    assert "LITELLM INTEGRATION" in response.text.upper()
    _assert_prompt_usage(session)


def test_litellm_adapter_parses_structured_output(adapter: LiteLLMAdapter) -> None:
    prompt_template = _build_structured_prompt()
    params = ReviewParams(text="Integration tests should remain deterministic.")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.text is None
    assert response.output.summary
    assert response.output.sentiment
    _assert_prompt_usage(session)


def test_litellm_adapter_with_typed_client_config(
    adapter_with_typed_config: LiteLLMAdapter,
) -> None:
    """Verify adapter instantiated with LiteLLMClientConfig works correctly."""
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="typed config integration tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter_with_typed_config.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_litellm_adapter_with_model_config(
    litellm_model: str, completion_config: LiteLLMClientConfig
) -> None:
    """Verify adapter with LiteLLMModelConfig applies model parameters."""
    model_config = LiteLLMModelConfig(
        temperature=0.3,
        max_tokens=150,
    )

    adapter = LiteLLMAdapter(
        model=litellm_model,
        completion_config=completion_config,
        model_config=model_config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="model config tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)
