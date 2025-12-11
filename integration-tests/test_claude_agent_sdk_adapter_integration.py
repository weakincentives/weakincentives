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

"""Integration tests for the Claude Agent SDK adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
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

pytest.importorskip("claude_agent_sdk")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in os.environ,
        reason="ANTHROPIC_API_KEY not set; skipping Claude Agent SDK integration tests.",
    ),
    pytest.mark.timeout(60),  # SDK tests may take longer due to agentic execution
]

_MODEL_ENV_VAR = "CLAUDE_AGENT_SDK_TEST_MODEL"
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_PROMPT_NS = "integration/claude-agent-sdk"


@pytest.fixture(scope="module")
def claude_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def client_config() -> ClaudeAgentSDKClientConfig:
    """Build a typed ClaudeAgentSDKClientConfig from environment variables."""
    return ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
    )


@pytest.fixture(scope="module")
def adapter(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> ClaudeAgentSDKAdapter:
    """Create a Claude Agent SDK adapter instance for basic evaluations."""
    return ClaudeAgentSDKAdapter(
        model=claude_model,
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


def _build_greeting_prompt() -> PromptTemplate[object]:
    greeting_section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}. "
            "Reply in a single sentence without any tool calls."
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


def _make_session_with_usage_tracking() -> Session:
    return Session()


def _assert_prompt_usage(session: Session) -> None:
    event = session.query(PromptExecuted).latest()
    assert event is not None, "Expected a PromptExecuted event."
    usage = event.usage
    assert usage is not None, "Expected token usage to be recorded."
    assert usage.total_tokens is not None and usage.total_tokens > 0


def test_claude_agent_sdk_adapter_returns_text(adapter: ClaudeAgentSDKAdapter) -> None:
    """Test that the adapter returns text from a simple prompt."""
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_processes_tool_invocation(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Test that the adapter processes custom tool invocations via MCP bridge."""
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="integration tests")

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=client_config,
    )

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None and response.text.strip()
    # The uppercase text should appear in the response
    assert params.text.upper() in response.text
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_parses_structured_output(
    adapter: ClaudeAgentSDKAdapter,
) -> None:
    """Test that the adapter parses structured output correctly."""
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
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_with_model_config(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Verify adapter with ClaudeAgentSDKModelConfig applies model parameters."""
    model_config = ClaudeAgentSDKModelConfig(
        model=claude_model,
        temperature=0.3,
        max_tokens=150,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
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


def test_claude_agent_sdk_adapter_with_max_turns(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Verify adapter respects max_turns configuration."""
    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="max turns test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_with_disallowed_tools(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Verify adapter can be configured with disallowed tools."""
    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=client_config,
        disallowed_tools=("Bash", "Write"),
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="disallowed tools test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    _assert_prompt_usage(session)
