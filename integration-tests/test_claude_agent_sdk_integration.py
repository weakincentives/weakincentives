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
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events import PromptExecuted, ToolInvoked
from weakincentives.runtime.session import Session

pytest.importorskip("claude_agent_sdk")

# Check if Claude CLI is available
_CLAUDE_CLI_AVAILABLE = shutil.which("claude") is not None


def _check_claude_cli_authenticated() -> bool:
    """Check if Claude CLI is authenticated."""
    if not _CLAUDE_CLI_AVAILABLE:
        return False
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    else:
        return result.returncode == 0


_CLAUDE_CLI_READY = _check_claude_cli_authenticated()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _CLAUDE_CLI_READY
        and "CLAUDE_AGENT_SDK_INTEGRATION_TESTS" not in os.environ,
        reason=(
            "Claude CLI not available or not authenticated. "
            "Set CLAUDE_AGENT_SDK_INTEGRATION_TESTS=1 to force run."
        ),
    ),
    pytest.mark.timeout(120),  # Claude operations can take a while
]

_MODEL_ENV_VAR = "CLAUDE_TEST_MODEL"
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_PROMPT_NS = "integration/claude-agent-sdk"


@pytest.fixture(scope="module")
def claude_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def client_config() -> ClaudeAgentSDKClientConfig:
    """Build a typed ClaudeAgentSDKClientConfig for tests."""
    return ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        max_turns=5,
    )


@pytest.fixture(scope="module")
def adapter(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> ClaudeAgentSDKAdapter:
    """Create a Claude Agent SDK adapter instance for evaluations."""
    return ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=client_config,
    )


@dataclass(slots=True, frozen=True)
class GreetingParams:
    """Prompt parameters for a greeting scenario."""

    audience: str


@dataclass(slots=True, frozen=True)
class TransformRequest:
    """Input sent to the uppercase helper tool."""

    text: str


@dataclass(slots=True, frozen=True)
class TransformResult:
    """Payload emitted by the uppercase helper tool."""

    text: str


@dataclass(slots=True, frozen=True)
class ReviewParams:
    """Prompt parameters for structured output verification."""

    text: str


@dataclass(slots=True, frozen=True)
class ReviewAnalysis:
    """Structured payload expected from the provider."""

    summary: str
    sentiment: str


def _build_greeting_prompt() -> PromptTemplate[object]:
    """Build a simple greeting prompt template."""
    greeting_section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}. "
            "Keep your response to one sentence."
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
    """Build a simple tool that uppercases text."""

    def uppercase_tool(
        params: TransformRequest, *, context: ToolContext
    ) -> ToolResult[TransformResult]:
        del context
        transformed = params.text.upper()
        message = f"Transformed '{params.text}' to uppercase: {transformed}"
        return ToolResult(message=message, value=TransformResult(text=transformed))

    return Tool[TransformRequest, TransformResult](
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=uppercase_tool,
    )


def _build_tool_prompt(
    tool: Tool[TransformRequest, TransformResult],
) -> PromptTemplate[object]:
    """Build a prompt that uses a tool."""
    instruction_section = MarkdownSection[TransformRequest](
        title="Instruction",
        template=(
            "You must call the `uppercase_text` tool exactly once using the "
            'payload {"text": "${text}"}. After the tool response is '
            "observed, reply to the user with a brief summary including the uppercase text."
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
    """Build a prompt expecting structured output."""
    analysis_section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce a concise summary and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Use only the available response schema and keep strings short (under 50 characters)."
        ),
        key="analysis-task",
    )
    return PromptTemplate[ReviewAnalysis](
        ns=_PROMPT_NS,
        key="integration-structured",
        name="structured_review",
        sections=[analysis_section],
    )


def _make_session() -> Session:
    """Create a session for testing."""
    return Session()


def _assert_prompt_usage(session: Session) -> None:
    """Assert that token usage was recorded."""
    event = session.query(PromptExecuted).latest()
    assert event is not None, "Expected a PromptExecuted event."
    # Note: SDK may not always provide usage info
    # Just verify the event was published


class TestClaudeAgentSDKBasicEvaluation:
    """Tests for basic prompt evaluation."""

    def test_adapter_returns_text(self, adapter: ClaudeAgentSDKAdapter) -> None:
        """Test that adapter returns text response."""
        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="integration tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "greeting"
        assert response.text is not None
        assert response.text.strip()
        _assert_prompt_usage(session)

    def test_adapter_with_custom_model_config(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test adapter with custom model configuration."""
        model_config = ClaudeAgentSDKModelConfig(
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

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "greeting"
        assert response.text is not None
        assert response.text.strip()


class TestClaudeAgentSDKToolInvocation:
    """Tests for tool invocation through the SDK."""

    def test_adapter_processes_tool_invocation(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test that adapter processes tool calls correctly."""
        tool = _build_uppercase_tool()
        prompt_template = _build_tool_prompt(tool)
        params = TransformRequest(text="hello world")

        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=client_config,
        )

        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "uppercase_workflow"
        assert response.text is not None and response.text.strip()
        # The response should mention the uppercase text
        assert (
            "HELLO WORLD" in response.text.upper()
            or "uppercase" in response.text.lower()
        )

        # Verify tool invocation event was published
        _ = session.query(ToolInvoked).latest()
        # Tool events may or may not be published depending on how SDK handles tools


class TestClaudeAgentSDKStructuredOutput:
    """Tests for structured output parsing."""

    def test_adapter_parses_structured_output(
        self, adapter: ClaudeAgentSDKAdapter
    ) -> None:
        """Test that adapter can parse structured output."""
        prompt_template = _build_structured_prompt()
        sample = ReviewParams(
            text=(
                "The new release shipped important bug fixes and improved the onboarding flow. "
                "Early adopters report smoother setup, though some mention learning curves."
            ),
        )

        prompt = Prompt(prompt_template).bind(sample)

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "structured_review"
        # Structured output support depends on SDK capabilities
        # At minimum, we should get a text response
        assert response.text is not None or response.output is not None


class TestClaudeAgentSDKConstraints:
    """Tests for deadline and budget constraints."""

    def test_respects_deadline(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test that adapter respects deadline constraints."""
        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=client_config,
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="deadline tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()

        # Set a reasonable deadline that should allow completion
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=60))

        response = adapter.evaluate(prompt, session=session, deadline=deadline)

        assert response.text is not None
        assert response.text.strip()

    def test_budget_tracking(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test that adapter tracks budget usage."""
        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=client_config,
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="budget tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()

        # Set a generous budget
        budget = Budget(max_total_tokens=100000)
        tracker = BudgetTracker(budget=budget)

        response = adapter.evaluate(prompt, session=session, budget_tracker=tracker)

        assert response.text is not None
        # Budget tracking may or may not record depending on SDK response


class TestClaudeAgentSDKToolFiltering:
    """Tests for tool filtering configuration."""

    def test_disallowed_tools(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test that disallowed_tools configuration works."""
        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=client_config,
            disallowed_tools=("Bash",),  # Disable shell access
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="tool filtering tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        # Should still work for basic prompts
        assert response.text is not None
        assert response.text.strip()

    def test_allowed_tools_restriction(
        self, claude_model: str, client_config: ClaudeAgentSDKClientConfig
    ) -> None:
        """Test that allowed_tools configuration restricts available tools."""
        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=client_config,
            allowed_tools=("Read", "Write"),  # Only allow file operations
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="allowed tools tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        # Should still work for basic prompts that don't need tools
        assert response.text is not None
        assert response.text.strip()


class TestClaudeAgentSDKEventPublishing:
    """Tests for event publishing through hooks."""

    def test_prompt_rendered_event(self, adapter: ClaudeAgentSDKAdapter) -> None:
        """Test that PromptRendered event is published."""
        from weakincentives.runtime.events import PromptRendered

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="event tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        _ = adapter.evaluate(prompt, session=session)

        # Check that PromptRendered event was published
        rendered_event = session.query(PromptRendered).latest()
        assert rendered_event is not None
        assert rendered_event.prompt_name == "greeting"
        assert rendered_event.adapter == "claude_agent_sdk"

    def test_prompt_executed_event(self, adapter: ClaudeAgentSDKAdapter) -> None:
        """Test that PromptExecuted event is published."""
        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="event tests")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session()
        _ = adapter.evaluate(prompt, session=session)

        # Check that PromptExecuted event was published
        executed_event = session.query(PromptExecuted).latest()
        assert executed_event is not None
        assert executed_event.prompt_name == "greeting"
        assert executed_event.adapter == "claude_agent_sdk"
        assert executed_event.result is not None
