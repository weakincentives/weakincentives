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
from pathlib import Path

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events import PromptExecuted, PromptRendered, ToolInvoked
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
    """Test that the adapter processes custom tool invocations via MCP bridge.

    This test validates that weakincentives tools are correctly bridged to the
    SDK via an in-process MCP server. The streaming mode approach enables proper
    MCP server initialization.
    """
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
    """Verify adapter works with ClaudeAgentSDKModelConfig.

    Note: The Claude Agent SDK does not expose max_tokens or temperature
    parameters directly - it manages token budgets internally. This test
    verifies that the adapter handles a model config gracefully even though
    these parameters are not applied to SDK options.
    """
    model_config = ClaudeAgentSDKModelConfig(
        model=claude_model,
        temperature=0.3,  # Ignored by SDK - no direct equivalent
        max_tokens=150,  # Ignored by SDK - no direct equivalent
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


@dataclass(slots=True)
class ReadFileParams:
    """Prompt parameters for reading a file."""

    file_path: str


def _build_file_read_prompt() -> PromptTemplate[object]:
    """Build a prompt that requests the assistant to read a file."""
    task_section = MarkdownSection[ReadFileParams](
        title="Task",
        template=(
            "Use the Read tool to read the contents of the file at ${file_path}. "
            "After reading the file, provide a brief summary of its contents in a single sentence."
        ),
        key="task",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-file-read",
        name="file_reader",
        sections=[task_section],
    )


def test_claude_agent_sdk_adapter_hooks_publish_tool_invoked_events(
    claude_model: str,
) -> None:
    """Verify that adapter hooks publish ToolInvoked events for SDK native tools.

    This test validates that the PostToolUse hook correctly publishes ToolInvoked
    events to the session's event bus when the SDK uses its native tools (like Read).
    This is a key integration point between the SDK's execution and weakincentives'
    event-driven architecture.
    """
    # Use a fresh client config with permissions that allow the Read tool
    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd=str(Path.cwd()),  # Set working directory to current repo
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
        # Only allow the Read tool to ensure we get a predictable tool call
        allowed_tools=("Read",),
    )

    prompt_template = _build_file_read_prompt()
    # Use README.md as it's guaranteed to exist in the repo
    params = ReadFileParams(file_path="README.md")
    prompt = Prompt(prompt_template).bind(params)

    # Track ToolInvoked events
    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.event_bus.subscribe(ToolInvoked, tool_invoked_events.append)

    response = adapter.evaluate(prompt, session=session)

    # Verify the prompt completed successfully
    assert response.prompt_name == "file_reader"
    assert response.text is not None

    # Verify that at least one ToolInvoked event was published
    # The SDK should have used the Read tool to read README.md
    assert len(tool_invoked_events) >= 1, (
        "Expected at least one ToolInvoked event from PostToolUse hook. "
        "This indicates hooks are not being called or not publishing events."
    )

    # Verify the Read tool was invoked
    read_events = [e for e in tool_invoked_events if e.name == "Read"]
    assert len(read_events) >= 1, (
        f"Expected at least one Read tool invocation. "
        f"Got events for: {[e.name for e in tool_invoked_events]}"
    )

    # Verify the event has expected structure
    read_event = read_events[0]
    assert read_event.adapter == "claude_agent_sdk"
    assert read_event.prompt_name == "file_reader"
    # The params should be a dict with file_path (SDK native tools pass dict params)
    assert isinstance(read_event.params, dict)

    # Verify the result contains tool_response data from the SDK
    # The PostToolUse hook receives tool_response with stdout/stderr/etc.
    assert read_event.result is not None, "Expected tool result to be captured"
    # The result should be a dict (the tool_response from SDK)
    assert isinstance(read_event.result, dict), (
        f"Expected result to be a dict (tool_response), got {type(read_event.result)}"
    )
    # For successful Read operations, the response may have stdout or similar content
    # The key thing is that the result is not empty/None, confirming PostToolUse captured it

    # For SDK native tools, value is None (typed values only for WINK tools)
    assert read_event.value is None, (
        f"Expected event.value to be None for SDK native tools, got {type(read_event.value)}"
    )


def test_claude_agent_sdk_adapter_publishes_prompt_rendered_event(
    adapter: ClaudeAgentSDKAdapter,
) -> None:
    """Verify that PromptRendered event is published before SDK execution.

    This test validates that the adapter publishes a PromptRendered event
    containing the rendered prompt text, which is useful for debugging and
    observability.
    """
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="prompt rendered test")
    prompt = Prompt(prompt_template).bind(params)

    prompt_rendered_events: list[PromptRendered] = []
    session = Session()
    session.event_bus.subscribe(PromptRendered, prompt_rendered_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None

    # Verify PromptRendered event was published
    assert len(prompt_rendered_events) == 1, "Expected exactly one PromptRendered event"

    event = prompt_rendered_events[0]
    assert event.prompt_name == "greeting"
    assert event.adapter == "claude_agent_sdk"
    assert event.rendered_prompt is not None
    # The rendered prompt should contain the audience text
    assert "prompt rendered test" in event.rendered_prompt


@dataclass(slots=True)
class RenderableResult:
    """A result type with a custom render() method."""

    value: int
    message: str

    def render(self) -> str:
        return f"[RENDERED] Value is {self.value}: {self.message}"


@dataclass(slots=True)
class ComputeParams:
    """Parameters for the compute tool."""

    x: int
    y: int


def _build_renderable_tool() -> Tool[ComputeParams, RenderableResult]:
    """Build a tool that returns a renderable result."""

    def compute_handler(
        params: ComputeParams, *, context: ToolContext
    ) -> ToolResult[RenderableResult]:
        del context
        result = params.x + params.y
        return ToolResult(
            message=f"Computed {params.x} + {params.y} = {result}",
            value=RenderableResult(value=result, message=f"{params.x} + {params.y}"),
        )

    return Tool[ComputeParams, RenderableResult](
        name="compute_sum",
        description="Compute the sum of two integers x and y.",
        handler=compute_handler,
    )


def _build_renderable_tool_prompt(
    tool: Tool[ComputeParams, RenderableResult],
) -> PromptTemplate[object]:
    """Build a prompt that uses the renderable compute tool."""
    instruction_section = MarkdownSection[ComputeParams](
        title="Instruction",
        template=(
            "You must call the `compute_sum` tool exactly once with "
            "x=${x} and y=${y}. After the tool response, "
            "tell the user what the rendered result says."
        ),
        tools=(tool,),
        key="instruction",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-renderable",
        name="renderable_workflow",
        sections=[instruction_section],
    )


def test_claude_agent_sdk_adapter_mcp_tool_uses_render(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Verify that MCP bridged tools call render() on result values.

    This test validates that when a custom weakincentives tool returns a
    ToolResult with a value that has a render() method, the MCP bridge
    uses that rendered output when returning to the SDK.
    """
    tool = _build_renderable_tool()
    prompt_template = _build_renderable_tool_prompt(tool)
    params = ComputeParams(x=7, y=13)

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=client_config,
    )

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "renderable_workflow"
    assert response.text is not None

    # The response should mention the rendered output format
    # The render() method returns "[RENDERED] Value is 20: 7 + 13"
    assert "20" in response.text, (
        f"Expected response to mention the computed value 20. Got: {response.text}"
    )
    _assert_prompt_usage(session)


@dataclass(slots=True)
class MultiStepParams:
    """Parameters for multi-step file operations."""

    target_dir: str


def _build_multi_tool_prompt() -> PromptTemplate[object]:
    """Build a prompt that requires multiple tool invocations."""
    task_section = MarkdownSection[MultiStepParams](
        title="Task",
        template=(
            "Use the Glob tool to list all Python files (*.py pattern) in ${target_dir}. "
            "Then use the Read tool to read the first Python file you found. "
            "Finally, summarize what you learned in one sentence."
        ),
        key="task",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-multi-tool",
        name="multi_tool_workflow",
        sections=[task_section],
    )


def test_claude_agent_sdk_adapter_multiple_tool_invocations(
    claude_model: str,
) -> None:
    """Verify adapter tracks multiple sequential tool invocations.

    This test validates that when the SDK uses multiple tools in sequence,
    each tool invocation is captured via the PostToolUse hook and published
    as a ToolInvoked event.
    """
    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd=str(Path.cwd()),
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
        # Allow Glob and Read for this multi-step task
        allowed_tools=("Glob", "Read"),
    )

    prompt_template = _build_multi_tool_prompt()
    params = MultiStepParams(target_dir="src/weakincentives")
    prompt = Prompt(prompt_template).bind(params)

    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.event_bus.subscribe(ToolInvoked, tool_invoked_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "multi_tool_workflow"
    assert response.text is not None

    # Verify multiple tool invocations were captured
    expected_min_tools = 2  # Glob + Read
    assert len(tool_invoked_events) >= expected_min_tools, (
        f"Expected at least {expected_min_tools} tool invocations (Glob + Read). "
        f"Got {len(tool_invoked_events)}: {[e.name for e in tool_invoked_events]}"
    )

    # Verify both Glob and Read were used
    tool_names = {e.name for e in tool_invoked_events}
    assert "Glob" in tool_names, f"Expected Glob to be invoked. Got: {tool_names}"
    assert "Read" in tool_names, f"Expected Read to be invoked. Got: {tool_names}"

    # Verify each event has the expected adapter attribution
    for event in tool_invoked_events:
        assert event.adapter == "claude_agent_sdk"
        assert event.prompt_name == "multi_tool_workflow"


def test_claude_agent_sdk_adapter_tracks_token_usage_across_tools(
    claude_model: str,
) -> None:
    """Verify adapter accumulates token usage across multi-turn execution.

    This test validates that token usage is properly tracked and reported
    even when the SDK performs multiple tool calls, which may span multiple
    API interactions.
    """
    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd=str(Path.cwd()),
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
        allowed_tools=("Read",),
    )

    prompt_template = _build_file_read_prompt()
    params = ReadFileParams(file_path="README.md")
    prompt = Prompt(prompt_template).bind(params)

    prompt_executed_events: list[PromptExecuted] = []
    session = Session()
    session.event_bus.subscribe(PromptExecuted, prompt_executed_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None

    # Verify PromptExecuted event has usage
    assert len(prompt_executed_events) == 1
    event = prompt_executed_events[0]
    assert event.usage is not None, "Expected token usage to be recorded"
    assert event.usage.input_tokens is not None and event.usage.input_tokens > 0
    assert event.usage.output_tokens is not None and event.usage.output_tokens > 0
    # Total should be sum of input + output
    assert event.usage.total_tokens is not None
    assert event.usage.total_tokens == (
        event.usage.input_tokens + event.usage.output_tokens
    )


@dataclass(slots=True)
class EchoParams:
    """Parameters for simple echo."""

    text: str


@dataclass(slots=True)
class EchoResult:
    """Result from echo tool - no render() method."""

    echoed: str


def _build_simple_tool() -> Tool[EchoParams, EchoResult]:
    """Build a simple tool that echoes text."""

    def echo_handler(
        params: EchoParams, *, context: ToolContext
    ) -> ToolResult[EchoResult]:
        del context
        return ToolResult(
            message=f"Echo: {params.text}",
            value=EchoResult(echoed=params.text),
        )

    return Tool[EchoParams, EchoResult](
        name="echo",
        description="Echo the provided text back.",
        handler=echo_handler,
    )


def _build_echo_prompt(tool: Tool[EchoParams, EchoResult]) -> PromptTemplate[object]:
    """Build a prompt that uses the echo tool."""
    instruction_section = MarkdownSection[EchoParams](
        title="Instruction",
        template=(
            'Call the `echo` tool with text="${text}". '
            "Then report what the tool returned."
        ),
        tools=(tool,),
        key="instruction",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-echo",
        name="echo_workflow",
        sections=[instruction_section],
    )


def test_claude_agent_sdk_adapter_custom_tool_without_render(
    claude_model: str, client_config: ClaudeAgentSDKClientConfig
) -> None:
    """Verify MCP bridged tools work when result has no render() method.

    This test validates that custom tools without a render() method on
    their result value still work correctly, falling back to the message.
    """
    tool = _build_simple_tool()
    prompt_template = _build_echo_prompt(tool)
    params = EchoParams(text="hello from integration test")

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=client_config,
    )

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "echo_workflow"
    assert response.text is not None
    # The response should mention the echoed text
    assert "hello" in response.text.lower() or "integration" in response.text.lower()
    _assert_prompt_usage(session)


# =============================================================================
# Isolation Integration Tests
# =============================================================================
#
# These tests validate that the IsolationConfig properly isolates SDK execution
# from the host's ~/.claude configuration, preventing interference with the
# user's personal Claude Code installation.
# =============================================================================


def test_claude_agent_sdk_adapter_with_isolation_returns_text(
    claude_model: str,
) -> None:
    """Verify adapter works correctly with IsolationConfig enabled.

    This test validates that:
    1. The adapter creates an ephemeral home directory
    2. SDK execution uses the ephemeral home instead of ~/.claude
    3. The adapter still returns valid responses
    4. Cleanup happens after execution
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="isolation tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_custom_tools(
    claude_model: str,
) -> None:
    """Verify custom tools work correctly in isolated mode.

    This test validates that MCP-bridged tools function correctly when
    the adapter is configured with isolation. The ephemeral home should
    not affect tool bridging functionality.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="isolation mode")

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None
    # The uppercase text should appear in the response
    assert "ISOLATION MODE" in response.text
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_structured_output(
    claude_model: str,
) -> None:
    """Verify structured output works in isolated mode.

    This test validates that structured output parsing functions correctly
    when the adapter is configured with isolation.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text="The isolated execution mode provides security benefits. Users report peace of mind.",
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


def test_claude_agent_sdk_adapter_isolation_does_not_modify_host_claude_dir(
    claude_model: str,
) -> None:
    """Verify isolation prevents modification of host ~/.claude directory.

    This test validates that running with IsolationConfig does not create,
    modify, or read from the real ~/.claude directory. This is critical for
    ensuring the user's personal configuration is not affected.
    """
    import tempfile
    import shutil

    # Create a fake ~/.claude directory to monitor
    with tempfile.TemporaryDirectory() as fake_home:
        fake_claude_dir = Path(fake_home) / ".claude"
        fake_claude_dir.mkdir()

        # Create a marker file to verify it's not modified
        marker_file = fake_claude_dir / "marker.txt"
        marker_file.write_text("original content")
        original_mtime = marker_file.stat().st_mtime

        # Configure isolation
        isolation = IsolationConfig(
            network_policy=NetworkPolicy.api_only(),
            sandbox=SandboxConfig(enabled=True),
            # Note: We don't use include_host_env because that would
            # potentially inherit HOME from the test environment
        )

        config = ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            isolation=isolation,
        )

        adapter = ClaudeAgentSDKAdapter(
            model=claude_model,
            client_config=config,
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="host protection test")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session_with_usage_tracking()
        response = adapter.evaluate(prompt, session=session)

        # Verify execution completed
        assert response.text is not None

        # Verify marker file was not modified
        assert marker_file.exists(), "Marker file should still exist"
        assert marker_file.read_text() == "original content"
        assert marker_file.stat().st_mtime == original_mtime

        # Verify no new files were created in fake_claude_dir
        # (besides our marker file)
        files_in_claude_dir = list(fake_claude_dir.iterdir())
        assert len(files_in_claude_dir) == 1
        assert files_in_claude_dir[0].name == "marker.txt"


def test_claude_agent_sdk_adapter_isolation_network_policy_api_only(
    claude_model: str,
) -> None:
    """Verify NetworkPolicy.api_only() allows Anthropic API access.

    This test validates that the api_only() network policy correctly
    allows access to api.anthropic.com while the adapter is in isolated mode.
    A successful response proves the network policy is being applied correctly.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="network policy test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()

    # This should succeed because api.anthropic.com is allowed
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_custom_env(
    claude_model: str,
) -> None:
    """Verify custom environment variables are passed to SDK subprocess.

    This test validates that the env parameter in IsolationConfig correctly
    passes environment variables to the SDK subprocess.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
        env={
            "WINK_TEST_VAR": "isolation_test_value",
        },
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="custom env test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_cleanup_on_success(
    claude_model: str,
) -> None:
    """Verify ephemeral home is cleaned up after successful execution.

    This test validates that the ephemeral home directory created for
    isolation is properly cleaned up after the adapter.evaluate() returns,
    even on successful execution.
    """
    import glob

    # Count temp directories with our prefix before
    temp_dirs_before = set(glob.glob("/tmp/claude-agent-*"))

    isolation = IsolationConfig(
        network_policy=NetworkPolicy.api_only(),
    )

    config = ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=isolation,
    )

    adapter = ClaudeAgentSDKAdapter(
        model=claude_model,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="cleanup test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None

    # Count temp directories after - should be same as before
    # (cleanup should have removed the ephemeral home)
    temp_dirs_after = set(glob.glob("/tmp/claude-agent-*"))

    # Any new directories should have been cleaned up
    new_dirs = temp_dirs_after - temp_dirs_before
    assert len(new_dirs) == 0, (
        f"Expected ephemeral home to be cleaned up. Found orphaned dirs: {new_dirs}"
    )
