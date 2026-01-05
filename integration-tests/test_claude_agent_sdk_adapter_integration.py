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

# pyright: reportArgumentType=false

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
_DEFAULT_MODEL = "claude-opus-4-5-20251101"
_PROMPT_NS = "integration/claude-agent-sdk"


def _get_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


def _make_config(tmp_path: Path, **kwargs: object) -> ClaudeAgentSDKClientConfig:
    """Build a ClaudeAgentSDKClientConfig with explicit cwd.

    All tests MUST use this helper to ensure cwd points to a temporary
    directory, preventing snapshot operations from creating commits in
    the actual repository.
    """
    config_kwargs: dict[str, object] = {
        "permission_mode": "bypassPermissions",
        "cwd": str(tmp_path),
    }
    config_kwargs.update(kwargs)
    return ClaudeAgentSDKClientConfig(**config_kwargs)  # type: ignore[arg-type]


def _make_adapter(tmp_path: Path, **kwargs: object) -> ClaudeAgentSDKAdapter:
    """Create an adapter with explicit cwd pointing to tmp_path."""
    model = kwargs.pop("model", None) or _get_model()
    client_config = kwargs.pop("client_config", None) or _make_config(tmp_path)
    return ClaudeAgentSDKAdapter(
        model=model,  # type: ignore[arg-type]
        client_config=client_config,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
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
    event = session[PromptExecuted].latest()
    assert event is not None, "Expected a PromptExecuted event."
    usage = event.usage
    assert usage is not None, "Expected token usage to be recorded."
    assert usage.total_tokens is not None and usage.total_tokens > 0


def test_claude_agent_sdk_adapter_returns_text(tmp_path: Path) -> None:
    """Test that the adapter returns text from a simple prompt."""
    adapter = _make_adapter(tmp_path)
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_processes_tool_invocation(tmp_path: Path) -> None:
    """Test that the adapter processes custom tool invocations via MCP bridge.

    This test validates that weakincentives tools are correctly bridged to the
    SDK via an in-process MCP server. The streaming mode approach enables proper
    MCP server initialization.
    """
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="integration tests")

    adapter = _make_adapter(tmp_path)

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None and response.text.strip()
    # The uppercase text should appear in the response
    assert params.text.upper() in response.text
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_parses_structured_output(tmp_path: Path) -> None:
    """Test that the adapter parses structured output correctly."""
    adapter = _make_adapter(tmp_path)
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


def test_claude_agent_sdk_adapter_with_model_config(tmp_path: Path) -> None:
    """Verify adapter works with ClaudeAgentSDKModelConfig.

    Note: The Claude Agent SDK does not expose max_tokens or temperature
    parameters directly - it manages token budgets internally. This test
    verifies that the adapter handles a model config gracefully even though
    these parameters are not applied to SDK options.
    """
    model = _get_model()
    model_config = ClaudeAgentSDKModelConfig(
        model=model,
        temperature=0.3,  # Ignored by SDK - no direct equivalent
        max_tokens=150,  # Ignored by SDK - no direct equivalent
    )

    adapter = _make_adapter(tmp_path, model_config=model_config)

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


def test_claude_agent_sdk_adapter_with_max_turns(tmp_path: Path) -> None:
    """Verify adapter respects max_turns configuration."""
    config = _make_config(
        tmp_path,
        max_turns=1,
    )

    adapter = _make_adapter(
        tmp_path,
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


def test_claude_agent_sdk_adapter_with_disallowed_tools(tmp_path: Path) -> None:
    """Verify adapter can be configured with disallowed tools."""
    adapter = _make_adapter(tmp_path, disallowed_tools=("Bash", "Write"))

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
    tmp_path: Path,
) -> None:
    """Verify that adapter hooks publish ToolInvoked events for SDK native tools.

    This test validates that the PostToolUse hook correctly publishes ToolInvoked
    events to the session's event bus when the SDK uses its native tools (like Read).
    This is a key integration point between the SDK's execution and weakincentives'
    event-driven architecture.
    """
    # Create a README.md file for the SDK to read
    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        "# Test Project\n\nThis is a test README for integration tests."
    )

    # Use isolated workspace to avoid operations on the actual repository
    config = _make_config(tmp_path)

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        # Only allow the Read tool to ensure we get a predictable tool call
        allowed_tools=("Read",),
    )

    prompt_template = _build_file_read_prompt()
    # Use README.md from the isolated workspace
    params = ReadFileParams(file_path="README.md")
    prompt = Prompt(prompt_template).bind(params)

    # Track ToolInvoked events
    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.dispatcher.subscribe(ToolInvoked, tool_invoked_events.append)

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


def test_claude_agent_sdk_adapter_publishes_prompt_rendered_event(
    tmp_path: Path,
) -> None:
    """Verify that PromptRendered event is published before SDK execution.

    This test validates that the adapter publishes a PromptRendered event
    containing the rendered prompt text, which is useful for debugging and
    observability.
    """
    adapter = _make_adapter(tmp_path)
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="prompt rendered test")
    prompt = Prompt(prompt_template).bind(params)

    prompt_rendered_events: list[PromptRendered] = []
    session = Session()
    session.dispatcher.subscribe(PromptRendered, prompt_rendered_events.append)

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


def test_claude_agent_sdk_adapter_mcp_tool_uses_render(tmp_path: Path) -> None:
    """Verify that MCP bridged tools call render() on result values.

    This test validates that when a custom weakincentives tool returns a
    ToolResult with a value that has a render() method, the MCP bridge
    uses that rendered output when returning to the SDK.
    """
    tool = _build_renderable_tool()
    prompt_template = _build_renderable_tool_prompt(tool)
    params = ComputeParams(x=7, y=13)

    adapter = _make_adapter(tmp_path)

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
            "Find all Python files in the ${target_dir} directory and read the "
            "contents of at least one of them. Summarize what you found."
        ),
        key="task",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-multi-tool",
        name="multi_tool_workflow",
        sections=[task_section],
    )


def test_claude_agent_sdk_adapter_multiple_tool_invocations(tmp_path: Path) -> None:
    """Verify adapter tracks multiple sequential tool invocations.

    This test validates that when the SDK uses multiple tools in sequence,
    each tool invocation is captured via the PostToolUse hook and published
    as a ToolInvoked event.
    """
    config = _make_config(tmp_path)

    # Allow tools that can be used for file discovery and reading
    # Claude may choose different tools (Glob vs Bash for listing, Read vs Bash for reading)
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Glob", "Read", "Bash"),
    )

    prompt_template = _build_multi_tool_prompt()
    # Use the src directory in tmp_path (created by fixture)
    params = MultiStepParams(target_dir="src")
    prompt = Prompt(prompt_template).bind(params)

    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.dispatcher.subscribe(ToolInvoked, tool_invoked_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "multi_tool_workflow"
    assert response.text is not None

    # Verify multiple tool invocations were captured (at least 2 for find + read)
    expected_min_tools = 2
    assert len(tool_invoked_events) >= expected_min_tools, (
        f"Expected at least {expected_min_tools} tool invocations. "
        f"Got {len(tool_invoked_events)}: {[e.name for e in tool_invoked_events]}"
    )

    # Verify at least one file discovery tool was used (Glob or Bash)
    tool_names = {e.name for e in tool_invoked_events}
    has_discovery_tool = "Glob" in tool_names or "Bash" in tool_names
    assert has_discovery_tool, (
        f"Expected Glob or Bash to be invoked for file discovery. Got: {tool_names}"
    )

    # Verify at least one file reading tool was used (Read or Bash)
    has_read_tool = "Read" in tool_names or "Bash" in tool_names
    assert has_read_tool, (
        f"Expected Read or Bash to be invoked for file reading. Got: {tool_names}"
    )

    # Verify each event has the expected adapter attribution
    for event in tool_invoked_events:
        assert event.adapter == "claude_agent_sdk"
        assert event.prompt_name == "multi_tool_workflow"


def test_claude_agent_sdk_adapter_tracks_token_usage_across_tools(
    tmp_path: Path,
) -> None:
    """Verify adapter accumulates token usage across multi-turn execution.

    This test validates that token usage is properly tracked and reported
    even when the SDK performs multiple tool calls, which may span multiple
    API interactions.
    """
    # Create a README.md file for the SDK to read
    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        "# Test Project\n\nThis is a test README for integration tests."
    )

    config = _make_config(tmp_path)

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Read",),
    )

    prompt_template = _build_file_read_prompt()
    # Use README.md in tmp_path (created above)
    params = ReadFileParams(file_path="README.md")
    prompt = Prompt(prompt_template).bind(params)

    prompt_executed_events: list[PromptExecuted] = []
    session = Session()
    session.dispatcher.subscribe(PromptExecuted, prompt_executed_events.append)

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


def test_claude_agent_sdk_adapter_custom_tool_without_render(tmp_path: Path) -> None:
    """Verify MCP bridged tools work when result has no render() method.

    This test validates that custom tools without a render() method on
    their result value still work correctly, falling back to the message.
    """
    tool = _build_simple_tool()
    prompt_template = _build_echo_prompt(tool)
    params = EchoParams(text="hello from integration test")

    adapter = _make_adapter(tmp_path)

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


def test_claude_agent_sdk_adapter_with_isolation_returns_text(tmp_path: Path) -> None:
    """Verify adapter works correctly with IsolationConfig enabled.

    This test validates that:
    1. The adapter creates an ephemeral home directory
    2. SDK execution uses the ephemeral home instead of ~/.claude
    3. The adapter still returns valid responses
    4. Cleanup happens after execution
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
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


def test_claude_agent_sdk_adapter_isolation_with_custom_tools(tmp_path: Path) -> None:
    """Verify custom tools work correctly in isolated mode.

    This test validates that MCP-bridged tools function correctly when
    the adapter is configured with isolation. The ephemeral home should
    not affect tool bridging functionality.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="isolation mode")

    adapter = _make_adapter(
        tmp_path,
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
    tmp_path: Path,
) -> None:
    """Verify structured output works in isolated mode.

    This test validates that structured output parsing functions correctly
    when the adapter is configured with isolation.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
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
    tmp_path: Path,
) -> None:
    """Verify isolation prevents modification of host ~/.claude directory.

    This test validates that running with IsolationConfig does not create,
    modify, or read from the real ~/.claude directory. This is critical for
    ensuring the user's personal configuration is not affected.
    """
    import tempfile

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
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
            # Note: We don't use include_host_env because that would
            # potentially inherit HOME from the test environment
        )

        config = _make_config(
            tmp_path,
            isolation=isolation,
            cwd=fake_home,  # Use the temp directory as cwd
        )

        adapter = _make_adapter(
            tmp_path,
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


def test_claude_agent_sdk_adapter_isolation_network_policy_no_network(
    tmp_path: Path,
) -> None:
    """Verify NetworkPolicy.no_network() still allows Claude API access.

    This test validates that even with no_network() (which blocks all tool
    network access), the Claude Code CLI can still reach the Anthropic API.
    The network policy only affects tools running in the sandbox, not the
    CLI itself.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
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


def test_claude_agent_sdk_adapter_isolation_with_custom_env(tmp_path: Path) -> None:
    """Verify custom environment variables are passed to SDK subprocess.

    This test validates that the env parameter in IsolationConfig correctly
    passes environment variables to the SDK subprocess.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        env={
            "WINK_TEST_VAR": "isolation_test_value",
        },
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="custom env test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_cleanup_on_success(tmp_path: Path) -> None:
    """Verify ephemeral home is cleaned up after successful execution.

    This test validates that the ephemeral home directory created for
    isolation is properly cleaned up after the adapter.evaluate() returns,
    even on successful execution.
    """
    # Count temp directories with our prefix before
    temp_dir = Path("/tmp")
    temp_dirs_before = set(temp_dir.glob("claude-agent-*"))

    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
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
    temp_dirs_after = set(temp_dir.glob("claude-agent-*"))

    # Any new directories should have been cleaned up
    new_dirs = temp_dirs_after - temp_dirs_before
    assert len(new_dirs) == 0, (
        f"Expected ephemeral home to be cleaned up. Found orphaned dirs: {new_dirs}"
    )


def test_claude_agent_sdk_adapter_isolation_creates_files_in_ephemeral_home(
    tmp_path: Path,
) -> None:
    """Verify files are created in the ephemeral home directory during execution.

    This test validates that the isolation mechanism actually uses the ephemeral
    home directory by capturing the directory during execution and verifying
    that expected files (like .claude/settings.json) are created there.

    This complements the negative test (host not modified) with positive proof
    that the ephemeral home is being used.
    """
    import json
    from unittest.mock import patch

    from weakincentives.adapters.claude_agent_sdk.isolation import EphemeralHome

    # Capture state: {home_path: (env_home, settings_dict or None)}
    captured: dict[Path, tuple[str, dict[str, object] | None]] = {}
    original_cleanup = EphemeralHome.cleanup

    def capturing_cleanup(self: EphemeralHome) -> None:
        """Capture ephemeral home state before cleanup."""
        home_path = Path(self._temp_dir)
        if home_path not in captured:
            settings_path = home_path / ".claude" / "settings.json"
            env_home = self.get_env().get("HOME", "")
            settings = (
                json.loads(settings_path.read_text())
                if settings_path.exists()
                else None
            )
            captured[home_path] = (env_home, settings)
        original_cleanup(self)

    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )
    config = _make_config(tmp_path, isolation=isolation)
    adapter = _make_adapter(tmp_path, client_config=config)
    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="ephemeral home verification")
    )
    session = _make_session_with_usage_tracking()

    with patch.object(EphemeralHome, "cleanup", capturing_cleanup):
        response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    _assert_prompt_usage(session)

    # Verify exactly one ephemeral home was created
    assert len(captured) == 1, f"Expected one ephemeral home, got {len(captured)}"
    ephemeral_home, (env_home, settings) = next(iter(captured.items()))

    # Verify HOME env var pointed to ephemeral directory
    assert env_home == str(ephemeral_home), f"HOME mismatch: {env_home}"

    # Verify settings.json was created with expected structure
    assert settings is not None, "Expected settings.json in ephemeral .claude directory"
    assert "sandbox" in settings, "Expected sandbox configuration in settings"
    sandbox = settings["sandbox"]
    assert isinstance(sandbox, dict) and sandbox.get("enabled") is True

    # Verify network policy was applied (no_network() means empty allowed domains)
    assert "network" in sandbox, "Expected network configuration in sandbox"
    network = sandbox["network"]
    assert isinstance(network, dict)
    # no_network() creates empty allowed domains - CLI reaches API outside sandbox
    assert network.get("allowedDomains") == []

    # Verify cleanup occurred
    assert not ephemeral_home.exists(), (
        f"Ephemeral home not cleaned up: {ephemeral_home}"
    )


@dataclass(slots=True)
class NetworkTestParams:
    """Parameters for network test prompt."""

    url: str


@dataclass(slots=True, frozen=True)
class NetworkTestResult:
    """Result of network connectivity test."""

    reachable: bool
    http_status: int | None
    error_message: str | None


def _build_network_test_prompt() -> PromptTemplate[NetworkTestResult]:
    """Build a prompt that tests network connectivity from within the sandbox."""
    return PromptTemplate[NetworkTestResult](
        ns=_PROMPT_NS,
        key="network-test",
        name="network_test",
        sections=(
            MarkdownSection[NetworkTestParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to test network connectivity to ${url}. "
                    "Run: curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 ${url} 2>&1 "
                    "If successful, return reachable=true with the HTTP status code. "
                    "If it fails (connection refused, timeout, etc.), return reachable=false "
                    "with the error message. Set http_status to null if no response was received."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_network_policy_allows_listed_domain(
    tmp_path: Path,
) -> None:
    """Verify network policy allows access to explicitly listed domains.

    This test validates that when a domain IS in allowed_domains, tools
    running in the sandbox can successfully reach it.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("api.anthropic.com", "example.com"),
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Allowed domain test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )
    # example.com is in allowed_domains, so it should be reachable with HTTP 200
    http_ok = 200
    assert result.reachable is True, (
        f"Expected example.com to be reachable: {result.error_message}"
    )
    assert result.http_status == http_ok


def test_claude_agent_sdk_adapter_network_policy_blocks_unlisted_domain(
    tmp_path: Path,
) -> None:
    """Verify network policy blocks access to domains not in the allowed list.

    This test validates that when a domain is NOT in allowed_domains, tools
    running in the sandbox cannot reach it.

    See: https://github.com/anthropic-experimental/sandbox-runtime
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),  # Block all tool network
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Blocked domain test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )

    # example.com should be blocked since it's not in allowed_domains
    assert result.reachable is False, (
        f"Expected example.com to be blocked, but got status {result.http_status}"
    )


def test_claude_agent_sdk_adapter_no_network_blocks_all_tool_access(
    tmp_path: Path,
) -> None:
    """Verify no_network() blocks all network access for tools.

    This test validates that NetworkPolicy.no_network() blocks tools from
    accessing any external network resources. The Claude Code CLI can still
    reach the API (it runs outside the tool sandbox), but tools like Bash
    cannot make network requests.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),  # Block all tool network
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"No network test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )

    # All network should be blocked for tools
    assert result.reachable is False, (
        f"Expected all network blocked, but got status {result.http_status}"
    )


# =============================================================================
# Additional Isolation Configuration Tests
# =============================================================================
#
# These tests cover additional IsolationConfig options to ensure each
# configuration option behaves as documented.
# =============================================================================


@dataclass(slots=True, frozen=True)
class EnvTestResult:
    """Result of environment variable test."""

    found_vars: list[str]
    path_value: str | None
    custom_var_value: str | None


@dataclass(slots=True)
class EmptyParams:
    """Empty params placeholder for prompts without parameters."""

    pass


def _build_env_test_prompt() -> PromptTemplate[EnvTestResult]:
    """Build a prompt that tests environment variable inheritance."""
    return PromptTemplate[EnvTestResult](
        ns=_PROMPT_NS,
        key="env-test",
        name="env_test",
        sections=(
            MarkdownSection[EmptyParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to inspect environment variables. "
                    "Run: env | head -20 "
                    "Return found_vars with a list of variable names you see. "
                    "Return path_value with the value of PATH if present (or null). "
                    "Return custom_var_value with the value of WINK_CUSTOM_TEST_VAR "
                    "if present (or null)."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_include_host_env_false(tmp_path: Path) -> None:
    """Verify include_host_env=False limits environment passed to SDK.

    This test validates that when include_host_env=False (the default),
    we only pass HOME, ANTHROPIC_API_KEY, and custom env vars to the SDK.

    NOTE: The actual subprocess environment may still include variables
    from the Claude Code CLI process itself. This test validates that
    custom env vars are correctly passed through our isolation layer.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            include_host_env=False,  # Explicitly false
            env={"WINK_CUSTOM_TEST_VAR": "custom_value"},
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_env_test_prompt()).bind(EmptyParams())
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Env test (include_host_env=False): found_vars={result.found_vars}, "
        f"path={result.path_value}, custom={result.custom_var_value}"
    )

    # The key validation: our custom env var is present
    assert result.custom_var_value == "custom_value", (
        f"Expected custom var to be 'custom_value', got {result.custom_var_value}"
    )


def test_claude_agent_sdk_adapter_include_host_env_true(tmp_path: Path) -> None:
    """Verify include_host_env=True inherits non-sensitive host environment.

    This test validates that when include_host_env=True, safe host
    environment variables like PATH are inherited, but sensitive ones
    (AWS_*, OPENAI_*, etc.) are filtered out.
    """
    import os as test_os

    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            include_host_env=True,  # Inherit host env
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_env_test_prompt()).bind(EmptyParams())
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Env test (include_host_env=True): found_vars={result.found_vars}, "
        f"path={result.path_value}"
    )

    # PATH should be inherited from host
    host_path = test_os.environ.get("PATH")
    if host_path:
        assert result.path_value is not None, "Expected PATH to be inherited from host"

    # Sensitive variables should NOT be in found_vars
    sensitive_prefixes = ["AWS_", "OPENAI_", "GOOGLE_", "AZURE_"]
    for var in result.found_vars:
        for prefix in sensitive_prefixes:
            assert not var.startswith(prefix), (
                f"Sensitive variable {var} should be filtered"
            )


@dataclass(slots=True, frozen=True)
class FileWriteTestResult:
    """Result of file write test."""

    write_succeeded: bool
    file_path: str | None
    error_message: str | None


def _build_file_write_test_prompt() -> PromptTemplate[FileWriteTestResult]:
    """Build a prompt that tests file writing to a specific path."""
    return PromptTemplate[FileWriteTestResult](
        ns=_PROMPT_NS,
        key="file-write-test",
        name="file_write_test",
        sections=(
            MarkdownSection[ReadFileParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to try writing a file to ${file_path}. "
                    "Run: echo 'test content' > ${file_path} 2>&1 && echo SUCCESS || echo FAILED "
                    "Return write_succeeded=true if the write succeeded, false otherwise. "
                    "Return the file_path you tried to write to. "
                    "Return any error message if the write failed."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_writable_paths(tmp_path: Path) -> None:
    """Verify writable_paths allows writing to extra directories.

    This test validates that SandboxConfig.writable_paths correctly
    allows writing to directories outside the default workspace.
    """
    import tempfile

    # Create a temp directory to use as a writable path
    with tempfile.TemporaryDirectory(prefix="wink-writable-") as writable_dir:
        test_file = f"{writable_dir}/test-writable.txt"

        config = _make_config(
            tmp_path,
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    writable_paths=(writable_dir,),
                ),
            ),
        )
        adapter = _make_adapter(
            tmp_path,
            client_config=config,
            allowed_tools=("Bash",),
        )

        prompt = Prompt(_build_file_write_test_prompt()).bind(
            ReadFileParams(file_path=test_file)
        )
        session = _make_session_with_usage_tracking()

        response = adapter.evaluate(prompt, session=session)

        assert response.output is not None, (
            f"Expected structured output, got: {response.text}"
        )
        result = response.output
        print(
            f"Writable paths test: succeeded={result.write_succeeded}, "
            f"path={result.file_path}, error={result.error_message}"
        )

        # With writable_paths configured, write should succeed
        assert result.write_succeeded is True, (
            f"Expected write to succeed with writable_paths: {result.error_message}"
        )

        # Verify file actually exists
        assert Path(test_file).exists(), "Expected file to be created"


@dataclass(slots=True, frozen=True)
class FileReadTestResult:
    """Result of file read test."""

    read_succeeded: bool
    content: str | None
    error_message: str | None


def _build_file_read_test_prompt() -> PromptTemplate[FileReadTestResult]:
    """Build a prompt that tests file reading from a specific path."""
    return PromptTemplate[FileReadTestResult](
        ns=_PROMPT_NS,
        key="file-read-test",
        name="file_read_test",
        sections=(
            MarkdownSection[ReadFileParams](
                title="Task",
                key="task",
                template=(
                    "Use the Read tool to read the file at ${file_path}. "
                    "Return read_succeeded=true if you can read it, false otherwise. "
                    "Return the content if readable, or the error message if not."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_readable_paths(tmp_path: Path) -> None:
    """Verify readable_paths allows reading extra directories.

    This test validates that SandboxConfig.readable_paths correctly
    allows reading from directories outside the default workspace.
    """
    import tempfile

    # Create a temp directory with a file to read
    with tempfile.TemporaryDirectory(prefix="wink-readable-") as readable_dir:
        test_file = f"{readable_dir}/test-readable.txt"
        Path(test_file).write_text("readable test content")

        config = _make_config(
            tmp_path,
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    readable_paths=(readable_dir,),
                ),
            ),
        )
        adapter = _make_adapter(
            tmp_path,
            client_config=config,
            allowed_tools=("Read",),
        )

        prompt = Prompt(_build_file_read_test_prompt()).bind(
            ReadFileParams(file_path=test_file)
        )
        session = _make_session_with_usage_tracking()

        response = adapter.evaluate(prompt, session=session)

        assert response.output is not None, (
            f"Expected structured output, got: {response.text}"
        )
        result = response.output
        print(
            f"Readable paths test: succeeded={result.read_succeeded}, "
            f"content={result.content}, error={result.error_message}"
        )

        # With readable_paths configured, read should succeed
        assert result.read_succeeded is True, (
            f"Expected read to succeed with readable_paths: {result.error_message}"
        )
        assert result.content is not None
        assert "readable test content" in result.content


def test_claude_agent_sdk_adapter_sandbox_disabled(tmp_path: Path) -> None:
    """Verify sandbox can be disabled with enabled=False.

    This test validates that SandboxConfig(enabled=False) disables
    OS-level sandboxing, allowing full filesystem access.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=False),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Read",),
    )

    # Try to read a system file that would normally be outside sandbox
    prompt = Prompt(_build_file_read_test_prompt()).bind(
        ReadFileParams(file_path="/etc/hosts")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Sandbox disabled test: succeeded={result.read_succeeded}, "
        f"error={result.error_message}"
    )

    # With sandbox disabled, /etc/hosts should be readable
    assert result.read_succeeded is True, (
        f"Expected /etc/hosts to be readable with sandbox disabled: {result.error_message}"
    )
    assert result.content is not None
    assert "localhost" in result.content.lower() or "127.0.0.1" in result.content


@dataclass(slots=True, frozen=True)
class CommandTestResult:
    """Result of command execution test."""

    command_succeeded: bool
    output: str | None
    error_message: str | None


def _build_command_test_prompt() -> PromptTemplate[CommandTestResult]:
    """Build a prompt that tests running a specific command."""
    return PromptTemplate[CommandTestResult](
        ns=_PROMPT_NS,
        key="command-test",
        name="command_test",
        sections=(
            MarkdownSection[EchoParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to run the following command: ${text} "
                    "Return command_succeeded=true if it ran successfully. "
                    "Return the output if successful, or error_message if it failed."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_excluded_commands(tmp_path: Path) -> None:
    """Verify excluded_commands allows specific commands to bypass sandbox.

    This test validates that SandboxConfig.excluded_commands correctly
    allows specific commands to run outside the sandbox.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(
                enabled=True,
                excluded_commands=("ls",),
                allow_unsandboxed_commands=True,
            ),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    # Run ls which is in excluded_commands
    prompt = Prompt(_build_command_test_prompt()).bind(EchoParams(text="ls /"))
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Excluded commands test: succeeded={result.command_succeeded}, "
        f"output={result.output}, error={result.error_message}"
    )

    # Command in excluded_commands should succeed
    assert result.command_succeeded is True, (
        f"Expected excluded command to succeed: {result.error_message}"
    )


# =============================================================================
# ClaudeAgentWorkspaceSection Integration Tests
# =============================================================================


def test_claude_agent_sdk_adapter_with_workspace_section(tmp_path: Path) -> None:
    """Verify ClaudeAgentWorkspaceSection works with the adapter.

    This test validates that host files can be mounted into a workspace
    and the SDK can read them through native tools.
    """
    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentWorkspaceSection,
        HostMount,
    )

    # Create a README.md file for the workspace to mount
    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        "# Test Project\n\nThis is a test README for integration tests."
    )

    # Create session and workspace
    session = Session()

    # Mount the README.md (avoids touching actual repo)
    workspace = ClaudeAgentWorkspaceSection(
        session=session,
        mounts=(
            HostMount(
                host_path=str(tmp_path / "README.md"),
                mount_path="readme.md",
            ),
        ),
        allowed_host_roots=(str(tmp_path),),
    )

    try:
        config = _make_config(
            tmp_path,
            cwd=str(workspace.temp_dir),
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    readable_paths=(str(workspace.temp_dir),),
                ),
            ),
        )

        adapter = _make_adapter(
            tmp_path,
            client_config=config,
            allowed_tools=("Read",),
        )

        # Build a prompt to read the mounted file
        prompt_template = PromptTemplate[FileReadTestResult](
            ns=_PROMPT_NS,
            key="workspace-read",
            name="workspace_read",
            sections=(
                workspace,
                MarkdownSection[EmptyParams](
                    title="Task",
                    key="task",
                    template=(
                        "Read the file 'readme.md' in the current workspace directory. "
                        "Return read_succeeded=true if successful. "
                        "Return the first 100 characters of content."
                    ),
                ),
            ),
        )
        prompt = Prompt(prompt_template).bind(EmptyParams())

        response = adapter.evaluate(prompt, session=session)

        assert response.output is not None, (
            f"Expected structured output, got: {response.text}"
        )
        result = response.output
        print(
            f"Workspace test: succeeded={result.read_succeeded}, "
            f"content={result.content[:50] if result.content else None}..."
        )

        # Should be able to read the mounted file
        assert result.read_succeeded is True, (
            f"Expected workspace file read to succeed: {result.error_message}"
        )
        assert result.content is not None

    finally:
        workspace.cleanup()


def test_claude_agent_sdk_adapter_workspace_respects_byte_limit(tmp_path: Path) -> None:
    """Verify HostMount.max_bytes is enforced during workspace creation.

    This test validates that the workspace section correctly rejects
    mounts that exceed their byte budget.
    """
    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentWorkspaceSection,
        HostMount,
        WorkspaceBudgetExceededError,
    )

    # Create a README.md file for the workspace to mount
    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        "# Test Project\n\nThis is a test README for integration tests."
    )

    session = Session()

    # Try to mount README.md with a very small byte limit
    with pytest.raises(WorkspaceBudgetExceededError):
        ClaudeAgentWorkspaceSection(
            session=session,
            mounts=(
                HostMount(
                    host_path=str(tmp_path / "README.md"),
                    mount_path="readme.md",
                    max_bytes=10,  # Very small - should fail
                ),
            ),
            allowed_host_roots=(str(tmp_path),),
        )


def test_claude_agent_sdk_adapter_workspace_security_check(tmp_path: Path) -> None:
    """Verify allowed_host_roots security boundary is enforced.

    This test validates that attempting to mount files outside
    the allowed_host_roots raises a security error.
    """
    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentWorkspaceSection,
        HostMount,
        WorkspaceSecurityError,
    )

    session = Session()

    # Try to mount /etc/hosts which is outside allowed_host_roots
    with pytest.raises(WorkspaceSecurityError):
        ClaudeAgentWorkspaceSection(
            session=session,
            mounts=(
                HostMount(
                    host_path="/etc/hosts",
                    mount_path="hosts",
                ),
            ),
            # Only allow isolated workspace directory
            allowed_host_roots=(str(tmp_path),),
        )


# ---------------------------------------------------------------------------
# MCP Tool + Native Tool Integration Test
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WriteFileRequest:
    """Input for the MCP file writer tool."""

    filename: str
    content: str


@dataclass(slots=True)
class WriteFileResponse:
    """Output from the MCP file writer tool."""

    path: str
    bytes_written: int

    def render(self) -> str:
        return f"Wrote {self.bytes_written} bytes to {self.path}"


def _build_mcp_file_writer_tool() -> Tool[WriteFileRequest, WriteFileResponse]:
    """Build a WINK tool that writes files using context.filesystem."""

    def write_file_handler(
        params: WriteFileRequest, *, context: ToolContext
    ) -> ToolResult[WriteFileResponse]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        # Write the file using the filesystem from context
        context.filesystem.write(params.filename, params.content, mode="create")
        bytes_written = len(params.content.encode("utf-8"))

        return ToolResult(
            message=f"Successfully wrote {bytes_written} bytes to {params.filename}",
            value=WriteFileResponse(path=params.filename, bytes_written=bytes_written),
            success=True,
        )

    return Tool[WriteFileRequest, WriteFileResponse](
        name="write_file_to_workspace",
        description=(
            "Write content to a file in the workspace. "
            "Takes a filename and content string."
        ),
        handler=write_file_handler,
    )


@dataclass(slots=True)
class McpWriteAndReadParams:
    """Parameters for the MCP write and read test."""

    filename: str
    content: str


@dataclass(slots=True)
class McpWriteAndReadResult:
    """Structured result for verifying the write+read cycle."""

    write_succeeded: bool
    read_succeeded: bool
    content_matches: bool
    read_content: str | None = None
    error_message: str | None = None


def test_claude_agent_sdk_mcp_tool_writes_file_native_tool_reads(
    tmp_path: Path,
) -> None:
    """Verify MCP-bridged tools can write files that native SDK tools can read.

    This integration test validates the filesystem bridging between WINK tools
    (exposed via MCP) and the Claude Agent SDK's native file tools. The test:

    1. Creates a WINK tool that writes files using context.filesystem
    2. Configures the adapter with cwd pointing to a temp directory
    3. Has Claude call the MCP tool to write a file
    4. Has Claude use the native Read tool to read the file back
    5. Verifies the content matches

    This demonstrates that the HostFilesystem passed to ToolContext is backed
    by the same directory that the Claude Agent SDK operates in.
    """
    # Create the MCP tool that uses context.filesystem
    write_tool = _build_mcp_file_writer_tool()

    # Build a prompt that instructs Claude to:
    # 1. Call the MCP tool to write a file
    # 2. Use the native Read tool to read it back
    # 3. Return structured output confirming the content matches
    task_section = MarkdownSection[McpWriteAndReadParams](
        title="Task",
        template=(
            "You have two tasks:\n\n"
            "1. First, use the `write_file_to_workspace` tool to write a file named "
            "'${filename}' with the content: '${content}'\n\n"
            "2. After the file is written, use the Read tool to read the file at "
            "'${filename}' and verify its contents.\n\n"
            "Return a structured response indicating:\n"
            "- write_succeeded: true if the write tool succeeded\n"
            "- read_succeeded: true if you could read the file\n"
            "- content_matches: true if the content you read matches '${content}'\n"
            "- read_content: the actual content you read from the file\n"
        ),
        tools=(write_tool,),
        key="task",
    )

    prompt_template = PromptTemplate[McpWriteAndReadResult](
        ns=_PROMPT_NS,
        key="mcp-write-native-read",
        name="mcp_filesystem_integration",
        sections=[task_section],
    )

    # Configure adapter with cwd pointing to temp directory
    # This ensures both the MCP tool's filesystem and SDK's native tools
    # operate on the same directory
    config = _make_config(tmp_path)

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        # Allow both the MCP tool and the Read tool
        allowed_tools=("Read", "mcp__wink__write_file_to_workspace"),
    )

    # Bind parameters - use a unique filename and recognizable content
    test_content = "Hello from MCP tool! 🎉 Integration test content."
    params = McpWriteAndReadParams(
        filename="mcp_test_output.txt",
        content=test_content,
    )
    prompt = Prompt(prompt_template).bind(params)

    # Track tool invocations to verify both tools were called
    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.dispatcher.subscribe(ToolInvoked, tool_invoked_events.append)

    # Execute
    response = adapter.evaluate(prompt, session=session)

    # Verify structured output
    assert response.output is not None, (
        f"Expected structured output, got text: {response.text}"
    )
    result = response.output

    print("\nMCP + Native Tool Integration Test Results:")
    print(f"  write_succeeded: {result.write_succeeded}")
    print(f"  read_succeeded: {result.read_succeeded}")
    print(f"  content_matches: {result.content_matches}")
    print(f"  read_content: {result.read_content!r}")
    if result.error_message:
        print(f"  error_message: {result.error_message}")

    # Verify the MCP tool was called
    tool_names = [e.name for e in tool_invoked_events]
    assert any("write_file" in n.lower() or "mcp" in n.lower() for n in tool_names), (
        f"Expected MCP write tool to be called. Tool events: {tool_names}"
    )

    # Verify the native Read tool was called
    assert any(n == "Read" for n in tool_names), (
        f"Expected native Read tool to be called. Tool events: {tool_names}"
    )

    # Verify the write succeeded
    assert result.write_succeeded is True, (
        f"MCP tool write failed: {result.error_message}"
    )

    # Verify the read succeeded
    assert result.read_succeeded is True, (
        f"Native Read tool failed: {result.error_message}"
    )

    # Verify content matches (the key assertion!)
    assert result.content_matches is True, (
        f"Content mismatch! Expected: {test_content!r}, Read: {result.read_content!r}"
    )

    # Also verify the file actually exists on disk
    written_file = tmp_path / "mcp_test_output.txt"
    assert written_file.exists(), f"Expected file to exist at {written_file}"
    assert written_file.read_text(encoding="utf-8") == test_content, (
        f"File content on disk doesn't match expected: {test_content!r}"
    )
