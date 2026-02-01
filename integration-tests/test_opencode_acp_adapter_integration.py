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

"""Integration tests for the OpenCode ACP adapter."""

# pyright: reportArgumentType=false

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from weakincentives.adapters.opencode_acp import (
        OpenCodeACPAdapter,
        OpenCodeACPClientConfig,
    )

pytest.importorskip("agent_client_protocol")


def _has_opencode_binary() -> bool:
    """Check if the opencode binary is available on PATH."""
    return shutil.which("opencode") is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_opencode_binary(),
        reason="opencode binary not found on PATH.",
    ),
    pytest.mark.timeout(120),  # ACP tests may take longer due to agentic execution
]

_PROMPT_NS = "integration/opencode-acp"


def _make_config(tmp_path: Path, **kwargs: object) -> OpenCodeACPClientConfig:
    """Build an OpenCodeACPClientConfig with explicit cwd.

    All tests MUST use this helper to ensure cwd points to a temporary
    directory, preventing operations from modifying the actual repository.
    """
    from weakincentives.adapters.opencode_acp import OpenCodeACPClientConfig

    config_kwargs: dict[str, object] = {
        "permission_mode": "auto",
        "cwd": str(tmp_path),
    }
    config_kwargs.update(kwargs)
    return OpenCodeACPClientConfig(**config_kwargs)


def _make_adapter(tmp_path: Path, **kwargs: object) -> OpenCodeACPAdapter[object]:
    """Create an adapter with explicit cwd pointing to tmp_path."""
    from weakincentives.adapters.opencode_acp import (
        OpenCodeACPAdapter,
        OpenCodeACPAdapterConfig,
    )

    client_config = kwargs.pop("client_config", None) or _make_config(tmp_path)
    adapter_config = kwargs.pop("adapter_config", None) or OpenCodeACPAdapterConfig()

    return OpenCodeACPAdapter(
        client_config=client_config,  # type: ignore[arg-type]
        adapter_config=adapter_config,  # type: ignore[arg-type]
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
        return ToolResult.ok(TransformResult(text=transformed), message=message)

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


# =============================================================================
# Basic Adapter Tests
# =============================================================================


def test_opencode_acp_adapter_returns_text(tmp_path: Path) -> None:
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


def test_opencode_acp_adapter_processes_tool_invocation(tmp_path: Path) -> None:
    """Test that the adapter processes custom tool invocations via MCP bridge.

    This test validates that weakincentives tools are correctly bridged to
    OpenCode via an in-process MCP server. The adapter creates an MCP server
    that exposes WINK tools to OpenCode.
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


def test_opencode_acp_adapter_parses_structured_output(tmp_path: Path) -> None:
    """Test that the adapter parses structured output correctly.

    The OpenCode ACP adapter uses a dedicated structured_output MCP tool
    rather than provider-native structured output.
    """
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


# =============================================================================
# Event Tracking Tests
# =============================================================================


def test_opencode_acp_adapter_publishes_tool_invoked_events(tmp_path: Path) -> None:
    """Verify that adapter publishes ToolInvoked events for WINK tool invocations.

    This test validates that when OpenCode invokes WINK tools via the MCP bridge,
    the adapter correctly publishes ToolInvoked events to the session.
    """
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="event tracking test")

    adapter = _make_adapter(tmp_path)

    prompt = Prompt(prompt_template).bind(params)

    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.dispatcher.subscribe(ToolInvoked, tool_invoked_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None

    # Verify that at least one ToolInvoked event was published for our tool
    assert len(tool_invoked_events) >= 1, (
        "Expected at least one ToolInvoked event. "
        "This indicates tool invocations are not being tracked."
    )

    # Find the uppercase_text tool invocation
    uppercase_events = [e for e in tool_invoked_events if e.name == "uppercase_text"]
    assert len(uppercase_events) >= 1, (
        f"Expected at least one uppercase_text tool invocation. "
        f"Got events for: {[e.name for e in tool_invoked_events]}"
    )

    # Verify the event has expected structure
    event = uppercase_events[0]
    assert event.adapter == "opencode_acp"
    assert event.prompt_name == "uppercase_workflow"
    assert event.result is not None


def test_opencode_acp_adapter_publishes_prompt_rendered_event(tmp_path: Path) -> None:
    """Verify that PromptRendered event is published before execution.

    This test validates that the adapter publishes a PromptRendered event
    containing the rendered prompt text.
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
    assert event.adapter == "opencode_acp"
    assert event.rendered_prompt is not None
    # The rendered prompt should contain the audience text
    assert "prompt rendered test" in event.rendered_prompt


def test_opencode_acp_adapter_tracks_token_usage(tmp_path: Path) -> None:
    """Verify adapter tracks token usage from OpenCode execution."""
    adapter = _make_adapter(tmp_path)
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="token tracking test")
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


# =============================================================================
# Configuration Tests
# =============================================================================


def test_opencode_acp_adapter_with_permission_mode_deny(tmp_path: Path) -> None:
    """Verify adapter works with permission_mode='deny'.

    This test validates that the adapter handles deny mode gracefully.
    Simple prompts that don't require permissions should still work.
    """
    config = _make_config(tmp_path, permission_mode="deny")

    adapter = _make_adapter(tmp_path, client_config=config)

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="deny mode test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    _assert_prompt_usage(session)


def test_opencode_acp_adapter_with_custom_opencode_args(tmp_path: Path) -> None:
    """Verify adapter accepts custom opencode arguments."""
    config = _make_config(
        tmp_path,
        opencode_args=("acp",),  # Default args, but explicitly specified
    )

    adapter = _make_adapter(tmp_path, client_config=config)

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="custom args test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    _assert_prompt_usage(session)


# =============================================================================
# Workspace Tests
# =============================================================================


@dataclass(slots=True)
class ReadFileParams:
    """Prompt parameters for reading a file."""

    file_path: str


def _build_file_read_prompt() -> PromptTemplate[object]:
    """Build a prompt that requests the assistant to read a file."""
    task_section = MarkdownSection[ReadFileParams](
        title="Task",
        template=(
            "Read the contents of the file at ${file_path}. "
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


def test_opencode_acp_adapter_with_workspace(tmp_path: Path) -> None:
    """Verify adapter works with workspace section for file access.

    This test validates that the OpenCodeWorkspaceSection properly exposes
    files to OpenCode via the cwd configuration.
    """
    from weakincentives.adapters.opencode_acp import (
        HostMount,
        OpenCodeWorkspaceSection,
    )

    # Create a test file to be read
    test_file = tmp_path / "test_content.txt"
    test_file.write_text("This is test content for workspace integration testing.")

    # Create workspace section with the test file mounted
    session = Session()
    workspace = OpenCodeWorkspaceSection(
        session=session,
        mounts=(HostMount(source=tmp_path, mount_path="workspace"),),
    )

    try:
        config = _make_config(
            workspace.temp_dir,  # Use workspace temp dir as cwd
            allow_file_reads=True,
        )

        adapter = _make_adapter(workspace.temp_dir, client_config=config)

        prompt_template = _build_file_read_prompt()
        params = ReadFileParams(file_path="workspace/test_content.txt")
        prompt = Prompt(prompt_template).bind(params)

        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "file_reader"
        assert response.text is not None
        # The response should mention something about the test content
        assert "test" in response.text.lower() or "content" in response.text.lower()
    finally:
        workspace.cleanup()


# =============================================================================
# Custom Tool Tests
# =============================================================================


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
        return ToolResult.ok(
            EchoResult(echoed=params.text), message=f"Echo: {params.text}"
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


def test_opencode_acp_adapter_custom_tool_without_render(tmp_path: Path) -> None:
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
        return ToolResult.ok(
            RenderableResult(value=result, message=f"{params.x} + {params.y}"),
            message=f"Computed {params.x} + {params.y} = {result}",
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


def test_opencode_acp_adapter_mcp_tool_uses_render(tmp_path: Path) -> None:
    """Verify that MCP bridged tools call render() on result values.

    This test validates that when a custom weakincentives tool returns a
    ToolResult with a value that has a render() method, the MCP bridge
    uses that rendered output when returning to OpenCode.
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

    # The response should mention the computed value 20
    assert "20" in response.text, (
        f"Expected response to mention the computed value 20. Got: {response.text}"
    )
    _assert_prompt_usage(session)


# =============================================================================
# Multi-Tool Tests
# =============================================================================


@dataclass(slots=True)
class MultiToolParams:
    """Parameters for multi-tool operations."""

    first_text: str
    second_text: str


def _build_multi_tool_prompt(
    tool1: Tool[TransformRequest, TransformResult],
    tool2: Tool[EchoParams, EchoResult],
) -> PromptTemplate[object]:
    """Build a prompt that requires multiple tool invocations."""
    task_section = MarkdownSection[MultiToolParams](
        title="Task",
        template=(
            "You must complete two tasks in sequence:\n"
            "1. First, call the `uppercase_text` tool with text='${first_text}'\n"
            "2. Then, call the `echo` tool with text='${second_text}'\n"
            "After both tools complete, summarize both results."
        ),
        tools=(tool1, tool2),
        key="task",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-multi-tool",
        name="multi_tool_workflow",
        sections=[task_section],
    )


def test_opencode_acp_adapter_multiple_tool_invocations(tmp_path: Path) -> None:
    """Verify adapter tracks multiple sequential tool invocations.

    This test validates that when OpenCode uses multiple WINK tools in sequence,
    each tool invocation is captured and published as a ToolInvoked event.
    """
    tool1 = _build_uppercase_tool()
    tool2 = _build_simple_tool()
    prompt_template = _build_multi_tool_prompt(tool1, tool2)
    params = MultiToolParams(first_text="hello", second_text="world")

    adapter = _make_adapter(tmp_path)

    prompt = Prompt(prompt_template).bind(params)

    tool_invoked_events: list[ToolInvoked] = []
    session = Session()
    session.dispatcher.subscribe(ToolInvoked, tool_invoked_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "multi_tool_workflow"
    assert response.text is not None

    # Verify multiple tool invocations were captured (at least 2)
    expected_min_tools = 2
    assert len(tool_invoked_events) >= expected_min_tools, (
        f"Expected at least {expected_min_tools} tool invocations. "
        f"Got {len(tool_invoked_events)}: {[e.name for e in tool_invoked_events]}"
    )

    # Verify both tools were invoked
    tool_names = {e.name for e in tool_invoked_events}
    assert "uppercase_text" in tool_names, (
        f"Expected uppercase_text to be invoked. Got: {tool_names}"
    )
    assert "echo" in tool_names, f"Expected echo to be invoked. Got: {tool_names}"

    # Verify each event has the expected adapter attribution
    for event in tool_invoked_events:
        assert event.adapter == "opencode_acp"
        assert event.prompt_name == "multi_tool_workflow"
