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

"""Integration tests for the Codex App Server adapter.

These tests exercise the full adapter lifecycle against a real ``codex
app-server`` subprocess. They are skipped when the ``codex`` CLI is not
found on PATH.
"""

# pyright: reportArgumentType=false

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    CodexWorkspaceSection,
)
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events import (
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import Session


def _has_codex() -> bool:
    return shutil.which("codex") is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_codex(), reason="codex CLI not found on PATH"),
    pytest.mark.timeout(90),
]

_MODEL_ENV_VAR: Final[str] = "CODEX_APP_SERVER_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/codex-app-server"


# =============================================================================
# Helpers
# =============================================================================


def _get_model() -> str:
    """Return the model name for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, "gpt-5.3-codex")


def _make_adapter(
    tmp_path: Path,
    *,
    reuse_thread: bool = False,
    approval_policy: str = "never",
) -> CodexAppServerAdapter:
    """Create an adapter configured for testing with cwd in tmp_path."""
    return CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(model=_get_model()),
        client_config=CodexAppServerClientConfig(
            cwd=str(tmp_path),
            approval_policy=approval_policy,
            reuse_thread=reuse_thread,
        ),
    )


def _make_session() -> Session:
    return Session(tags={"suite": "integration"})


def _assert_prompt_usage(session: Session) -> None:
    event = session[PromptExecuted].latest()
    assert event is not None, "Expected a PromptExecuted event."
    assert event.adapter == "codex_app_server"


# =============================================================================
# Prompt builders
# =============================================================================


@dataclass(slots=True)
class GreetingParams:
    """Prompt parameters for a greeting scenario."""

    audience: str


def _build_greeting_prompt() -> PromptTemplate[object]:
    section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting "
            "for ${audience}. Reply in a single sentence without any tool calls."
        ),
        key="greeting",
    )
    return PromptTemplate(
        ns=_PROMPT_NS,
        key="integration-greeting",
        name="greeting",
        sections=[section],
    )


@dataclass(slots=True)
class TransformRequest:
    """Input sent to the uppercase helper tool."""

    text: str


@dataclass(slots=True)
class TransformResult:
    """Payload emitted by the uppercase helper tool."""

    text: str


def _build_uppercase_tool() -> Tool[TransformRequest, TransformResult]:
    def uppercase_tool(
        params: TransformRequest, *, context: ToolContext
    ) -> ToolResult[TransformResult]:
        del context
        transformed = params.text.upper()
        return ToolResult.ok(
            TransformResult(text=transformed),
            message=f"Transformed to uppercase: {transformed}",
        )

    return Tool[TransformRequest, TransformResult](
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=uppercase_tool,
    )


def _build_tool_prompt(
    tool: Tool[TransformRequest, TransformResult],
) -> PromptTemplate[object]:
    section = MarkdownSection[TransformRequest](
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
        sections=[section],
    )


@dataclass(slots=True)
class ReviewParams:
    """Prompt parameters for structured output verification."""

    text: str


@dataclass(slots=True, frozen=True)
class ReviewAnalysis:
    """Structured payload expected from the provider."""

    summary: str
    sentiment: str


def _build_structured_prompt() -> PromptTemplate[ReviewAnalysis]:
    section = MarkdownSection[ReviewParams](
        title="Analysis Task",
        template=(
            "Review the provided passage and produce a concise summary "
            "and sentiment label.\n"
            "Passage:\n${text}\n\n"
            "Use only the available response schema and keep strings short."
        ),
        key="analysis-task",
    )
    return PromptTemplate[ReviewAnalysis](
        ns=_PROMPT_NS,
        key="integration-structured",
        name="structured_review",
        sections=[section],
    )


# =============================================================================
# Tests
# =============================================================================


def test_codex_adapter_returns_text(tmp_path: Path) -> None:
    """Verify that the adapter returns text from a simple prompt."""
    adapter = _make_adapter(tmp_path)
    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="integration tests")
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_codex_adapter_publishes_prompt_rendered_event(tmp_path: Path) -> None:
    """Verify that PromptRendered is published before execution."""
    adapter = _make_adapter(tmp_path)
    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="prompt rendered test")
    )

    rendered_events: list[PromptRendered] = []
    session = _make_session()
    session.dispatcher.subscribe(PromptRendered, rendered_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert len(rendered_events) == 1
    event = rendered_events[0]
    assert event.prompt_name == "greeting"
    assert event.adapter == "codex_app_server"
    assert "prompt rendered test" in event.rendered_prompt


def test_codex_adapter_processes_dynamic_tool(tmp_path: Path) -> None:
    """Verify that WINK tools bridged as dynamic tools are invoked."""
    tool = _build_uppercase_tool()
    prompt = Prompt(_build_tool_prompt(tool)).bind(
        TransformRequest(text="integration tests")
    )

    tool_events: list[ToolInvoked] = []
    session = _make_session()
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    adapter = _make_adapter(tmp_path)
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None and response.text.strip()
    _assert_prompt_usage(session)


def test_codex_adapter_parses_structured_output(tmp_path: Path) -> None:
    """Verify that the adapter parses structured output correctly."""
    adapter = _make_adapter(tmp_path)
    prompt = Prompt(_build_structured_prompt()).bind(
        ReviewParams(
            text=(
                "The new release shipped important bug fixes and improved "
                "the onboarding flow. Early adopters report smoother setup, "
                "though some mention learning curves."
            ),
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    _assert_prompt_usage(session)


def test_codex_adapter_publishes_tool_invoked_for_native_tools(
    tmp_path: Path,
) -> None:
    """Verify that ToolInvoked events are published for native Codex tools.

    Codex uses its own tools (commandExecution, fileChange, etc.) and the
    adapter should publish ToolInvoked events for each.
    """
    # Write a file so the agent has something to read
    readme = tmp_path / "README.md"
    readme.write_text("# Test\n\nSample content for integration tests.\n")

    section = MarkdownSection(
        title="Task",
        template=(
            "Read the file README.md in the current directory and summarise "
            "its contents in a single sentence."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="native-tool-test",
            name="native_tool",
            sections=[section],
        )
    )

    tool_events: list[ToolInvoked] = []
    session = _make_session()
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    adapter = _make_adapter(tmp_path)
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    # The agent should have used at least one native tool (command or file)
    assert len(tool_events) >= 1, (
        "Expected at least one ToolInvoked event from native Codex tools. "
        f"Got: {tool_events!r}"
    )
    _assert_prompt_usage(session)


def test_codex_adapter_thread_resume(tmp_path: Path) -> None:
    """Verify thread resume: a second evaluate reuses the existing thread."""
    adapter = _make_adapter(tmp_path, reuse_thread=True)

    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="thread resume test (turn 1)")
    )

    session = _make_session()
    response1 = adapter.evaluate(prompt, session=session)
    assert response1.text is not None

    # Check that a thread ID was stored in adapter state
    assert adapter._last_thread is not None
    first_thread_id = adapter._last_thread.thread_id
    assert first_thread_id is not None

    # Second evaluation with the same adapter should resume the thread
    prompt2 = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="thread resume test (turn 2)")
    )
    response2 = adapter.evaluate(prompt2, session=session)
    assert response2.text is not None


def test_codex_adapter_with_workspace_section(tmp_path: Path) -> None:
    """Verify adapter works with a CodexWorkspaceSection."""
    session = _make_session()
    workspace = CodexWorkspaceSection(session=session)

    try:
        adapter = CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=_get_model()),
            client_config=CodexAppServerClientConfig(
                approval_policy="never",
            ),
        )

        section = MarkdownSection(
            title="Task",
            template="Say hello. Reply in one sentence.",
            key="task",
        )
        prompt = Prompt(
            PromptTemplate(
                ns=_PROMPT_NS,
                key="workspace-test",
                name="ws_test",
                sections=[workspace, section],
            )
        )

        response = adapter.evaluate(prompt, session=session)
        assert response.text is not None
        assert response.text.strip()
    finally:
        workspace.cleanup()


def test_codex_adapter_with_model_config(tmp_path: Path) -> None:
    """Verify adapter works with explicit model configuration."""
    adapter = CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(
            model=_get_model(),
            effort="high",
        ),
        client_config=CodexAppServerClientConfig(
            cwd=str(tmp_path),
            approval_policy="never",
        ),
    )

    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="model config test")
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)
