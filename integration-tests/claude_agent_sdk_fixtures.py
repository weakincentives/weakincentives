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

"""Shared fixtures and helpers for Claude Agent SDK integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    get_default_model,
)
from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events import PromptExecuted
from weakincentives.runtime.session import Session

_MODEL_ENV_VAR = "CLAUDE_AGENT_SDK_TEST_MODEL"
_PROMPT_NS = "integration/claude-agent-sdk"


def _is_bedrock_mode() -> bool:
    """Check if running in Bedrock mode based on environment."""
    return os.getenv("CLAUDE_CODE_USE_BEDROCK") == "1" and "AWS_REGION" in os.environ


def _has_credentials() -> bool:
    """Check if Bedrock or Anthropic API credentials are available."""
    return _is_bedrock_mode() or "ANTHROPIC_API_KEY" in os.environ


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_credentials(),
        reason=(
            "Neither CLAUDE_CODE_USE_BEDROCK+AWS_REGION nor ANTHROPIC_API_KEY set."
        ),
    ),
    pytest.mark.timeout(60),
]


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
class ReadFileParams:
    """Prompt parameters for reading or writing a file path."""

    file_path: str


@dataclass(slots=True)
class EchoParams:
    """Parameters for simple echo or command prompts."""

    text: str


@dataclass(slots=True)
class EmptyParams:
    """Empty params placeholder for prompts without parameters."""

    pass


@dataclass(slots=True, frozen=True)
class FileReadTestResult:
    """Result of file read test."""

    read_succeeded: bool
    content: str | None
    error_message: str | None


def _get_model() -> str:
    """Return the model name used for integration tests.

    Uses get_default_model() which returns Sonnet 4.5 in the appropriate
    format based on whether Bedrock is configured.
    """
    return os.environ.get(_MODEL_ENV_VAR, get_default_model())


def _make_config(tmp_path: Path, **kwargs: object) -> ClaudeAgentSDKClientConfig:
    """Build a ClaudeAgentSDKClientConfig with explicit cwd.

    All tests MUST use this helper to ensure cwd points to a temporary
    directory, preventing snapshot operations from creating commits in
    the actual repository.

    The IsolationConfig automatically inherits host authentication
    (Bedrock or Anthropic API) when no explicit api_key is provided.
    """
    config_kwargs: dict[str, object] = {
        "permission_mode": "bypassPermissions",
        "cwd": str(tmp_path),
    }
    config_kwargs.update(kwargs)
    return ClaudeAgentSDKClientConfig(**config_kwargs)


def _make_adapter(tmp_path: Path, **kwargs: object) -> ClaudeAgentSDKAdapter:
    """Create an adapter with explicit cwd pointing to tmp_path."""
    model = kwargs.pop("model", None) or _get_model()
    client_config = kwargs.pop("client_config", None) or _make_config(tmp_path)
    return ClaudeAgentSDKAdapter(
        model=model,  # type: ignore[arg-type]
        client_config=client_config,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _make_session_with_usage_tracking() -> Session:
    return Session()


def _assert_prompt_usage(session: Session) -> None:
    event = session[PromptExecuted].latest()
    assert event is not None, "Expected a PromptExecuted event."
    usage = event.usage
    assert usage is not None, "Expected token usage to be recorded."
    assert usage.total_tokens is not None and usage.total_tokens > 0


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
