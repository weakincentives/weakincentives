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

"""Shared fixtures and mocks for Claude Agent SDK adapter tests."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncGenerator, AsyncIterable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.claude_agent_sdk._hooks import HookContext
from weakincentives.adapters.claude_agent_sdk._transcript_collector import (
    TranscriptCollector,
    TranscriptCollectorConfig,
)
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt


# ---------------------------------------------------------------------------
# Mock SDK types
# ---------------------------------------------------------------------------
@dataclass
class MockResultMessage:
    """Mock ResultMessage for testing."""

    result: str | None = None
    structured_output: dict[str, object] | None = None
    usage: dict[str, int] | None = None


class MockClaudeAgentOptions:
    """Mock ClaudeAgentOptions for testing."""

    model: str | None
    cwd: str | None
    permission_mode: str | None
    max_turns: int | None
    max_budget_usd: float | None
    betas: list[str] | None
    output_format: dict[str, object] | None
    allowed_tools: list[str] | None
    disallowed_tools: list[str] | None
    reasoning: str | None
    mcp_servers: dict[str, object] | None
    hooks: dict[str, list[object]] | None
    can_use_tool: object | None

    def __init__(self, **kwargs: object) -> None:
        self.can_use_tool = None
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class StrictClaudeAgentOptionsNoReasoning:
    """Mock options type that rejects unsupported keyword arguments."""

    model: str | None = None
    cwd: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    betas: list[str] | None = None
    output_format: dict[str, object] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    mcp_servers: dict[str, object] | None = None
    hooks: dict[str, list[object]] | None = None
    can_use_tool: object | None = None
    setting_sources: list[str] | None = None
    env: dict[str, str] | None = None
    stderr: object | None = None


@dataclass
class MockHookMatcher:
    """Mock HookMatcher for testing."""

    matcher: str | None = None
    hooks: list[object] | None = None


class MockSDKQuery:
    """Mock for sdk[] async generator function."""

    captured_options: ClassVar[list[MockClaudeAgentOptions]] = []
    captured_prompts: ClassVar[list[str]] = []
    _results: ClassVar[list[object]] = []
    _error: ClassVar[Exception | None] = None

    @classmethod
    def reset(cls) -> None:
        """Reset captured options and results between tests."""
        cls.captured_options = []
        cls.captured_prompts = []
        cls._results = []
        cls._error = None

    @classmethod
    def set_results(cls, results: list[object]) -> None:
        """Set the results to yield from query."""
        cls._results = results

    @classmethod
    def set_error(cls, error: Exception) -> None:
        """Set an error to raise during iteration."""
        cls._error = error

    @classmethod
    async def query(
        cls,
        *,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: MockClaudeAgentOptions,
    ) -> AsyncGenerator[object, None]:
        """Mock sdk[] that yields configured results."""
        if not isinstance(prompt, str):
            prompt_content = ""
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    message = msg["message"]
                    if isinstance(message, dict) and "content" in message:
                        prompt_content = message["content"]
            cls.captured_prompts.append(prompt_content)
        else:
            cls.captured_prompts.append(prompt)

        cls.captured_options.append(options)

        if cls._error is not None:
            raise cls._error

        for result in cls._results:
            yield result


class MockTransport:
    """Mock transport for ClaudeSDKClient."""

    async def end_input(self) -> None:
        """Mock end_input - signals EOF to subprocess."""


class MockClaudeSDKClient:
    """Mock for ClaudeSDKClient."""

    def __init__(
        self,
        options: MockClaudeAgentOptions | None = None,
        transport: object | None = None,
    ) -> None:
        """Initialize mock client."""
        self.options = options or MockClaudeAgentOptions()
        self._connected = False
        self._transport: MockTransport | None = None
        MockSDKQuery.captured_options.append(self.options)

    async def connect(
        self, prompt: str | AsyncIterable[dict[str, Any]] | None = None
    ) -> None:
        """Mock connect method."""
        self._connected = True
        self._transport = MockTransport()
        self._prompt_stream = prompt
        if prompt is not None and not isinstance(prompt, str):
            prompt_content = ""
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    message = msg["message"]
                    if isinstance(message, dict) and "content" in message:
                        prompt_content = message["content"]
                break
            MockSDKQuery.captured_prompts.append(prompt_content)
        elif isinstance(prompt, str):
            MockSDKQuery.captured_prompts.append(prompt)

    async def disconnect(self) -> None:
        """Mock disconnect method."""
        self._connected = False
        self._transport = None

    async def query(self, prompt: str, session_id: str = "default") -> None:
        """Mock query method for sending messages."""
        MockSDKQuery.captured_prompts.append(prompt)

    async def receive_messages(self) -> AsyncGenerator[object, None]:
        """Mock receive_messages that yields configured results."""
        if MockSDKQuery._error is not None:
            raise MockSDKQuery._error

        for result in MockSDKQuery._results:
            yield result

    async def receive_response(self) -> AsyncGenerator[object, None]:
        """Mock receive_response that yields results until ResultMessage."""
        if MockSDKQuery._error is not None:
            raise MockSDKQuery._error

        for result in MockSDKQuery._results:
            yield result
            if isinstance(result, MockResultMessage):
                return


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class SimpleOutput:
    message: str


@dataclass(slots=True, frozen=True)
class NullableOutput:
    count: int | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_mock_query(results: list[object]) -> None:
    """Set up MockSDKQuery with specific results."""
    MockSDKQuery.reset()
    MockSDKQuery.set_results(results)


def create_sdk_mock() -> MagicMock:
    """Create mock SDK module with query function."""
    mock_sdk = MagicMock()
    mock_sdk.query = MockSDKQuery.query
    return mock_sdk


@contextmanager
def sdk_patches() -> Generator[None, None, None]:
    """Context manager for common SDK patches."""
    with (
        patch(
            "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
            return_value=create_sdk_mock(),
        ),
        patch(
            "claude_agent_sdk.ClaudeSDKClient",
            MockClaudeSDKClient,
        ),
        patch(
            "claude_agent_sdk.types.ClaudeAgentOptions",
            MockClaudeAgentOptions,
        ),
        patch(
            "claude_agent_sdk.types.HookMatcher",
            MockHookMatcher,
        ),
        patch(
            "claude_agent_sdk.types.ResultMessage",
            MockResultMessage,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        prompt=cast("PromptProtocol[object]", _make_prompt()),
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )


@pytest.fixture
def simple_prompt() -> Prompt[SimpleOutput]:
    template = PromptTemplate[SimpleOutput](
        ns="test",
        key="simple",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template="Say hello",
            ),
        ],
    )
    return Prompt(template)


@pytest.fixture
def untyped_prompt() -> Prompt[None]:
    template = PromptTemplate[None](
        ns="test",
        key="untyped",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template="Do something",
            ),
        ],
    )
    return Prompt(template)


@pytest.fixture
def nullable_prompt() -> Prompt[NullableOutput]:
    template = PromptTemplate[NullableOutput](
        ns="test",
        key="nullable",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template="Return optional count",
            ),
        ],
    )
    return Prompt(template)


# ---------------------------------------------------------------------------
# Bridge test fixtures and shared helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str


@dataclass(slots=True, frozen=True)
class SearchResult:
    matches: int

    def render(self) -> str:
        return f"Found {self.matches} matches"


@dataclass(slots=True, frozen=True)
class EmptyRenderResult:
    """Result with empty render()."""

    def render(self) -> str:
        return ""


def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    del context
    return ToolResult.ok(
        SearchResult(matches=5), message=f"Found matches for {params.query}"
    )


def failing_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    del context
    raise RuntimeError("Handler failed")


search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for content",
    handler=search_handler,
)

failing_tool = Tool[SearchParams, SearchResult](
    name="failing",
    description="A tool that fails",
    handler=failing_handler,
)

no_handler_tool = Tool[SearchParams, SearchResult](
    name="no_handler",
    description="A tool without handler",
    handler=None,
)


def _make_prompt_with_resources(
    resources: dict[type[object], object],
) -> Prompt[object]:
    """Create a prompt with resources bound in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="bridge-test"))
    prompt = prompt.bind(resources=resources)
    prompt.resources.__enter__()
    return prompt


@pytest.fixture
def bridge_prompt() -> Prompt[object]:
    """Create a prompt in active context for bridge tests."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="bridge-test"))
    prompt.resources.__enter__()
    return prompt


@pytest.fixture
def mock_adapter() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Transcript collector test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transcript_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test transcripts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def transcript_config() -> TranscriptCollectorConfig:
    """Create a test configuration."""
    return TranscriptCollectorConfig(
        poll_interval=0.01,  # Fast polling for tests
        subagent_discovery_interval=0.02,
        max_read_bytes=1024,
        emit_raw=True,
    )


@pytest.fixture
def transcript_collector(
    transcript_config: TranscriptCollectorConfig,
) -> TranscriptCollector:
    """Create a test collector."""
    return TranscriptCollector(
        prompt_name="test-prompt",
        config=transcript_config,
    )
