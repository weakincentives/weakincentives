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

"""Tests for Claude Agent SDK MCP server creation and visibility expansion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    create_mcp_server,
    make_async_handler,
)
from weakincentives.adapters.claude_agent_sdk._visibility_signal import (
    VisibilityExpansionSignal,
)
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
    SectionVisibility,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


def _make_prompt_with_resources(
    resources: dict[type[object], object],
) -> Prompt[object]:
    """Create a prompt with resources bound in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="bridge-test"))
    prompt = prompt.bind(resources=resources)
    prompt.resources.__enter__()
    return prompt


@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str


@dataclass(slots=True, frozen=True)
class SearchResult:
    matches: int

    def render(self) -> str:
        return f"Found {self.matches} matches"


def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    del context
    return ToolResult.ok(
        SearchResult(matches=5), message=f"Found matches for {params.query}"
    )


search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for content",
    handler=search_handler,
)


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def prompt() -> Prompt[object]:
    """Create a prompt in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="bridge-test"))
    prompt.resources.__enter__()
    return prompt


@pytest.fixture
def mock_adapter() -> MagicMock:
    return MagicMock()


class TestMakeAsyncHandler:
    """Tests for make_async_handler function."""

    def test_creates_async_wrapper(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that make_async_handler creates an async wrapper."""
        bridged = BridgedTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        async_handler = make_async_handler(bridged)

        # Verify it's a coroutine function
        assert asyncio.iscoroutinefunction(async_handler)

    def test_async_handler_returns_result(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that async handler executes and returns result."""
        bridged = BridgedTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        async_handler = make_async_handler(bridged)
        result = asyncio.run(async_handler({"query": "test"}))

        assert result["isError"] is False
        # Output uses render() from SearchResult which returns "Found 5 matches"
        assert "Found 5 matches" in result["content"][0]["text"]

    def test_async_handler_uses_to_thread(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """make_async_handler delegates to asyncio.to_thread to avoid blocking."""
        bridged = BridgedTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        async_handler = make_async_handler(bridged)
        called_with: list[tuple[object, ...]] = []
        original_to_thread = asyncio.to_thread

        async def tracking_to_thread(*args: object, **kwargs: object) -> object:
            called_with.append(args)
            return await original_to_thread(*args, **kwargs)  # type: ignore[arg-type]

        async def _run() -> dict[str, object]:
            with patch(
                "weakincentives.adapters._shared._bridge.asyncio.to_thread",
                side_effect=tracking_to_thread,
            ):
                return await async_handler({"query": "test"})

        result = asyncio.run(_run())
        assert result["isError"] is False
        assert len(called_with) == 1
        assert called_with[0][0] is bridged


class TestCreateMcpServer:
    """Tests for create_mcp_server function."""

    def test_creates_mcp_server_config(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that create_mcp_server creates an MCP server config."""
        bridged = BridgedTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        mock_sdk_tool = MagicMock(return_value=lambda f: f)
        mock_create_server = MagicMock(return_value={"type": "sdk"})

        # Mock SdkMcpTool as a simple type alias
        class MockSdkMcpTool:
            pass

        with patch.dict(
            "sys.modules",
            {
                "claude_agent_sdk": MagicMock(
                    create_sdk_mcp_server=mock_create_server,
                    tool=mock_sdk_tool,
                    SdkMcpTool=MockSdkMcpTool,
                )
            },
        ):
            result = create_mcp_server((bridged,), server_name="test-server")

        assert result == {"type": "sdk"}
        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["name"] == "test-server"
        assert call_kwargs["version"] == "1.0.0"
        assert len(call_kwargs["tools"]) == 1

    def test_raises_import_error_when_sdk_missing(self) -> None:
        """Test that create_mcp_server raises ImportError when SDK is missing."""
        with (
            patch.dict("sys.modules", {"claude_agent_sdk": None}),
            pytest.raises(ImportError, match="claude-agent-sdk is required"),
        ):
            create_mcp_server(())


class TestVisibilityExpansionRequiredPropagation:
    """Tests for VisibilityExpansionRequired exception handling via signal."""

    def test_bridged_tool_stores_visibility_expansion_in_signal(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that BridgedTool stores VisibilityExpansionRequired in signal."""

        def expanding_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            del context
            raise VisibilityExpansionRequired(
                "Model requested expansion",
                requested_overrides={("section", "key"): SectionVisibility.FULL},
                reason="Need more details",
                section_keys=("section.key",),
            )

        expanding_tool = Tool[SearchParams, SearchResult](
            name="expanding",
            description="Tool that requests expansion",
            handler=expanding_handler,
        )

        visibility_signal = VisibilityExpansionSignal()

        bridged = BridgedTool(
            name="expanding",
            description="Tool that requests expansion",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=expanding_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            visibility_signal=visibility_signal,
        )

        # Should return success response (not error - the tool worked correctly)
        result = bridged({"query": "test"})

        # Verify success response format
        assert result["isError"] is False
        assert "content" in result
        assert result["content"][0]["type"] == "text"
        assert "section.key" in result["content"][0]["text"]

        # Verify the exception was stored in signal
        stored_exc = visibility_signal.get_and_clear()
        assert stored_exc is not None
        assert isinstance(stored_exc, VisibilityExpansionRequired)
        assert stored_exc.requested_overrides == {
            ("section", "key"): SectionVisibility.FULL
        }
        assert stored_exc.section_keys == ("section.key",)
        assert stored_exc.reason == "Need more details"

    def test_bridged_tool_without_signal_returns_success_response(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that BridgedTool without signal still returns success response."""

        def expanding_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            del context
            raise VisibilityExpansionRequired(
                "Expansion required",
                requested_overrides={("a", "b"): SectionVisibility.FULL},
                reason="Test reason",
                section_keys=("a.b",),
            )

        expanding_tool = Tool[SearchParams, SearchResult](
            name="expanding",
            description="Tool that requests expansion",
            handler=expanding_handler,
        )

        # No visibility_signal provided
        bridged = BridgedTool(
            name="expanding",
            description="Tool that requests expansion",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=expanding_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        # Should return success response (tool worked correctly)
        result = bridged({"query": "test"})
        assert result["isError"] is False
        assert "a.b" in result["content"][0]["text"]

    def test_async_handler_stores_visibility_expansion_in_signal(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that async handler wrapper uses signal for VisibilityExpansionRequired."""

        def expanding_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            del context
            raise VisibilityExpansionRequired(
                "Expansion required",
                requested_overrides={("a", "b"): SectionVisibility.FULL},
                reason="Test reason",
                section_keys=("a.b",),
            )

        expanding_tool = Tool[SearchParams, SearchResult](
            name="expanding",
            description="Tool that requests expansion",
            handler=expanding_handler,
        )

        visibility_signal = VisibilityExpansionSignal()

        bridged = BridgedTool(
            name="expanding",
            description="Tool that requests expansion",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=expanding_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            visibility_signal=visibility_signal,
        )

        async_handler = make_async_handler(bridged)

        # The async wrapper should return success response
        result = asyncio.run(async_handler({"query": "test"}))
        assert result["isError"] is False

        # And the exception should be in the signal
        stored_exc = visibility_signal.get_and_clear()
        assert stored_exc is not None
        assert stored_exc.section_keys == ("a.b",)

    def test_passes_filesystem_to_tool_context(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that filesystem is accessed via prompt resources."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        captured_filesystem: list[object] = []

        def capture_context_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            captured_filesystem.append(context.filesystem)
            return ToolResult.ok(
                SearchResult(matches=3), message=f"Searched for {params.query}"
            )

        capture_tool = Tool[SearchParams, SearchResult](
            name="capture",
            description="Tool that captures context",
            handler=capture_context_handler,
        )

        test_filesystem = InMemoryFilesystem()
        prompt = _make_prompt_with_resources({Filesystem: test_filesystem})

        bridged = BridgedTool(
            name="capture",
            description="Tool that captures context",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=capture_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        _ = bridged({"query": "test"})

        assert len(captured_filesystem) == 1
        assert captured_filesystem[0] is test_filesystem


class TestVisibilityExpansionSignal:
    """Tests for VisibilityExpansionSignal thread-safe container."""

    def test_signal_stores_exception(self) -> None:
        """Test that signal stores exception correctly."""
        signal = VisibilityExpansionSignal()

        exc = VisibilityExpansionRequired(
            "Test",
            requested_overrides={("a",): SectionVisibility.FULL},
            reason="Test reason",
            section_keys=("a",),
        )

        signal.set(exc)
        assert signal.is_set()

        stored = signal.get_and_clear()
        assert stored is exc
        assert not signal.is_set()

    def test_signal_first_one_wins(self) -> None:
        """Test that only the first exception is stored."""
        signal = VisibilityExpansionSignal()

        exc1 = VisibilityExpansionRequired(
            "First",
            requested_overrides={("first",): SectionVisibility.FULL},
            reason="First reason",
            section_keys=("first",),
        )
        exc2 = VisibilityExpansionRequired(
            "Second",
            requested_overrides={("second",): SectionVisibility.FULL},
            reason="Second reason",
            section_keys=("second",),
        )

        signal.set(exc1)
        signal.set(exc2)  # Should be ignored

        stored = signal.get_and_clear()
        assert stored is exc1
        assert stored.section_keys == ("first",)

    def test_signal_get_and_clear_clears(self) -> None:
        """Test that get_and_clear clears the signal."""
        signal = VisibilityExpansionSignal()

        exc = VisibilityExpansionRequired(
            "Test",
            requested_overrides={("a",): SectionVisibility.FULL},
            reason="Test",
            section_keys=("a",),
        )

        signal.set(exc)
        assert signal.is_set()

        _ = signal.get_and_clear()
        assert not signal.is_set()

        # Second call should return None
        assert signal.get_and_clear() is None

    def test_signal_empty_returns_none(self) -> None:
        """Test that empty signal returns None."""
        signal = VisibilityExpansionSignal()

        assert not signal.is_set()
        assert signal.get_and_clear() is None
