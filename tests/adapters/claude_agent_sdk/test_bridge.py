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

"""Tests for Claude Agent SDK MCP tool bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    create_bridged_tools,
)
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
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


class TestBridgedTool:
    def test_executes_handler_successfully(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
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

        result = bridged({"query": "test"})

        assert result["isError"] is False
        # Output uses render() from SearchResult which returns "Found 5 matches"
        assert "Found 5 matches" in result["content"][0]["text"]

    def test_returns_error_for_no_handler(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        bridged = BridgedTool(
            name="no_handler",
            description="No handler",
            input_schema={},
            tool=no_handler_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({})

        assert result["isError"] is True
        assert "no handler" in result["content"][0]["text"]

    def test_returns_validation_error(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        bridged = BridgedTool(
            name="search",
            description="Search",
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

        result = bridged({"wrong_field": "value"})

        assert result["isError"] is True
        assert "error" in result["content"][0]["text"].lower()

    def test_catches_handler_exception(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        bridged = BridgedTool(
            name="failing",
            description="Fails",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=failing_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    def test_uses_render_method_on_result_value(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that bridged tool uses render() on result value when available."""
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

        result = bridged({"query": "test"})

        # SearchResult.render() returns "Found 5 matches"
        assert result["content"][0]["text"] == "Found 5 matches"

    def test_falls_back_to_message_when_render_empty(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that bridged tool falls back to message when render() returns empty."""

        def empty_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[EmptyRenderResult]:
            del context
            return ToolResult.ok(
                EmptyRenderResult(), message=f"Searched for {params.query}"
            )

        empty_tool = Tool[SearchParams, EmptyRenderResult](
            name="empty_render",
            description="Tool with empty render",
            handler=empty_handler,
        )

        bridged = BridgedTool(
            name="empty_render",
            description="Tool with empty render",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=empty_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        # Falls back to message since render() returned empty
        assert result["content"][0]["text"] == "Searched for test"

    def test_exclude_value_from_context_uses_message_only(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that exclude_value_from_context skips value rendering."""

        def data_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            del context
            return ToolResult(
                message="Query processed",
                value=SearchResult(matches=100),
                success=True,
                exclude_value_from_context=True,
            )

        data_tool = Tool[SearchParams, SearchResult](
            name="data_tool",
            description="Tool returning large data",
            handler=data_handler,
        )

        bridged = BridgedTool(
            name="data_tool",
            description="Tool returning large data",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=data_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        # Should use message only, not the rendered value
        assert result["content"][0]["text"] == "Query processed"
        assert "100" not in result["content"][0]["text"]


class TestCreateBridgedTools:
    def test_creates_bridged_tools(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        tools = (search_tool,)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].name == "search"
        assert bridged[0].description == "Search for content"

    def test_skips_tools_without_handlers(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        tools = (search_tool, no_handler_tool)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].name == "search"

    def test_generates_input_schema(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        tools = (search_tool,)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        schema = bridged[0].input_schema
        assert "properties" in schema
        assert "query" in schema["properties"]

    def test_handles_none_params_type(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        def null_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
            del params, context
            return ToolResult.ok(None, message="ok")

        null_tool = Tool[None, None](
            name="null_tool",
            description="A tool with no params",
            handler=null_handler,
        )

        bridged = create_bridged_tools(
            (null_tool,),
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].input_schema == {"type": "object", "properties": {}}

    def test_executes_tool_with_none_params_type(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that calling a bridged tool with None params works."""

        def null_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
            del params, context
            return ToolResult.ok(None, message="executed")

        null_tool = Tool[None, None](
            name="null_tool",
            description="A tool with no params",
            handler=null_handler,
        )

        bridged = BridgedTool(
            name="null_tool",
            description="A tool with no params",
            input_schema={"type": "object", "properties": {}},
            tool=null_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({})

        assert result["isError"] is False
        assert "executed" in result["content"][0]["text"]

    def test_empty_tools_returns_empty(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        bridged = create_bridged_tools(
            (),
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert bridged == ()


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


class TestCreateBridgedToolsWithFilesystem:
    def test_passes_filesystem_to_bridged_tools(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that create_bridged_tools passes filesystem via prompt resources."""
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

        bridged_tools = create_bridged_tools(
            (capture_tool,),
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged_tools) == 1
        _ = bridged_tools[0]({"query": "test"})

        assert len(captured_filesystem) == 1
        assert captured_filesystem[0] is test_filesystem


class TestBudgetTrackerInResourceRegistry:
    def test_passes_budget_tracker_to_tool_context_via_resources(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """Test that budget_tracker in resources is accessible via context.budget_tracker."""
        from weakincentives.budget import Budget, BudgetTracker

        captured_budget_tracker: list[BudgetTracker | None] = []

        def capture_context_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            captured_budget_tracker.append(context.budget_tracker)
            return ToolResult.ok(
                SearchResult(matches=3), message=f"Searched for {params.query}"
            )

        capture_tool = Tool[SearchParams, SearchResult](
            name="capture",
            description="Tool that captures context",
            handler=capture_context_handler,
        )

        test_budget = Budget(max_total_tokens=1000)
        test_tracker = BudgetTracker(budget=test_budget)

        # Budget tracker must be in prompt resources to be accessible via context
        prompt = _make_prompt_with_resources({BudgetTracker: test_tracker})

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
            budget_tracker=None,  # Now unused; tracker is in resources
        )

        _ = bridged({"query": "test"})

        assert len(captured_budget_tracker) == 1
        assert captured_budget_tracker[0] is test_tracker

    def test_create_bridged_tools_passes_budget_tracker_via_resources(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """Test that create_bridged_tools passes budget_tracker via resources."""
        from weakincentives.budget import Budget, BudgetTracker

        captured_budget_tracker: list[BudgetTracker | None] = []

        def capture_context_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            captured_budget_tracker.append(context.budget_tracker)
            return ToolResult.ok(
                SearchResult(matches=3), message=f"Searched for {params.query}"
            )

        capture_tool = Tool[SearchParams, SearchResult](
            name="capture",
            description="Tool that captures context",
            handler=capture_context_handler,
        )

        test_budget = Budget(max_total_tokens=1000)
        test_tracker = BudgetTracker(budget=test_budget)

        # Budget tracker must be in prompt resources to be accessible via context
        prompt = _make_prompt_with_resources({BudgetTracker: test_tracker})

        bridged_tools = create_bridged_tools(
            (capture_tool,),
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,  # Now unused; tracker is in resources
        )

        assert len(bridged_tools) == 1
        _ = bridged_tools[0]({"query": "test"})

        assert len(captured_budget_tracker) == 1
        assert captured_budget_tracker[0] is test_tracker


class TestBridgedToolTransactionalExecution:
    """Tests for BridgedTool transactional execution."""

    def test_restores_state_on_tool_failure(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that state is restored when tool returns success=False."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        test_fs = InMemoryFilesystem()
        test_fs.write("/test.txt", "initial content")

        prompt = _make_prompt_with_resources({Filesystem: test_fs})

        def failing_result_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            # Modify filesystem before returning failure
            if context.filesystem is not None:
                context.filesystem.write("/test.txt", "modified content")
            return ToolResult(
                message="Tool failed",
                value=SearchResult(matches=0),
                success=False,
            )

        fail_tool = Tool[SearchParams, SearchResult](
            name="fail_tool",
            description="Tool that returns failure",
            handler=failing_result_handler,
        )

        bridged = BridgedTool(
            name="fail_tool",
            description="Tool that returns failure",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=fail_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        # Filesystem should be restored to initial content
        assert test_fs.read("/test.txt").content == "initial content"

    def test_restores_state_on_exception(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that state is restored when tool raises exception."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        test_fs = InMemoryFilesystem()
        test_fs.write("/test.txt", "initial content")

        prompt = _make_prompt_with_resources({Filesystem: test_fs})

        def exception_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            # Modify filesystem before raising exception
            if context.filesystem is not None:
                context.filesystem.write("/test.txt", "modified content")
            raise RuntimeError("Handler exploded")

        exception_tool = Tool[SearchParams, SearchResult](
            name="exception_tool",
            description="Tool that raises exception",
            handler=exception_handler,
        )

        bridged = BridgedTool(
            name="exception_tool",
            description="Tool that raises exception",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=exception_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        # Filesystem should be restored to initial content
        assert test_fs.read("/test.txt").content == "initial content"

    def test_restores_state_on_validation_error(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that state is restored when validation fails."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        test_fs = InMemoryFilesystem()
        test_fs.write("/test.txt", "initial content")

        prompt = _make_prompt_with_resources({Filesystem: test_fs})

        bridged = BridgedTool(
            name="search",
            description="Search tool",
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

        # Pass invalid args to trigger validation error
        result = bridged({"wrong_field": "value"})

        assert result["isError"] is True
        assert "error" in result["content"][0]["text"].lower()
        # Filesystem should remain at initial content (no restore needed since
        # validation fails before handler runs, but state is restored)
        assert test_fs.read("/test.txt").content == "initial content"

    def test_restores_state_on_visibility_expansion(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that state is restored when VisibilityExpansionRequired is raised."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        test_fs = InMemoryFilesystem()
        test_fs.write("/test.txt", "initial content")

        prompt = _make_prompt_with_resources({Filesystem: test_fs})

        def visibility_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            # Modify filesystem before raising visibility expansion
            if context.filesystem is not None:
                context.filesystem.write("/test.txt", "modified content")
            raise VisibilityExpansionRequired(
                "Need more context",
                requested_overrides={("section", "key"): SectionVisibility.FULL},
                reason="More details needed",
                section_keys=("section.key",),
            )

        visibility_tool = Tool[SearchParams, SearchResult](
            name="visibility_tool",
            description="Tool that requests visibility expansion",
            handler=visibility_handler,
        )

        visibility_signal = VisibilityExpansionSignal()

        bridged = BridgedTool(
            name="visibility_tool",
            description="Tool that requests visibility expansion",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=visibility_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            visibility_signal=visibility_signal,
        )

        # Should return success response (tool worked correctly)
        result = bridged({"query": "test"})
        assert result["isError"] is False

        # Filesystem should be restored to initial content
        assert test_fs.read("/test.txt").content == "initial content"

        # Exception should be stored in signal
        stored_exc = visibility_signal.get_and_clear()
        assert stored_exc is not None

    def test_no_filesystem_when_not_in_resources(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """Test that when no filesystem is in resources, context.filesystem is None."""
        captured_filesystem: list[object] = []

        def capture_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            captured_filesystem.append(context.filesystem)
            return ToolResult.ok(SearchResult(matches=0), message="Tool executed")

        capture_tool = Tool[SearchParams, SearchResult](
            name="capture_tool",
            description="Tool that captures context",
            handler=capture_handler,
        )

        # Create prompt without filesystem in resources
        prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="no-fs-test"))
        prompt.resources.__enter__()

        bridged = BridgedTool(
            name="capture_tool",
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

        result = bridged({"query": "test"})

        assert result["isError"] is False
        # Without filesystem in resources, context.filesystem should be None
        assert captured_filesystem == [None]


class TestCreateBridgedToolsWithSession:
    """Tests for create_bridged_tools with session parameter."""

    def test_passes_session_to_bridged_tools(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Test that create_bridged_tools passes session to BridgedTool."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        test_fs = InMemoryFilesystem()
        test_fs.write("/test.txt", "initial content")

        prompt = _make_prompt_with_resources({Filesystem: test_fs})

        def failing_result_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            if context.filesystem is not None:
                context.filesystem.write("/test.txt", "modified content")
            return ToolResult(
                message="Tool failed",
                value=SearchResult(matches=0),
                success=False,
            )

        fail_tool = Tool[SearchParams, SearchResult](
            name="fail_tool",
            description="Tool that returns failure",
            handler=failing_result_handler,
        )

        bridged_tools = create_bridged_tools(
            (fail_tool,),
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged_tools) == 1
        result = bridged_tools[0]({"query": "test"})

        assert result["isError"] is True
        # Filesystem should be restored to initial content
        assert test_fs.read("/test.txt").content == "initial content"


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


class TestMCPToolExecutionState:
    """Tests for MCPToolExecutionState shared state between hooks and bridge."""

    def test_bridged_tool_uses_call_id_from_mcp_state(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """BridgedTool includes call_id from mcp_tool_state in ToolInvoked event."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        mcp_state = MCPToolExecutionState()
        # Enqueue with same params that will be passed to bridged tool
        mcp_state.enqueue("search", {"query": "test"}, "call-from-hook-789")

        bridged = BridgedTool(
            name="search",
            description="Search tool",
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
            mcp_tool_state=mcp_state,
        )

        bridged({"query": "test"})

        assert len(events) == 1
        assert events[0].call_id == "call-from-hook-789"
        assert events[0].name == "search"

    def test_bridged_tool_without_mcp_state_has_none_call_id(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """BridgedTool without mcp_tool_state has None call_id in ToolInvoked event."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        bridged = BridgedTool(
            name="search",
            description="Search tool",
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
            # No mcp_tool_state
        )

        bridged({"query": "test"})

        assert len(events) == 1
        assert events[0].call_id is None

    def test_mcp_state_defaults_to_empty(self) -> None:
        """MCPToolExecutionState defaults to empty queues."""
        state = MCPToolExecutionState()
        assert state.dequeue("any_tool", {}) is None

    def test_mcp_state_enqueue_dequeue_fifo(self) -> None:
        """MCPToolExecutionState uses FIFO ordering for same tool+params."""
        state = MCPToolExecutionState()
        params = {"key": "value"}

        state.enqueue("my_tool", params, "call-1")
        state.enqueue("my_tool", params, "call-2")
        state.enqueue("my_tool", params, "call-3")

        # Dequeue returns in FIFO order
        assert state.dequeue("my_tool", params) == "call-1"
        assert state.dequeue("my_tool", params) == "call-2"
        assert state.dequeue("my_tool", params) == "call-3"
        assert state.dequeue("my_tool", params) is None

    def test_mcp_state_handles_prefix_normalization(self) -> None:
        """MCPToolExecutionState normalizes mcp__wink__ prefix."""
        state = MCPToolExecutionState()
        params = {"plan": "test"}

        # Enqueue with prefix, dequeue without
        state.enqueue("mcp__wink__planning_setup_plan", params, "call-123")
        assert state.dequeue("planning_setup_plan", params) == "call-123"

        # Enqueue without prefix, dequeue with prefix
        state.enqueue("search", {"query": "test"}, "call-456")
        assert state.dequeue("mcp__wink__search", {"query": "test"}) == "call-456"

    def test_mcp_state_different_params_different_queues(self) -> None:
        """MCPToolExecutionState uses different queues for different params."""
        state = MCPToolExecutionState()

        # Same tool, different params - should be separate queues
        state.enqueue("search", {"query": "foo"}, "call-foo")
        state.enqueue("search", {"query": "bar"}, "call-bar")

        # Each dequeue gets the correct call_id based on params
        assert state.dequeue("search", {"query": "bar"}) == "call-bar"
        assert state.dequeue("search", {"query": "foo"}) == "call-foo"

    def test_mcp_state_supports_multiple_concurrent_tools(self) -> None:
        """MCPToolExecutionState can track multiple tools concurrently."""
        state = MCPToolExecutionState()

        state.enqueue("tool_a", {"x": 1}, "call-a")
        state.enqueue("tool_b", {"x": 2}, "call-b")
        state.enqueue("tool_c", {"x": 3}, "call-c")

        # Can dequeue in any order - each tool has its own queue
        assert state.dequeue("tool_b", {"x": 2}) == "call-b"
        assert state.dequeue("tool_a", {"x": 1}) == "call-a"
        assert state.dequeue("tool_c", {"x": 3}) == "call-c"


class TestBridgedToolPolicyEnforcement:
    """Tests for tool policy enforcement in BridgedTool._execute_handler."""

    def test_denying_policy_prevents_handler_execution(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """A denying policy blocks handler and returns isError."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision

        calls: list[str] = []

        def recording_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            calls.append(params.query)
            return ToolResult.ok(SearchResult(matches=1), message="ok")

        recording_tool = Tool[SearchParams, SearchResult](
            name="search",
            description="Search",
            handler=recording_handler,
        )

        @dataclass(frozen=True)
        class DenyPolicy:
            @property
            def name(self) -> str:
                return "deny_all"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.deny("Blocked by test policy.")

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del tool, params, result, context

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(recording_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-deny",
            sections=[section],
            policies=[DenyPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=recording_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        assert "Blocked by test policy" in result["content"][0]["text"]
        assert calls == []  # Handler never called

    def test_denying_policy_with_suggestions_includes_them_in_output(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """A denial with suggestions appends them to the error message."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision

        def dummy_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            return ToolResult.ok(SearchResult(matches=0), message="ok")

        recording_tool = Tool[SearchParams, SearchResult](
            name="search",
            description="Search",
            handler=dummy_handler,
        )

        @dataclass(frozen=True)
        class DenySuggestPolicy:
            @property
            def name(self) -> str:
                return "deny_suggest"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.deny(
                    "Missing prerequisite.",
                    suggestions=("Run lint first.",),
                )

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del tool, params, result, context

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(recording_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-deny-suggest",
            sections=[section],
            policies=[DenySuggestPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=recording_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "Missing prerequisite" in text
        assert "Suggestions:" in text
        assert "Run lint first" in text

    def test_allowing_policy_lets_handler_execute(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """An allowing policy permits handler execution."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision

        calls: list[str] = []

        def recording_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            calls.append(params.query)
            return ToolResult.ok(SearchResult(matches=1), message="ok")

        recording_tool = Tool[SearchParams, SearchResult](
            name="search",
            description="Search",
            handler=recording_handler,
        )

        @dataclass(frozen=True)
        class AllowPolicy:
            @property
            def name(self) -> str:
                return "allow_all"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.allow()

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del tool, params, result, context

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(recording_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-allow",
            sections=[section],
            policies=[AllowPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=recording_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is False
        assert calls == ["test"]

    def test_on_result_called_after_successful_execution(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """on_result is called after successful handler execution."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision

        on_result_calls: list[str] = []

        @dataclass(frozen=True)
        class TrackingPolicy:
            @property
            def name(self) -> str:
                return "tracking"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.allow()

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del params, context
                on_result_calls.append(f"{tool.name}:{result.success}")

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(search_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-track",
            sections=[section],
            policies=[TrackingPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        bridged({"query": "test"})

        assert on_result_calls == ["search:True"]

    def test_on_result_not_called_after_denial(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """on_result is NOT called when policy denies the call."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision

        on_result_calls: list[str] = []

        @dataclass(frozen=True)
        class DenyAndTrackPolicy:
            @property
            def name(self) -> str:
                return "deny_and_track"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.deny("Denied.")

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del params, context
                on_result_calls.append(f"{tool.name}:{result.success}")

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(search_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-deny-track",
            sections=[section],
            policies=[DenyAndTrackPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        assert on_result_calls == []  # on_result never called

    def test_policy_denial_dispatches_tool_invoked(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """Policy denial still dispatches a ToolInvoked event."""
        from weakincentives.prompt import MarkdownSection, PolicyDecision
        from weakincentives.runtime.events.types import ToolInvoked

        @dataclass(frozen=True)
        class DenyPolicy:
            @property
            def name(self) -> str:
                return "deny_all"

            def check(
                self,
                tool: Tool[object, object],
                params: object,
                *,
                context: ToolContext,
            ) -> PolicyDecision:
                del tool, params, context
                return PolicyDecision.deny("Blocked.")

            def on_result(
                self,
                tool: Tool[object, object],
                params: object,
                result: ToolResult[object],
                *,
                context: ToolContext,
            ) -> None:
                del tool, params, result, context

        section = MarkdownSection[SearchParams](
            title="Task",
            template="Search for ${query}",
            tools=(search_tool,),
            key="task",
        )
        template = PromptTemplate(
            ns="tests",
            key="policy-deny-event",
            sections=[section],
            policies=[DenyPolicy()],
        )
        prompt_with_policy: Prompt[object] = Prompt(template)
        prompt_with_policy.resources.__enter__()

        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, events.append)

        bridged = BridgedTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tool=search_tool,
            session=session,
            adapter=mock_adapter,
            prompt=cast("PromptProtocol[object]", prompt_with_policy),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        assert len(events) == 1
        event = events[0]
        assert event.name == "search"
        assert not event.result.success
        assert "Blocked" in event.rendered_output

    def test_no_policies_allows_execution(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """When no policies are configured, handler executes normally."""
        bridged = BridgedTool(
            name="search",
            description="Search",
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

        result = bridged({"query": "test"})

        assert result["isError"] is False
