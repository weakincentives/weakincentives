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

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    _make_async_handler,
    create_bridged_tools,
    create_mcp_server,
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
    """Tests for _make_async_handler function."""

    def test_creates_async_wrapper(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that _make_async_handler creates an async wrapper."""
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

        async_handler = _make_async_handler(bridged)

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

        async_handler = _make_async_handler(bridged)
        result = asyncio.run(async_handler({"query": "test"}))

        assert result["isError"] is False
        # Output uses render() from SearchResult which returns "Found 5 matches"
        assert "Found 5 matches" in result["content"][0]["text"]


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
    """Tests for VisibilityExpansionRequired exception propagation."""

    def test_bridged_tool_propagates_visibility_expansion_required(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that BridgedTool re-raises VisibilityExpansionRequired."""

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

        # The exception should propagate, not be caught
        with pytest.raises(VisibilityExpansionRequired) as exc_info:
            bridged({"query": "test"})

        # Verify the exception has the expected attributes
        exc = exc_info.value
        assert isinstance(exc, VisibilityExpansionRequired)
        assert exc.requested_overrides == {("section", "key"): SectionVisibility.FULL}
        assert exc.section_keys == ("section.key",)
        assert exc.reason == "Need more details"

    def test_async_handler_propagates_visibility_expansion_required(
        self,
        session: Session,
        prompt: Prompt[object],
        mock_adapter: MagicMock,
    ) -> None:
        """Test that async handler wrapper propagates VisibilityExpansionRequired."""

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

        async_handler = _make_async_handler(bridged)

        # The async wrapper should also propagate the exception
        with pytest.raises(VisibilityExpansionRequired):
            asyncio.run(async_handler({"query": "test"}))

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
        )

        with pytest.raises(VisibilityExpansionRequired):
            bridged({"query": "test"})

        # Filesystem should be restored to initial content
        assert test_fs.read("/test.txt").content == "initial content"

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
