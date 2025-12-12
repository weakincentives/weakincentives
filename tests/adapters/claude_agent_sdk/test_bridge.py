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
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    _make_async_handler,
    create_bridged_tools,
    create_mcp_server,
)
from weakincentives.prompt import Tool, ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


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
    return ToolResult(
        message=f"Found matches for {params.query}",
        value=SearchResult(matches=5),
        success=True,
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
    bus = InProcessEventBus()
    return Session(bus=bus)


@pytest.fixture
def mock_adapter() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_prompt() -> MagicMock:
    return MagicMock()


class TestBridgedTool:
    def test_executes_handler_successfully(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is False
        # Output uses render() from SearchResult which returns "Found 5 matches"
        assert "Found 5 matches" in result["content"][0]["text"]

    def test_returns_error_for_no_handler(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        bridged = BridgedTool(
            name="no_handler",
            description="No handler",
            input_schema={},
            tool=no_handler_tool,
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({})

        assert result["isError"] is True
        assert "no handler" in result["content"][0]["text"]

    def test_returns_validation_error(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"wrong_field": "value"})

        assert result["isError"] is True
        assert "error" in result["content"][0]["text"].lower()

    def test_catches_handler_exception(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    def test_uses_render_method_on_result_value(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        # SearchResult.render() returns "Found 5 matches"
        assert result["content"][0]["text"] == "Found 5 matches"

    def test_falls_back_to_message_when_render_empty(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test that bridged tool falls back to message when render() returns empty."""

        def empty_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[EmptyRenderResult]:
            del context
            return ToolResult(
                message=f"Searched for {params.query}",
                value=EmptyRenderResult(),
                success=True,
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        # Falls back to message since render() returned empty
        assert result["content"][0]["text"] == "Searched for test"


class TestCreateBridgedTools:
    def test_creates_bridged_tools(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        tools = (search_tool,)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].name == "search"
        assert bridged[0].description == "Search for content"

    def test_skips_tools_without_handlers(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        tools = (search_tool, no_handler_tool)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].name == "search"

    def test_generates_input_schema(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        tools = (search_tool,)

        bridged = create_bridged_tools(
            tools,
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        schema = bridged[0].input_schema
        assert "properties" in schema
        assert "query" in schema["properties"]

    def test_handles_none_params_type(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        def null_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
            del params, context
            return ToolResult(message="ok", value=None, success=True)

        null_tool = Tool[None, None](
            name="null_tool",
            description="A tool with no params",
            handler=null_handler,
        )

        bridged = create_bridged_tools(
            (null_tool,),
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert len(bridged) == 1
        assert bridged[0].input_schema == {"type": "object", "properties": {}}

    def test_executes_tool_with_none_params_type(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test that calling a bridged tool with None params works."""

        def null_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
            del params, context
            return ToolResult(message="executed", value=None, success=True)

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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({})

        assert result["isError"] is False
        assert "executed" in result["content"][0]["text"]

    def test_empty_tools_returns_empty(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
    ) -> None:
        bridged = create_bridged_tools(
            (),
            session=session,
            adapter=mock_adapter,
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        assert bridged == ()


class TestMakeAsyncHandler:
    """Tests for _make_async_handler function."""

    def test_creates_async_wrapper(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        async_handler = _make_async_handler(bridged)

        # Verify it's a coroutine function
        assert asyncio.iscoroutinefunction(async_handler)

    def test_async_handler_returns_result(
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
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
        self, session: Session, mock_adapter: MagicMock, mock_prompt: MagicMock
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
            prompt=mock_prompt,
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
