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
