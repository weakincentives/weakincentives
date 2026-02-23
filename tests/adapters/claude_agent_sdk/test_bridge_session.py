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

"""Tests for Claude Agent SDK bridge: session, MCP state, and policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    MCPToolExecutionState,
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
from weakincentives.runtime.events.types import ToolInvoked
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
