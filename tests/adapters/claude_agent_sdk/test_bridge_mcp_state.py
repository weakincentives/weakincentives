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

"""Tests for MCPToolExecutionState shared state between hooks and bridge."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    MCPToolExecutionState,
)
from weakincentives.prompt import (
    Prompt,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from .conftest import (
    search_tool,
)


class TestMCPToolExecutionState:
    """Tests for MCPToolExecutionState shared state between hooks and bridge."""

    def test_bridged_tool_uses_call_id_from_mcp_state(
        self,
        session: Session,
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
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
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
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
