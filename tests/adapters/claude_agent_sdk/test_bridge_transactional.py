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

"""Tests for BridgedTool transactional execution, filesystem, session, and budget tracker."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    create_bridged_tools,
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
from weakincentives.runtime.session import Session

from .conftest import (
    SearchParams,
    SearchResult,
    _make_prompt_with_resources,
    search_tool,
)


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
