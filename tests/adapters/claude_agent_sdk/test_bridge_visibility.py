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

"""Tests for visibility expansion signal handling in the bridge."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import MagicMock

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    make_async_handler,
)
from weakincentives.adapters.claude_agent_sdk._visibility_signal import (
    VisibilityExpansionSignal,
)
from weakincentives.prompt import (
    Prompt,
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
)


class TestVisibilityExpansionRequiredPropagation:
    """Tests for VisibilityExpansionRequired exception handling via signal."""

    def test_bridged_tool_stores_visibility_expansion_in_signal(
        self,
        session: Session,
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
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
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
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
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
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
