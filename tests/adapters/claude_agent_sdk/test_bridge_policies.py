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

"""Tests for tool policy enforcement in BridgedTool._execute_handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
)
from weakincentives.prompt import (
    MarkdownSection,
    PolicyDecision,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.session import Session

from .conftest import (
    SearchParams,
    SearchResult,
    search_tool,
)


class TestBridgedToolPolicyEnforcement:
    """Tests for tool policy enforcement in BridgedTool._execute_handler."""

    def test_denying_policy_prevents_handler_execution(
        self,
        session: Session,
        mock_adapter: MagicMock,
    ) -> None:
        """A denying policy blocks handler and returns isError."""
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
        bridge_prompt: Prompt[object],
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
            prompt=cast("PromptProtocol[object]", bridge_prompt),
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
        )

        result = bridged({"query": "test"})

        assert result["isError"] is False
