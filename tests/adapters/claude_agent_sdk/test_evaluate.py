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

"""Tests for ClaudeAgentSDKAdapter.evaluate and output format building."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    NullableOutput,
    SimpleOutput,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    ClaudeAgentSDKAdapter,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.events import PromptExecuted, PromptRendered
from weakincentives.runtime.session import Session


class TestClaudeAgentSDKAdapterEvaluate:
    def test_raises_when_deadline_expired(
        self,
        session: Session,
        simple_prompt: Prompt[SimpleOutput],
    ) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        adapter = ClaudeAgentSDKAdapter()

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(simple_prompt, session=session, deadline=deadline)

    def test_raises_when_sdk_not_installed(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        adapter = ClaudeAgentSDKAdapter()

        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            with patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                side_effect=ImportError("claude-agent-sdk is not installed"),
            ):
                with pytest.raises(ImportError, match="claude-agent-sdk"):
                    adapter.evaluate(simple_prompt, session=session)

    def test_publishes_prompt_rendered_event(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        events: list[PromptRendered] = []
        session.dispatcher.subscribe(PromptRendered, lambda e: events.append(e))

        setup_mock_query(
            [MockResultMessage(result="Hello!", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        event = events[0]
        assert event.prompt_ns == "test"
        assert event.prompt_key == "simple"
        assert event.adapter == CLAUDE_AGENT_SDK_ADAPTER_NAME

    def test_publishes_rendered_tools_event(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that RenderedTools event is dispatched during evaluate."""
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        events: list[RenderedTools] = []
        session.dispatcher.subscribe(RenderedTools, lambda e: events.append(e))

        setup_mock_query(
            [MockResultMessage(result="Hello!", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        event = events[0]
        assert event.prompt_ns == "test"
        assert event.prompt_key == "simple"
        assert event.tools == ()

    def test_rendered_tools_event_correlates_with_prompt_rendered(
        self, session: Session
    ) -> None:
        """Test that RenderedTools event has matching render_event_id with PromptRendered."""
        from tests.adapters.claude_agent_sdk.conftest import search_tool
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        template_with_tools = PromptTemplate[SimpleOutput](
            ns="test",
            key="with_tools",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template="Use the tool",
                    tools=(search_tool,),
                ),
            ],
        )
        prompt_with_tools = Prompt(template_with_tools)

        prompt_rendered_events: list[PromptRendered] = []
        rendered_tools_events: list[RenderedTools] = []

        session.dispatcher.subscribe(
            PromptRendered, lambda e: prompt_rendered_events.append(e)
        )
        session.dispatcher.subscribe(
            RenderedTools, lambda e: rendered_tools_events.append(e)
        )

        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(prompt_with_tools, session=session)

        assert len(prompt_rendered_events) == 1
        assert len(rendered_tools_events) == 1

        prompt_event = prompt_rendered_events[0]
        tools_event = rendered_tools_events[0]

        assert prompt_event.event_id is not None
        assert tools_event.render_event_id is not None
        assert prompt_event.event_id == tools_event.render_event_id
        assert prompt_event.session_id == tools_event.session_id
        assert prompt_event.created_at == tools_event.created_at

    def test_rendered_tools_extracts_correct_schemas(self, session: Session) -> None:
        """Test that tool schemas are correctly extracted from rendered tools."""
        from tests.adapters.claude_agent_sdk.conftest import search_tool
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        template_with_tools = PromptTemplate[SimpleOutput](
            ns="test",
            key="with_tools",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template="Use the tool",
                    tools=(search_tool,),
                ),
            ],
        )
        prompt_with_tools = Prompt(template_with_tools)

        events: list[RenderedTools] = []
        session.dispatcher.subscribe(RenderedTools, lambda e: events.append(e))

        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(prompt_with_tools, session=session)

        assert len(events) == 1
        event = events[0]

        assert len(event.tools) == 1
        tool_schema = event.tools[0]
        assert tool_schema.name == "search"
        assert "Search" in tool_schema.description
        assert "properties" in tool_schema.parameters
        assert "query" in tool_schema.parameters["properties"]

    def test_rendered_tools_dispatch_failure_logs_error(
        self,
        session: Session,
        simple_prompt: Prompt[SimpleOutput],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that RenderedTools dispatch failures are logged."""
        import logging

        from weakincentives.runtime.session.rendered_tools import RenderedTools

        def failing_handler(event: RenderedTools) -> None:
            raise RuntimeError("Subscriber error")

        session.dispatcher.subscribe(RenderedTools, failing_handler)

        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()
        caplog.set_level(logging.ERROR)

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Done"
        assert any(
            "rendered_tools_dispatch_failed" in record.message
            for record in caplog.records
        )

    def test_returns_prompt_response(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [
                MockResultMessage(
                    result="Hello, world!",
                    usage={"input_tokens": 10, "output_tokens": 5},
                    structured_output=None,
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.prompt_name == "test:simple"
        assert response.text == "Hello, world!"

    def test_extracts_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [
                MockResultMessage(
                    result="Hello!",
                    usage=None,
                    structured_output={"message": "structured hello"},
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.output is not None
        assert response.output.message == "structured hello"

    def test_handles_invalid_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [
                MockResultMessage(
                    result="Hello!",
                    usage=None,
                    structured_output="not a dict",  # type: ignore[arg-type]
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.output is None
        assert response.text == "Hello!"

    def test_handles_no_structured_output(
        self, session: Session, untyped_prompt: Prompt[None]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(untyped_prompt, session=session)

        assert response.output is None

    def test_raises_on_empty_structured_result(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Structured prompts with no text/output should fail deterministically."""
        setup_mock_query([MockResultMessage(result=None, usage=None)])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(
                PromptEvaluationError,
                match="Structured output prompt returned no text and no structured output",
            ):
                adapter.evaluate(simple_prompt, session=session)

    def test_accumulates_token_usage(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        msg1 = MagicMock()
        msg1.usage = {"input_tokens": 100, "output_tokens": 50}

        setup_mock_query(
            [
                msg1,
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 50, "output_tokens": 25},
                    structured_output=None,
                ),
            ]
        )

        events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, lambda e: events.append(e))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        usage = events[0].usage
        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75


class TestAdapterName:
    def test_adapter_name_constant(self) -> None:
        assert CLAUDE_AGENT_SDK_ADAPTER_NAME == "claude_agent_sdk"


class TestBuildOutputFormat:
    def test_none_output_type_returns_none(self, untyped_prompt: Prompt[None]) -> None:
        from weakincentives.adapters.claude_agent_sdk._result_extraction import (
            build_output_format,
        )

        rendered = untyped_prompt.render()
        result = build_output_format(rendered)
        assert result is None

    def test_dataclass_output_type_returns_schema(
        self, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        from weakincentives.adapters.claude_agent_sdk._result_extraction import (
            build_output_format,
        )

        rendered = simple_prompt.render()
        result = build_output_format(rendered)

        assert result is not None
        assert result["type"] == "json_schema"
        assert "schema" in result
        assert "properties" in result["schema"]
        assert "message" in result["schema"]["properties"]

    def test_nullable_fields_collapse_anyof_for_claude(
        self, nullable_prompt: Prompt[NullableOutput]
    ) -> None:
        from weakincentives.adapters.claude_agent_sdk._result_extraction import (
            build_output_format,
        )

        rendered = nullable_prompt.render()
        result = build_output_format(rendered)

        assert result is not None
        count_schema = result["schema"]["properties"]["count"]
        assert count_schema["type"] == ["integer", "null"]
        assert "anyOf" not in count_schema
