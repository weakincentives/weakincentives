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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow
from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    ProviderAdapter,
)
from weakincentives.adapters.shared import (
    NativeToolCall,
    ResponseParser,
    ToolExecutor,
    parse_tool_arguments,
    tool_to_spec,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, ToolContext
from weakincentives.prompt._types import SupportsDataclass, SupportsToolResult
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    EventBus,
    PublishResult,
    ToolInvoked,
)
from weakincentives.runtime.events._types import EventHandler
from weakincentives.runtime.session.session import Session
from weakincentives.tools.web_search import (
    WebSearchFilters,
    WebSearchLocation,
    build_web_search_tool,
)


class RecordingBus(EventBus):
    def __init__(self) -> None:
        self.events: list[object] = []

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        pass

    def publish(self, event: object) -> PublishResult:
        self.events.append(event)
        return PublishResult(event=event, handlers_invoked=(), errors=())


@dataclass
class EchoParams:
    value: str


@dataclass
class EchoPayload:
    value: str


def echo_handler(
    params: EchoParams, *, context: ToolContext
) -> ToolResult[EchoPayload]:
    return ToolResult(message="echoed", value=EchoPayload(value=params.value))


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object | None = None
) -> object:
    return {"message": result.message, "payload": payload}


def test_tool_executor_success() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]], {tool.name: tool}
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"value": "hello"}'),
    )

    messages, next_choice = executor.execute([cast(Any, tool_call)], None)
    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call-1"
    assert messages[0]["content"] == {"message": "echoed", "payload": None}
    assert next_choice == "auto"
    assert len(tool_events) == 1
    assert len(executor.tool_message_records) == 1


def test_response_parser_text_only() -> None:
    rendered = RenderedPrompt(text="system")
    parser = ResponseParser[object](
        prompt_name="test",
        rendered=rendered,
        parse_output=False,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content="Hello")
    output, text = parser.parse(message, None)

    assert output is None
    assert text == "Hello"


def test_tool_to_spec_for_web_search() -> None:
    tool = build_web_search_tool(
        provider_type="web_search_2025_08_26",
        search_context_size="high",
        filters=WebSearchFilters(allowed_domains=("example.com", "openai.com")),
        user_location=WebSearchLocation(
            city="San Francisco",
            country="US",
            region=None,
            timezone="America/Los_Angeles",
        ),
    )

    spec = tool_to_spec(cast(Tool[SupportsDataclass, SupportsToolResult], tool))

    assert spec["type"] == "web_search_2025_08_26"
    assert spec["web_search"]["search_context_size"] == "high"
    assert spec["web_search"]["filters"] == {
        "allowed_domains": ["example.com", "openai.com"],
    }
    assert spec["web_search"]["user_location"] == {
        "city": "San Francisco",
        "country": "US",
        "timezone": "America/Los_Angeles",
        "type": "approximate",
    }


def test_record_native_calls_publishes_events() -> None:
    tool = build_web_search_tool()
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]], {tool.name: tool}
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    call = NativeToolCall(
        name="web_search",
        arguments={
            "id": "call-1",
            "status": "completed",
            "type": "web_search_call",
            "action": {"type": "search", "query": "latest"},
        },
        call_id="call-1",
        success=True,
    )

    executor.record_native_calls((call,), provider_payload={})

    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]
    assert tool_events
    assert tool_events[0].name == "web_search"


def test_record_native_calls_validates_tool_registration() -> None:
    rendered = RenderedPrompt(text="system")
    bus = RecordingBus()
    session = Session(bus=bus)

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry={},
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    native_call = NativeToolCall(
        name="missing",
        arguments={"id": "1", "status": "completed", "action": {}},
        call_id=None,
        success=True,
    )

    with pytest.raises(PromptEvaluationError):
        executor.record_native_calls((native_call,), provider_payload=None)

    non_native_tool = Tool[EchoParams, EchoPayload](
        name="echo", description="Echo", handler=echo_handler
    )
    non_native_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
        {non_native_tool.name: non_native_tool},
    )
    non_native_executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=non_native_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    with pytest.raises(PromptEvaluationError):
        non_native_executor.record_native_calls((native_call,), provider_payload=None)


def test_record_native_calls_rejects_non_native_tool_type() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo", description="Echo", handler=echo_handler
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
        {tool.name: tool},
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    native_call = NativeToolCall(
        name="echo",
        arguments={"id": "1", "status": "completed", "action": {}},
        call_id=None,
        success=True,
    )

    with pytest.raises(PromptEvaluationError):
        executor.record_native_calls((native_call,), provider_payload=None)


def test_record_native_calls_handles_validation_errors() -> None:
    tool = build_web_search_tool()
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]], {tool.name: tool}
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    invalid_call = NativeToolCall(
        name="web_search",
        arguments={"id": "bad"},
        call_id="bad",
        success=False,
    )

    executor.record_native_calls((invalid_call,), provider_payload={})

    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]
    assert tool_events


@dataclass
class StructuredOutput:
    answer: str


def test_response_parser_structured_output() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        parse_output=True,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content=None, parsed={"answer": "42"})
    output, text = parser.parse(message, None)

    assert output == StructuredOutput(answer="42")
    assert text is None


def test_response_parser_structured_output_failure() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        parse_output=True,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content="Not JSON", parsed=None)

    with pytest.raises(PromptEvaluationError) as excinfo:
        parser.parse(message, None)

    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_RESPONSE


def test_tool_executor_raises_when_deadline_expired(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]], {tool.name: tool}
    )

    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    frozen_utcnow.advance(timedelta(seconds=10))

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(ns="test", key="tool"),
        prompt_name="test",
        rendered=rendered,
        bus=bus,
        session=session,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
        deadline=deadline,
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"value": "hello"}'),
    )

    with pytest.raises(PromptEvaluationError) as excinfo:
        executor.execute([cast(Any, tool_call)], None)

    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }
