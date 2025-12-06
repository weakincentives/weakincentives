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

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    ProviderAdapter,
)
from weakincentives.adapters.shared import (
    ResponseParser,
    ToolExecutionContext,
    ToolExecutionOutcome,
    ToolExecutor,
    _parse_tool_params,
    _publish_tool_invocation,
    parse_tool_arguments,
    tool_to_spec,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    EventBus,
    PublishResult,
    ToolInvoked,
)
from weakincentives.runtime.logging import get_logger
from weakincentives.runtime.session.session import Session
from weakincentives.tools.errors import ToolValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.helpers import FrozenUtcNow
    from weakincentives.prompt._types import (
        SupportsDataclass,
        SupportsDataclassOrNone,
        SupportsToolResult,
    )
    from weakincentives.runtime.events._types import EventHandler


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


def test_tool_to_spec_accepts_none_params() -> None:
    def handler(params: None, *, context: ToolContext) -> ToolResult[EchoPayload]:
        return ToolResult(message="ok", value=EchoPayload(value="hi"))

    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=handler,
    )

    spec = tool_to_spec(cast("Tool[SupportsDataclassOrNone, SupportsToolResult]", tool))

    assert spec["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def test_parse_tool_params_rejects_arguments_for_none_params() -> None:
    def no_param_handler(
        params: None, *, context: ToolContext
    ) -> ToolResult[EchoPayload]:
        del context
        return ToolResult(message="ok", value=EchoPayload(value="hi"))

    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=no_param_handler,
    )

    with pytest.raises(ToolValidationError, match="does not accept any arguments"):
        _parse_tool_params(
            tool=cast("Tool[SupportsDataclassOrNone, SupportsToolResult]", tool),
            arguments_mapping={"unexpected": "value"},
        )


def test_parse_tool_params_returns_none_for_empty_arguments() -> None:
    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=None,
    )

    parsed = _parse_tool_params(
        tool=cast("Tool[SupportsDataclassOrNone, SupportsToolResult]", tool),
        arguments_mapping={},
    )

    assert parsed is None


def test_tool_executor_success() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            "tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]",
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        "Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]",
        {tool.name: tool},
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast("ProviderAdapter[Any]", object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
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

    messages, next_choice = executor.execute([cast("Any", tool_call)], None)
    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call-1"
    assert messages[0]["content"] == {"message": "echoed", "payload": None}
    assert next_choice == "auto"
    assert len(tool_events) == 1
    assert len(executor.tool_message_records) == 1


def test_publish_tool_invocation_attaches_usage() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    params = EchoParams(value="hello")
    result = ToolResult(message="echoed", value=EchoPayload(value="hello"))
    log = get_logger(__name__)

    bus = RecordingBus()
    session = Session(bus=bus)
    typed_tool = cast("Tool[SupportsDataclassOrNone, SupportsToolResult]", tool)
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]] = {
        tool.name: typed_tool
    }

    context = ToolExecutionContext(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast("ProviderAdapter[Any]", object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
        rendered_prompt=None,
        tool_registry=tool_registry,
        bus=bus,
        session=session,
        prompt_name="test",
        parse_arguments=parse_tool_arguments,
        format_publish_failures=lambda errors: "",
        deadline=None,
    ).with_provider_payload(
        {
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "cached_tokens": 2,
            }
        }
    )

    outcome = ToolExecutionOutcome(
        tool=typed_tool,
        params=cast("SupportsDataclass", params),
        result=cast("ToolResult[SupportsToolResult]", result),
        call_id="call-usage",
        log=log,
    )

    invocation = _publish_tool_invocation(context=context, outcome=outcome)

    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]

    assert invocation.usage is not None
    assert invocation.usage.input_tokens == 5
    assert invocation.usage.output_tokens == 7
    assert invocation.usage.cached_tokens == 2
    assert tool_events == [invocation]


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

    assert isinstance(excinfo.value, PromptEvaluationError)
    error = excinfo.value
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
            "tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]",
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    tool_registry = cast(
        "Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]",
        {tool.name: tool},
    )

    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    frozen_utcnow.advance(timedelta(seconds=10))

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast("ProviderAdapter[Any]", object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
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
        executor.execute([cast("Any", tool_call)], None)

    error = cast("PromptEvaluationError", excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }
