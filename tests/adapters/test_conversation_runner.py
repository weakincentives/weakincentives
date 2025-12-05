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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tests.helpers.adapters import DUMMY_ADAPTER_NAME
from weakincentives.adapters.core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.adapters.shared import (
    ConversationRunner,
    ThrottlePolicy,
    ToolChoice,
    new_throttle_policy,
    token_usage_from_payload,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
from weakincentives.prompt._types import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from weakincentives.prompt.overrides import PromptDescriptor
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    EventBus,
    HandlerFailure,
    PromptExecuted,
    PromptRendered,
    PublishResult,
    TokenUsage,
    ToolInvoked,
)
from weakincentives.runtime.events._types import EventHandler
from weakincentives.runtime.session.protocols import SnapshotProtocol
from weakincentives.runtime.session.session import Session
from weakincentives.runtime.session.snapshots import Snapshot

from ._test_stubs import DummyChoice, DummyMessage, DummyResponse, DummyToolCall


class DummyAdapter(ProviderAdapter[object]):
    def evaluate(
        self,
        prompt: Prompt[object],
        *,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol | None = None,
        deadline: Deadline | None = None,
    ) -> PromptResponse[object]:
        raise NotImplementedError


class ProviderStub:
    def __init__(self, responses: Sequence[DummyResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tool_specs": list(tool_specs),
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object | None = None
) -> object:
    return {"message": result.message, "payload": payload}


class RecordingBus(EventBus):
    def __init__(
        self,
        *,
        fail_rendered: bool = False,
        fail_tool: bool = False,
        fail_prompt: bool = False,
    ) -> None:
        self.events: list[object] = []
        self.fail_rendered = fail_rendered
        self.fail_tool = fail_tool
        self.fail_prompt = fail_prompt
        self.subscriptions: list[tuple[type[object], EventHandler]] = []

    def subscribe(
        self, event_type: type[object], handler: EventHandler
    ) -> None:  # pragma: no cover - unused
        self.subscriptions.append((event_type, handler))

    def publish(self, event: object) -> PublishResult:
        self.events.append(event)
        if self.fail_rendered and isinstance(event, PromptRendered):
            return self._failure_result(event, "prompt rendered publish failure")
        if self.fail_tool and isinstance(event, ToolInvoked):
            return self._failure_result(event, "reducer failure")
        if self.fail_prompt and isinstance(event, PromptExecuted):
            return self._failure_result(event, "prompt publish failure")
        return PublishResult(event=event, handlers_invoked=(), errors=())

    @staticmethod
    def _failure_result(event: object, message: str) -> PublishResult:
        def failure_handler(_event: object) -> None:  # pragma: no cover - defensive
            return None

        failure = HandlerFailure(handler=failure_handler, error=RuntimeError(message))
        return PublishResult(
            event=event,
            handlers_invoked=(failure_handler,),
            errors=(failure,),
        )


def build_runner(
    *,
    rendered: RenderedPrompt[object],
    provider: ProviderStub,
    bus: RecordingBus,
    tool_choice: ToolChoice = "auto",
    parse_output: bool = False,
    response_format: Mapping[str, Any] | None = None,
    session: SessionProtocol | None = None,
    render_inputs: tuple[SupportsDataclass, ...] | None = None,
    throttle_policy: ThrottlePolicy | None = None,
) -> ConversationRunner[object]:
    template = PromptTemplate(ns="tests", key="example")
    prompt = Prompt(template, params=render_inputs or ())
    session_arg: SessionProtocol = session if session is not None else Session(bus=bus)
    return ConversationRunner[object](
        adapter_name=DUMMY_ADAPTER_NAME,
        adapter=DummyAdapter(),
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=parse_output,
        bus=bus,
        session=session_arg,
        tool_choice=tool_choice,
        response_format=response_format,
        require_structured_output_text=False,
        call_provider=provider,
        select_choice=lambda response: response.choices[0],
        serialize_tool_message_fn=serialize_tool_message,
        throttle_policy=throttle_policy or new_throttle_policy(),
    )


def test_conversation_runner_success() -> None:
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)
    response = runner.run()

    assert response.text == "Hello"
    assert response.output is None
    assert isinstance(bus.events[-1], PromptExecuted)
    assert provider.calls[0]["messages"][0]["content"] == "system"


def test_conversation_runner_includes_usage_in_event() -> None:
    rendered = RenderedPrompt(text="system")
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="Hello"))],
            usage={"input_tokens": 12, "output_tokens": 5, "cached_tokens": 3},
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)
    _ = runner.run()

    prompt_event = cast(PromptExecuted, bus.events[-1])
    assert prompt_event.usage == TokenUsage(
        input_tokens=12, output_tokens=5, cached_tokens=3
    )


def test_token_usage_from_payload_handles_missing_counts() -> None:
    payload = {
        "usage": {
            "input_tokens": None,
            "output_tokens": "unknown",
            "cached_tokens": False,
        }
    }

    assert token_usage_from_payload(payload) is None


def test_conversation_runner_includes_prompt_descriptor_in_event() -> None:
    descriptor = PromptDescriptor(ns="tests", key="example", sections=[], tools=[])
    rendered = RenderedPrompt(text="system", descriptor=descriptor)
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)
    runner.run()

    rendered_event = next(
        event for event in bus.events if isinstance(event, PromptRendered)
    )
    assert rendered_event.descriptor is descriptor


def test_conversation_runner_publishes_prompt_rendered_event() -> None:
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    params = EchoParams(value="hello")

    runner = build_runner(
        rendered=rendered,
        provider=provider,
        bus=bus,
        render_inputs=(params,),
    )
    runner.run()

    assert provider.calls
    assert bus.events and isinstance(bus.events[0], PromptRendered)
    event = cast(PromptRendered, bus.events[0])
    assert event.rendered_prompt == "system"
    assert event.render_inputs == (params,)
    assert event.prompt_ns == "tests"
    assert event.prompt_key == "example"


def test_conversation_runner_continues_on_prompt_rendered_publish_failure() -> None:
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus(fail_rendered=True)
    params = EchoParams(value="blocked")

    runner = build_runner(
        rendered=rendered,
        provider=provider,
        bus=bus,
        render_inputs=(params,),
    )
    response = runner.run()

    assert response.text == "Hello"
    assert provider.calls
    assert bus.events and isinstance(bus.events[0], PromptRendered)
    assert isinstance(bus.events[-1], PromptExecuted)


def test_conversation_runner_raises_on_prompt_publish_failure() -> None:
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus(fail_prompt=True)

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)

    with pytest.raises(ExceptionGroup) as exc_info:
        runner.run()

    assert "prompt publish failure" in str(exc_info.value)
    assert isinstance(bus.events[-1], PromptExecuted)
    assert provider.calls[0]["messages"][0]["content"] == "system"


@dataclass
class EchoParams:
    value: str


@dataclass
class EchoPayload:
    value: str


def echo_handler(
    params: EchoParams, *, context: ToolContext
) -> ToolResult[EchoPayload]:
    del context
    return ToolResult(message="initial", value=EchoPayload(value=params.value))


def tool_rendered_prompt(tool: Tool[EchoParams, EchoPayload]) -> RenderedPrompt[object]:
    return RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
            (tool,),
        ),
    )


def build_tool_responses() -> Sequence[DummyResponse]:
    first = DummyResponse(
        [
            DummyChoice(
                DummyMessage(
                    content="",
                    tool_calls=(DummyToolCall("call-1", "echo", '{"value": "hello"}'),),
                )
            )
        ]
    )
    second = DummyResponse([DummyChoice(DummyMessage(content="All done"))])
    return (first, second)


def test_conversation_runner_formats_publish_failures() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo the provided value.",
        handler=echo_handler,
    )
    rendered = tool_rendered_prompt(tool)
    provider = ProviderStub(build_tool_responses())
    bus = RecordingBus(fail_tool=True)

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)
    response = runner.run()

    assert response.text == "All done"
    tool_event = next(event for event in bus.events if isinstance(event, ToolInvoked))
    failure_message = tool_event.result.message
    assert "Reducer errors prevented applying tool result" in failure_message


@dataclass
class StructuredOutput:
    answer: str


def test_conversation_runner_parses_structured_output() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    responses = [
        DummyResponse(
            [
                DummyChoice(
                    DummyMessage(
                        content=None,
                        parsed={"answer": "42"},
                    )
                )
            ]
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()

    runner = build_runner(
        rendered=rendered,
        provider=provider,
        bus=bus,
        parse_output=True,
    )
    response = runner.run()

    assert response.text is None
    assert response.output == StructuredOutput(answer="42")


class SessionStub(SessionProtocol):
    def __init__(self) -> None:
        self.snapshots: list[SnapshotProtocol] = []
        self.rollbacks: list[SnapshotProtocol] = []

    def snapshot(self) -> SnapshotProtocol:
        snapshot = Snapshot(created_at=datetime.now(UTC))
        self.snapshots.append(snapshot)
        return snapshot

    def rollback(self, snapshot: SnapshotProtocol) -> None:
        self.rollbacks.append(snapshot)

    def reset(self) -> None:
        pass


def test_conversation_runner_rolls_back_on_publish_failure() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo the provided value.",
        handler=echo_handler,
    )
    rendered = tool_rendered_prompt(tool)
    provider = ProviderStub(build_tool_responses())
    bus = RecordingBus(fail_tool=True)
    session = SessionStub()

    runner = build_runner(
        rendered=rendered,
        provider=provider,
        bus=bus,
        session=session,
    )
    response = runner.run()

    assert response.text == "All done"
    assert session.snapshots and session.rollbacks
    assert session.rollbacks == session.snapshots


def test_conversation_runner_requires_message_payload() -> None:
    rendered = RenderedPrompt(text="system")

    class MissingMessageResponse(DummyResponse):
        def __init__(self) -> None:
            super().__init__(choices=[DummyChoice(DummyMessage(content=None))])
            self.choices[0].message = None

    provider = ProviderStub([MissingMessageResponse()])
    bus = RecordingBus()
    runner = build_runner(rendered=rendered, provider=provider, bus=bus)

    with pytest.raises(PromptEvaluationError):
        runner.run()
