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

from weakincentives.adapters.core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.adapters.shared import ConversationRunner, ToolChoice
from weakincentives.events import (
    EventBus,
    HandlerFailure,
    PromptExecuted,
    PublishResult,
    ToolInvoked,
)
from weakincentives.events._types import EventHandler
from weakincentives.prompt import Prompt, ToolContext
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.session.protocols import SnapshotProtocol
from weakincentives.session.session import Session
from weakincentives.session.snapshots import Snapshot

from ._test_stubs import DummyChoice, DummyMessage, DummyResponse, DummyToolCall


class DummyAdapter(ProviderAdapter[object]):
    def evaluate(
        self,
        prompt: Prompt[object],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol | None = None,
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
    result: ToolResult[SupportsDataclass], *, payload: object | None = None
) -> object:
    return {"message": result.message, "payload": payload}


class RecordingBus(EventBus):
    def __init__(self, *, fail_tool: bool = False, fail_prompt: bool = False) -> None:
        self.events: list[object] = []
        self.fail_tool = fail_tool
        self.fail_prompt = fail_prompt

    def subscribe(
        self, event_type: type[object], handler: EventHandler
    ) -> None:  # pragma: no cover - unused
        del event_type, handler

    def publish(self, event: object) -> PublishResult:
        self.events.append(event)
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
) -> ConversationRunner[object]:
    prompt = Prompt(ns="tests", key="example")
    session_arg: SessionProtocol = session if session is not None else Session(bus=bus)
    return ConversationRunner[object](
        adapter_name="dummy",
        adapter=DummyAdapter(),
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
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
    )


def test_conversation_runner_success() -> None:
    rendered = RenderedPrompt(
        text="system",
        output_type=None,
        container=None,
        allow_extra_keys=None,
    )
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()

    runner = build_runner(rendered=rendered, provider=provider, bus=bus)
    response = runner.run()

    assert response.text == "Hello"
    assert response.output is None
    assert isinstance(bus.events[-1], PromptExecuted)
    assert provider.calls[0]["messages"][0]["content"] == "system"


def test_conversation_runner_raises_on_prompt_publish_failure() -> None:
    rendered = RenderedPrompt(
        text="system",
        output_type=None,
        container=None,
        allow_extra_keys=None,
    )
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
        output_type=None,
        container=None,
        allow_extra_keys=None,
        _tools=cast(
            tuple[Tool[SupportsDataclass, SupportsDataclass], ...],
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
    assert len(response.tool_results) == 1
    failure_message = response.tool_results[0].result.message
    assert "Reducer errors prevented applying tool result" in failure_message


@dataclass
class StructuredOutput:
    answer: str


def test_conversation_runner_parses_structured_output() -> None:
    rendered = RenderedPrompt(
        text="system",
        output_type=StructuredOutput,
        container="object",
        allow_extra_keys=False,
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
    rendered = RenderedPrompt(
        text="system",
        output_type=None,
        container=None,
        allow_extra_keys=None,
    )

    class MissingMessageResponse(DummyResponse):
        def __init__(self) -> None:
            super().__init__(choices=[DummyChoice(DummyMessage(content=None))])
            self.choices[0].message = None

    provider = ProviderStub([MissingMessageResponse()])
    bus = RecordingBus()
    runner = build_runner(rendered=rendered, provider=provider, bus=bus)

    with pytest.raises(PromptEvaluationError):
        runner.run()
