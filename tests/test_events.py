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

"""Tests for the prompt event bus primitives."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pytest

from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    HandlerFailure,
    InProcessEventBus,
    NullEventBus,
    PromptExecuted,
    PromptRendered,
    PublishResult,
    ToolInvoked,
)


def make_prompt_response(prompt_name: str) -> PromptResponse[object]:
    return cast(
        PromptResponse[object],
        PromptResponse(
            prompt_name=prompt_name,
            text="hello",
            output="hello",
            tool_results=(),
            provider_payload=None,
        ),
    )


def test_null_event_bus_is_noop() -> None:
    bus = NullEventBus()

    events: list[PromptExecuted] = []

    def record_event(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        events.append(event)

    bus.subscribe(PromptExecuted, record_event)
    result = bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter="test",
            result=make_prompt_response("demo"),
            session_id="session-1",
            created_at=datetime.now(UTC),
            value=None,
        )
    )

    assert events == []
    assert isinstance(result, PublishResult)
    assert result.ok
    assert result.handlers_invoked == ()
    assert result.handled_count == 0
    assert result.errors == ()


def test_publish_without_subscribers_returns_success_result() -> None:
    bus = InProcessEventBus()

    event = PromptExecuted(
        prompt_name="demo",
        adapter="test",
        result=make_prompt_response("demo"),
        session_id="session-1",
        created_at=datetime.now(UTC),
        value=None,
    )

    result = bus.publish(event)

    assert result.ok
    assert result.handlers_invoked == ()
    assert result.errors == ()
    assert result.handled_count == 0
    result.raise_if_errors()


def test_publish_prompt_rendered_returns_success() -> None:
    bus = InProcessEventBus()

    event = PromptRendered(
        prompt_ns="demo",
        prompt_key="greeting",
        prompt_name="Demo Prompt",
        adapter="test",
        session_id="session-1",
        render_inputs=(_Params(value=1),),
        rendered_prompt="Render result",
        created_at=datetime.now(UTC),
    )

    result = bus.publish(event)

    assert result.ok
    assert result.errors == ()


def test_in_process_bus_delivers_in_order() -> None:
    bus = InProcessEventBus()
    delivered: list[PromptExecuted] = []

    def first_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    def second_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    bus.subscribe(PromptExecuted, first_handler)
    bus.subscribe(PromptExecuted, second_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter="test",
        result=make_prompt_response("demo"),
        session_id="session-1",
        created_at=datetime.now(UTC),
        value=None,
    )
    result = bus.publish(event)

    assert delivered == [event, event]
    assert result.handlers_invoked == (first_handler, second_handler)
    assert result.errors == ()
    assert result.ok
    assert result.handled_count == 2


def test_in_process_bus_isolates_handler_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = InProcessEventBus()
    received: list[PromptExecuted] = []

    def bad_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        raise RuntimeError("boom")

    def good_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        received.append(event)

    bus.subscribe(PromptExecuted, bad_handler)
    bus.subscribe(PromptExecuted, good_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter="test",
        result=make_prompt_response("demo"),
        session_id="session-1",
        created_at=datetime.now(UTC),
        value=None,
    )
    with caplog.at_level(logging.ERROR, logger="weakincentives.runtime.events"):
        result = bus.publish(event)

    assert received == [event]
    assert any(
        getattr(record, "event", None) == "event_delivery_failed"
        and getattr(record, "context", {}).get("event_type") == "PromptExecuted"
        for record in caplog.records
    )
    assert result.handlers_invoked == (bad_handler, good_handler)
    assert len(result.errors) == 1
    failure = result.errors[0]
    assert isinstance(failure, HandlerFailure)
    assert failure.handler is bad_handler
    assert isinstance(failure.error, RuntimeError)
    assert not result.ok
    assert result.handled_count == 2


def test_publish_result_raise_if_errors() -> None:
    bus = InProcessEventBus()

    def first_handler(_: object) -> None:
        raise ValueError("first")

    def second_handler(_: object) -> None:
        raise RuntimeError("second")

    bus.subscribe(PromptExecuted, first_handler)
    bus.subscribe(PromptExecuted, second_handler)

    result = bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter="test",
            result=make_prompt_response("demo"),
            session_id="session-1",
            created_at=datetime.now(UTC),
            value=None,
        )
    )

    assert not result.ok
    with pytest.raises(ExceptionGroup) as excinfo:
        result.raise_if_errors()

    group = excinfo.value
    assert isinstance(group, ExceptionGroup)
    assert len(group.exceptions) == 2
    assert "first_handler" in str(group)
    assert any(isinstance(error, ValueError) for error in group.exceptions)
    assert any(isinstance(error, RuntimeError) for error in group.exceptions)


@dataclass(slots=True)
class _Params:
    value: int


@dataclass(slots=True)
class _Payload:
    value: str


def test_tool_invoked_event_fields() -> None:
    raw_result = ToolResult(message="ok", value=_Payload(value="data"))
    result = cast(ToolResult[object], raw_result)
    event = ToolInvoked(
        prompt_name="demo",
        adapter="test",
        name="tool",
        params=_Params(value=1),
        result=result,
        call_id="abc123",
        session_id="session-123",
        created_at=datetime.now(UTC),
        value=_Payload(value="data"),
    )

    assert event.prompt_name == "demo"
    assert event.adapter == "test"
    assert event.name == "tool"
    assert isinstance(event.params, _Params)
    assert event.params.value == 1
    assert event.session_id == "session-123"
    assert isinstance(event.created_at, datetime)
    assert isinstance(event.value, _Payload)
    assert event.result is result
    assert event.call_id == "abc123"
