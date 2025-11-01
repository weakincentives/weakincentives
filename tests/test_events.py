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
from typing import cast

import pytest

from weakincentives.adapters.core import PromptResponse
from weakincentives.events import (
    InProcessEventBus,
    NullEventBus,
    PromptExecuted,
    ToolInvoked,
)
from weakincentives.prompt.tool import ToolResult


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
    bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter="test",
            result=make_prompt_response("demo"),
        )
    )

    assert events == []


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
    )
    bus.publish(event)

    assert delivered == [event, event]


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
    )
    with caplog.at_level(logging.ERROR, logger="weakincentives.events"):
        bus.publish(event)

    assert received == [event]
    assert any("Error delivering event" in record.message for record in caplog.records)


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
    )

    assert event.prompt_name == "demo"
    assert event.adapter == "test"
    assert event.name == "tool"
    assert isinstance(event.params, _Params)
    assert event.params.value == 1
    assert event.result is result
    assert event.call_id == "abc123"
