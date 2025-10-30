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
from typing import cast

from weakincentives.adapters.core import PromptResponse
from weakincentives.events import InProcessEventBus, PromptExecuted, ToolInvoked
from weakincentives.prompts.tool import ToolResult
from weakincentives.session import (
    DataEvent,
    Session,
    append,
    replace_latest,
    select_all,
    select_latest,
    select_where,
    upsert_by,
)


@dataclass(slots=True, frozen=True)
class ExampleParams:
    value: int


@dataclass(slots=True, frozen=True)
class ExamplePayload:
    value: int


@dataclass(slots=True, frozen=True)
class ExampleOutput:
    text: str


def make_tool_event(value: int) -> ToolInvoked:
    tool_result = cast(
        ToolResult[object],
        ToolResult(message="ok", payload=ExamplePayload(value=value)),
    )
    return ToolInvoked(
        prompt_name="example",
        adapter="adapter",
        name="tool",
        params=ExampleParams(value=value),
        result=tool_result,
    )


def make_prompt_event(output: object) -> PromptExecuted:
    response = PromptResponse(
        prompt_name="example",
        text="done",
        output=output,
        tool_results=(),
        provider_payload=None,
    )
    return PromptExecuted(
        prompt_name="example",
        adapter="adapter",
        response=response,
    )


def test_tool_invoked_appends_payload_once() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    event = make_tool_event(1)
    bus.publish(event)
    bus.publish(event)

    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)


def test_prompt_executed_emits_multiple_dataclasses() -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    bus = InProcessEventBus()
    session = Session(bus=bus)

    bus.publish(make_prompt_event(outputs))

    assert session.select_all(ExampleOutput) == tuple(outputs)


def test_reducers_run_in_registration_order() -> None:
    @dataclass(slots=True, frozen=True)
    class FirstSlice:
        value: str

    @dataclass(slots=True, frozen=True)
    class SecondSlice:
        value: str

    bus = InProcessEventBus()
    session = Session(bus=bus)

    call_order: list[str] = []

    def first(
        slice_values: tuple[FirstSlice, ...], event: DataEvent
    ) -> tuple[FirstSlice, ...]:
        call_order.append("first")
        value = cast(ExampleOutput, event.value)
        return slice_values + (FirstSlice(value.text),)

    def second(
        slice_values: tuple[SecondSlice, ...], event: DataEvent
    ) -> tuple[SecondSlice, ...]:
        call_order.append("second")
        value = cast(ExampleOutput, event.value)
        return slice_values + (SecondSlice(value.text),)

    session.register_reducer(ExampleOutput, first, slice_type=FirstSlice)
    session.register_reducer(ExampleOutput, second, slice_type=SecondSlice)

    bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session.select_all(FirstSlice) == (FirstSlice("hello"),)
    assert session.select_all(SecondSlice) == (SecondSlice("hello"),)


def test_default_append_used_when_no_custom_reducer() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert session.select_all(ExampleOutput) == (ExampleOutput(text="hello"),)


def test_non_dataclass_payloads_are_ignored() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    event = make_tool_event(1)
    non_dataclass_event = ToolInvoked(
        prompt_name="example",
        adapter="adapter",
        name="tool",
        params=ExampleParams(value=2),
        result=cast(
            ToolResult[object], ToolResult(message="ok", payload="not a dataclass")
        ),
    )

    bus.publish(event)
    bus.publish(non_dataclass_event)

    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session.register_reducer(ExamplePayload, upsert_by(lambda payload: payload.value))

    bus.publish(make_tool_event(1))
    bus.publish(make_tool_event(1))
    bus.publish(make_tool_event(2))

    assert session.select_all(ExamplePayload) == (
        ExamplePayload(value=1),
        ExamplePayload(value=2),
    )


def test_replace_latest_keeps_only_newest_value() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session.register_reducer(ExamplePayload, replace_latest)

    bus.publish(make_tool_event(1))
    bus.publish(make_tool_event(2))

    assert session.select_all(ExamplePayload) == (ExamplePayload(value=2),)


def test_selector_helpers_delegate_to_session() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    assert select_latest(session, ExampleOutput) is None

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert select_all(session, ExampleOutput) == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )
    assert select_latest(session, ExampleOutput) == ExampleOutput(text="second")
    assert select_where(
        session, ExampleOutput, lambda value: value.text == "first"
    ) == (ExampleOutput(text="first"),)


def test_reducer_failure_leaves_previous_slice_unchanged() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session.register_reducer(ExampleOutput, append)

    def faulty(
        slice_values: tuple[ExampleOutput, ...], event: DataEvent
    ) -> tuple[ExampleOutput, ...]:
        raise RuntimeError("boom")

    session.register_reducer(ExampleOutput, faulty)

    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)

    # Second publish should leave slice unchanged due to faulty reducer
    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)
