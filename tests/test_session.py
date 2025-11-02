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

import pytest

from weakincentives.adapters.core import PromptResponse
from weakincentives.events import InProcessEventBus, PromptExecuted, ToolInvoked
from weakincentives.prompt.tool import ToolResult
from weakincentives.session import (
    DataEvent,
    Session,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    ToolData,
    append,
    replace_latest,
    select_all,
    select_latest,
    select_where,
    upsert_by,
)
from weakincentives.session.session import PromptData, _append_tool_data


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
        ToolResult(message="ok", value=ExamplePayload(value=value)),
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
        result=response,
    )


def test_tool_invoked_appends_payload_once() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    event = make_tool_event(1)
    first_result = bus.publish(event)
    second_result = bus.publish(event)

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)


def test_prompt_executed_emits_multiple_dataclasses() -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    bus = InProcessEventBus()
    session = Session(bus=bus)

    result = bus.publish(make_prompt_event(outputs))

    assert result.ok
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

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session.select_all(FirstSlice) == (FirstSlice("hello"),)
    assert session.select_all(SecondSlice) == (SecondSlice("hello"),)
    assert result.ok


def test_default_append_used_when_no_custom_reducer() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
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
            ToolResult[object], ToolResult(message="ok", value="not a dataclass")
        ),
    )

    first_result = bus.publish(event)
    second_result = bus.publish(non_dataclass_event)

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session.register_reducer(ExamplePayload, upsert_by(lambda payload: payload.value))

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(1))
    third_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert third_result.ok
    assert session.select_all(ExamplePayload) == (
        ExamplePayload(value=1),
        ExamplePayload(value=2),
    )


def test_replace_latest_keeps_only_newest_value() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session.register_reducer(ExamplePayload, replace_latest)

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=2),)


def test_tool_data_slice_records_failures() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    bus.publish(make_tool_event(1))

    failure = cast(
        ToolResult[object],
        ToolResult(message="failed", value=None, success=False),
    )
    failure_event = ToolInvoked(
        prompt_name="example",
        adapter="adapter",
        name="tool",
        params=ExampleParams(value=2),
        result=failure,
    )
    bus.publish(failure_event)

    tool_events = session.select_all(ToolData)
    assert len(tool_events) == 2
    assert tool_events[0].value == ExamplePayload(value=1)
    assert tool_events[1].value is None
    assert tool_events[1].source.result.success is False


def test_append_tool_data_ignores_prompt_data() -> None:
    prompt_event = make_prompt_event(ExampleOutput(text="hello"))
    prompt_data = PromptData(value=ExampleOutput(text="hello"), source=prompt_event)

    appended = _append_tool_data((), cast(DataEvent, prompt_data))
    assert appended == ()


def test_selector_helpers_delegate_to_session() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    assert select_latest(session, ExampleOutput) is None

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
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

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)
    assert first_result.handled_count == 1

    # Second publish should leave slice unchanged due to faulty reducer
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert second_result.handled_count == 1
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)


def test_snapshot_round_trip_restores_state() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
    original_state = session.select_all(ExampleOutput)

    snapshot = session.snapshot()
    raw = snapshot.to_json()
    restored = Snapshot.from_json(raw)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="third")))
    assert session.select_all(ExampleOutput) != original_state
    assert third_result.ok

    session.rollback(restored)

    assert session.select_all(ExampleOutput) == original_state


def test_snapshot_preserves_custom_reducer_behavior() -> None:
    @dataclass(slots=True, frozen=True)
    class Summary:
        entries: tuple[str, ...]

    bus = InProcessEventBus()
    session = Session(bus=bus)

    def aggregate(
        slice_values: tuple[Summary, ...], event: DataEvent
    ) -> tuple[Summary, ...]:
        value = cast(ExampleOutput, event.value)
        entries = slice_values[-1].entries if slice_values else ()
        return (Summary(entries + (value.text,)),)

    session.register_reducer(ExampleOutput, aggregate, slice_type=Summary)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="start")))
    snapshot = session.snapshot()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="after")))
    assert session.select_all(Summary)[0].entries == ("start", "after")

    session.rollback(snapshot)

    assert session.select_all(Summary)[0].entries == ("start",)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="again")))

    assert session.select_all(Summary)[0].entries == ("start", "again")
    assert first_result.ok
    assert second_result.ok
    assert third_result.ok


def test_snapshot_rollback_requires_registered_slices() -> None:
    bus = InProcessEventBus()
    source = Session(bus=bus)
    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session()

    with pytest.raises(SnapshotRestoreError):
        target.rollback(snapshot)

    assert target.select_all(ExampleOutput) == ()


def test_snapshot_rejects_non_dataclass_values() -> None:
    session = Session()
    session.seed_slice(str, ("value",))

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()
