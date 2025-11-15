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

import logging
from dataclasses import dataclass, is_dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse
from weakincentives.dbc import dbc_enabled
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    append,
    replace_latest,
    select_all,
    select_latest,
    select_where,
    upsert_by,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


DEFAULT_SESSION_ID = uuid4()


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
    payload = ExamplePayload(value=value)
    tool_result = cast(
        ToolResult[object],
        ToolResult(message="ok", value=payload),
    )
    return ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=value),
        result=tool_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=payload,
    )


def make_prompt_event(output: object) -> PromptExecuted:
    response = PromptResponse(
        prompt_name="example",
        text="done",
        output=output,
        tool_results=(),
        provider_payload=None,
    )
    prompt_value = (
        cast(SupportsDataclass, output)
        if output is not None and is_dataclass(output)
        else None
    )
    return PromptExecuted(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        result=response,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=prompt_value,
    )


def make_prompt_rendered(
    *,
    rendered_prompt: str,
    session_id: UUID | None = None,
    params_value: int = 1,
) -> PromptRendered:
    return PromptRendered(
        prompt_ns="example",
        prompt_key="example",
        prompt_name="Example",
        adapter=GENERIC_ADAPTER_NAME,
        session_id=session_id,
        render_inputs=(ExampleParams(value=params_value),),
        rendered_prompt=rendered_prompt,
        created_at=datetime.now(UTC),
    )


def test_tool_invoked_appends_payload_once(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    first_result = bus.publish(event)
    second_result = bus.publish(event)

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)
    assert isinstance(event.event_id, UUID)


def test_tool_invoked_enriches_missing_value(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    payload = ExamplePayload(value=7)
    enriched_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=7),
        result=cast(ToolResult[object], ToolResult(message="ok", value=payload)),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
    )

    bus.publish(enriched_event)

    tool_events = session.select_all(ToolInvoked)
    assert tool_events[0].value == payload
    assert session.select_all(ExamplePayload) == (payload,)


def test_prompt_rendered_appends_start_event(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    event = make_prompt_rendered(
        rendered_prompt="Rendered prompt text",
        session_id=uuid4(),
    )

    result = bus.publish(event)

    assert result.ok
    lifecycle = session.select_all(PromptRendered)
    assert lifecycle == (event,)
    assert lifecycle[0].rendered_prompt == "Rendered prompt text"
    assert lifecycle[0].value is event
    assert isinstance(event.event_id, UUID)


def test_prompt_executed_emits_multiple_dataclasses(
    session_factory: SessionFactory,
) -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    session, bus = session_factory()

    result = bus.publish(make_prompt_event(outputs))

    assert result.ok
    assert session.select_all(ExampleOutput) == tuple(outputs)


def test_reducers_run_in_registration_order(session_factory: SessionFactory) -> None:
    @dataclass(slots=True, frozen=True)
    class FirstSlice:
        value: str

    @dataclass(slots=True, frozen=True)
    class SecondSlice:
        value: str

    session, bus = session_factory()

    call_order: list[str] = []

    def first(
        slice_values: tuple[FirstSlice, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[FirstSlice, ...]:
        del context
        call_order.append("first")
        value = cast(ExampleOutput, event.value)
        return (*slice_values, FirstSlice(value.text))

    def second(
        slice_values: tuple[SecondSlice, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SecondSlice, ...]:
        del context
        call_order.append("second")
        value = cast(ExampleOutput, event.value)
        return (*slice_values, SecondSlice(value.text))

    session.register_reducer(ExampleOutput, first, slice_type=FirstSlice)
    session.register_reducer(ExampleOutput, second, slice_type=SecondSlice)

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session.select_all(FirstSlice) == (FirstSlice("hello"),)
    assert session.select_all(SecondSlice) == (SecondSlice("hello"),)
    assert result.ok


def test_default_append_used_when_no_custom_reducer(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="hello"),)


def test_prompt_executed_enriches_missing_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    output = ExampleOutput(text="filled")
    enriched_event = PromptExecuted(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        result=cast(
            PromptResponse[object],
            PromptResponse(
                prompt_name="example",
                text="done",
                output=output,
                tool_results=(),
                provider_payload=None,
            ),
        ),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
    )

    bus.publish(enriched_event)

    prompt_events = session.select_all(PromptExecuted)
    assert prompt_events[0].value == output
    assert session.select_all(ExampleOutput) == (output,)


def test_non_dataclass_payloads_are_ignored(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    non_dataclass_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=2),
        result=cast(
            ToolResult[object], ToolResult(message="ok", value="not a dataclass")
        ),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
    )

    first_result = bus.publish(event)
    second_result = bus.publish(non_dataclass_event)

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

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


def test_replace_latest_keeps_only_newest_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session.register_reducer(ExamplePayload, replace_latest)

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert session.select_all(ExamplePayload) == (ExamplePayload(value=2),)


def test_tool_data_slice_records_failures(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_tool_event(1))

    failure = cast(
        ToolResult[object],
        ToolResult(message="failed", value=None, success=False),
    )
    failure_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=2),
        result=failure,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
    )
    bus.publish(failure_event)

    tool_events = session.select_all(ToolInvoked)
    assert len(tool_events) == 2
    assert tool_events[0].value == ExamplePayload(value=1)
    assert tool_events[1].value is None
    assert tool_events[1].result.success is False


def test_selector_helpers_delegate_to_session(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

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


def test_selector_helpers_respect_dbc_purity(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    with dbc_enabled():
        assert select_all(session, ExampleOutput) == (
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        )
        assert select_latest(session, ExampleOutput) == ExampleOutput(text="second")
        assert select_where(
            session, ExampleOutput, lambda value: value.text.startswith("f")
        ) == (ExampleOutput(text="first"),)


def test_select_where_logs_violate_purity_contract(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    logger = logging.getLogger(__name__)

    def predicate(value: ExampleOutput) -> bool:
        logger.warning("Saw %s", value)
        return True

    with dbc_enabled(), pytest.raises(AssertionError):
        select_where(session, ExampleOutput, predicate)


def test_reducer_failure_leaves_previous_slice_unchanged(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session.register_reducer(ExampleOutput, append)

    def faulty(
        slice_values: tuple[ExampleOutput, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[ExampleOutput, ...]:
        del context
        raise RuntimeError("boom")

    session.register_reducer(ExampleOutput, faulty)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)
    assert first_result.handled_count == 1

    # Second publish should leave slice unchanged due to faulty reducer
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert second_result.handled_count == 1
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)


def test_snapshot_round_trip_restores_state(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

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


def test_snapshot_preserves_custom_reducer_behavior(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class Summary:
        entries: tuple[str, ...]

    session, bus = session_factory()

    def aggregate(
        slice_values: tuple[Summary, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[Summary, ...]:
        del context
        value = cast(ExampleOutput, event.value)
        entries = slice_values[-1].entries if slice_values else ()
        return (Summary((*entries, value.text)),)

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


def test_snapshot_rollback_requires_registered_slices(
    session_factory: SessionFactory,
) -> None:
    source, bus = session_factory()
    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session(bus=InProcessEventBus())

    with pytest.raises(SnapshotRestoreError):
        target.rollback(snapshot)

    assert target.select_all(ExampleOutput) == ()


def test_snapshot_rejects_non_dataclass_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()
    session.seed_slice(
        cast(type[SupportsDataclass], str),
        cast(tuple[SupportsDataclass, ...], ("value",)),
    )

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()


def test_reset_clears_registered_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.register_reducer(ExampleOutput, append)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert first_result.ok
    assert session.select_all(ExampleOutput)

    session.reset()

    assert session.select_all(ExampleOutput) == ()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert second_result.ok
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="second"),)


def test_clone_preserves_state_and_reducer_registration(
    session_factory: SessionFactory,
) -> None:
    provided_session_id = uuid4()
    provided_created_at = datetime.now(UTC)
    session, bus = session_factory(
        session_id=provided_session_id, created_at=provided_created_at
    )

    session.register_reducer(ExampleOutput, replace_latest)

    result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert result.ok

    clone_bus = InProcessEventBus()
    clone = session.clone(bus=clone_bus)

    assert clone.session_id == provided_session_id
    assert clone.created_at == provided_created_at
    assert clone.select_all(ExampleOutput) == (ExampleOutput(text="first"),)
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)
    assert clone._reducers.keys() == session._reducers.keys()

    clone_bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert clone.select_all(ExampleOutput)[-1] == ExampleOutput(text="second")
    assert session.select_all(ExampleOutput) == (ExampleOutput(text="first"),)

    bus.publish(make_prompt_event(ExampleOutput(text="third")))

    assert session.select_all(ExampleOutput)[-1] == ExampleOutput(text="third")
    assert clone.select_all(ExampleOutput)[-1] == ExampleOutput(text="second")


def test_clone_attaches_to_new_bus_when_provided(
    session_factory: SessionFactory,
) -> None:
    session, source_bus = session_factory()

    source_bus.publish(make_prompt_event(ExampleOutput(text="first")))

    target_bus = InProcessEventBus()
    clone_session_id = uuid4()
    clone_created_at = datetime.now(UTC)
    clone = session.clone(
        bus=target_bus, session_id=clone_session_id, created_at=clone_created_at
    )

    assert clone.session_id == clone_session_id
    assert clone.created_at == clone_created_at
    assert clone.select_all(ExampleOutput) == session.select_all(ExampleOutput)

    target_bus.publish(make_prompt_event(ExampleOutput(text="from clone")))

    assert clone.select_all(ExampleOutput)[-1] == ExampleOutput(text="from clone")
    assert session.select_all(ExampleOutput)[-1] == ExampleOutput(text="first")

    source_bus.publish(make_prompt_event(ExampleOutput(text="original")))

    assert session.select_all(ExampleOutput)[-1] == ExampleOutput(text="original")


def test_session_requires_timezone_aware_created_at() -> None:
    bus = InProcessEventBus()
    naive_timestamp = datetime.now()

    with pytest.raises(ValueError):
        Session(bus=bus, created_at=naive_timestamp)
