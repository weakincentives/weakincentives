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
    QueryBuilder,
    ReducerContextProtocol,
    ReducerEvent,
    ReducerEventWithValue,
    Session,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    append,
    replace_latest,
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
    rendered_output = tool_result.render()
    return ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=value),
        result=tool_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=payload,
        rendered_output=rendered_output,
    )


def make_prompt_event(output: object) -> PromptExecuted:
    response = PromptResponse(
        prompt_name="example",
        text="done",
        output=output,
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
    assert session.query(ExamplePayload).all() == (ExamplePayload(value=1),)
    assert isinstance(event.event_id, UUID)


def test_tool_invoked_enriches_missing_value(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    payload = ExamplePayload(value=7)
    enriched_result = cast(ToolResult[object], ToolResult(message="ok", value=payload))
    enriched_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=7),
        result=enriched_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
        rendered_output=enriched_result.render(),
    )

    bus.publish(enriched_event)

    tool_events = session.query(ToolInvoked).all()
    assert tool_events[0].value == payload
    assert session.query(ExamplePayload).all() == (payload,)


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
    lifecycle = session.query(PromptRendered).all()
    assert lifecycle == (event,)
    assert lifecycle[0].rendered_prompt == "Rendered prompt text"
    assert isinstance(event.event_id, UUID)


def test_prompt_executed_emits_multiple_dataclasses(
    session_factory: SessionFactory,
) -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    session, bus = session_factory()

    result = bus.publish(make_prompt_event(outputs))

    assert result.ok
    assert session.query(ExampleOutput).all() == tuple(outputs)


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
        assert isinstance(event, ReducerEventWithValue)
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
        assert isinstance(event, ReducerEventWithValue)
        value = cast(ExampleOutput, event.value)
        return (*slice_values, SecondSlice(value.text))

    session.mutate(FirstSlice).register(ExampleOutput, first)
    session.mutate(SecondSlice).register(ExampleOutput, second)

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session.query(FirstSlice).all() == (FirstSlice("hello"),)
    assert session.query(SecondSlice).all() == (SecondSlice("hello"),)
    assert result.ok


def test_default_append_used_when_no_custom_reducer(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="hello"),)


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
            ),
        ),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
    )

    bus.publish(enriched_event)

    prompt_events = session.query(PromptExecuted).all()
    assert prompt_events[0].value == output
    assert session.query(ExampleOutput).all() == (output,)


def test_non_dataclass_payloads_are_ignored(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    non_dataclass_result = cast(
        ToolResult[object], ToolResult(message="ok", value="not a dataclass")
    )
    non_dataclass_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=2),
        result=non_dataclass_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        value=None,
        rendered_output=non_dataclass_result.render(),
    )

    first_result = bus.publish(event)
    second_result = bus.publish(non_dataclass_event)

    assert first_result.ok
    assert second_result.ok
    assert session.query(ExamplePayload).all() == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExamplePayload).register(
        ExamplePayload, upsert_by(lambda payload: payload.value)
    )

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(1))
    third_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert third_result.ok
    assert session.query(ExamplePayload).all() == (
        ExamplePayload(value=1),
        ExamplePayload(value=2),
    )


def test_replace_latest_keeps_only_newest_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session.mutate(ExamplePayload).register(ExamplePayload, replace_latest)

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert session.query(ExamplePayload).all() == (ExamplePayload(value=2),)


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
        rendered_output=failure.render(),
    )
    bus.publish(failure_event)

    tool_events = session.query(ToolInvoked).all()
    assert len(tool_events) == 2
    assert tool_events[0].value == ExamplePayload(value=1)
    assert tool_events[1].value is None
    assert tool_events[1].result.success is False


def test_reducer_failure_leaves_previous_slice_unchanged(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append)

    def faulty(
        slice_values: tuple[ExampleOutput, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[ExampleOutput, ...]:
        del context
        raise RuntimeError("boom")

    session.mutate(ExampleOutput).register(ExampleOutput, faulty)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert first_result.handled_count == 1

    # Second publish should leave slice unchanged due to faulty reducer
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert second_result.handled_count == 1
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)


def test_snapshot_round_trip_restores_state(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
    original_state = session.query(ExampleOutput).all()

    snapshot = session.snapshot()
    raw = snapshot.to_json()
    restored = Snapshot.from_json(raw)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="third")))
    assert session.query(ExampleOutput).all() != original_state
    assert third_result.ok

    session.mutate().rollback(restored)

    assert session.query(ExampleOutput).all() == original_state


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
        assert isinstance(event, ReducerEventWithValue)
        value = cast(ExampleOutput, event.value)
        entries = slice_values[-1].entries if slice_values else ()
        return (Summary((*entries, value.text)),)

    session.mutate(Summary).register(ExampleOutput, aggregate)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="start")))
    snapshot = session.snapshot()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="after")))
    assert session.query(Summary).all()[0].entries == ("start", "after")

    session.mutate().rollback(snapshot)

    assert session.query(Summary).all()[0].entries == ("start",)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="again")))

    assert session.query(Summary).all()[0].entries == ("start", "again")
    assert first_result.ok
    assert second_result.ok
    assert third_result.ok


def test_snapshot_includes_event_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    rendered_event = make_prompt_rendered(rendered_prompt="Rendered prompt text")
    executed_event = make_prompt_event(ExampleOutput(text="complete"))
    tool_event = make_tool_event(4)

    bus.publish(rendered_event)
    bus.publish(executed_event)
    bus.publish(tool_event)

    snapshot = session.snapshot()
    serialized = snapshot.to_json()
    restored = Snapshot.from_json(serialized)

    rendered_snapshot = cast(
        tuple[PromptRendered, ...], snapshot.slices[PromptRendered]
    )
    executed_snapshot = cast(
        tuple[PromptExecuted, ...], snapshot.slices[PromptExecuted]
    )
    tools_snapshot = cast(tuple[ToolInvoked, ...], snapshot.slices[ToolInvoked])

    assert rendered_snapshot == (rendered_event,)
    assert executed_snapshot[0].value == ExampleOutput(text="complete")
    assert tools_snapshot[0].value == ExamplePayload(value=4)

    restored_rendered = cast(
        tuple[PromptRendered, ...], restored.slices[PromptRendered]
    )
    restored_executed_slice = cast(
        tuple[PromptExecuted, ...], restored.slices[PromptExecuted]
    )
    restored_tool_slice = cast(tuple[ToolInvoked, ...], restored.slices[ToolInvoked])

    assert restored_rendered[0].event_id == rendered_event.event_id
    restored_executed = restored_executed_slice[0]
    assert restored_executed.event_id == executed_event.event_id
    assert restored_executed.result["prompt_name"] == "example"
    assert restored_executed.value == {"text": "complete"}

    restored_tool = restored_tool_slice[0]
    assert restored_tool.event_id == tool_event.event_id
    assert restored_tool.name == "tool"
    assert restored_tool.params == {"value": 4}
    assert restored_tool.result["value"] == {"value": 4}


def test_snapshot_tracks_relationship_ids(session_factory: SessionFactory) -> None:
    parent, bus = session_factory()

    first_child = Session(bus=bus, parent=parent)
    parent_snapshot = parent.snapshot()

    assert parent_snapshot.parent_id is None
    assert parent_snapshot.children_ids == (first_child.session_id,)

    child_snapshot = first_child.snapshot()

    assert child_snapshot.parent_id == parent.session_id
    assert child_snapshot.children_ids == ()

    second_child = Session(bus=bus, parent=parent)

    parent.mutate().rollback(parent_snapshot)

    assert parent.children == (first_child, second_child)


def test_snapshot_rollback_requires_registered_slices(
    session_factory: SessionFactory,
) -> None:
    source, bus = session_factory()
    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session(bus=InProcessEventBus())

    with pytest.raises(SnapshotRestoreError):
        target.mutate().rollback(snapshot)

    assert target.query(ExampleOutput).all() == ()


def test_snapshot_rejects_non_dataclass_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()
    session.mutate(cast(type[SupportsDataclass], str)).seed(
        cast(tuple[SupportsDataclass, ...], ("value",)),
    )

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()


def test_reset_clears_registered_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert first_result.ok
    assert session.query(ExampleOutput).all()

    session.mutate().reset()

    assert session.query(ExampleOutput).all() == ()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert second_result.ok
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="second"),)


def test_clone_preserves_state_and_reducer_registration(
    session_factory: SessionFactory,
) -> None:
    provided_session_id = uuid4()
    provided_created_at = datetime.now(UTC)
    session, bus = session_factory(
        session_id=provided_session_id, created_at=provided_created_at
    )

    session.mutate(ExampleOutput).register(ExampleOutput, replace_latest)

    result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert result.ok

    clone_bus = InProcessEventBus()
    clone = session.clone(bus=clone_bus)

    assert clone.session_id == provided_session_id
    assert clone.created_at == provided_created_at
    assert clone.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert clone._reducers.keys() == session._reducers.keys()

    clone_bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="second")
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)

    bus.publish(make_prompt_event(ExampleOutput(text="third")))

    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="third")
    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="second")


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
    assert clone.query(ExampleOutput).all() == session.query(ExampleOutput).all()

    target_bus.publish(make_prompt_event(ExampleOutput(text="from clone")))

    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="from clone")
    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="first")

    source_bus.publish(make_prompt_event(ExampleOutput(text="original")))

    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="original")


def test_session_requires_timezone_aware_created_at() -> None:
    bus = InProcessEventBus()
    naive_timestamp = datetime.now()

    with pytest.raises(ValueError):
        Session(bus=bus, created_at=naive_timestamp)


def test_session_instantiates_default_bus_when_none_provided() -> None:
    session = Session()
    assert isinstance(session.event_bus, InProcessEventBus)


def test_query_returns_query_builder(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    builder = session.query(ExampleOutput)

    assert isinstance(builder, QueryBuilder)


def test_query_all_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).all()

    assert result == ()


def test_query_latest_returns_none_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).latest()

    assert result is None


def test_query_where_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).where(lambda x: True)

    assert result == ()


def test_query_all_returns_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    result = session.query(ExampleOutput).all()

    assert result == (ExampleOutput(text="first"), ExampleOutput(text="second"))


def test_query_latest_returns_most_recent_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    result = session.query(ExampleOutput).latest()

    assert result == ExampleOutput(text="second")


def test_query_where_filters_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="apple")))
    bus.publish(make_prompt_event(ExampleOutput(text="banana")))
    bus.publish(make_prompt_event(ExampleOutput(text="apricot")))

    result = session.query(ExampleOutput).where(lambda x: x.text.startswith("a"))

    assert result == (ExampleOutput(text="apple"), ExampleOutput(text="apricot"))


def test_query_respects_dbc_purity(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    with dbc_enabled():
        assert session.query(ExampleOutput).all() == (
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        )
        assert session.query(ExampleOutput).latest() == ExampleOutput(text="second")
        assert session.query(ExampleOutput).where(lambda x: x.text.startswith("f")) == (
            ExampleOutput(text="first"),
        )


def test_query_where_logs_violate_purity_contract(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    logger = logging.getLogger(__name__)

    def predicate(value: ExampleOutput) -> bool:
        logger.warning("Saw %s", value)
        return True

    with dbc_enabled(), pytest.raises(AssertionError):
        session.query(ExampleOutput).where(predicate)


# ──────────────────────────────────────────────────────────────────────
# Mutation API tests
# ──────────────────────────────────────────────────────────────────────


def test_mutate_returns_mutation_builder(session_factory: SessionFactory) -> None:
    from weakincentives.runtime.session import GlobalMutationBuilder, MutationBuilder

    session, _ = session_factory()

    slice_builder = session.mutate(ExampleOutput)
    global_builder = session.mutate()

    assert isinstance(slice_builder, MutationBuilder)
    assert isinstance(global_builder, GlobalMutationBuilder)


def test_mutate_seed_single_value(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).seed(ExampleOutput(text="seeded"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="seeded"),)


def test_mutate_seed_iterable_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).seed(
        [
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        ]
    )

    assert session.query(ExampleOutput).all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_clear_removes_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert session.query(ExampleOutput).all()

    session.mutate(ExampleOutput).clear()

    assert session.query(ExampleOutput).all() == ()


def test_mutate_clear_with_predicate(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="apple")))
    bus.publish(make_prompt_event(ExampleOutput(text="banana")))
    bus.publish(make_prompt_event(ExampleOutput(text="apricot")))

    session.mutate(ExampleOutput).clear(lambda x: x.text.startswith("a"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="banana"),)


def test_mutate_dispatch_triggers_registered_reducer(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    @dataclass(slots=True, frozen=True)
    class SetText:
        text: str

    def set_text_reducer(
        slice_values: tuple[ExampleOutput, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[ExampleOutput, ...]:
        del context
        if isinstance(event, ReducerEventWithValue) and isinstance(
            event.value, SetText
        ):
            return (ExampleOutput(text=event.value.text),)
        if isinstance(event, SetText):
            return (ExampleOutput(text=event.text),)
        return slice_values

    session.mutate(ExampleOutput).register(SetText, set_text_reducer)
    session.mutate(ExampleOutput).dispatch(SetText(text="dispatched"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="dispatched"),)


def test_mutate_append_uses_default_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).append(ExampleOutput(text="first"))
    session.mutate(ExampleOutput).append(ExampleOutput(text="second"))

    assert session.query(ExampleOutput).all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_register_adds_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, replace_latest)

    session.mutate(ExampleOutput).append(ExampleOutput(text="first"))
    session.mutate(ExampleOutput).append(ExampleOutput(text="second"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="second"),)


def test_mutate_reset_clears_all_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append)
    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert session.query(ExampleOutput).all()

    session.mutate().reset()

    assert session.query(ExampleOutput).all() == ()


def test_mutate_rollback_restores_snapshot(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append)
    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    snapshot = session.snapshot()

    bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert session.query(ExampleOutput).latest() == ExampleOutput(text="second")

    session.mutate().rollback(snapshot)

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
