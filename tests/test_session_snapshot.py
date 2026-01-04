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

"""Tests for session snapshot and restore functionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from tests.helpers.session import (
    ExampleOutput,
    ExamplePayload,
    make_prompt_event,
    make_prompt_rendered,
    make_tool_event,
)
from weakincentives.runtime.events import (
    InProcessDispatcher,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    Replace,
    Session,
    SlicePolicy,
    SliceView,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    append_all,
)
from weakincentives.types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


def test_snapshot_round_trip_restores_state(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
    original_state = session[ExampleOutput].all()

    snapshot = session.snapshot(include_all=True)
    raw = snapshot.to_json()
    restored = Snapshot.from_json(raw)

    third_result = bus.dispatch(make_prompt_event(ExampleOutput(text="third")))
    assert session[ExampleOutput].all() != original_state
    assert third_result.ok

    session.restore(restored)

    assert session[ExampleOutput].all() == original_state


def test_snapshot_preserves_custom_reducer_behavior(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class Summary:
        entries: tuple[str, ...]

    session, bus = session_factory()

    def aggregate(
        view: SliceView[Summary],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[Summary]:
        del context
        value = cast(ExampleOutput, event)
        latest = view.latest()
        entries = latest.entries if latest else ()
        return Replace((Summary((*entries, value.text)),))

    session[Summary].register(ExampleOutput, aggregate)

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="start")))
    snapshot = session.snapshot(include_all=True)

    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="after")))
    assert session[Summary].all()[0].entries == ("start", "after")

    session.restore(snapshot)

    assert session[Summary].all()[0].entries == ("start",)

    third_result = bus.dispatch(make_prompt_event(ExampleOutput(text="again")))

    assert session[Summary].all()[0].entries == ("start", "again")
    assert first_result.ok
    assert second_result.ok
    assert third_result.ok


def test_snapshot_includes_event_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    rendered_event = make_prompt_rendered(rendered_prompt="Rendered prompt text")
    executed_event = make_prompt_event(ExampleOutput(text="complete"))
    tool_event = make_tool_event(4)

    bus.dispatch(rendered_event)
    bus.dispatch(executed_event)
    bus.dispatch(tool_event)

    snapshot = session.snapshot(
        policies=frozenset({SlicePolicy.STATE, SlicePolicy.LOG})
    )
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
    assert executed_snapshot[0].result.output == ExampleOutput(text="complete")
    assert tools_snapshot[0].result.value == ExamplePayload(value=4)

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
    assert restored_executed.result["output"] == {"text": "complete"}

    restored_tool = restored_tool_slice[0]
    assert restored_tool.event_id == tool_event.event_id
    assert restored_tool.name == "tool"
    assert restored_tool.params == {"value": 4}
    assert restored_tool.result["value"] == {"value": 4}


def test_snapshot_filters_log_slices_by_default(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_tool_event(1))
    bus.dispatch(make_prompt_event(ExampleOutput(text="state")))

    snapshot = session.snapshot()

    assert ToolInvoked not in snapshot.slices
    assert snapshot.slices[ExampleOutput] == (ExampleOutput(text="state"),)


def test_snapshot_respects_registered_log_policy(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class LogEntry:
        message: str

    session, _ = session_factory()

    session[LogEntry].register(LogEntry, append_all, policy=SlicePolicy.LOG)
    session[LogEntry].seed((LogEntry(message="hello"),))

    snapshot = session.snapshot()
    assert LogEntry not in snapshot.slices

    snapshot_with_logs = session.snapshot(
        policies=frozenset({SlicePolicy.STATE, SlicePolicy.LOG})
    )
    assert snapshot_with_logs.slices[LogEntry] == (LogEntry(message="hello"),)


def test_rollback_preserves_log_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_event = make_tool_event(1)
    second_event = make_tool_event(2)

    bus.dispatch(first_event)
    snapshot = session.snapshot()

    bus.dispatch(second_event)

    session.restore(snapshot)

    assert session[ToolInvoked].all() == (first_event, second_event)


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

    parent.restore(parent_snapshot)

    assert parent.children == (first_child, second_child)


def test_snapshot_rollback_requires_registered_slices(
    session_factory: SessionFactory,
) -> None:
    source, bus = session_factory()
    result = bus.dispatch(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session(bus=InProcessDispatcher())

    with pytest.raises(SnapshotRestoreError):
        target.restore(snapshot)

    assert target[ExampleOutput].all() == ()


def test_snapshot_rejects_non_dataclass_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()
    session[cast(type[SupportsDataclass], str)].seed(
        cast(tuple[SupportsDataclass, ...], ("value",)),
    )

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()


def test_mutate_rollback_restores_snapshot(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    snapshot = session.snapshot()

    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert session[ExampleOutput].latest() == ExampleOutput(text="second")

    session.restore(snapshot)

    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)
