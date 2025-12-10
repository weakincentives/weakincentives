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

"""Tests for inner_messages module."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from weakincentives.runtime.events import EventBus
from weakincentives.runtime.inner_messages import (
    InnerMessage,
    ToolCallRecord,
    enable_inner_message_recording,
    get_inner_messages,
    get_latest_evaluation_id,
    get_pending_tool_calls,
    inner_message_append,
)
from weakincentives.runtime.session import Session
from weakincentives.runtime.session._types import ReducerContextProtocol


@dataclass(slots=True)
class _Context(ReducerContextProtocol):
    session: Session
    event_bus: EventBus


class TestToolCallRecord:
    def test_defaults_to_pending_status(self) -> None:
        record = ToolCallRecord(call_id="c1", name="test_tool", arguments="{}")
        assert record.status == "pending"

    def test_can_set_status(self) -> None:
        record = ToolCallRecord(
            call_id="c1", name="test_tool", arguments="{}", status="completed"
        )
        assert record.status == "completed"

    def test_is_immutable(self) -> None:
        record = ToolCallRecord(call_id="c1", name="test_tool", arguments="{}")
        updated = replace(record, status="completed")
        assert record.status == "pending"
        assert updated.status == "completed"


class TestInnerMessage:
    def test_creates_with_required_fields(self) -> None:
        msg = InnerMessage(
            role="system",
            content="You are a helpful assistant.",
            evaluation_id="eval-1",
            sequence=0,
        )
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
        assert msg.evaluation_id == "eval-1"
        assert msg.sequence == 0

    def test_defaults_for_optional_fields(self) -> None:
        msg = InnerMessage(
            role="assistant",
            content="Hello",
            evaluation_id="eval-1",
            sequence=1,
        )
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None
        assert msg.tool_name is None
        assert msg.turn == 0
        assert msg.prompt_ns == ""
        assert msg.prompt_key == ""
        assert msg.message_id is not None

    def test_assistant_message_with_tool_calls(self) -> None:
        tool_call = ToolCallRecord(
            call_id="c1", name="search", arguments='{"q": "test"}'
        )
        msg = InnerMessage(
            role="assistant",
            content="Let me search for that.",
            evaluation_id="eval-1",
            sequence=1,
            tool_calls=(tool_call,),
            turn=1,
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_result_message(self) -> None:
        msg = InnerMessage(
            role="tool",
            content='{"results": []}',
            evaluation_id="eval-1",
            sequence=2,
            tool_call_id="c1",
            tool_name="search",
            turn=1,
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "c1"
        assert msg.tool_name == "search"


class TestInnerMessageAppendReducer:
    def test_appends_to_empty_slice(self) -> None:
        session = Session()
        context = _Context(session=session, event_bus=session.event_bus)
        msg = InnerMessage(
            role="system",
            content="test",
            evaluation_id="eval-1",
            sequence=0,
        )

        result = inner_message_append((), msg, context=context)

        assert len(result) == 1
        assert result[0] is msg

    def test_deduplicates_by_message_id(self) -> None:
        session = Session()
        context = _Context(session=session, event_bus=session.event_bus)
        msg = InnerMessage(
            role="system",
            content="test",
            evaluation_id="eval-1",
            sequence=0,
        )

        result = inner_message_append((msg,), msg, context=context)

        assert len(result) == 1
        assert result[0] is msg

    def test_maintains_sequence_order_within_evaluation(self) -> None:
        session = Session()
        context = _Context(session=session, event_bus=session.event_bus)

        msg0 = InnerMessage(
            role="system", content="system", evaluation_id="eval-1", sequence=0
        )
        msg2 = InnerMessage(
            role="tool", content="result", evaluation_id="eval-1", sequence=2
        )
        msg1 = InnerMessage(
            role="assistant", content="call", evaluation_id="eval-1", sequence=1
        )

        # Insert out of order
        result = inner_message_append((msg0, msg2), msg1, context=context)

        assert len(result) == 3
        assert result[0].sequence == 0
        assert result[1].sequence == 1
        assert result[2].sequence == 2

    def test_appends_to_same_evaluation(self) -> None:
        session = Session()
        context = _Context(session=session, event_bus=session.event_bus)

        msg0 = InnerMessage(
            role="system", content="system", evaluation_id="eval-1", sequence=0
        )
        msg1 = InnerMessage(
            role="assistant", content="hello", evaluation_id="eval-1", sequence=1
        )

        result = inner_message_append((msg0,), msg1, context=context)

        assert len(result) == 2
        assert result[1] is msg1

    def test_inserts_earlier_evaluation_before_later(self) -> None:
        session = Session()
        context = _Context(session=session, event_bus=session.event_bus)

        # Start with message from eval-2
        msg_eval2 = InnerMessage(
            role="system", content="system", evaluation_id="eval-2", sequence=0
        )
        # Insert message from eval-1 (should come before eval-2)
        msg_eval1 = InnerMessage(
            role="system", content="system", evaluation_id="eval-1", sequence=0
        )

        result = inner_message_append((msg_eval2,), msg_eval1, context=context)

        assert len(result) == 2
        assert result[0].evaluation_id == "eval-1"
        assert result[1].evaluation_id == "eval-2"


class TestGetInnerMessages:
    def test_returns_empty_for_no_messages(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        result = get_inner_messages(session)

        assert result == ()

    def test_returns_all_messages_sorted(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        # Add messages out of order
        msg1 = InnerMessage(
            role="assistant", content="hello", evaluation_id="eval-1", sequence=1
        )
        msg0 = InnerMessage(
            role="system", content="system", evaluation_id="eval-1", sequence=0
        )

        session.mutate(InnerMessage).dispatch(msg1)
        session.mutate(InnerMessage).dispatch(msg0)

        result = get_inner_messages(session)

        assert len(result) == 2
        assert result[0].sequence == 0
        assert result[1].sequence == 1

    def test_filters_by_evaluation_id(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        msg1 = InnerMessage(
            role="system", content="sys1", evaluation_id="eval-1", sequence=0
        )
        msg2 = InnerMessage(
            role="system", content="sys2", evaluation_id="eval-2", sequence=0
        )

        session.mutate(InnerMessage).dispatch(msg1)
        session.mutate(InnerMessage).dispatch(msg2)

        result = get_inner_messages(session, evaluation_id="eval-1")

        assert len(result) == 1
        assert result[0].evaluation_id == "eval-1"


class TestGetLatestEvaluationId:
    def test_returns_none_for_no_messages(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        result = get_latest_evaluation_id(session)

        assert result is None

    def test_returns_none_for_no_system_messages(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        # Add only assistant message
        msg = InnerMessage(
            role="assistant", content="hello", evaluation_id="eval-1", sequence=1
        )
        session.mutate(InnerMessage).dispatch(msg)

        result = get_latest_evaluation_id(session)

        assert result is None

    def test_returns_latest_by_created_at(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        now = datetime.now(UTC)
        earlier = now - timedelta(hours=1)

        msg1 = InnerMessage(
            role="system",
            content="old",
            evaluation_id="eval-1",
            sequence=0,
            created_at=earlier,
        )
        msg2 = InnerMessage(
            role="system",
            content="new",
            evaluation_id="eval-2",
            sequence=0,
            created_at=now,
        )

        session.mutate(InnerMessage).dispatch(msg1)
        session.mutate(InnerMessage).dispatch(msg2)

        result = get_latest_evaluation_id(session)

        assert result == "eval-2"


class TestGetPendingToolCalls:
    def test_returns_empty_for_no_messages(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        result = get_pending_tool_calls(session, "eval-1")

        assert result == ()

    def test_returns_empty_when_all_completed(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        tool_call = ToolCallRecord(call_id="c1", name="search", arguments="{}")
        assistant_msg = InnerMessage(
            role="assistant",
            content="",
            evaluation_id="eval-1",
            sequence=1,
            tool_calls=(tool_call,),
        )
        tool_result = InnerMessage(
            role="tool",
            content="result",
            evaluation_id="eval-1",
            sequence=2,
            tool_call_id="c1",
        )

        session.mutate(InnerMessage).dispatch(assistant_msg)
        session.mutate(InnerMessage).dispatch(tool_result)

        result = get_pending_tool_calls(session, "eval-1")

        assert result == ()

    def test_returns_pending_tool_calls(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        tool_call1 = ToolCallRecord(call_id="c1", name="search", arguments="{}")
        tool_call2 = ToolCallRecord(call_id="c2", name="fetch", arguments="{}")
        assistant_msg = InnerMessage(
            role="assistant",
            content="",
            evaluation_id="eval-1",
            sequence=1,
            tool_calls=(tool_call1, tool_call2),
        )
        # Only first tool completed
        tool_result = InnerMessage(
            role="tool",
            content="result",
            evaluation_id="eval-1",
            sequence=2,
            tool_call_id="c1",
        )

        session.mutate(InnerMessage).dispatch(assistant_msg)
        session.mutate(InnerMessage).dispatch(tool_result)

        result = get_pending_tool_calls(session, "eval-1")

        assert len(result) == 1
        assert result[0].call_id == "c2"
        assert result[0].name == "fetch"


class TestEnableInnerMessageRecording:
    def test_registers_reducer(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        # Verify we can dispatch messages
        msg = InnerMessage(
            role="system", content="test", evaluation_id="eval-1", sequence=0
        )
        session.mutate(InnerMessage).dispatch(msg)

        result = session.select_all(InnerMessage)
        assert len(result) == 1

    def test_multiple_registrations_safe(self) -> None:
        session = Session()
        enable_inner_message_recording(session)
        enable_inner_message_recording(session)

        msg = InnerMessage(
            role="system", content="test", evaluation_id="eval-1", sequence=0
        )
        session.mutate(InnerMessage).dispatch(msg)

        # Should not duplicate
        result = session.select_all(InnerMessage)
        assert len(result) == 1


class TestSessionSnapshot:
    def test_messages_survive_snapshot_roundtrip(self) -> None:
        session = Session()
        enable_inner_message_recording(session)

        # Add messages
        msg = InnerMessage(
            role="system",
            content="test",
            evaluation_id="eval-1",
            sequence=0,
            prompt_ns="ns",
            prompt_key="key",
        )
        session.mutate(InnerMessage).dispatch(msg)

        # Take snapshot
        snapshot = session.snapshot()
        json_str = snapshot.to_json()

        # Restore to new session
        from weakincentives.runtime.session import Snapshot

        restored_snapshot = Snapshot.from_json(json_str)
        new_session = Session()
        enable_inner_message_recording(new_session)
        new_session.mutate().rollback(restored_snapshot)

        # Verify messages restored
        result = get_inner_messages(new_session)
        assert len(result) == 1
        assert result[0].content == "test"
        assert result[0].evaluation_id == "eval-1"
        assert result[0].prompt_ns == "ns"
        assert result[0].prompt_key == "key"
