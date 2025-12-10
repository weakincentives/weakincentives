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

"""Inner message storage for conversation replay and resume support.

This module provides dataclasses and utilities for capturing conversation
history in session snapshots, enabling prompt evaluations to resume after
process restarts.
"""

from __future__ import annotations

from dataclasses import field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from ..dbc import pure
from .session._types import ReducerContextProtocol, ReducerEvent

if TYPE_CHECKING:
    from .session.protocols import SessionProtocol


@FrozenDataclass()
class ToolCallRecord:
    """Record of a tool call made by the assistant."""

    call_id: str
    name: str
    arguments: str  # JSON-encoded arguments
    status: Literal["pending", "completed", "failed"] = "pending"


@FrozenDataclass()
class InnerMessage:
    """A single message in a conversation, stored as a session slice item.

    Provider-neutral representation enabling snapshot-based resume.
    """

    role: Literal["system", "assistant", "user", "tool"]
    content: str

    # Ordering fields
    evaluation_id: str
    sequence: int  # Global sequence within evaluation, starts at 0

    # Tool-related fields
    tool_calls: tuple[ToolCallRecord, ...] = ()
    tool_call_id: str | None = None  # For role="tool", references the call
    tool_name: str | None = None  # For role="tool"

    # Metadata
    turn: int = 0  # Provider round-trip number
    prompt_ns: str = ""
    prompt_key: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    message_id: UUID = field(default_factory=uuid4)


@pure
def inner_message_append(
    slice_values: tuple[InnerMessage, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[InnerMessage, ...]:
    """Append message maintaining sequence order within evaluation.

    Messages are ordered by (evaluation_id, sequence). Messages from different
    evaluations are kept separate but ordered by their first message's timestamp.
    Duplicate messages (same message_id) are ignored.
    """
    del context

    value = cast(InnerMessage, event)

    # Deduplicate by message_id
    existing_ids = {msg.message_id for msg in slice_values}
    if value.message_id in existing_ids:
        return slice_values

    # Find insertion point maintaining order
    result = list(slice_values)
    insert_idx = len(result)

    for i, msg in enumerate(result):
        if msg.evaluation_id == value.evaluation_id:
            if msg.sequence > value.sequence:
                insert_idx = i
                break
        elif msg.evaluation_id > value.evaluation_id:
            # Different evaluation, maintain evaluation order
            insert_idx = i
            break

    result.insert(insert_idx, value)
    return tuple(result)


def get_inner_messages(
    session: SessionProtocol,
    evaluation_id: str | None = None,
) -> tuple[InnerMessage, ...]:
    """Get conversation messages, optionally filtered by evaluation.

    Args:
        session: The session to query.
        evaluation_id: Optional filter for a specific evaluation's messages.

    Returns:
        Messages sorted by (evaluation_id, sequence).
    """
    messages = cast(tuple[InnerMessage, ...], session.select_all(InnerMessage))

    if evaluation_id is not None:
        messages = tuple(m for m in messages if m.evaluation_id == evaluation_id)

    return tuple(sorted(messages, key=lambda m: (m.evaluation_id, m.sequence)))


def get_latest_evaluation_id(session: SessionProtocol) -> str | None:
    """Get the evaluation_id of the most recent conversation.

    Args:
        session: The session to query.

    Returns:
        The evaluation_id of the latest system message, or None if no messages.
    """
    messages = cast(tuple[InnerMessage, ...], session.select_all(InnerMessage))
    if not messages:
        return None

    # Find latest by created_at of system message
    system_messages = [m for m in messages if m.role == "system"]
    if not system_messages:
        return None

    latest = max(system_messages, key=lambda m: m.created_at)
    return latest.evaluation_id


def get_pending_tool_calls(
    session: SessionProtocol,
    evaluation_id: str,
) -> tuple[ToolCallRecord, ...]:
    """Get tool calls that haven't completed.

    Args:
        session: The session to query.
        evaluation_id: The evaluation to check for pending tools.

    Returns:
        Tool call records that started but have no corresponding result message.
    """
    messages = get_inner_messages(session, evaluation_id)

    # Find all tool_call_ids that have results
    completed_ids = {
        m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id
    }

    # Find pending from assistant messages
    pending = [
        tc
        for msg in messages
        if msg.role == "assistant"
        for tc in msg.tool_calls
        if tc.call_id not in completed_ids
    ]

    return tuple(pending)


def enable_inner_message_recording(session: SessionProtocol) -> None:
    """Register the inner message reducer on a session.

    This is a convenience function that enables message recording for
    snapshot-based resume support.

    Args:
        session: The session to configure.
    """
    session.mutate(InnerMessage).register(InnerMessage, inner_message_append)


__all__ = [
    "InnerMessage",
    "ToolCallRecord",
    "enable_inner_message_recording",
    "get_inner_messages",
    "get_latest_evaluation_id",
    "get_pending_tool_calls",
    "inner_message_append",
]
