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

"""Notification dataclasses for Claude Agent SDK hook inputs.

These dataclasses capture the input data from various SDK hooks
(SubagentStart, SubagentStop, PreCompact, Notification) and store
them in the Session for debugging purposes.
"""

from __future__ import annotations

from dataclasses import field
from datetime import UTC, datetime
from typing import Any, Final, Literal
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass

__all__ = [
    "Notification",
    "NotificationSource",
    "PreCompactInput",
    "SubagentStartInput",
    "SubagentStopInput",
    "UserNotificationInput",
]

_RENDER_TRUNCATE_LENGTH: Final[int] = 100
"""Maximum length for rendered text fields before truncation."""

NotificationSource = Literal[
    "subagent_start",
    "subagent_stop",
    "pre_compact",
    "notification",
]
"""Source hook that generated the notification."""


@FrozenDataclass()
class SubagentStartInput:
    """Typed input from SubagentStart hook.

    Captures details when a subagent is launched during execution.
    """

    session_id: str = field(
        metadata={"description": "Session identifier from the SDK."}
    )
    parent_session_id: str | None = field(
        default=None,
        metadata={"description": "Parent session ID if this is a nested subagent."},
    )
    subagent_type: str = field(
        default="",
        metadata={"description": "Type/name of the subagent being launched."},
    )
    prompt: str = field(
        default="",
        metadata={"description": "Initial prompt given to the subagent."},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SubagentStartInput:
        """Parse a dict into SubagentStartInput."""
        if data is None:
            return cls(session_id="")
        return cls(
            session_id=str(data.get("session_id", "")),
            parent_session_id=data.get("parent_session_id"),
            subagent_type=str(data.get("subagent_type", data.get("type", ""))),
            prompt=str(data.get("prompt", "")),
        )

    def render(self) -> str:
        lines = [f"SubagentStart: {self.subagent_type or 'unknown'}"]
        lines.append(f"  session_id: {self.session_id}")
        if self.parent_session_id:
            lines.append(f"  parent_session_id: {self.parent_session_id}")
        if self.prompt:
            truncated = (
                self.prompt[:_RENDER_TRUNCATE_LENGTH] + "..."
                if len(self.prompt) > _RENDER_TRUNCATE_LENGTH
                else self.prompt
            )
            lines.append(f"  prompt: {truncated}")
        return "\n".join(lines)


@FrozenDataclass()
class SubagentStopInput:
    """Typed input from SubagentStop hook.

    Captures details when a subagent completes execution.
    """

    session_id: str = field(
        metadata={"description": "Session identifier from the SDK."}
    )
    stop_reason: str = field(
        default="",
        metadata={"description": "Reason the subagent stopped."},
    )
    result: str = field(
        default="",
        metadata={"description": "Result or output from the subagent."},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SubagentStopInput:
        """Parse a dict into SubagentStopInput."""
        if data is None:
            return cls(session_id="")
        return cls(
            session_id=str(data.get("session_id", "")),
            stop_reason=str(data.get("stop_reason", data.get("stopReason", ""))),
            result=str(data.get("result", "")),
        )

    def render(self) -> str:
        lines = [f"SubagentStop: {self.stop_reason or 'unknown reason'}"]
        lines.append(f"  session_id: {self.session_id}")
        if self.result:
            truncated = (
                self.result[:_RENDER_TRUNCATE_LENGTH] + "..."
                if len(self.result) > _RENDER_TRUNCATE_LENGTH
                else self.result
            )
            lines.append(f"  result: {truncated}")
        return "\n".join(lines)


@FrozenDataclass()
class PreCompactInput:
    """Typed input from PreCompact hook.

    Captures details before context compaction occurs.
    """

    session_id: str = field(
        metadata={"description": "Session identifier from the SDK."}
    )
    message_count: int = field(
        default=0,
        metadata={"description": "Number of messages before compaction."},
    )
    token_count: int = field(
        default=0,
        metadata={"description": "Approximate token count before compaction."},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreCompactInput:
        """Parse a dict into PreCompactInput."""
        if data is None:
            return cls(session_id="")
        return cls(
            session_id=str(data.get("session_id", "")),
            message_count=int(data.get("message_count", data.get("messageCount", 0))),
            token_count=int(data.get("token_count", data.get("tokenCount", 0))),
        )

    def render(self) -> str:
        return (
            f"PreCompact: {self.message_count} messages, "
            f"~{self.token_count} tokens (session: {self.session_id})"
        )


@FrozenDataclass()
class UserNotificationInput:
    """Typed input from Notification hook.

    Captures user-facing notifications from the SDK.
    """

    session_id: str = field(
        metadata={"description": "Session identifier from the SDK."}
    )
    message: str = field(
        default="",
        metadata={"description": "Notification message text."},
    )
    level: str = field(
        default="info",
        metadata={"description": "Notification level (info, warning, error)."},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UserNotificationInput:
        """Parse a dict into UserNotificationInput."""
        if data is None:
            return cls(session_id="")
        return cls(
            session_id=str(data.get("session_id", "")),
            message=str(data.get("message", "")),
            level=str(data.get("level", "info")),
        )

    def render(self) -> str:
        return (
            f"Notification [{self.level}]: {self.message} (session: {self.session_id})"
        )


NotificationPayload = (
    SubagentStartInput | SubagentStopInput | PreCompactInput | UserNotificationInput
)
"""Union of all notification payload types."""


@FrozenDataclass()
class Notification:
    """Generic notification record from Claude Agent SDK hooks.

    Stores the details from SubagentStart, SubagentStop, PreCompact,
    and Notification hooks for debugging and observability.
    """

    source: NotificationSource = field(
        metadata={"description": "Hook that generated this notification."}
    )
    payload: NotificationPayload = field(
        metadata={"description": "Typed payload from the hook input."}
    )
    raw_input: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Raw input dict from the SDK hook."},
    )
    prompt_name: str = field(
        default="",
        metadata={"description": "Name of the prompt being evaluated."},
    )
    adapter_name: str = field(
        default="",
        metadata={"description": "Name of the adapter that received the hook."},
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
        metadata={"description": "Timestamp when the notification was created."},
    )
    notification_id: UUID = field(
        default_factory=uuid4,
        metadata={"description": "Unique identifier for this notification."},
    )

    def render(self) -> str:
        """Render the notification for logging."""
        lines = [
            f"[{self.source}] {self.created_at.isoformat()}",
            f"  prompt: {self.prompt_name}",
            f"  adapter: {self.adapter_name}",
        ]
        if hasattr(self.payload, "render"):
            lines.append(f"  {self.payload.render()}")
        return "\n".join(lines)
