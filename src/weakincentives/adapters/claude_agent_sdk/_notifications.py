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

"""Notification dataclass for Claude Agent SDK hook inputs.

Captures the raw input data from SDK hooks (SubagentStart, SubagentStop,
PreCompact, Notification) and stores them in the Session for debugging.
"""

from __future__ import annotations

from dataclasses import field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass
from ...runtime.annotations import SliceMeta, register_annotations

__all__ = [
    "Notification",
    "NotificationSource",
]

NotificationSource = Literal[
    "subagent_start",
    "subagent_stop",
    "pre_compact",
    "notification",
]
"""Source hook that generated the notification."""


@FrozenDataclass()
class Notification:
    """Notification record from Claude Agent SDK hooks.

    Stores the raw input data from SubagentStart, SubagentStop, PreCompact,
    and Notification hooks for debugging and observability.
    """

    __slice_meta__ = SliceMeta(
        label="SDK Notification",
        description="Hook inputs from Claude Agent SDK for debugging.",
        icon="bell",
        sort_key="created_at",
        sort_order="desc",
    )

    source: NotificationSource = field(
        metadata={
            "display": "primary",
            "label": "Hook",
            "description": "Hook that generated this notification.",
        }
    )
    payload: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "display": "secondary",
            "format": "json",
            "label": "Data",
            "description": "Raw input dict from the SDK hook.",
        },
    )
    prompt_name: str = field(
        default="",
        metadata={
            "display": "primary",
            "label": "Prompt",
            "description": "Name of the prompt being evaluated.",
        },
    )
    adapter_name: str = field(
        default="",
        metadata={
            "display": "hidden",
            "description": "Name of the adapter that received the hook.",
        },
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
        metadata={
            "display": "secondary",
            "description": "Timestamp when the notification was created.",
        },
    )
    notification_id: UUID = field(
        default_factory=uuid4,
        metadata={
            "display": "hidden",
            "description": "Unique identifier for this notification.",
        },
    )

    def render(self) -> str:
        """Render the notification for logging."""
        return (
            f"[{self.source}] {self.created_at.isoformat()} prompt={self.prompt_name}"
        )


# Register annotations at module import time
register_annotations(Notification)
