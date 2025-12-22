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

"""Core types for mailbox operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReplyState(Enum):
    """State of a reply entry."""

    PENDING = "pending"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ReplyEntry[T]:
    """Entry in a reply store tracking reply state."""

    id: str
    value: T | None
    state: ReplyState
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MessageData[T, R]:
    """Immutable message data container.

    This dataclass holds the core message data. The Message class wraps this
    with methods that interact with the mailbox.
    """

    id: str
    body: T
    receipt_handle: str
    delivery_count: int
    enqueued_at: datetime
    attributes: Mapping[str, str] = field(default_factory=lambda: {})
    # Using Any here to avoid circular import with _protocols.py
    # At runtime, this is InMemoryReplyChannel or None
    reply_channel: Any = None


__all__ = [
    "MessageData",
    "ReplyEntry",
    "ReplyState",
]
