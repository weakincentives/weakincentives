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

"""Event types for optimization observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True, frozen=True)
class OptimizationStarted:
    """Event emitted when an optimizer begins work.

    Published through the ``OptimizationContext.dispatcher`` when
    ``optimize()`` is invoked.
    """

    optimizer_type: str
    """The class name or identifier of the optimizer."""

    prompt_ns: str
    """Namespace of the prompt being optimized."""

    prompt_key: str
    """Key of the prompt being optimized."""

    prompt_name: str | None
    """Optional name of the prompt."""

    session_id: UUID | None
    """Session ID from the caller's session, if available."""

    created_at: datetime
    """Timestamp when optimization started."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this event."""


@dataclass(slots=True, frozen=True)
class OptimizationCompleted:
    """Event emitted on successful optimization completion.

    Published through the ``OptimizationContext.dispatcher`` after
    ``optimize()`` returns successfully.
    """

    optimizer_type: str
    """The class name or identifier of the optimizer."""

    prompt_ns: str
    """Namespace of the prompt that was optimized."""

    prompt_key: str
    """Key of the prompt that was optimized."""

    prompt_name: str | None
    """Optional name of the prompt."""

    session_id: UUID | None
    """Session ID from the caller's session, if available."""

    created_at: datetime
    """Timestamp when optimization completed."""

    result_summary: dict[str, Any]
    """Algorithm-specific summary of the result."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this event."""


@dataclass(slots=True, frozen=True)
class OptimizationFailed:
    """Event emitted when optimization raises an exception.

    Published through the ``OptimizationContext.dispatcher`` when
    ``optimize()`` encounters an error.
    """

    optimizer_type: str
    """The class name or identifier of the optimizer."""

    prompt_ns: str
    """Namespace of the prompt being optimized."""

    prompt_key: str
    """Key of the prompt being optimized."""

    prompt_name: str | None
    """Optional name of the prompt."""

    session_id: UUID | None
    """Session ID from the caller's session, if available."""

    created_at: datetime
    """Timestamp when the failure occurred."""

    error_type: str
    """The type name of the exception that was raised."""

    error_message: str
    """The exception message."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this event."""


__all__ = [
    "OptimizationCompleted",
    "OptimizationFailed",
    "OptimizationStarted",
]
