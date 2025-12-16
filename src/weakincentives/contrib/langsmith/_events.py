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

"""LangSmith-specific events published to the event bus."""

from __future__ import annotations

from dataclasses import field
from datetime import datetime
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass


@FrozenDataclass()
class LangSmithTraceStarted:
    """Event emitted when a new LangSmith trace is started.

    Published at the beginning of prompt evaluation when telemetry captures
    a :class:`~weakincentives.runtime.events.PromptRendered` event.
    """

    trace_id: UUID
    session_id: UUID | None
    project: str
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)


@FrozenDataclass()
class LangSmithTraceCompleted:
    """Event emitted when a LangSmith trace is completed.

    Published after prompt evaluation completes (following
    :class:`~weakincentives.runtime.events.PromptExecuted` event).
    """

    trace_id: UUID
    run_count: int
    total_tokens: int
    trace_url: str | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)


@FrozenDataclass()
class LangSmithUploadFailed:
    """Event emitted when LangSmith upload fails.

    Published when telemetry data cannot be uploaded to LangSmith.
    These failures are logged but do not interrupt agent execution.
    """

    trace_id: UUID | None
    error: str
    retry_count: int
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)


__all__ = [
    "LangSmithTraceCompleted",
    "LangSmithTraceStarted",
    "LangSmithUploadFailed",
]
