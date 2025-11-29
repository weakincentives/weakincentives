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

"""Runtime namespace exposing the :mod:`weakincentives.runtime.api` surface."""

# pyright: reportImportCycles=false

from __future__ import annotations

from . import api
from .api import (
    DataEvent,
    EventBus,
    HandlerFailure,
    InProcessEventBus,
    PromptExecuted,
    PublishResult,
    ReducerContext,
    ReducerContextProtocol,
    ReducerEvent,
    ReducerEventWithValue,
    Session,
    SessionProtocol,
    Snapshot,
    SnapshotProtocol,
    SnapshotRestoreError,
    SnapshotSerializationError,
    StructuredLogger,
    ToolInvoked,
    TypedReducer,
    append,
    build_reducer_context,
    configure_logging,
    events,
    get_logger,
    replace_latest,
    select_all,
    select_latest,
    select_where,
    session,
    upsert_by,
)

__all__ = [
    "DataEvent",
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "PromptExecuted",
    "PublishResult",
    "ReducerContext",
    "ReducerContextProtocol",
    "ReducerEvent",
    "ReducerEventWithValue",
    "Session",
    "SessionProtocol",
    "Snapshot",
    "SnapshotProtocol",
    "SnapshotRestoreError",
    "SnapshotSerializationError",
    "StructuredLogger",
    "ToolInvoked",
    "TypedReducer",
    "api",
    "append",
    "build_reducer_context",
    "configure_logging",
    "events",
    "get_logger",
    "replace_latest",
    "select_all",
    "select_latest",
    "select_where",
    "session",
    "upsert_by",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
