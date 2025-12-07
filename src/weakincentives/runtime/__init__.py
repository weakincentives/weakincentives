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

"""Runtime primitives for :mod:`weakincentives`."""

from __future__ import annotations

from . import events, main_loop, session
from .events import (
    EventBus,
    HandlerFailure,
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    PublishResult,
    TokenUsage,
    ToolInvoked,
)
from .logging import StructuredLogger, configure_logging, get_logger
from .main_loop import (
    MainLoop,
    MainLoopCompleted,
    MainLoopConfig,
    MainLoopFailed,
    MainLoopRequest,
)
from .session import (
    DataEvent,
    QueryBuilder,
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
    TypedReducer,
    append,
    build_reducer_context,
    iter_sessions_bottom_up,
    replace_latest,
    replace_latest_by,
    upsert_by,
)

__all__ = [
    "DataEvent",
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "MainLoop",
    "MainLoopCompleted",
    "MainLoopConfig",
    "MainLoopFailed",
    "MainLoopRequest",
    "PromptExecuted",
    "PromptRendered",
    "PublishResult",
    "QueryBuilder",
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
    "TokenUsage",
    "ToolInvoked",
    "TypedReducer",
    "append",
    "build_reducer_context",
    "configure_logging",
    "events",
    "get_logger",
    "iter_sessions_bottom_up",
    "main_loop",
    "replace_latest",
    "replace_latest_by",
    "session",
    "upsert_by",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
