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

from importlib import import_module
from typing import TYPE_CHECKING

from .logging import StructuredLogger, configure_logging, get_logger

if TYPE_CHECKING:
    from . import events, session
    from .events import (
        EventBus,
        HandlerFailure,
        InProcessEventBus,
        PromptExecuted,
        PromptRendered,
        PublishResult,
        ToolInvoked,
    )
    from .session import (
        DataEvent,
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
        replace_latest,
        select_all,
        select_latest,
        select_where,
        upsert_by,
    )

__all__ = [
    "DataEvent",
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "PromptExecuted",
    "PromptRendered",
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

_MODULE_EXPORTS: dict[str, tuple[str, str | None]] = {
    "DataEvent": ("weakincentives.runtime.session", "DataEvent"),
    "EventBus": ("weakincentives.runtime.events", "EventBus"),
    "HandlerFailure": ("weakincentives.runtime.events", "HandlerFailure"),
    "InProcessEventBus": ("weakincentives.runtime.events", "InProcessEventBus"),
    "PromptExecuted": ("weakincentives.runtime.events", "PromptExecuted"),
    "PromptRendered": ("weakincentives.runtime.events", "PromptRendered"),
    "PublishResult": ("weakincentives.runtime.events", "PublishResult"),
    "ReducerContext": ("weakincentives.runtime.session", "ReducerContext"),
    "ReducerContextProtocol": (
        "weakincentives.runtime.session",
        "ReducerContextProtocol",
    ),
    "ReducerEvent": ("weakincentives.runtime.session", "ReducerEvent"),
    "ReducerEventWithValue": (
        "weakincentives.runtime.session",
        "ReducerEventWithValue",
    ),
    "Session": ("weakincentives.runtime.session", "Session"),
    "SessionProtocol": ("weakincentives.runtime.session", "SessionProtocol"),
    "Snapshot": ("weakincentives.runtime.session", "Snapshot"),
    "SnapshotProtocol": ("weakincentives.runtime.session", "SnapshotProtocol"),
    "SnapshotRestoreError": (
        "weakincentives.runtime.session",
        "SnapshotRestoreError",
    ),
    "SnapshotSerializationError": (
        "weakincentives.runtime.session",
        "SnapshotSerializationError",
    ),
    "ToolInvoked": ("weakincentives.runtime.events", "ToolInvoked"),
    "TypedReducer": ("weakincentives.runtime.session", "TypedReducer"),
    "append": ("weakincentives.runtime.session", "append"),
    "build_reducer_context": (
        "weakincentives.runtime.session",
        "build_reducer_context",
    ),
    "events": ("weakincentives.runtime.events", None),
    "replace_latest": ("weakincentives.runtime.session", "replace_latest"),
    "select_all": ("weakincentives.runtime.session", "select_all"),
    "select_latest": ("weakincentives.runtime.session", "select_latest"),
    "select_where": ("weakincentives.runtime.session", "select_where"),
    "session": ("weakincentives.runtime.session", None),
    "upsert_by": ("weakincentives.runtime.session", "upsert_by"),
}


def __getattr__(name: str) -> object:
    try:
        module_path, attr_name = _MODULE_EXPORTS[name]
    except KeyError as error:  # pragma: no cover - convenience shim
        raise AttributeError(name) from error
    module = import_module(module_path)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
