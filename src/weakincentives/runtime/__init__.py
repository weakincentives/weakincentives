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

"""Runtime primitives for coordinating sessions and events."""

# pyright: reportImportCycles=false
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import *  # noqa: F403

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
    "get_logger",
    "replace_latest",
    "select_all",
    "select_latest",
    "select_where",
    "upsert_by",
]

api: object | None = None


def __getattr__(name: str) -> object:
    module = globals().get("api")
    if module is None:
        module = import_module(f"{__name__}.api")
        globals()["api"] = module
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
