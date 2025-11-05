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

"""Session state container for agent runs."""

from ._types import ReducerEvent, TypedReducer
from .reducer_context import ReducerContext
from .reducers import append, replace_latest, upsert_by
from .selectors import select_all, select_latest, select_where
from .session import DataEvent, PromptData, Session, ToolData
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
)

__all__ = [
    "DataEvent",
    "PromptData",
    "ReducerContext",
    "ReducerEvent",
    "Session",
    "Snapshot",
    "SnapshotRestoreError",
    "SnapshotSerializationError",
    "ToolData",
    "TypedReducer",
    "append",
    "replace_latest",
    "select_all",
    "select_latest",
    "select_where",
    "upsert_by",
]
