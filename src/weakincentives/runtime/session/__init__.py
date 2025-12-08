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

from ._mutation_types import MutationProvider
from ._types import (
    ReducerContextProtocol,
    ReducerEvent,
    ReducerEventWithValue,
    SimpleReducer,
    TypedReducer,
)
from .mutation import GlobalMutationBuilder, MutationBuilder
from .protocols import SessionProtocol, SnapshotProtocol
from .query import QueryBuilder
from .reducer_context import ReducerContext, build_reducer_context
from .reducers import (
    append,
    as_typed_reducer,
    replace_latest,
    replace_latest_by,
    upsert_by,
)
from .session import DataEvent, Session, iter_sessions_bottom_up
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
)

__all__ = [
    "DataEvent",
    "GlobalMutationBuilder",
    "MutationBuilder",
    "MutationProvider",
    "QueryBuilder",
    "ReducerContext",
    "ReducerContextProtocol",
    "ReducerEvent",
    "ReducerEventWithValue",
    "Session",
    "SessionProtocol",
    "SimpleReducer",
    "Snapshot",
    "SnapshotProtocol",
    "SnapshotRestoreError",
    "SnapshotSerializationError",
    "TypedReducer",
    "append",
    "as_typed_reducer",
    "build_reducer_context",
    "iter_sessions_bottom_up",
    "replace_latest",
    "replace_latest_by",
    "upsert_by",
]
