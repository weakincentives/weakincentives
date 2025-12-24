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

from ._types import (
    ReducerContextProtocol,
    ReducerEvent,
    TypedReducer,
)
from .protocols import SessionProtocol, SessionViewProtocol, SnapshotProtocol
from .reducer_context import ReducerContext, build_reducer_context
from .reducers import (
    append_all,
    replace_latest,
    replace_latest_by,
    upsert_by,
)
from .session import DataEvent, Session, iter_sessions_bottom_up
from .session_view import SessionView, as_view
from .slice_accessor import ReadOnlySliceAccessor, SliceAccessor
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .slices import (
    Append,
    Clear,
    Extend,
    JsonlSlice,
    JsonlSliceFactory,
    JsonlSliceView,
    MemorySlice,
    MemorySliceFactory,
    MemorySliceView,
    Replace,
    Slice,
    SliceFactory,
    SliceFactoryConfig,
    SliceOp,
    SliceView,
    default_slice_config,
)
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
)
from .state_slice import (
    ReducerMeta,
    install_state_slice,
    reducer,
)
from .visibility_overrides import (
    ClearAllVisibilityOverrides,
    ClearVisibilityOverride,
    SetVisibilityOverride,
    VisibilityOverrides,
    get_session_visibility_override,
    register_visibility_reducers,
)

__all__ = [  # noqa: RUF022
    "Append",
    "Clear",
    "ClearAllVisibilityOverrides",
    "ClearVisibilityOverride",
    "DEFAULT_SNAPSHOT_POLICIES",
    "DataEvent",
    "Extend",
    "JsonlSlice",
    "JsonlSliceFactory",
    "JsonlSliceView",
    "MemorySlice",
    "MemorySliceFactory",
    "MemorySliceView",
    "ReadOnlySliceAccessor",
    "ReducerContext",
    "ReducerContextProtocol",
    "ReducerEvent",
    "ReducerMeta",
    "Replace",
    "Session",
    "SessionProtocol",
    "SessionView",
    "SessionViewProtocol",
    "SetVisibilityOverride",
    "Slice",
    "SliceAccessor",
    "SliceFactory",
    "SliceFactoryConfig",
    "SliceOp",
    "SlicePolicy",
    "SliceView",
    "Snapshot",
    "SnapshotProtocol",
    "SnapshotRestoreError",
    "SnapshotSerializationError",
    "TypedReducer",
    "VisibilityOverrides",
    "append_all",
    "as_view",
    "build_reducer_context",
    "default_slice_config",
    "get_session_visibility_override",
    "install_state_slice",
    "iter_sessions_bottom_up",
    "reducer",
    "register_visibility_reducers",
    "replace_latest",
    "replace_latest_by",
    "upsert_by",
]
