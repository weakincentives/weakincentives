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

"""Protocol definitions for weakincentives.

This package provides zero-dependency protocol definitions that can be
imported by any module without risk of circular imports.

Dependency rule: This package may only import from:
- Standard library
- weakincentives.types (pure type definitions)
"""

from __future__ import annotations

from .dispatcher import (
    ControlDispatcher,
    Dispatcher,
    DispatchResultProtocol,
    EventHandler,
    HandlerFailureProtocol,
    TelemetryDispatcher,
)
from .reducer import ReducerContextProtocol, ReducerEvent, TypedReducer
from .session import (
    SessionProtocol,
    SessionViewProtocol,
    SliceAccessorProtocol,
    SliceObserver,
    SnapshotProtocol,
    Subscription,
)
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy

__all__ = [
    "DEFAULT_SNAPSHOT_POLICIES",
    "ControlDispatcher",
    "DispatchResultProtocol",
    "Dispatcher",
    "EventHandler",
    "HandlerFailureProtocol",
    "ReducerContextProtocol",
    "ReducerEvent",
    "SessionProtocol",
    "SessionViewProtocol",
    "SliceAccessorProtocol",
    "SliceObserver",
    "SlicePolicy",
    "SnapshotProtocol",
    "Subscription",
    "TelemetryDispatcher",
    "TypedReducer",
]
