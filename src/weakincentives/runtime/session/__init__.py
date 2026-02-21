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

"""Redux-style session state container with typed slices and reducers.

This package provides the core state management infrastructure for agent
workflows. Sessions store dataclass payloads from prompt executions and
tool invocations in typed slices, with mutations flowing through pure
reducer functions.

Architecture Overview
---------------------

The session system follows a Redux-inspired architecture:

**Slices**
    Typed collections of frozen dataclass instances. Each slice stores
    items of a single type and supports query, append, and replace operations.

**Reducers**
    Pure functions that receive the current slice state and an event,
    returning a SliceOp describing the mutation (Append, Replace, Clear).

**Events**
    Frozen dataclass instances dispatched to trigger state changes.
    Events are routed to all registered reducers for their type.

**Snapshots**
    Immutable captures of session state that can be serialized and
    restored for transaction rollback and debugging.

Basic Usage
-----------

Create a session and access state through typed indexing::

    from weakincentives.runtime.session import Session

    session = Session()

    # Query operations (returns SliceAccessor)
    plan = session[AgentPlan].latest()    # Most recent item or None
    all_plans = session[AgentPlan].all()  # All items as tuple
    active = session[AgentPlan].where(lambda p: p.active)  # Filtered iterator

    # Direct mutations (bypass reducers)
    session[AgentPlan].seed(initial_plan)  # Set if empty
    session[AgentPlan].append(new_plan)    # Add item
    session[AgentPlan].clear()             # Remove all items

    # Register reducer for event type
    session[AgentPlan].register(AddStep, add_step_reducer)

    # Dispatch events to reducers
    session.dispatch(AddStep(step="implement feature"))

Reducer Pattern
---------------

Reducers are pure functions that compute state transitions. They receive
a readonly view of the current slice, the event, and a context object::

    from weakincentives.runtime.session import SliceView, ReducerContextProtocol
    from weakincentives.runtime.session.slices import Append, Replace

    def add_step_reducer(
        view: SliceView[AgentPlan],
        event: AddStep,
        *,
        context: ReducerContextProtocol,
    ) -> Append[AgentPlan]:
        # Get current state
        current = view.latest()
        if current is None:
            return Append(AgentPlan(steps=(event.step,)))

        # Return SliceOp describing the mutation
        new_plan = replace(current, steps=(*current.steps, event.step))
        return Replace((new_plan,))

Built-in reducers are provided for common patterns:

- :func:`append_all` - Ledger semantics, always append (default)
- :func:`replace_latest` - Keep only the most recent value
- :func:`replace_latest_by` - Keep latest per derived key
- :func:`upsert_by` - Update or insert by derived key

Declarative Reducers
--------------------

For co-located reducer definitions, use the ``@reducer`` decorator::

    from dataclasses import dataclass, replace
    from weakincentives.runtime.session import reducer
    from weakincentives.runtime.session.slices import Replace

    @dataclass(frozen=True)
    class AgentPlan:
        steps: tuple[str, ...]
        current_step: int = 0

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
            new_plan = replace(self, steps=(*self.steps, event.step))
            return Replace((new_plan,))

        @reducer(on=CompleteStep)
        def complete(self, event: CompleteStep) -> Replace["AgentPlan"]:
            new_plan = replace(self, current_step=self.current_step + 1)
            return Replace((new_plan,))

    # Install registers all @reducer methods automatically
    session.install(AgentPlan)

Slice Operations
----------------

Reducers return :class:`SliceOp` values describing mutations:

- :class:`Append` - Add a single item (O(1) for file-backed slices)
- :class:`Extend` - Add multiple items
- :class:`Replace` - Replace entire slice contents
- :class:`Clear` - Remove items (optionally filtered by predicate)

Example::

    from weakincentives.runtime.session.slices import Append, Replace, Clear

    # Append single item (efficient for logs)
    return Append(new_event)

    # Replace all items
    return Replace((item1, item2, item3))

    # Clear items matching predicate
    return Clear(predicate=lambda x: x.expired)

Slice Policies
--------------

Slices have retention policies that affect snapshot behavior:

- :attr:`SlicePolicy.STATE` - Core state, included in snapshots (default)
- :attr:`SlicePolicy.LOG` - Event logs, excluded from snapshots by default
- :attr:`SlicePolicy.TRANSIENT` - Never included in snapshots

Set policy when registering reducers::

    session[Event].register(LogEvent, log_reducer, policy=SlicePolicy.LOG)

Snapshots
---------

Capture and restore session state for transaction rollback::

    from weakincentives.runtime.session import Snapshot

    # Capture current state
    snapshot = session.snapshot()

    # Make changes
    session.dispatch(SomeEvent(...))

    # Restore previous state
    session.restore(snapshot)

    # Control what's included
    snapshot = session.snapshot(
        policies=frozenset({SlicePolicy.STATE}),  # Only STATE slices
        include_all=True,  # Override policy filtering
    )

Session Hierarchy
-----------------

Sessions can form parent-child hierarchies for scoped state::

    root = Session()
    child = Session(parent=root)

    # Navigate hierarchy
    child.parent  # Returns root
    root.children  # Returns (child,)

    # Iterate bottom-up for cleanup
    for session in iter_sessions_bottom_up(root):
        session.reset()

Slice Storage Backends
----------------------

The ``slices`` subpackage provides storage backends:

- :class:`MemorySlice` - In-memory tuple storage (default)
- :class:`JsonlSlice` - File-backed JSONL storage for large datasets

Configure via SliceFactoryConfig::

    from weakincentives.runtime.session.slices import (
        MemorySliceFactory,
        JsonlSliceFactory,
        SliceFactoryConfig,
    )

    config = SliceFactoryConfig(
        state_factory=MemorySliceFactory(),
        log_factory=JsonlSliceFactory(base_dir=Path("/tmp/logs")),
    )
    session = Session(slice_config=config)

Exports
-------

**Core:**
    - :class:`Session` - Main state container
    - :class:`SessionProtocol` - Protocol for session implementations
    - :class:`SessionView` - Read-only session view
    - :class:`SessionViewProtocol` - Protocol for read-only views
    - :class:`DataEvent` - Union of prompt/tool telemetry events
    - :func:`iter_sessions_bottom_up` - Iterate session hierarchy

**Slices:**
    - :class:`Slice` - Storage backend protocol
    - :class:`SliceView` - Read-only lazy view for reducers
    - :class:`SliceFactory` - Factory protocol for creating slices
    - :class:`SliceFactoryConfig` - Configuration for slice backends
    - :class:`MemorySlice` - In-memory storage
    - :class:`JsonlSlice` - File-backed JSONL storage
    - :func:`default_slice_config` - Default configuration factory

**Slice Operations:**
    - :class:`SliceOp` - Union of all operation types
    - :class:`Append` - Add single item
    - :class:`Extend` - Add multiple items
    - :class:`Replace` - Replace all items
    - :class:`Clear` - Remove items

**Reducers:**
    - :data:`TypedReducer` - Reducer function type alias
    - :func:`append_all` - Ledger semantics
    - :func:`replace_latest` - Keep most recent
    - :func:`replace_latest_by` - Keep latest per key
    - :func:`upsert_by` - Update or insert by key
    - :func:`reducer` - Decorator for declarative reducers
    - :func:`install_state_slice` - Install decorated slice class
    - :class:`ReducerMeta` - Metadata for reducer methods

**Context:**
    - :class:`ReducerContext` - Context passed to reducers
    - :class:`ReducerContextProtocol` - Protocol for reducer context
    - :func:`build_reducer_context` - Create reducer context

**Snapshots:**
    - :class:`Snapshot` - Immutable state capture
    - :class:`SnapshotProtocol` - Protocol for snapshots
    - :exc:`SnapshotRestoreError` - Restore operation failed
    - :exc:`SnapshotSerializationError` - Serialization failed

**Policies:**
    - :class:`SlicePolicy` - STATE, LOG, or TRANSIENT
    - :data:`DEFAULT_SNAPSHOT_POLICIES` - Policies included by default

**Accessors:**
    - :class:`SliceAccessor` - Full slice access (query + mutation)
    - :class:`ReadOnlySliceAccessor` - Query-only slice access

**Visibility Overrides:**
    - :class:`VisibilityOverrides` - Section visibility state
    - :class:`SetVisibilityOverride` - Set visibility event
    - :class:`ClearVisibilityOverride` - Clear visibility event
    - :class:`ClearAllVisibilityOverrides` - Clear all event
    - :func:`get_session_visibility_override` - Query visibility
    - :func:`register_visibility_reducers` - Register built-in reducers

**Tools:**
    - :class:`RenderedTools` - Tool schema collection
    - :class:`ToolSchema` - Individual tool schema

**Mutations:**
    - :class:`InitializeSlice` - System event to initialize slice
    - :class:`ClearSlice` - System event to clear slice
"""

from ._session_helpers import DataEvent, iter_sessions_bottom_up
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
from .rendered_tools import (
    RenderedTools,
    ToolSchema,
)
from .session import Session
from .session_view import SessionView, as_view
from .slice_accessor import ReadOnlySliceAccessor, SliceAccessor
from .slice_mutations import ClearSlice, InitializeSlice
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
    "ClearSlice",
    "ClearVisibilityOverride",
    "DEFAULT_SNAPSHOT_POLICIES",
    "DataEvent",
    "Extend",
    "InitializeSlice",
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
    "RenderedTools",
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
    "ToolSchema",
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
