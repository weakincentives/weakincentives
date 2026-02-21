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

# pyright: reportImportCycles=false

"""Session state container synchronized with the event dispatcher.

Session is a thin facade coordinating specialized subsystems:
- SliceStore: Slice storage with policy-based factories
- ReducerRegistry: Event-to-reducer routing
- SessionSnapshotter: Snapshot/restore functionality

Thread safety is provided by Session's lock. Subsystems are not
thread-safe on their own and must only be accessed while holding
the lock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import RLock
from typing import Any, cast, override
from uuid import UUID, uuid4

from ...clock import SYSTEM_CLOCK
from ...dbc import invariant
from ...types.dataclass import SupportsDataclass
from ..events import (
    DispatchResult,
    PromptExecuted,
    PromptRendered,
    TelemetryDispatcher,
    ToolInvoked,
)
from ..logging import StructuredLogger, get_logger
from ._session_helpers import (
    PROMPT_EXECUTED_TYPE,
    PROMPT_RENDERED_TYPE,
    RENDERED_TOOLS_TYPE,
    TOOL_INVOKED_TYPE,
    created_at_has_tz,
    created_at_is_utc,
    normalize_tags,
    session_id_is_well_formed,
)
from ._slice_types import SessionSliceType
from ._types import ReducerEvent, TypedReducer
from .protocols import SessionProtocol, SnapshotProtocol
from .reducer_registry import ReducerRegistration, ReducerRegistry
from .reducers import append_all
from .rendered_tools import RenderedTools
from .session_cloning import (
    apply_policies_to_clone,
    apply_state_to_clone,
    copy_reducers_to_clone,
    resolve_clone_created,
    resolve_clone_id,
    resolve_clone_parent,
    resolve_clone_tags,
    snapshot_reducers_and_state,
)
from .session_dispatch import apply_slice_op
from .session_snapshotter import SessionSnapshotter
from .session_telemetry import (
    handle_prompt_executed,
    handle_prompt_rendered,
    handle_rendered_tools,
    handle_tool_invoked,
)
from .slice_accessor import SliceAccessor
from .slice_mutations import ClearSlice, InitializeSlice
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .slice_store import SliceStore
from .slices import (
    Slice,
    SliceFactoryConfig,
)

logger: StructuredLogger = get_logger(__name__, context={"component": "session"})


@invariant(
    session_id_is_well_formed,
    created_at_has_tz,
    created_at_is_utc,
)
class Session(SessionProtocol):
    """Collect dataclass payloads from prompt executions and tool invocations.

    Session provides a Redux-style state container with typed slices, reducers,
    and event-driven dispatch. Access slices via indexing::

        # Query operations
        session[Plan].latest()
        session[Plan].all()
        session[Plan].where(lambda p: p.active)

        # Direct mutations
        session[Plan].seed(initial_plan)
        session[Plan].clear()
        session[Plan].register(AddStep, reducer)

    For dispatch (routes to all reducers for the event type)::

        session.dispatch(AddStep(step="x"))

    Global operations are available directly on the session::

        session.reset()                  # Clear all slices
        session.restore(snapshot)        # Restore from snapshot

    Architecture
    ------------

    Session is a thin facade coordinating three specialized subsystems:

    - **SliceStore**: Thread-safe slice storage with policy-based factories.
      Manages slice creation, access, and policy configuration.

    - **ReducerRegistry**: Event-to-reducer routing. Tracks which reducers
      should be invoked for each event type.

    - **SessionSnapshotter**: Snapshot/restore functionality. Captures and
      restores immutable state snapshots for transaction rollback.

    """

    def __init__(
        self,
        *,
        dispatcher: TelemetryDispatcher | None = None,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        tags: Mapping[object, object] | None = None,
        slice_config: SliceFactoryConfig | None = None,
    ) -> None:
        super().__init__()
        resolved_session_id = session_id if session_id is not None else uuid4()
        resolved_created_at = (
            created_at if created_at is not None else SYSTEM_CLOCK.utcnow()
        )
        if resolved_created_at.tzinfo is None:
            msg = "Session created_at must be timezone-aware."
            raise ValueError(msg)

        self.session_id: UUID = resolved_session_id
        self.created_at: datetime = resolved_created_at.astimezone(UTC)

        if dispatcher is None:
            from ..events import InProcessDispatcher

            self._dispatcher: TelemetryDispatcher = InProcessDispatcher()
        else:
            self._dispatcher = dispatcher

        self._lock = RLock()
        self._parent = parent
        self._children: list[Session] = []
        self._tags = normalize_tags(tags, session_id=self.session_id, parent=parent)
        self._subscriptions_attached = False

        # Initialize subsystems
        initial_policies: dict[SessionSliceType, SlicePolicy] = {
            PROMPT_RENDERED_TYPE: SlicePolicy.LOG,
            PROMPT_EXECUTED_TYPE: SlicePolicy.LOG,
            TOOL_INVOKED_TYPE: SlicePolicy.LOG,
            RENDERED_TOOLS_TYPE: SlicePolicy.LOG,
        }
        self._store = SliceStore(slice_config, initial_policies=initial_policies)
        self._registry = ReducerRegistry()
        self._snapshotter = SessionSnapshotter(
            store=self._store,
            registry=self._registry,
            lock=self._lock,
        )

        if parent is self:
            msg = "Session cannot be its own parent."
            raise ValueError(msg)
        if parent is not None:
            parent._register_child(self)
        self._attach_to_dispatcher(self._dispatcher)
        self._register_builtin_reducers()

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    # ──────────────────────────────────────────────────────────────────────
    # Slice Access (delegated to SliceStore)
    # ──────────────────────────────────────────────────────────────────────

    def _get_or_create_slice[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> Slice[T]:
        """Get existing slice or create one using the appropriate factory.

        Caller must hold Session's lock.
        """
        return self._store.get_or_create(slice_type)

    def _select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type.

        Internal method used by SliceAccessor. Use ``session[SliceType].all()``
        for public access.
        """
        with self._lock:
            return self._store.select_all(slice_type)

    @override
    def __getitem__[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> SliceAccessor[S]:
        """Access a slice for querying and mutation operations."""
        return SliceAccessor(self, slice_type)

    @override
    def install[S: SupportsDataclass](
        self,
        slice_type: type[S],
        *,
        initial: Callable[[], S] | None = None,
    ) -> None:
        """Install a declarative state slice.

        Auto-registers all reducers defined with ``@reducer`` decorators
        on the slice class.
        """
        # Lazy import to avoid import cycle
        from .state_slice import install_state_slice

        install_state_slice(self, slice_type, initial=initial)

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch API
    # ──────────────────────────────────────────────────────────────────────

    @override
    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type."""
        event_type = type(event)
        logger.debug(
            "session.dispatch",
            event="session.dispatch",
            context={
                "session_id": str(self.session_id),
                "event_type": event_type.__qualname__,
            },
        )
        self._dispatch_data_event(event_type, event)
        return self._dispatcher.dispatch(event)

    # ──────────────────────────────────────────────────────────────────────
    # Global Mutation Operations
    # ──────────────────────────────────────────────────────────────────────

    @override
    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations."""
        with self._lock:
            slice_types = self._store.all_slice_types()
            slice_types.update(self._registry.all_target_slice_types())
            logger.debug(
                "session.reset",
                event="session.reset",
                context={
                    "session_id": str(self.session_id),
                    "slice_count": len(slice_types),
                    "slice_types": [st.__qualname__ for st in slice_types],
                },
            )
            self._store.clear_all(slice_types)

    @override
    def restore(
        self, snapshot: SnapshotProtocol, *, preserve_logs: bool = True
    ) -> None:
        """Restore session slices from the provided snapshot."""
        logger.debug(
            "session.restore",
            event="session.restore",
            context={
                "session_id": str(self.session_id),
                "preserve_logs": preserve_logs,
                "snapshot_slice_count": len(snapshot.slices),
            },
        )
        self._snapshotter.restore(snapshot, preserve_logs=preserve_logs)

    # ──────────────────────────────────────────────────────────────────────
    # Private Mutation Methods (used by SliceAccessor)
    # ──────────────────────────────────────────────────────────────────────

    def _mutation_register_reducer[S: SupportsDataclass](
        self,
        data_type: SessionSliceType,
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
        policy: SlicePolicy | None = None,
    ) -> None:
        """Register a reducer for the provided data type."""
        target_slice_type: SessionSliceType = (
            data_type if slice_type is None else slice_type
        )
        reducer_name = getattr(reducer, "__qualname__", repr(reducer))
        logger.debug(
            "session.register_reducer",
            event="session.register_reducer",
            context={
                "session_id": str(self.session_id),
                "data_type": data_type.__qualname__,
                "slice_type": target_slice_type.__qualname__,
                "reducer": reducer_name,
                "policy": policy.name if policy is not None else None,
            },
        )
        with self._lock:
            self._registry.register(
                data_type,
                reducer,
                target_slice=cast(type[S], target_slice_type),
            )
            # Ensure slice exists
            _ = self._store.get_or_create(target_slice_type)
            self._store.ensure_policy(target_slice_type, policy)

    def _mutation_dispatch_event(
        self, slice_type: SessionSliceType, event: SupportsDataclass
    ) -> None:
        """Dispatch an event to be processed by registered reducers."""
        self._dispatch_data_event(slice_type, event)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    @override
    def dispatcher(self) -> TelemetryDispatcher:
        """Return the dispatcher backing this session."""
        return self._dispatcher

    @property
    @override
    def parent(self) -> Session | None:
        """Return the parent session if one was provided."""
        return self._parent

    @property
    @override
    def tags(self) -> Mapping[str, str]:
        """Return immutable tags associated with this session."""
        return self._tags

    @property
    @override
    def children(self) -> tuple[Session, ...]:
        """Return direct child sessions in registration order."""
        with self.locked():
            return tuple(self._children)

    @override
    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state."""
        del tag
        with self._lock:
            parent_id = self._parent.session_id if self._parent is not None else None
            children_ids = tuple(child.session_id for child in self._children)
            tags_snapshot = self.tags
            return self._snapshotter.create_snapshot(
                parent_id=parent_id,
                children_ids=children_ids,
                tags=tags_snapshot,
                policies=policies,
                include_all=include_all,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Clone Support (uses session_cloning module)
    # ──────────────────────────────────────────────────────────────────────

    def clone(
        self,
        *,
        dispatcher: TelemetryDispatcher,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        tags: Mapping[object, object] | None = None,
        slice_config: SliceFactoryConfig | None = None,
    ) -> Session:
        """Return a new session that mirrors the current state and reducers."""
        reducer_snapshot, state_snapshot, policy_snapshot = snapshot_reducers_and_state(
            self
        )
        clone = Session(
            dispatcher=dispatcher,
            parent=resolve_clone_parent(self, parent),
            session_id=resolve_clone_id(self, session_id),
            created_at=resolve_clone_created(self, created_at),
            tags=resolve_clone_tags(self, tags),
            slice_config=slice_config
            if slice_config is not None
            else self._store.config,
        )
        copy_reducers_to_clone(clone, reducer_snapshot)
        apply_state_to_clone(clone, state_snapshot)
        apply_policies_to_clone(clone, policy_snapshot)
        return clone

    # ──────────────────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────────────────

    def _register_child(self, child: Session) -> None:
        with self.locked():
            for registered in self._children:
                if registered is child:
                    return
            self._children.append(child)

    def _handle_system_mutation_event(self, event: ReducerEvent) -> bool:
        """Handle system mutation events (InitializeSlice, ClearSlice)."""
        if isinstance(event, InitializeSlice):
            init_event = cast("InitializeSlice[Any]", event)
            slice_type: SessionSliceType = init_event.slice_type
            values = init_event.values
            logger.debug(
                "session.initialize_slice",
                event="session.initialize_slice",
                context={
                    "session_id": str(self.session_id),
                    "slice_type": slice_type.__qualname__,
                    "value_count": len(values),
                },
            )
            with self.locked():
                slice_instance = self._store.get_or_create(slice_type)
                slice_instance.replace(values)
            return True

        if isinstance(event, ClearSlice):
            clear_event = cast("ClearSlice[Any]", event)
            slice_type = clear_event.slice_type
            predicate: Callable[[Any], bool] | None = clear_event.predicate
            logger.debug(
                "session.clear_slice",
                event="session.clear_slice",
                context={
                    "session_id": str(self.session_id),
                    "slice_type": slice_type.__qualname__,
                    "has_predicate": predicate is not None,
                },
            )
            with self.locked():
                slice_instance = self._store.get_or_create(slice_type)
                slice_instance.clear(predicate)
            return True

        return False

    def _dispatch_data_event(
        self,
        data_type: SessionSliceType,
        event: ReducerEvent,
    ) -> None:
        """Dispatch a data event to registered reducers."""
        if self._handle_system_mutation_event(event):
            return

        from .reducer_context import build_reducer_context
        from .session_view import SessionView

        with self.locked():
            registrations = list(self._registry.get_registrations(data_type))

            if not registrations:
                registrations = [
                    ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append_all),
                        slice_type=data_type,
                    )
                ]

        logger.debug(
            "session.dispatch_data_event",
            event="session.dispatch_data_event",
            context={
                "session_id": str(self.session_id),
                "data_type": data_type.__qualname__,
                "reducer_count": len(registrations),
            },
        )

        view = SessionView(self)
        context = build_reducer_context(session=view)

        for registration in registrations:
            slice_type = registration.slice_type
            reducer_name = getattr(
                registration.reducer, "__qualname__", repr(registration.reducer)
            )
            with self.locked():
                slice_instance = self._store.get_or_create(slice_type)
                slice_view = slice_instance.view()
            try:
                op = registration.reducer(slice_view, event, context=context)
                with self.locked():
                    apply_slice_op(op, slice_instance)
            except Exception:
                logger.exception(
                    "Reducer application failed.",
                    event="session_reducer_failed",
                    context={
                        "reducer": reducer_name,
                        "data_type": data_type.__qualname__,
                        "slice_type": slice_type.__qualname__,
                    },
                )
                continue

    def _on_tool_invoked(self, event: object) -> None:
        tool_event = cast(ToolInvoked, event)
        handle_tool_invoked(self, tool_event)

    def _on_prompt_executed(self, event: object) -> None:
        prompt_event = cast(PromptExecuted, event)
        handle_prompt_executed(self, prompt_event)

    def _on_prompt_rendered(self, event: object) -> None:
        start_event = cast(PromptRendered, event)
        handle_prompt_rendered(self, start_event)

    def _on_rendered_tools(self, event: object) -> None:
        tools_event = cast(RenderedTools, event)
        handle_rendered_tools(self, tools_event)

    def _attach_to_dispatcher(self, dispatcher: TelemetryDispatcher) -> None:
        with self.locked():
            if self._subscriptions_attached and self._dispatcher is dispatcher:
                return
            self._dispatcher = dispatcher
            self._subscriptions_attached = True
            dispatcher.subscribe(ToolInvoked, self._on_tool_invoked)
            dispatcher.subscribe(PromptExecuted, self._on_prompt_executed)
            dispatcher.subscribe(PromptRendered, self._on_prompt_rendered)
            dispatcher.subscribe(RenderedTools, self._on_rendered_tools)

    def _register_builtin_reducers(self) -> None:
        """Register built-in reducers for prompt visibility overrides."""
        from .visibility_overrides import (
            SetVisibilityOverride,
            register_visibility_reducers,
        )

        with self.locked():
            if self._registry.has_registrations(SetVisibilityOverride):
                return

        register_visibility_reducers(self)


__all__ = [
    "Session",
    "SliceAccessor",
    "TypedReducer",
]
