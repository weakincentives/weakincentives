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
from types import MappingProxyType
from typing import Any, Final, cast, override
from uuid import UUID, uuid4

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


type DataEvent = PromptExecuted | PromptRendered | ToolInvoked


SESSION_ID_BYTE_LENGTH: Final[int] = 16


def iter_sessions_bottom_up(root: Session) -> Iterator[Session]:
    """Yield sessions from the leaves up to the provided root session."""

    visited: set[Session] = set()

    def _walk(node: Session) -> Iterator[Session]:
        if node in visited:
            return
        visited.add(node)
        children = node.children
        if children:
            for child in children:
                yield from _walk(child)
        yield node

    yield from _walk(root)


# Type casts for telemetry event types used in slice policy initialization
_PROMPT_RENDERED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptRendered
)
_TOOL_INVOKED_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolInvoked)
_PROMPT_EXECUTED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecuted
)
_RENDERED_TOOLS_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], RenderedTools
)


def _session_id_is_well_formed(session: Session) -> bool:
    return len(session.session_id.bytes) == SESSION_ID_BYTE_LENGTH


def _created_at_has_tz(session: Session) -> bool:
    return session.created_at.tzinfo is not None


def _created_at_is_utc(session: Session) -> bool:
    return session.created_at.tzinfo == UTC


def _normalize_tags(
    tags: Mapping[object, object] | None,
    *,
    session_id: UUID,
    parent: Session | None,
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}

    if tags is not None:
        for key, value in tags.items():
            if not isinstance(key, str) or not isinstance(value, str):
                msg = "Session tags must be string key/value pairs."
                raise TypeError(msg)
            normalized[key] = value

    normalized["session_id"] = str(session_id)
    if parent is not None:
        _ = normalized.setdefault("parent_session_id", str(parent.session_id))

    return MappingProxyType(normalized)


@invariant(
    _session_id_is_well_formed,
    _created_at_has_tz,
    _created_at_is_utc,
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
            created_at if created_at is not None else datetime.now(UTC)
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
        self._tags = _normalize_tags(tags, session_id=self.session_id, parent=parent)
        self._subscriptions_attached = False

        # Initialize subsystems
        initial_policies: dict[SessionSliceType, SlicePolicy] = {
            _PROMPT_RENDERED_TYPE: SlicePolicy.LOG,
            _PROMPT_EXECUTED_TYPE: SlicePolicy.LOG,
            _TOOL_INVOKED_TYPE: SlicePolicy.LOG,
            _RENDERED_TOOLS_TYPE: SlicePolicy.LOG,
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
        """Access a slice for querying and mutation operations.

        This is the primary API for working with session state. All slice
        operations are available through the returned accessor.

        Usage::

            # Query operations
            session[Plan].latest()
            session[Plan].all()
            session[Plan].where(lambda p: p.active)

            # Direct mutations (bypass reducers)
            session[Plan].seed(initial_plan)
            session[Plan].clear()
            session[Plan].append(new_plan)

            # Reducer registration
            session[Plan].register(AddStep, add_step_reducer)

        For dispatch (routes to all reducers for the event type)::

            session.dispatch(AddStep(step="x"))

        """
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
        on the slice class. The ``@state_slice`` decorator is optional.

        Args:
            slice_type: A frozen dataclass with ``@reducer`` decorated methods.
            initial: Optional factory function to create initial state when empty.

        Raises:
            TypeError: If the class is not a frozen dataclass.
            ValueError: If no @reducer methods are found.

        Example::

            @dataclass(frozen=True)
            class AgentPlan:
                steps: tuple[str, ...]

                @reducer(on=AddStep)
                def add_step(self, event: AddStep) -> "AgentPlan":
                    return replace(self, steps=(*self.steps, event.step))

            session.install(AgentPlan)
            session[AgentPlan].latest()

        """
        # Lazy import to avoid import cycle
        from .state_slice import install_state_slice

        install_state_slice(self, slice_type, initial=initial)

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch API
    # ──────────────────────────────────────────────────────────────────────

    @override
    def dispatch(self, event: SupportsDataclass) -> DispatchResult:
        """Dispatch an event to all reducers registered for its type.

        This routes by event type and runs all registrations for that type,
        regardless of which slice they target. Use this for cross-cutting
        events that affect multiple slices.

        Args:
            event: The event to dispatch. All reducers registered for
                ``type(event)`` will be executed.

        Returns:
            DispatchResult containing dispatch outcome and any handler errors.

        Example::

            # Dispatches to ALL reducers registered for AddStep
            result = session.dispatch(AddStep(step="implement feature"))

        """
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
        """Clear all stored slices while preserving reducer registrations.

        This is useful for resetting session state between operations while
        keeping the same reducer configuration.

        Example::

            session.reset()  # All slices are now empty

        """
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
        """Restore session slices from the provided snapshot.

        All slice types in the snapshot must be registered in the session.

        Args:
            snapshot: A snapshot previously captured via ``session.snapshot()``.
            preserve_logs: If True, slices marked as LOG are not modified.

        Raises:
            SnapshotRestoreError: If the snapshot contains unregistered slice types.

        Example::

            snapshot = session.snapshot()
            # ... operations that modify state ...
            session.restore(snapshot)  # Restore previous state

        """
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
        """Capture an immutable snapshot of the current session state.

        Args:
            tag: Optional label for the snapshot (unused, reserved for future use).
            policies: Slice policies to include when include_all is False.
            include_all: If True, snapshot all slices regardless of policy.
        """
        del tag
        with self._lock:
            # Capture all state under lock to prevent concurrent modifications
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
        """Handle system mutation events (InitializeSlice, ClearSlice).

        These events bypass normal reducer dispatch and directly mutate state,
        ensuring consistent behavior regardless of registered reducers.

        Returns:
            True if the event was a system mutation event and was handled,
            False otherwise.
        """
        if isinstance(event, InitializeSlice):
            # Use cast to work around generic type parameter inference
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
            # Use cast to work around generic type parameter inference
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
        # Handle system mutation events specially
        if self._handle_system_mutation_event(event):
            return

        from .reducer_context import build_reducer_context
        from .session_view import SessionView

        with self.locked():
            registrations = list(self._registry.get_registrations(data_type))

            if not registrations:
                # Default: ledger semantics (always append)
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
                op_type = type(op).__name__
                logger.debug(
                    "session.reducer_applied",
                    event="session.reducer_applied",
                    context={
                        "session_id": str(self.session_id),
                        "reducer": reducer_name,
                        "slice_type": slice_type.__qualname__,
                        "operation": op_type,
                    },
                )
                # Apply the slice operation
                with self.locked():
                    apply_slice_op(op, slice_instance)
            except Exception:  # log and continue
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
        """Register built-in reducers for prompt visibility overrides.

        Called once during Session initialization. Safe to call multiple times
        as it guards against re-registration.
        """
        from .visibility_overrides import (
            SetVisibilityOverride,
            register_visibility_reducers,
        )

        # Guard against re-registration (e.g., during clone)
        with self.locked():
            if self._registry.has_registrations(SetVisibilityOverride):
                return

        register_visibility_reducers(self)

    # ──────────────────────────────────────────────────────────────────────
    # Backward Compatibility Attributes (Internal Use Only)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def _slices(self) -> dict[SessionSliceType, Slice[Any]]:
        """Access internal slices dict for backward compatibility.

        WARNING: Caller MUST hold Session's lock (via locked() context manager).
        Returns a direct mutable reference - not a defensive copy.

        This property is for internal use by session_cloning module only.
        """
        return self._store._slices  # pyright: ignore[reportPrivateUsage]

    @property
    def _slice_policies(self) -> dict[SessionSliceType, SlicePolicy]:
        """Access internal policies dict for backward compatibility.

        WARNING: Caller MUST hold Session's lock (via locked() context manager).
        Returns a direct mutable reference - not a defensive copy.

        This property is for internal use by session_cloning module only.
        """
        return self._store._slice_policies  # pyright: ignore[reportPrivateUsage]

    @_slice_policies.setter
    def _slice_policies(self, value: dict[SessionSliceType, SlicePolicy]) -> None:
        """Set internal policies dict for backward compatibility.

        WARNING: Caller MUST hold Session's lock (via locked() context manager).
        """
        self._store._slice_policies = value  # pyright: ignore[reportPrivateUsage]

    @property
    def _reducers(self) -> dict[SessionSliceType, list[ReducerRegistration]]:
        """Access internal reducers dict for backward compatibility.

        WARNING: Caller MUST hold Session's lock (via locked() context manager).
        Returns a direct mutable reference - not a defensive copy.

        This property is for internal use by session_cloning module only.
        """
        return self._registry._reducers  # pyright: ignore[reportPrivateUsage]

    @property
    def _slice_config(self) -> SliceFactoryConfig:
        """Access internal slice config for backward compatibility."""
        return self._store.config


__all__ = [
    "DataEvent",
    "Session",
    "SliceAccessor",
    "TypedReducer",
]
