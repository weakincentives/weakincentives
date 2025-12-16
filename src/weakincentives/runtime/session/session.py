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

"""Session state container synchronized with the event bus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from threading import RLock
from types import MappingProxyType
from typing import Any, Concatenate, Final, cast, overload, override
from uuid import UUID, uuid4

from ...dbc import invariant
from ...prompt._types import SupportsDataclass
from ..events import PromptExecuted, PromptRendered, TelemetryBus, ToolInvoked
from ..logging import StructuredLogger, get_logger
from ._observer_types import SliceObserver, Subscription
from ._slice_types import SessionSlice, SessionSliceType
from ._types import ReducerEvent, TypedReducer
from .dataclasses import is_dataclass_instance
from .mutation import GlobalMutationBuilder, MutationBuilder
from .protocols import SessionProtocol, SnapshotProtocol
from .query import QueryBuilder
from .reducers import append_all
from .slice_accessor import SliceAccessor
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    SnapshotState,
    normalize_snapshot_state,
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


def _locked_method[SessionT: "Session", **P, R](
    func: Callable[Concatenate[SessionT, P], R],
) -> Callable[Concatenate[SessionT, P], R]:
    @wraps(func)
    def wrapper(session: SessionT, *args: P.args, **kwargs: P.kwargs) -> R:
        with session.locked():
            return func(session, *args, **kwargs)

    return wrapper


_PROMPT_RENDERED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptRendered
)
_TOOL_INVOKED_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolInvoked)
_PROMPT_EXECUTED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecuted
)

EMPTY_SLICE: SessionSlice = ()


@dataclass(slots=True)
class _ReducerRegistration:
    reducer: TypedReducer[Any]
    slice_type: SessionSliceType


@dataclass(slots=True)
class _ObserverRegistration:
    observer: SliceObserver[Any]
    subscription: Subscription


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
    """Collect dataclass payloads from prompt executions and tool invocations."""

    def __init__(
        self,
        *,
        bus: TelemetryBus | None = None,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        tags: Mapping[object, object] | None = None,
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

        if bus is None:
            from ..events import InProcessEventBus

            self._bus: TelemetryBus = InProcessEventBus()
        else:
            self._bus = bus

        self._reducers: dict[SessionSliceType, list[_ReducerRegistration]] = {}
        self._observers: dict[SessionSliceType, list[_ObserverRegistration]] = {}
        self._state: dict[SessionSliceType, SessionSlice] = {}
        self._lock = RLock()
        self._parent = parent
        self._children: list[Session] = []
        self._tags = _normalize_tags(tags, session_id=self.session_id, parent=parent)
        self._subscriptions_attached = False
        if parent is self:
            msg = "Session cannot be its own parent."
            raise ValueError(msg)
        if parent is not None:
            parent._register_child(self)
        self._attach_to_bus(self._bus)
        self._register_builtin_reducers()

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def clone(
        self,
        *,
        bus: TelemetryBus,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        tags: Mapping[object, object] | None = None,
    ) -> Session:
        """Return a new session that mirrors the current state and reducers."""

        with self.locked():
            reducer_snapshot = [
                (data_type, tuple(registrations))
                for data_type, registrations in self._reducers.items()
            ]
            state_snapshot = dict(self._state)

        clone = Session(
            bus=bus,
            parent=self._parent if parent is None else parent,
            session_id=session_id if session_id is not None else self.session_id,
            created_at=created_at if created_at is not None else self.created_at,
            tags=cast(
                Mapping[object, object] | None,
                self.tags if tags is None else tags,
            ),
        )

        # Copy non-builtin reducers from original session.
        # Builtin reducers (visibility overrides) are already registered by __init__.
        for data_type, registrations in reducer_snapshot:
            if data_type in clone._reducers:
                # Skip event types already registered by __init__
                continue
            for registration in registrations:
                clone.mutation_register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )

        with clone.locked():
            clone._state = state_snapshot

        return clone

    @override
    @_locked_method
    def select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type."""

        return cast(tuple[S, ...], self._state.get(slice_type, EMPTY_SLICE))

    @override
    def query[S: SupportsDataclass](self, slice_type: type[S]) -> QueryBuilder[S]:
        """Return a query builder for fluent slice queries.

        Usage::

            session.query(Plan).latest()
            session.query(Plan).all()
            session.query(Plan).where(lambda p: p.active)

        """
        return QueryBuilder(self, slice_type)

    @overload
    def mutate[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> MutationBuilder[S]: ...

    @overload
    def mutate(self) -> GlobalMutationBuilder: ...

    @override
    def mutate[S: SupportsDataclass](
        self, slice_type: type[S] | None = None
    ) -> MutationBuilder[S] | GlobalMutationBuilder:
        """Return a mutation builder for fluent slice mutations.

        Usage::

            # Slice-specific mutations
            session.mutate(Plan).seed(initial_plan)
            session.mutate(Plan).clear()
            session.mutate(Plan).register(AddStep, add_step_reducer)

            # Session-wide mutations
            session.mutate().reset()
            session.mutate().rollback(snapshot)

        For event-driven dispatch, use the explicit APIs::

            session.apply(event)           # Broadcast to all reducers
            session[Plan].apply(event)     # Targeted to Plan slice

        """
        if slice_type is None:
            return GlobalMutationBuilder(self)
        return MutationBuilder(self, slice_type)

    @override
    def observe[S: SupportsDataclass](
        self,
        slice_type: type[S],
        observer: SliceObserver[S],
    ) -> Subscription:
        """Register an observer called when the slice changes.

        The observer receives ``(old_values, new_values)`` after each state
        update. Returns a :class:`Subscription` handle that can be used to
        unsubscribe.

        Usage::

            def on_plan_change(old: tuple[Plan, ...], new: tuple[Plan, ...]) -> None:
                print(f"Plan changed: {len(old)} -> {len(new)} items")

            subscription = session.observe(Plan, on_plan_change)

            # Later, to stop observing:
            subscription.unsubscribe()

        """
        subscription_id = uuid4()

        def unsubscribe() -> None:
            with self.locked():
                registrations = self._observers.get(slice_type, [])
                self._observers[slice_type] = [
                    reg
                    for reg in registrations
                    if reg.subscription.subscription_id != subscription_id
                ]

        subscription = Subscription(
            unsubscribe_fn=unsubscribe, subscription_id=subscription_id
        )

        registration = _ObserverRegistration(
            observer=cast(SliceObserver[Any], observer),
            subscription=subscription,
        )
        with self.locked():
            bucket = self._observers.setdefault(slice_type, [])
            bucket.append(registration)

        return subscription

    @override
    def __getitem__[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> SliceAccessor[S]:
        """Convenient access to slice for querying and targeted dispatch.

        Supports declarative state slices with natural syntax.

        Usage::

            # Query operations
            session[Plan].latest()
            session[Plan].all()
            session[Plan].where(lambda p: p.active)

            # Targeted dispatch (slice-scoped)
            session[Plan].apply(AddStep(step="x"))

        For broadcast dispatch (event-type routed)::

            session.apply(AddStep(step="x"))

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
    # Dispatch API (apply / apply_to_slice)
    # ──────────────────────────────────────────────────────────────────────

    @override
    def apply(self, event: SupportsDataclass) -> None:
        """Broadcast an event to all reducers registered for its type.

        This routes by event type and runs all registrations for that type,
        regardless of which slice they target. Use this for cross-cutting
        events that affect multiple slices.

        For targeted dispatch that only runs reducers for a specific slice::

            session[Plan].apply(AddStep(step="x"))

        Args:
            event: The event to dispatch. All reducers registered for
                ``type(event)`` will be executed.

        Example::

            # Broadcasts to ALL reducers registered for AddStep
            session.apply(AddStep(step="implement feature"))

        """
        self._dispatch_data_event(type(event), event)

    @override
    def apply_to_slice(
        self,
        slice_type: type[SupportsDataclass],
        event: SupportsDataclass,
    ) -> None:
        """Dispatch an event to reducers targeting a specific slice.

        This filters registrations by ``(event_type, slice_type)`` so only
        reducers that handle this event type AND target this slice type
        will be executed.

        This method is used internally by :class:`SliceAccessor`. Prefer
        using the indexing syntax for clarity::

            session[Plan].apply(AddStep(step="x"))

        Args:
            slice_type: The target slice type to filter reducers.
            event: The event to dispatch.

        """
        self._dispatch_data_event(type(event), event, target_slice_type=slice_type)

    # ──────────────────────────────────────────────────────────────────────
    # MutationProvider implementation (used by MutationBuilder)
    # ──────────────────────────────────────────────────────────────────────

    @_locked_method
    def mutation_seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None:
        """Initialize or replace the stored tuple for the provided type.

        This method implements :class:`MutationProvider` for use by
        :class:`MutationBuilder`.
        """
        self._state[slice_type] = tuple(values)

    @_locked_method
    def mutation_clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        This method implements :class:`MutationProvider` for use by
        :class:`MutationBuilder`.
        """
        existing = cast(tuple[S, ...], self._state.get(slice_type, EMPTY_SLICE))
        if not existing:
            return
        if predicate is None:
            self._state[slice_type] = EMPTY_SLICE
            return
        filtered = tuple(value for value in existing if not predicate(value))
        self._state[slice_type] = filtered

    @_locked_method
    def mutation_reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations.

        This method implements :class:`MutationProvider` for use by
        :class:`GlobalMutationBuilder`.
        """
        slice_types: set[SessionSliceType] = set(self._state)
        for registrations in self._reducers.values():
            for registration in registrations:
                slice_types.add(registration.slice_type)
        self._state = dict.fromkeys(slice_types, EMPTY_SLICE)

    def mutation_rollback(self, snapshot: SnapshotProtocol) -> None:
        """Restore session slices from the provided snapshot.

        This method implements :class:`MutationProvider` for use by
        :class:`GlobalMutationBuilder`.
        """
        registered_slices = self._registered_slice_types()
        missing = [
            slice_type
            for slice_type in snapshot.slices
            if slice_type not in registered_slices
        ]
        if missing:
            missing_names = ", ".join(sorted(cls.__qualname__ for cls in missing))
            msg = f"Slice types not registered: {missing_names}"
            raise SnapshotRestoreError(msg)

        with self.locked():
            new_state: dict[SessionSliceType, SessionSlice] = dict(self._state)
            for slice_type in registered_slices:
                new_state[slice_type] = snapshot.slices.get(slice_type, EMPTY_SLICE)
            self._state = new_state

    @_locked_method
    def mutation_register_reducer[S: SupportsDataclass](
        self,
        data_type: SessionSliceType,
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
    ) -> None:
        """Register a reducer for the provided data type.

        This method implements :class:`MutationProvider` for use by
        :class:`MutationBuilder`.
        """
        target_slice_type: SessionSliceType = (
            data_type if slice_type is None else slice_type
        )
        registration = _ReducerRegistration(
            reducer=cast(TypedReducer[Any], reducer),
            slice_type=target_slice_type,
        )
        bucket = self._reducers.setdefault(data_type, [])
        bucket.append(registration)
        _ = self._state.setdefault(target_slice_type, EMPTY_SLICE)

    def mutation_dispatch_event(
        self, slice_type: SessionSliceType, event: SupportsDataclass
    ) -> None:
        """Dispatch an event to be processed by registered reducers.

        This method implements :class:`MutationProvider` for use by
        :class:`MutationBuilder`.
        """
        self._dispatch_data_event(slice_type, event)

    @property
    @override
    def event_bus(self) -> TelemetryBus:
        """Return the telemetry bus backing this session."""

        return self._bus

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
    def snapshot(self) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state."""

        with self.locked():
            state_snapshot: dict[SessionSliceType, SessionSlice] = dict(self._state)
            parent_id = self._parent.session_id if self._parent is not None else None
            children_ids = tuple(child.session_id for child in self._children)
        try:
            normalized: SnapshotState = normalize_snapshot_state(state_snapshot)
        except ValueError as error:
            msg = "Unable to serialize session slices"
            raise SnapshotSerializationError(msg) from error

        created_at = datetime.now(UTC)
        return Snapshot(
            created_at=created_at,
            parent_id=parent_id,
            children_ids=children_ids,
            slices=normalized,
            tags=self.tags,
        )

    def _registered_slice_types(self) -> set[SessionSliceType]:
        with self.locked():
            types: set[SessionSliceType] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def _register_child(self, child: Session) -> None:
        with self.locked():
            for registered in self._children:
                if registered is child:
                    return
            self._children.append(child)

    def _on_tool_invoked(self, event: object) -> None:
        tool_event = cast(ToolInvoked, event)
        self._handle_tool_invoked(tool_event)

    def _on_prompt_executed(self, event: object) -> None:
        prompt_event = cast(PromptExecuted, event)
        self._handle_prompt_executed(prompt_event)

    def _on_prompt_rendered(self, event: object) -> None:
        start_event = cast(PromptRendered, event)
        self._handle_prompt_rendered(start_event)

    def _handle_tool_invoked(self, event: ToolInvoked) -> None:
        # Dispatch ToolInvoked to any reducers registered for ToolInvoked events
        self._dispatch_data_event(
            _TOOL_INVOKED_TYPE,
            cast(ReducerEvent, event),
        )

        # Extract payload from ToolResult for slice dispatch
        result = event.result
        if hasattr(result, "value"):
            # ToolResult dataclass
            payload = result.value
        elif isinstance(result, dict):
            # Raw dict from SDK native tools - no typed value
            payload = None
        else:
            payload = None

        # Dispatch payload directly to slice reducers
        if is_dataclass_instance(payload):
            # Narrow for ty: payload is SupportsDataclass after TypeGuard
            narrowed = cast(SupportsDataclass, payload)  # pyright: ignore[reportUnnecessaryCast]
            self._dispatch_data_event(type(narrowed), narrowed)

    def _handle_prompt_executed(self, event: PromptExecuted) -> None:
        # Dispatch PromptExecuted to any reducers registered for PromptExecuted events
        self._dispatch_data_event(
            _PROMPT_EXECUTED_TYPE,
            cast(ReducerEvent, event),
        )

        # Dispatch output directly to slice reducers
        output = event.result.output
        if is_dataclass_instance(output):
            self._dispatch_data_event(type(output), output)
            return

        # Handle iterable outputs (dispatch each item directly)
        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            for item in cast(Iterable[object], output):
                if is_dataclass_instance(item):
                    # Narrow for ty: item is SupportsDataclass after TypeGuard
                    narrowed_item = cast(SupportsDataclass, item)  # pyright: ignore[reportUnnecessaryCast]
                    self._dispatch_data_event(type(narrowed_item), narrowed_item)

    def _handle_prompt_rendered(self, event: PromptRendered) -> None:
        self._dispatch_data_event(
            _PROMPT_RENDERED_TYPE,
            cast(ReducerEvent, event),
        )

    def _dispatch_data_event(
        self,
        data_type: SessionSliceType,
        event: ReducerEvent,
        *,
        target_slice_type: SessionSliceType | None = None,
    ) -> None:
        from .reducer_context import build_reducer_context

        with self.locked():
            all_registrations = list(self._reducers.get(data_type, ()))

            # Filter by target slice type if specified (targeted dispatch)
            if target_slice_type is not None:
                registrations = [
                    reg
                    for reg in all_registrations
                    if reg.slice_type == target_slice_type
                ]
            else:
                registrations = all_registrations

            if not registrations:
                # Default: ledger semantics (always append)
                # For targeted dispatch, use the target slice type; otherwise event type
                default_slice = (
                    target_slice_type if target_slice_type is not None else data_type
                )
                registrations = [
                    _ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append_all),
                        slice_type=default_slice,
                    )
                ]

        context = build_reducer_context(session=self)

        # Track state changes for observer notification
        state_changes: dict[SessionSliceType, tuple[SessionSlice, SessionSlice]] = {}

        for registration in registrations:
            slice_type = registration.slice_type
            while True:
                with self.locked():
                    previous = self._state.get(slice_type, EMPTY_SLICE)
                try:
                    result = registration.reducer(previous, event, context=context)
                except Exception:  # log and continue
                    reducer_name = getattr(
                        registration.reducer, "__qualname__", repr(registration.reducer)
                    )
                    logger.exception(
                        "Reducer application failed.",
                        event="session_reducer_failed",
                        context={
                            "reducer": reducer_name,
                            "data_type": data_type.__qualname__,
                            "slice_type": slice_type.__qualname__,
                        },
                    )
                    break
                normalized = tuple(result)
                with self.locked():
                    current = self._state.get(slice_type, EMPTY_SLICE)
                    if current is previous or current == normalized:
                        self._state[slice_type] = normalized
                        # Track change if state actually changed
                        if previous != normalized:
                            state_changes[slice_type] = (previous, normalized)
                        break

        # Notify observers of state changes
        self._notify_observers(state_changes)

    def _notify_observers(
        self, state_changes: dict[SessionSliceType, tuple[SessionSlice, SessionSlice]]
    ) -> None:
        """Call registered observers for slices that changed."""
        for slice_type, (old_values, new_values) in state_changes.items():
            with self.locked():
                observer_registrations = list(self._observers.get(slice_type, ()))

            for registration in observer_registrations:
                try:
                    registration.observer(old_values, new_values)
                except Exception:
                    observer_name = getattr(
                        registration.observer,
                        "__qualname__",
                        repr(registration.observer),
                    )
                    logger.exception(
                        "Observer invocation failed.",
                        event="session_observer_failed",
                        context={
                            "observer": observer_name,
                            "slice_type": slice_type.__qualname__,
                        },
                    )

    def _attach_to_bus(self, bus: TelemetryBus) -> None:
        with self.locked():
            if self._subscriptions_attached and self._bus is bus:
                return
            self._bus = bus
            self._subscriptions_attached = True
            bus.subscribe(ToolInvoked, self._on_tool_invoked)
            bus.subscribe(PromptExecuted, self._on_prompt_executed)
            bus.subscribe(PromptRendered, self._on_prompt_rendered)

    def _register_builtin_reducers(self) -> None:
        """Register built-in reducers for prompt visibility overrides.

        Called once during Session initialization. Safe to call multiple times
        as it guards against re-registration.
        """
        from ...prompt.visibility_overrides import (
            SetVisibilityOverride,
            register_visibility_reducers,
        )

        # Guard against re-registration (e.g., during clone)
        with self.locked():
            if SetVisibilityOverride in self._reducers:
                return

        register_visibility_reducers(self)


__all__ = [
    "DataEvent",
    "GlobalMutationBuilder",
    "MutationBuilder",
    "QueryBuilder",
    "Session",
    "SliceAccessor",
    "SliceObserver",
    "Subscription",
    "TypedReducer",
]
