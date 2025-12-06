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

"""Session state container synchronized with the event bus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import wraps
from threading import RLock
from types import MappingProxyType
from typing import Any, Concatenate, Final, cast, overload, override
from uuid import UUID, uuid4

from ...dbc import invariant
from ...prompt._types import SupportsDataclass
from ..events import EventBus, PromptExecuted, PromptRendered, ToolInvoked
from ..logging import StructuredLogger, get_logger
from ._slice_types import SessionSlice, SessionSliceType
from ._types import ReducerContextProtocol, ReducerEvent, TypedReducer
from .dataclasses import is_dataclass_instance
from .mutation import GlobalMutationBuilder, MutationBuilder
from .protocols import SessionProtocol, SnapshotProtocol
from .query import QueryBuilder
from .reducers import append
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


def _append_event(
    slice_values: tuple[SupportsDataclass, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[SupportsDataclass, ...]:
    del context
    appended = cast(SupportsDataclass, event)
    return (*slice_values, appended)


@dataclass(slots=True)
class _ReducerRegistration:
    reducer: TypedReducer[Any]
    slice_type: SessionSliceType


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


@invariant(  # noqa: PLR0904
    _session_id_is_well_formed,
    _created_at_has_tz,
    _created_at_is_utc,
)
class Session(SessionProtocol):
    """Collect dataclass payloads from prompt executions and tool invocations."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
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

            self._bus: EventBus = InProcessEventBus()
        else:
            self._bus = bus

        self._reducers: dict[SessionSliceType, list[_ReducerRegistration]] = {}
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

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def clone(
        self,
        *,
        bus: EventBus,
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

        for data_type, registrations in reducer_snapshot:
            for registration in registrations:
                clone.register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )

        with clone.locked():
            clone._state = state_snapshot

        return clone

    def register_reducer[S: SupportsDataclass](
        self,
        data_type: SessionSliceType,
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
    ) -> None:
        """Register a reducer for the provided data type.

        Prefer ``session.mutate(T).register(E, reducer)`` for new code.
        """
        self.mutation_register_reducer(data_type, reducer, slice_type=slice_type)

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
            session.mutate(Plan).dispatch(SetupPlan(objective="..."))
            session.mutate(Plan).register(AddStep, add_step_reducer)

            # Session-wide mutations
            session.mutate().reset()
            session.mutate().rollback(snapshot)

        """
        if slice_type is None:
            return GlobalMutationBuilder(self)
        return MutationBuilder(self, slice_type)

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
        self._dispatch_data_event(slice_type, cast(ReducerEvent, event))

    # ──────────────────────────────────────────────────────────────────────
    # Legacy methods (delegate to internal implementations)
    # ──────────────────────────────────────────────────────────────────────

    @override
    def seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None:
        """Initialize or replace the stored tuple for the provided type.

        Prefer ``session.mutate(T).seed(values)`` for new code.
        """
        self.mutation_seed_slice(slice_type, values)

    @override
    def clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        Prefer ``session.mutate(T).clear(predicate)`` for new code.
        """
        self.mutation_clear_slice(slice_type, predicate)

    @override
    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations.

        Prefer ``session.mutate().reset()`` for new code.
        """
        self.mutation_reset()

    @property
    @override
    def event_bus(self) -> EventBus:
        """Return the event bus backing this session."""

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

    @override
    def rollback(self, snapshot: SnapshotProtocol) -> None:
        """Restore session slices from the provided snapshot.

        Prefer ``session.mutate().rollback(snapshot)`` for new code.
        """
        self.mutation_rollback(snapshot)

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
        normalized_event = event
        payload = event.value if event.value is not None else event.result.value
        if event.value is None and is_dataclass_instance(payload):
            normalized_event = replace(event, value=payload)

        self._dispatch_data_event(
            _TOOL_INVOKED_TYPE,
            cast(ReducerEvent, normalized_event),
        )

        if normalized_event.value is not None:
            value_type = cast(SessionSliceType, type(normalized_event.value))
            self._dispatch_data_event(
                value_type,
                cast(ReducerEvent, normalized_event),
            )

    def _handle_prompt_executed(self, event: PromptExecuted) -> None:
        normalized_event = event
        output = event.result.output
        if event.value is None and is_dataclass_instance(output):
            normalized_event = replace(event, value=output)

        self._dispatch_data_event(
            _PROMPT_EXECUTED_TYPE,
            cast(ReducerEvent, normalized_event),
        )

        if normalized_event.value is not None:
            value_type = cast(SessionSliceType, type(normalized_event.value))
            self._dispatch_data_event(
                value_type,
                cast(ReducerEvent, normalized_event),
            )
            return

        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            for item in cast(Iterable[object], output):
                if is_dataclass_instance(item):
                    enriched_event = replace(normalized_event, value=item)
                    self._dispatch_data_event(
                        type(item),
                        cast(ReducerEvent, enriched_event),
                    )

    def _handle_prompt_rendered(self, event: PromptRendered) -> None:
        self._dispatch_data_event(
            _PROMPT_RENDERED_TYPE,
            cast(ReducerEvent, event),
        )

    def _dispatch_data_event(
        self, data_type: SessionSliceType, event: ReducerEvent
    ) -> None:
        from .reducer_context import build_reducer_context

        with self.locked():
            registrations = list(self._reducers.get(data_type, ()))
            if not registrations:
                default_reducer: TypedReducer[Any]
                if data_type in {_TOOL_INVOKED_TYPE, _PROMPT_EXECUTED_TYPE}:
                    default_reducer = cast(TypedReducer[Any], _append_event)
                else:
                    default_reducer = cast(TypedReducer[Any], append)
                registrations = [
                    _ReducerRegistration(
                        reducer=default_reducer,
                        slice_type=data_type,
                    )
                ]
            event_bus = self._bus

        context = build_reducer_context(session=self, event_bus=event_bus)

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
                        break

    def _attach_to_bus(self, bus: EventBus) -> None:
        with self.locked():
            if self._subscriptions_attached and self._bus is bus:
                return
            self._bus = bus
            self._subscriptions_attached = True
            bus.subscribe(ToolInvoked, self._on_tool_invoked)
            bus.subscribe(PromptExecuted, self._on_prompt_executed)
            bus.subscribe(PromptRendered, self._on_prompt_rendered)


__all__ = [
    "DataEvent",
    "GlobalMutationBuilder",
    "MutationBuilder",
    "QueryBuilder",
    "Session",
    "TypedReducer",
]
