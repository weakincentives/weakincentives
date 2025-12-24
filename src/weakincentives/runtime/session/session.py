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
from typing import Any, Concatenate, Final, cast, override
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
from ._slice_types import SessionSlice, SessionSliceType
from ._types import ReducerEvent, TypedReducer
from .dataclasses import is_dataclass_instance
from .protocols import SessionProtocol, SnapshotProtocol
from .reducers import append_all
from .slice_accessor import SliceAccessor
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .slices import (
    Append,
    Clear,
    Extend,
    Replace,
    Slice,
    SliceFactoryConfig,
    SliceOp,
    default_slice_config,
)
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

    """

    def __init__(
        self,
        *,
        bus: TelemetryDispatcher | None = None,
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

        if bus is None:
            from ..events import InProcessDispatcher

            self._bus: TelemetryDispatcher = InProcessDispatcher()
        else:
            self._bus = bus

        self._slice_config = (
            slice_config if slice_config is not None else default_slice_config()
        )
        self._reducers: dict[SessionSliceType, list[_ReducerRegistration]] = {}
        self._slices: dict[SessionSliceType, Slice[Any]] = {}
        self._slice_policies: dict[SessionSliceType, SlicePolicy] = {
            _PROMPT_RENDERED_TYPE: SlicePolicy.LOG,
            _PROMPT_EXECUTED_TYPE: SlicePolicy.LOG,
            _TOOL_INVOKED_TYPE: SlicePolicy.LOG,
        }
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
        self._attach_to_dispatcher(self._bus)
        self._register_builtin_reducers()

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    @staticmethod
    def _copy_reducers_to_clone(
        clone: Session,
        reducer_snapshot: list[
            tuple[SessionSliceType, tuple[_ReducerRegistration, ...]]
        ],
    ) -> None:
        """Copy non-builtin reducers from snapshot to clone."""
        for data_type, registrations in reducer_snapshot:
            if data_type in clone._reducers:
                continue
            for registration in registrations:
                clone._mutation_register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )

    def _get_or_create_slice[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> Slice[T]:
        """Get existing slice or create one using the appropriate factory."""
        if slice_type not in self._slices:
            policy = self._slice_policies.get(slice_type, SlicePolicy.STATE)
            factory = self._slice_config.factory_for_policy(policy)
            self._slices[slice_type] = factory.create(slice_type)
        return cast(Slice[T], self._slices[slice_type])

    def _snapshot_reducers_and_state(
        self,
    ) -> tuple[
        list[tuple[SessionSliceType, tuple[_ReducerRegistration, ...]]],
        dict[SessionSliceType, SessionSlice],
        dict[SessionSliceType, SlicePolicy],
    ]:
        """Create a snapshot of reducers and state while holding the lock."""
        with self.locked():
            reducer_snapshot = [
                (data_type, tuple(registrations))
                for data_type, registrations in self._reducers.items()
            ]
            # Convert slices to tuples for snapshot
            state_snapshot = {
                slice_type: slice_instance.snapshot()
                for slice_type, slice_instance in self._slices.items()
            }
            policy_snapshot = dict(self._slice_policies)
        return reducer_snapshot, state_snapshot, policy_snapshot

    def _resolve_clone_parent(self, parent: Session | None) -> Session | None:
        return self._parent if parent is None else parent

    def _resolve_clone_id(self, session_id: UUID | None) -> UUID:
        return session_id if session_id is not None else self.session_id

    def _resolve_clone_created(self, created_at: datetime | None) -> datetime:
        return created_at if created_at is not None else self.created_at

    def _resolve_clone_tags(
        self, tags: Mapping[object, object] | None
    ) -> Mapping[object, object] | None:
        if tags is None:
            return cast(Mapping[object, object], self.tags)
        return tags  # pragma: no cover - covered by test_clone_with_custom_tags

    @staticmethod
    def _apply_state_to_clone(
        clone: Session,
        state_snapshot: dict[SessionSliceType, SessionSlice],
    ) -> None:
        """Apply state snapshot to cloned session."""
        with clone.locked():
            for slice_type, items in state_snapshot.items():
                slice_instance = clone._get_or_create_slice(slice_type)
                slice_instance.replace(items)

    @staticmethod
    def _apply_policies_to_clone(
        clone: Session,
        policy_snapshot: dict[SessionSliceType, SlicePolicy],
    ) -> None:
        """Apply slice policy snapshot to cloned session."""
        with clone.locked():
            clone._slice_policies = dict(policy_snapshot)

    def clone(
        self,
        *,
        bus: TelemetryDispatcher,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        tags: Mapping[object, object] | None = None,
        slice_config: SliceFactoryConfig | None = None,
    ) -> Session:
        """Return a new session that mirrors the current state and reducers."""
        reducer_snapshot, state_snapshot, policy_snapshot = (
            self._snapshot_reducers_and_state()
        )
        clone = Session(
            bus=bus,
            parent=self._resolve_clone_parent(parent),
            session_id=self._resolve_clone_id(session_id),
            created_at=self._resolve_clone_created(created_at),
            tags=self._resolve_clone_tags(tags),
            slice_config=slice_config
            if slice_config is not None
            else self._slice_config,
        )
        self._copy_reducers_to_clone(clone, reducer_snapshot)
        self._apply_state_to_clone(clone, state_snapshot)
        self._apply_policies_to_clone(clone, policy_snapshot)
        return clone

    @_locked_method
    def _select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type.

        Internal method used by SliceAccessor. Use ``session[SliceType].all()``
        for public access.
        """
        slice_instance = self._get_or_create_slice(slice_type)
        return slice_instance.all()

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

        For broadcast dispatch (routes to all reducers for the event type)::

            session.broadcast(AddStep(step="x"))

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
        self._dispatch_data_event(type(event), event)
        return self._bus.dispatch(event)

    # ──────────────────────────────────────────────────────────────────────
    # Global Mutation Operations
    # ──────────────────────────────────────────────────────────────────────

    @override
    @_locked_method
    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations.

        This is useful for resetting session state between operations while
        keeping the same reducer configuration.

        Example::

            session.reset()  # All slices are now empty

        """
        slice_types: set[SessionSliceType] = set(self._slices)
        for registrations in self._reducers.values():
            for registration in registrations:
                slice_types.add(registration.slice_type)
        for slice_type in slice_types:
            slice_instance = self._get_or_create_slice(slice_type)
            slice_instance.clear()

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
            for slice_type in registered_slices:
                policy = snapshot.policies.get(
                    slice_type,
                    self._slice_policies.get(slice_type, SlicePolicy.STATE),
                )
                if preserve_logs and policy is SlicePolicy.LOG:
                    continue
                items = snapshot.slices.get(slice_type, EMPTY_SLICE)
                slice_instance = self._get_or_create_slice(slice_type)
                slice_instance.replace(items)

    # ──────────────────────────────────────────────────────────────────────
    # Private Mutation Methods (used by SliceAccessor)
    # ──────────────────────────────────────────────────────────────────────

    @_locked_method
    def _mutation_seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None:
        """Initialize or replace the stored tuple for the provided type."""
        slice_instance = self._get_or_create_slice(slice_type)
        slice_instance.replace(tuple(values))

    @_locked_method
    def _mutation_clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None:
        """Remove items from the slice, optionally filtering by predicate."""
        slice_instance = self._get_or_create_slice(slice_type)
        slice_instance.clear(predicate)

    @_locked_method
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
        registration = _ReducerRegistration(
            reducer=cast(TypedReducer[Any], reducer),
            slice_type=target_slice_type,
        )
        bucket = self._reducers.setdefault(data_type, [])
        bucket.append(registration)
        # Ensure slice exists
        _ = self._get_or_create_slice(target_slice_type)
        if policy is not None:
            self._slice_policies[target_slice_type] = policy
        else:
            _ = self._slice_policies.setdefault(target_slice_type, SlicePolicy.STATE)

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

        with self.locked():
            # Convert slices to tuples for snapshot
            state_snapshot: dict[SessionSliceType, SessionSlice] = {
                slice_type: slice_instance.snapshot()
                for slice_type, slice_instance in self._slices.items()
            }
            parent_id = self._parent.session_id if self._parent is not None else None
            children_ids = tuple(child.session_id for child in self._children)
            registered = set(state_snapshot)
            for registrations in self._reducers.values():
                for registration in registrations:
                    registered.add(registration.slice_type)
            policy_snapshot = {
                slice_type: self._slice_policies.get(slice_type, SlicePolicy.STATE)
                for slice_type in registered
            }
        if include_all:
            snapshot_state = state_snapshot
        else:
            snapshot_state = {
                slice_type: values
                for slice_type, values in state_snapshot.items()
                if policy_snapshot.get(slice_type, SlicePolicy.STATE) in policies
            }
        try:
            normalized: SnapshotState = normalize_snapshot_state(snapshot_state)
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
            policies=policy_snapshot,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────────────────

    def _registered_slice_types(self) -> set[SessionSliceType]:
        with self.locked():
            types: set[SessionSliceType] = set(self._slices)
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
    ) -> None:
        from .reducer_context import build_reducer_context
        from .session_view import SessionView

        with self.locked():
            registrations = list(self._reducers.get(data_type, ()))

            if not registrations:
                # Default: ledger semantics (always append)
                registrations = [
                    _ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append_all),
                        slice_type=data_type,
                    )
                ]

        view = SessionView(self)
        context = build_reducer_context(session=view)

        for registration in registrations:
            slice_type = registration.slice_type
            with self.locked():
                slice_instance = self._get_or_create_slice(slice_type)
                view = slice_instance.view()
            try:
                op = registration.reducer(view, event, context=context)
                # Apply the slice operation
                with self.locked():
                    self._apply_slice_op(op, slice_instance)
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
                continue

    @staticmethod
    def _apply_slice_op[S: SupportsDataclass](
        op: SliceOp[S],
        slice_instance: Slice[S],
    ) -> None:
        """Apply slice operation using optimal method."""
        match op:
            case Append(item=item):
                slice_instance.append(item)
            case Extend(items=items):
                slice_instance.extend(items)
            case Replace(items=items):
                slice_instance.replace(items)
            case Clear(predicate=pred):
                slice_instance.clear(pred)

    def _attach_to_dispatcher(self, bus: TelemetryDispatcher) -> None:
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
        from .visibility_overrides import (
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
    "Session",
    "SliceAccessor",
    "TypedReducer",
]
