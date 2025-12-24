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

"""Internal state manager for Session.

SessionState encapsulates all state management internals including slices,
reducers, observers, and their thread-safe access patterns. Session delegates
to this helper while maintaining its public API and bus integration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from ...types.dataclass import SupportsDataclass
from ..logging import StructuredLogger, get_logger
from ._observer_types import SliceObserver, Subscription
from ._slice_types import SessionSlice, SessionSliceType
from ._types import ReducerEvent, TypedReducer
from .reducers import append_all
from .slice_policy import SlicePolicy

if TYPE_CHECKING:
    from .protocols import SessionProtocol

logger: StructuredLogger = get_logger(__name__, context={"component": "session_state"})


EMPTY_SLICE: SessionSlice = ()


@dataclass(slots=True)
class ReducerRegistration:
    """Internal registration record for a reducer."""

    reducer: TypedReducer[Any]
    slice_type: SessionSliceType


@dataclass(slots=True)
class ObserverRegistration:
    """Internal registration record for an observer."""

    observer: SliceObserver[Any]
    subscription: Subscription


class SessionState:
    """Internal state manager for Session.

    Owns all mutable state and provides thread-safe access patterns.
    Session delegates to this helper for state operations.

    Thread Safety:
        All state access and mutations are protected by an RLock.
        The lock is reentrant to allow nested calls within reducers.
    """

    __slots__ = (
        "_lock",
        "_observers",
        "_reducers",
        "_slice_policies",
        "_state",
    )

    def __init__(
        self,
        *,
        initial_policies: dict[SessionSliceType, SlicePolicy] | None = None,
    ) -> None:
        """Initialize the state manager.

        Args:
            initial_policies: Optional initial slice policies to install.
        """
        super().__init__()
        self._lock = RLock()
        self._state: dict[SessionSliceType, SessionSlice] = {}
        self._reducers: dict[SessionSliceType, list[ReducerRegistration]] = {}
        self._observers: dict[SessionSliceType, list[ObserverRegistration]] = {}
        self._slice_policies: dict[SessionSliceType, SlicePolicy] = (
            dict(initial_policies) if initial_policies else {}
        )

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Context manager for acquiring the state lock."""
        with self._lock:
            yield

    # ──────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────

    def select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type."""
        with self.locked():
            return cast(tuple[S, ...], self._state.get(slice_type, EMPTY_SLICE))

    def registered_slice_types(self) -> set[SessionSliceType]:
        """Return all registered slice types."""
        with self.locked():
            types: set[SessionSliceType] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def get_policy(
        self, slice_type: SessionSliceType, default: SlicePolicy = SlicePolicy.STATE
    ) -> SlicePolicy:
        """Return the policy for a slice type."""
        with self.locked():
            return self._slice_policies.get(slice_type, default)

    def get_policies_snapshot(self) -> dict[SessionSliceType, SlicePolicy]:
        """Return a snapshot of all slice policies."""
        with self.locked():
            registered = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    registered.add(registration.slice_type)
            return {
                slice_type: self._slice_policies.get(slice_type, SlicePolicy.STATE)
                for slice_type in registered
            }

    def get_state_snapshot(self) -> dict[SessionSliceType, SessionSlice]:
        """Return a snapshot of all state."""
        with self.locked():
            return dict(self._state)

    def get_reducers_snapshot(
        self,
    ) -> list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]]:
        """Return a snapshot of all reducer registrations."""
        with self.locked():
            return [
                (data_type, tuple(registrations))
                for data_type, registrations in self._reducers.items()
            ]

    # ──────────────────────────────────────────────────────────────────────
    # Mutation Operations
    # ──────────────────────────────────────────────────────────────────────

    def seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None:
        """Initialize or replace the stored tuple for the provided type."""
        with self.locked():
            self._state[slice_type] = tuple(values)

    def clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None:
        """Remove items from the slice, optionally filtering by predicate."""
        with self.locked():
            existing = cast(tuple[S, ...], self._state.get(slice_type, EMPTY_SLICE))
            if not existing:
                return
            if predicate is None:
                self._state[slice_type] = EMPTY_SLICE
                return
            filtered = tuple(value for value in existing if not predicate(value))
            self._state[slice_type] = filtered

    def register_reducer[S: SupportsDataclass](
        self,
        data_type: SessionSliceType,
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
        policy: SlicePolicy | None = None,
    ) -> None:
        """Register a reducer for the provided data type."""
        with self.locked():
            target_slice_type: SessionSliceType = (
                data_type if slice_type is None else slice_type
            )
            registration = ReducerRegistration(
                reducer=cast(TypedReducer[Any], reducer),
                slice_type=target_slice_type,
            )
            bucket = self._reducers.setdefault(data_type, [])
            bucket.append(registration)
            _ = self._state.setdefault(target_slice_type, EMPTY_SLICE)
            if policy is not None:
                self._slice_policies[target_slice_type] = policy
            else:
                _ = self._slice_policies.setdefault(
                    target_slice_type, SlicePolicy.STATE
                )

    def has_reducer(self, data_type: SessionSliceType) -> bool:
        """Check if a reducer is registered for the given data type."""
        with self.locked():
            return data_type in self._reducers

    # ──────────────────────────────────────────────────────────────────────
    # Observer Operations
    # ──────────────────────────────────────────────────────────────────────

    def register_observer[S: SupportsDataclass](
        self,
        slice_type: type[S],
        observer: SliceObserver[S],
    ) -> Subscription:
        """Register an observer called when the slice changes."""
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

        registration = ObserverRegistration(
            observer=cast(SliceObserver[Any], observer),
            subscription=subscription,
        )
        with self.locked():
            bucket = self._observers.setdefault(slice_type, [])
            bucket.append(registration)

        return subscription

    def notify_observers(
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

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch Operations
    # ──────────────────────────────────────────────────────────────────────

    def dispatch_data_event(
        self,
        data_type: SessionSliceType,
        event: ReducerEvent,
        *,
        session: SessionProtocol,
    ) -> None:
        """Dispatch an event to all registered reducers.

        Args:
            data_type: The event type being dispatched.
            event: The event instance.
            session: The session owning this state (for reducer context).
        """
        from .reducer_context import build_reducer_context

        with self.locked():
            registrations = list(self._reducers.get(data_type, ()))

            if not registrations:
                # Default: ledger semantics (always append)
                registrations = [
                    ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append_all),
                        slice_type=data_type,
                    )
                ]

        context = build_reducer_context(session=session)

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
                        registration.reducer,
                        "__qualname__",
                        repr(registration.reducer),
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
        self.notify_observers(state_changes)

    # ──────────────────────────────────────────────────────────────────────
    # Global State Operations
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations."""
        with self.locked():
            slice_types: set[SessionSliceType] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    slice_types.add(registration.slice_type)
            self._state = dict.fromkeys(slice_types, EMPTY_SLICE)

    def apply_state(self, state_snapshot: dict[SessionSliceType, SessionSlice]) -> None:
        """Apply a state snapshot directly."""
        with self.locked():
            self._state = state_snapshot

    def apply_policies(
        self, policy_snapshot: dict[SessionSliceType, SlicePolicy]
    ) -> None:
        """Apply a policy snapshot directly."""
        with self.locked():
            self._slice_policies = dict(policy_snapshot)

    def restore_state(
        self,
        snapshot_slices: Mapping[SessionSliceType, SessionSlice],
        snapshot_policies: Mapping[SessionSliceType, SlicePolicy],
        *,
        preserve_logs: bool = True,
    ) -> None:
        """Restore state from snapshot data.

        Args:
            snapshot_slices: The slice data from the snapshot.
            snapshot_policies: The policy data from the snapshot.
            preserve_logs: If True, slices marked as LOG are not modified.
        """
        registered_slices = self.registered_slice_types()
        with self.locked():
            new_state: dict[SessionSliceType, SessionSlice] = dict(self._state)
            for slice_type in registered_slices:
                policy = snapshot_policies.get(
                    slice_type,
                    self._slice_policies.get(slice_type, SlicePolicy.STATE),
                )
                if preserve_logs and policy is SlicePolicy.LOG:
                    continue
                new_state[slice_type] = snapshot_slices.get(slice_type, EMPTY_SLICE)
            self._state = new_state

    # ──────────────────────────────────────────────────────────────────────
    # Clone Support
    # ──────────────────────────────────────────────────────────────────────

    def copy_reducers_from(
        self,
        reducer_snapshot: list[
            tuple[SessionSliceType, tuple[ReducerRegistration, ...]]
        ],
    ) -> None:
        """Copy non-builtin reducers from snapshot."""
        for data_type, registrations in reducer_snapshot:
            with self.locked():
                if data_type in self._reducers:
                    continue
            for registration in registrations:
                self.register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )


__all__ = [
    "EMPTY_SLICE",
    "ObserverRegistration",
    "ReducerRegistration",
    "SessionState",
]
