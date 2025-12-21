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

"""State storage manager for session slices and reducers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from typing import Any, cast

from ...types.dataclass import SupportsDataclass
from ..logging import StructuredLogger, get_logger
from ._slice_types import SessionSlice, SessionSliceType
from ._types import ReducerContextProtocol, ReducerEvent, TypedReducer
from .reducers import append_all
from .slice_policy import SlicePolicy

logger: StructuredLogger = get_logger(__name__, context={"component": "state_store"})

EMPTY_SLICE: SessionSlice = ()


@dataclass(slots=True)
class ReducerRegistration:
    """Registration entry linking a reducer to its target slice type."""

    reducer: TypedReducer[Any]
    slice_type: SessionSliceType


class StateStore:
    """Thread-safe storage for session state, reducers, and slice policies.

    Manages:
    - State slices (dict mapping slice types to tuples of values)
    - Reducer registrations (dict mapping event types to reducer lists)
    - Slice policies (dict mapping slice types to STATE or LOG policy)
    """

    __slots__ = ("_lock", "_reducers", "_slice_policies", "_state")

    def __init__(self, lock: RLock) -> None:
        """Initialize the state store.

        Args:
            lock: Shared RLock for thread-safe access.
        """
        super().__init__()
        self._lock = lock
        self._state: dict[SessionSliceType, SessionSlice] = {}
        self._reducers: dict[SessionSliceType, list[ReducerRegistration]] = {}
        self._slice_policies: dict[SessionSliceType, SlicePolicy] = {}

    def select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type.

        Thread-safe: Acquires lock during access.
        """
        with self._lock:
            return cast(tuple[S, ...], self._state.get(slice_type, EMPTY_SLICE))

    def seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None:
        """Initialize or replace the stored tuple for the provided type.

        Thread-safe: Acquires lock during mutation.
        """
        with self._lock:
            self._state[slice_type] = tuple(values)

    def clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None:
        """Remove items from the slice, optionally filtering by predicate.

        Thread-safe: Acquires lock during mutation.
        """
        with self._lock:
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
        """Register a reducer for the provided data type.

        Thread-safe: Acquires lock during registration.
        """
        with self._lock:
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

    def registered_slice_types(self) -> set[SessionSliceType]:
        """Return the set of all registered slice types.

        Thread-safe: Acquires lock during access.
        """
        with self._lock:
            types: set[SessionSliceType] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def reset(self) -> None:
        """Clear all stored slices while preserving reducer registrations.

        Thread-safe: Acquires lock during mutation.
        """
        with self._lock:
            slice_types: set[SessionSliceType] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    slice_types.add(registration.slice_type)
            self._state = dict.fromkeys(slice_types, EMPTY_SLICE)

    def dispatch_event(
        self,
        data_type: SessionSliceType,
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> dict[SessionSliceType, tuple[SessionSlice, SessionSlice]]:
        """Dispatch an event to registered reducers and return state changes.

        Returns a dict mapping slice types to (old_values, new_values) tuples
        for slices that changed.

        Thread-safe: Acquires lock at key points during dispatch.
        """
        with self._lock:
            registrations = list(self._reducers.get(data_type, ()))

            if not registrations:
                # Default: ledger semantics (always append)
                registrations = [
                    ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append_all),
                        slice_type=data_type,
                    )
                ]

        # Track state changes for observer notification
        state_changes: dict[SessionSliceType, tuple[SessionSlice, SessionSlice]] = {}

        for registration in registrations:
            slice_type = registration.slice_type
            while True:
                with self._lock:
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
                with self._lock:
                    current = self._state.get(slice_type, EMPTY_SLICE)
                    if current is previous or current == normalized:
                        self._state[slice_type] = normalized
                        # Track change if state actually changed
                        if previous != normalized:
                            state_changes[slice_type] = (previous, normalized)
                        break

        return state_changes

    def snapshot_for_clone(
        self,
    ) -> tuple[
        list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]],
        dict[SessionSliceType, SessionSlice],
        dict[SessionSliceType, SlicePolicy],
    ]:
        """Create atomic snapshots of reducers, state, and policies for cloning.

        Thread-safe: Acquires lock during snapshot.
        """
        with self._lock:
            reducer_snapshot = [
                (data_type, tuple(registrations))
                for data_type, registrations in self._reducers.items()
            ]
            state_snapshot = dict(self._state)
            policy_snapshot = dict(self._slice_policies)
        return reducer_snapshot, state_snapshot, policy_snapshot

    def copy_reducers_from_snapshot(
        self,
        reducer_snapshot: list[
            tuple[SessionSliceType, tuple[ReducerRegistration, ...]]
        ],
    ) -> None:
        """Copy non-builtin reducers from snapshot (for cloning).

        Thread-safe: Acquires lock during copy.
        """
        with self._lock:
            for data_type, registrations in reducer_snapshot:
                if data_type in self._reducers:
                    continue
                for registration in registrations:
                    self.register_reducer(
                        data_type,
                        registration.reducer,
                        slice_type=registration.slice_type,
                    )

    def apply_state_snapshot(
        self, state_snapshot: dict[SessionSliceType, SessionSlice]
    ) -> None:
        """Apply state snapshot to store (for cloning).

        Thread-safe: Acquires lock during application.
        """
        with self._lock:
            self._state = state_snapshot

    def apply_policy_snapshot(
        self, policy_snapshot: dict[SessionSliceType, SlicePolicy]
    ) -> None:
        """Apply slice policy snapshot to store (for cloning).

        Thread-safe: Acquires lock during application.
        """
        with self._lock:
            self._slice_policies = dict(policy_snapshot)

    def set_initial_policies(
        self, policies: dict[SessionSliceType, SlicePolicy]
    ) -> None:
        """Set initial slice policies during initialization.

        Thread-safe: Acquires lock during setting.
        """
        with self._lock:
            self._slice_policies.update(policies)

    def has_reducer_for(self, data_type: SessionSliceType) -> bool:
        """Check if a reducer is registered for the given data type.

        Thread-safe: Acquires lock during check.
        """
        with self._lock:
            return data_type in self._reducers

    # ──────────────────────────────────────────────────────────────────────
    # Snapshot Access (for SnapshotManager)
    # ──────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict[SessionSliceType, SessionSlice]:
        """Get a copy of the current state for snapshotting.

        Thread-safe: Acquires lock during access.
        """
        with self._lock:
            return dict(self._state)

    def get_reducer_slice_types(self) -> set[SessionSliceType]:
        """Get all slice types registered through reducers.

        Thread-safe: Acquires lock during access.
        """
        with self._lock:
            types: set[SessionSliceType] = set()
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def get_policies_snapshot(
        self, slice_types: set[SessionSliceType]
    ) -> dict[SessionSliceType, SlicePolicy]:
        """Get policies for the given slice types.

        Thread-safe: Acquires lock during access.
        """
        with self._lock:
            return {
                slice_type: self._slice_policies.get(slice_type, SlicePolicy.STATE)
                for slice_type in slice_types
            }

    def set_state(self, state: dict[SessionSliceType, SessionSlice]) -> None:
        """Set the state directly (for snapshot restore).

        Thread-safe: Acquires lock during mutation.
        """
        with self._lock:
            self._state = state


__all__ = ["ReducerRegistration", "StateStore"]
