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

"""Snapshot management for session state capture and restoration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from threading import RLock
from typing import TYPE_CHECKING
from uuid import UUID

from ._slice_types import SessionSlice, SessionSliceType
from .protocols import SnapshotProtocol
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    normalize_snapshot_state,
)

if TYPE_CHECKING:
    from .state_store import StateStore


class SnapshotManager:
    """Manager for session snapshot capture and restoration.

    Handles:
    - Capturing immutable snapshots of session state
    - Restoring session state from snapshots
    """

    __slots__ = ("_lock", "_state_store")

    def __init__(self, lock: RLock, state_store: StateStore) -> None:
        """Initialize the snapshot manager.

        Args:
            lock: Shared RLock for thread-safe access.
            state_store: Reference to the session's StateStore.
        """
        super().__init__()
        self._lock = lock
        self._state_store = state_store

    def capture(  # noqa: PLR0913
        self,
        *,
        session_id: UUID,
        parent_id: UUID | None,
        children_ids: tuple[UUID, ...],
        tags: Mapping[str, str],
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state.

        Args:
            session_id: The session's UUID.
            parent_id: Parent session UUID if exists.
            children_ids: Child session UUIDs.
            tags: Session tags.
            policies: Slice policies to include when include_all is False.
            include_all: If True, snapshot all slices regardless of policy.

        Thread-safe: Acquires lock during capture.
        """
        # Get state snapshot through public accessors
        state_snapshot = self._state_store.get_state_snapshot()
        registered = set(state_snapshot)
        registered.update(self._state_store.get_reducer_slice_types())
        policy_snapshot = self._state_store.get_policies_snapshot(registered)

        if include_all:
            snapshot_state = state_snapshot
        else:
            snapshot_state = {
                slice_type: values
                for slice_type, values in state_snapshot.items()
                if policy_snapshot.get(slice_type, SlicePolicy.STATE) in policies
            }
        try:
            normalized = normalize_snapshot_state(snapshot_state)
        except ValueError as error:
            msg = "Unable to serialize session slices"
            raise SnapshotSerializationError(msg) from error

        created_at = datetime.now(UTC)
        return Snapshot(
            created_at=created_at,
            parent_id=parent_id,
            children_ids=children_ids,
            slices=normalized,
            tags=tags,
            policies=policy_snapshot,
        )

    def restore(
        self,
        snapshot: SnapshotProtocol,
        *,
        preserve_logs: bool = True,
    ) -> None:
        """Restore session slices from the provided snapshot.

        All slice types in the snapshot must be registered in the session.

        Args:
            snapshot: A snapshot previously captured via ``session.snapshot()``.
            preserve_logs: If True, slices marked as LOG are not modified.

        Raises:
            SnapshotRestoreError: If the snapshot contains unregistered slice types.

        Thread-safe: Acquires lock during restoration.
        """
        registered_slices = self._state_store.registered_slice_types()
        missing = [
            slice_type
            for slice_type in snapshot.slices
            if slice_type not in registered_slices
        ]
        if missing:
            missing_names = ", ".join(sorted(cls.__qualname__ for cls in missing))
            msg = f"Slice types not registered: {missing_names}"
            raise SnapshotRestoreError(msg)

        # Get current state and policies through public accessors
        current_state = self._state_store.get_state_snapshot()
        current_policies = self._state_store.get_policies_snapshot(registered_slices)

        new_state: dict[SessionSliceType, SessionSlice] = dict(current_state)
        for slice_type in registered_slices:
            policy = snapshot.policies.get(
                slice_type,
                current_policies.get(slice_type, SlicePolicy.STATE),
            )
            if preserve_logs and policy is SlicePolicy.LOG:
                continue
            empty: SessionSlice = ()
            new_state[slice_type] = snapshot.slices.get(slice_type, empty)
        self._state_store.set_state(new_state)


__all__ = ["SnapshotManager"]
