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

"""Snapshot and restore functionality for session state.

SessionSnapshotter provides methods for capturing immutable snapshots of
session state and restoring from those snapshots for transaction rollback.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from threading import RLock
from typing import TYPE_CHECKING
from uuid import UUID  # used in type annotations

from ._slice_types import SessionSlice, SessionSliceType
from .slice_policy import DEFAULT_SNAPSHOT_POLICIES, SlicePolicy
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    SnapshotState,
    normalize_snapshot_state,
)

if TYPE_CHECKING:
    from .protocols import SnapshotProtocol
    from .reducer_registry import ReducerRegistry
    from .slice_store import SliceStore

EMPTY_SLICE: SessionSlice = ()


class SessionSnapshotter:
    """Handles snapshot creation and restoration for sessions.

    SessionSnapshotter encapsulates the logic for creating immutable
    snapshots of session state and restoring sessions from those snapshots.
    It works with SliceStore and ReducerRegistry to gather and apply state.

    Example::

        snapshotter = SessionSnapshotter(
            store=slice_store,
            registry=reducer_registry,
            lock=session_lock,
        )

        # Capture snapshot
        snapshot = snapshotter.create_snapshot(
            session_id=session.session_id,
            parent_id=parent.session_id if parent else None,
            children_ids=tuple(c.session_id for c in children),
            tags=session.tags,
        )

        # Restore from snapshot
        snapshotter.restore(snapshot)

    """

    __slots__ = ("_lock", "_registry", "_store")

    def __init__(
        self,
        *,
        store: SliceStore,
        registry: ReducerRegistry,
        lock: RLock,
    ) -> None:
        """Initialize the snapshotter.

        Args:
            store: The slice store to snapshot/restore.
            registry: The reducer registry to determine registered types.
            lock: The session lock for thread-safe operations.
        """
        super().__init__()
        self._store = store
        self._registry = registry
        self._lock = lock

    def _get_registered_slice_types(self) -> set[SessionSliceType]:
        """Get all registered slice types from store and registry."""
        types = self._store.all_slice_types()
        types.update(self._registry.all_target_slice_types())
        return types

    def create_snapshot(
        self,
        *,
        parent_id: UUID | None,
        children_ids: tuple[UUID, ...],
        tags: Mapping[str, str],
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state.

        Args:
            parent_id: The parent session ID if any.
            children_ids: Tuple of child session IDs.
            tags: Immutable tags associated with the session.
            policies: Slice policies to include when include_all is False.
            include_all: If True, snapshot all slices regardless of policy.

        Returns:
            An immutable Snapshot of current state.

        Raises:
            SnapshotSerializationError: If slices cannot be serialized.
        """

        with self._lock:
            state_snapshot = self._store.snapshot_slices()
            registered = self._get_registered_slice_types()
            policy_snapshot = self._store.snapshot_policies(registered)

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
            snapshot: A snapshot previously captured via create_snapshot.
            preserve_logs: If True, slices marked as LOG are not modified.

        Raises:
            SnapshotRestoreError: If the snapshot contains unregistered types.
        """
        registered_slices = self._get_registered_slice_types()
        missing = [
            slice_type
            for slice_type in snapshot.slices
            if slice_type not in registered_slices
        ]
        if missing:
            missing_names = ", ".join(sorted(cls.__qualname__ for cls in missing))
            msg = f"Slice types not registered: {missing_names}"
            raise SnapshotRestoreError(msg)

        with self._lock:
            for slice_type in registered_slices:
                policy = snapshot.policies.get(
                    slice_type,
                    self._store.get_policy(slice_type),
                )
                if preserve_logs and policy is SlicePolicy.LOG:
                    continue
                items = snapshot.slices.get(slice_type, EMPTY_SLICE)
                slice_instance = self._store.get_or_create(slice_type)
                slice_instance.replace(items)


__all__ = ["SessionSnapshotter"]
