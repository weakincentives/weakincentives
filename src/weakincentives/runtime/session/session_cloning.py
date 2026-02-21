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

# pyright: reportPrivateUsage=false

"""Session cloning and snapshot helper functions.

This module provides helper functions for cloning sessions and managing
state snapshots during the clone process.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from ._slice_types import SessionSlice, SessionSliceType
from .reducer_registry import ReducerRegistration
from .slice_policy import SlicePolicy

if TYPE_CHECKING:
    from .session import Session


def snapshot_reducers_and_state(
    session: Session,
) -> tuple[
    list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]],
    dict[SessionSliceType, SessionSlice],
    dict[SessionSliceType, SlicePolicy],
]:
    """Create a snapshot of reducers and state while holding the lock.

    Args:
        session: The session to snapshot.

    Returns:
        A tuple of (reducer_snapshot, state_snapshot, policy_snapshot).
    """
    with session.locked():
        reducer_snapshot = session._registry.snapshot()
        state_snapshot = session._store.snapshot_slices()
        registered = session._store.all_slice_types()
        registered.update(session._registry.all_target_slice_types())
        policy_snapshot = session._store.snapshot_policies(registered)
    return reducer_snapshot, state_snapshot, policy_snapshot


def copy_reducers_to_clone(
    clone: Session,
    reducer_snapshot: list[tuple[SessionSliceType, tuple[ReducerRegistration, ...]]],
) -> None:
    """Copy non-builtin reducers from snapshot to clone.

    Args:
        clone: The cloned session to copy reducers to.
        reducer_snapshot: List of (data_type, registrations) tuples.
    """
    with clone.locked():
        for data_type, registrations in reducer_snapshot:
            if clone._registry.has_registrations(data_type):
                continue
            for registration in registrations:
                clone._mutation_register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )


def apply_state_to_clone(
    clone: Session,
    state_snapshot: dict[SessionSliceType, SessionSlice],
) -> None:
    """Apply state snapshot to cloned session.

    Args:
        clone: The cloned session to apply state to.
        state_snapshot: Dict mapping slice types to their values.
    """
    with clone.locked():
        for slice_type, items in state_snapshot.items():
            slice_instance = clone._get_or_create_slice(slice_type)
            slice_instance.replace(items)


def apply_policies_to_clone(
    clone: Session,
    policy_snapshot: dict[SessionSliceType, SlicePolicy],
) -> None:
    """Apply slice policy snapshot to cloned session.

    Args:
        clone: The cloned session to apply policies to.
        policy_snapshot: Dict mapping slice types to their policies.
    """
    with clone.locked():
        clone._store.apply_policies(policy_snapshot)


def resolve_clone_parent(session: Session, parent: Session | None) -> Session | None:
    """Resolve the parent for a cloned session.

    Args:
        session: The source session.
        parent: Optional explicit parent override.

    Returns:
        The resolved parent session.
    """
    return session._parent if parent is None else parent


def resolve_clone_id(session: Session, session_id: UUID | None) -> UUID:
    """Resolve the session ID for a cloned session.

    Args:
        session: The source session.
        session_id: Optional explicit session ID override.

    Returns:
        The resolved session ID.
    """
    return session_id if session_id is not None else session.session_id


def resolve_clone_created(session: Session, created_at: datetime | None) -> datetime:
    """Resolve the created_at timestamp for a cloned session.

    Args:
        session: The source session.
        created_at: Optional explicit timestamp override.

    Returns:
        The resolved created_at timestamp.
    """
    return created_at if created_at is not None else session.created_at


def resolve_clone_tags(
    session: Session, tags: Mapping[object, object] | None
) -> Mapping[object, object] | None:
    """Resolve tags for a cloned session.

    Args:
        session: The source session.
        tags: Optional explicit tags override.

    Returns:
        The resolved tags mapping.
    """
    if tags is None:
        return cast(Mapping[object, object], session.tags)
    return tags  # pragma: no cover - covered by test_clone_with_custom_tags


__all__ = [
    "apply_policies_to_clone",
    "apply_state_to_clone",
    "copy_reducers_to_clone",
    "resolve_clone_created",
    "resolve_clone_id",
    "resolve_clone_parent",
    "resolve_clone_tags",
    "snapshot_reducers_and_state",
]
