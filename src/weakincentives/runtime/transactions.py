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

"""Transaction support for tool execution.

This module provides transactional semantics for tool execution by managing
composite snapshots of session state and snapshotable resources. It enables
atomic rollback on tool failure.

Example usage::

    from weakincentives.runtime.transactions import tool_transaction, restore_snapshot

    # Use as context manager for automatic rollback on exception
    # Note: pass prompt.resources.context (the ScopedResourceContext)
    with tool_transaction(session, prompt.resources.context, tag="my_tool") as snapshot:
        result = execute_tool(...)
        if not result.success:
            restore_snapshot(session, prompt.resources.context, snapshot)
        return result

For hook-based native tool execution, use the PendingToolTracker class to
manage snapshots across pre_tool_use and post_tool_use hooks.
"""

from __future__ import annotations

import json
import threading
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from ..errors import RestoreFailedError
from ..serde import TYPE_REF_KEY, dump, parse, resolve_type_identifier, type_identifier
from ..types import JSONValue
from .session.protocols import SessionProtocol
from .session.snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
)
from .snapshotable import Snapshotable

if TYPE_CHECKING:
    from ..resources.context import ScopedResourceContext

# Schema version for composite snapshot serialization
COMPOSITE_SNAPSHOT_SCHEMA_VERSION = "1"


@FrozenDataclass()
class SnapshotMetadata:
    """Context for when and why a snapshot was taken.

    SnapshotMetadata captures the circumstances around a snapshot creation,
    including optional tool call information when snapshots are taken as
    part of tool transaction boundaries.

    Attributes:
        tag: Optional human-readable label for the snapshot.
        tool_call_id: ID of the tool call that triggered this snapshot.
        tool_name: Name of the tool being executed.
        phase: When the snapshot was taken relative to tool execution.
    """

    tag: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    phase: Literal["pre_tool", "post_tool", "manual"] = "manual"


@FrozenDataclass()
class CompositeSnapshot:
    """Consistent snapshot of session and snapshotable resources.

    CompositeSnapshot captures a point-in-time view of session slices and all
    registered snapshotable resources (e.g., filesystem). This enables atomic
    rollback on tool failure.

    Attributes:
        snapshot_id: Unique identifier for this snapshot.
        created_at: Timestamp when the snapshot was taken.
        session: Snapshot of session slice state.
        resources: Mapping of resource types to their snapshots.
        metadata: Optional context about when/why this snapshot was taken.
    """

    snapshot_id: UUID
    created_at: datetime
    session: Snapshot
    resources: Mapping[type[object], object] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    metadata: SnapshotMetadata | None = None

    def to_json(self) -> str:
        """Serialize the composite snapshot to a JSON string.

        Returns:
            JSON representation of the composite snapshot.

        Raises:
            SnapshotSerializationError: If serialization fails.
        """
        try:
            # Serialize session snapshot
            session_json = self.session.to_json()

            # Serialize resource snapshots
            resource_entries: list[dict[str, JSONValue]] = []
            for resource_type, resource_snapshot in sorted(
                self.resources.items(),
                key=lambda item: type_identifier(item[0]),
            ):
                try:
                    serialized_snapshot = dump(
                        resource_snapshot,
                        include_dataclass_type=True,
                        type_key=TYPE_REF_KEY,
                    )
                except Exception as error:  # pragma: no cover - defensive
                    msg = f"Failed to serialize resource snapshot for {resource_type.__qualname__}"
                    raise SnapshotSerializationError(msg) from error

                resource_entries.append(
                    {
                        "resource_type": type_identifier(resource_type),
                        "snapshot": cast(JSONValue, serialized_snapshot),
                    }
                )

            # Serialize metadata
            metadata_payload: dict[str, JSONValue] | None = None
            if self.metadata is not None:  # pragma: no branch - tested separately
                metadata_payload = {
                    "tag": self.metadata.tag,
                    "tool_call_id": self.metadata.tool_call_id,
                    "tool_name": self.metadata.tool_name,
                    "phase": self.metadata.phase,
                }

            payload: dict[str, JSONValue] = {
                "version": COMPOSITE_SNAPSHOT_SCHEMA_VERSION,
                "snapshot_id": str(self.snapshot_id),
                "created_at": self.created_at.isoformat(),
                "session": json.loads(session_json),
                "resources": resource_entries,
                "metadata": metadata_payload,
            }
            return json.dumps(payload, sort_keys=True)
        except SnapshotSerializationError:  # pragma: no cover - defensive
            raise
        except Exception as error:  # pragma: no cover - defensive
            msg = "Failed to serialize composite snapshot"
            raise SnapshotSerializationError(msg) from error

    @classmethod
    def from_json(cls, raw: str) -> CompositeSnapshot:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Deserialize a composite snapshot from its JSON representation.

        Args:
            raw: JSON string to deserialize.

        Returns:
            Deserialized CompositeSnapshot.

        Raises:
            SnapshotRestoreError: If deserialization fails.
        """
        try:
            payload_obj: JSONValue = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotRestoreError("Invalid composite snapshot JSON") from error

        if not isinstance(payload_obj, Mapping):
            raise SnapshotRestoreError("Composite snapshot payload must be an object")

        payload = cast(Mapping[str, JSONValue], payload_obj)

        # Validate version
        version = payload.get("version")
        if version != COMPOSITE_SNAPSHOT_SCHEMA_VERSION:
            msg = (
                f"Composite snapshot schema version mismatch: "
                f"expected {COMPOSITE_SNAPSHOT_SCHEMA_VERSION}, got {version!r}"
            )
            raise SnapshotRestoreError(msg)

        # Parse snapshot_id
        snapshot_id_str = payload.get("snapshot_id")
        if not isinstance(snapshot_id_str, str):  # pragma: no cover - defensive
            raise SnapshotRestoreError(
                "Composite snapshot snapshot_id must be a string"
            )
        try:
            snapshot_id = UUID(snapshot_id_str)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid snapshot_id") from error

        # Parse created_at
        created_at_str = payload.get("created_at")
        if not isinstance(created_at_str, str):  # pragma: no cover - defensive
            raise SnapshotRestoreError("Composite snapshot created_at must be a string")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid created_at timestamp") from error

        # Parse session snapshot
        session_payload = payload.get("session")
        if not isinstance(session_payload, Mapping):
            raise SnapshotRestoreError("Session snapshot must be an object")
        try:
            session_snapshot = Snapshot.from_json(json.dumps(session_payload))
        except Exception as error:  # pragma: no cover - defensive
            raise SnapshotRestoreError("Failed to parse session snapshot") from error

        # Parse resource snapshots
        resources_payload = payload.get("resources", [])
        if not isinstance(resources_payload, list):
            raise SnapshotRestoreError("Resources must be a list")

        resources: dict[type[object], object] = {}
        for entry in resources_payload:
            if not isinstance(entry, Mapping):
                raise SnapshotRestoreError("Resource entry must be an object")

            resource_type_str = entry.get("resource_type")
            if not isinstance(resource_type_str, str):  # pragma: no cover - defensive
                raise SnapshotRestoreError("Resource type must be a string")

            try:
                resource_type = resolve_type_identifier(resource_type_str)
            except (TypeError, ValueError, ImportError) as error:  # pragma: no cover
                raise SnapshotRestoreError(
                    f"Unknown resource type: {resource_type_str}"
                ) from error

            snapshot_data = entry.get("snapshot")
            if not isinstance(snapshot_data, Mapping):  # pragma: no cover - defensive
                raise SnapshotRestoreError("Resource snapshot must be an object")

            # Determine snapshot type from the data
            snapshot_type_str = snapshot_data.get(TYPE_REF_KEY)
            if not isinstance(snapshot_type_str, str):  # pragma: no cover - defensive
                raise SnapshotRestoreError(
                    "Resource snapshot must include type reference"
                )

            try:
                snapshot_type = resolve_type_identifier(snapshot_type_str)
            except (TypeError, ValueError, ImportError) as error:  # pragma: no cover
                raise SnapshotRestoreError(
                    f"Unknown snapshot type: {snapshot_type_str}"
                ) from error

            try:
                resource_snapshot = parse(
                    snapshot_type,
                    snapshot_data,
                    allow_dataclass_type=True,
                    type_key=TYPE_REF_KEY,
                )
            except Exception as error:  # pragma: no cover - defensive
                raise SnapshotRestoreError(
                    f"Failed to parse resource snapshot for {resource_type_str}"
                ) from error

            resources[resource_type] = resource_snapshot

        # Parse metadata
        metadata_payload = payload.get("metadata")
        metadata: SnapshotMetadata | None = None
        if metadata_payload is not None:  # pragma: no branch - tested separately
            if not isinstance(metadata_payload, Mapping):
                raise SnapshotRestoreError("Metadata must be an object")

            metadata_dict = cast(Mapping[str, JSONValue], metadata_payload)
            tag = metadata_dict.get("tag")
            tool_call_id = metadata_dict.get("tool_call_id")
            tool_name = metadata_dict.get("tool_name")
            phase = metadata_dict.get("phase", "manual")

            if tag is not None and not isinstance(tag, str):  # pragma: no cover
                raise SnapshotRestoreError("Metadata tag must be a string")
            if tool_call_id is not None and not isinstance(
                tool_call_id, str
            ):  # pragma: no cover
                raise SnapshotRestoreError("Metadata tool_call_id must be a string")
            if tool_name is not None and not isinstance(
                tool_name, str
            ):  # pragma: no cover
                raise SnapshotRestoreError("Metadata tool_name must be a string")
            if phase not in {"pre_tool", "post_tool", "manual"}:
                raise SnapshotRestoreError("Metadata phase must be valid")

            metadata = SnapshotMetadata(
                tag=tag,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                phase=cast(Literal["pre_tool", "post_tool", "manual"], phase),
            )

        return cls(
            snapshot_id=snapshot_id,
            created_at=created_at,
            session=session_snapshot,
            resources=types.MappingProxyType(resources),
            metadata=metadata,
        )


@dataclass(slots=True, frozen=True)
class PendingToolExecution:
    """Metadata for an in-flight native tool execution.

    Stored by PendingToolTracker between begin_tool_execution()
    and end_tool_execution() calls.

    Attributes:
        tool_use_id: Unique identifier for this tool invocation.
        tool_name: Name of the tool being executed.
        snapshot: Composite snapshot taken before tool execution.
        started_at: Timestamp when tool execution began.
    """

    tool_use_id: str
    tool_name: str
    snapshot: CompositeSnapshot
    started_at: datetime


def create_snapshot(
    session: SessionProtocol,
    resources: ScopedResourceContext,
    *,
    tag: str | None = None,
) -> CompositeSnapshot:
    """Capture consistent snapshot of session and all snapshotable resources.

    Takes a point-in-time snapshot of the session state and all registered
    resources that implement the Snapshotable protocol.

    Args:
        session: Session to snapshot.
        resources: Resource context containing snapshotable resources.
        tag: Optional human-readable label for the snapshot.

    Returns:
        CompositeSnapshot containing session and resource snapshots.
    """
    resource_snapshots: dict[type[object], object] = {}

    for resource_type, resource in resources.singleton_cache.items():
        if isinstance(resource, Snapshotable):
            resource_snapshots[resource_type] = resource.snapshot(tag=tag)

    session_snapshot = session.snapshot()

    return CompositeSnapshot(
        snapshot_id=uuid4(),
        created_at=datetime.now(UTC),
        session=session_snapshot,
        resources=types.MappingProxyType(resource_snapshots),
        metadata=SnapshotMetadata(tag=tag),
    )


def restore_snapshot(
    session: SessionProtocol,
    resources: ScopedResourceContext,
    snapshot: CompositeSnapshot,
) -> None:
    """Restore session and all snapshotable resources from a composite snapshot.

    Restores the session state first, then restores each snapshotable resource.
    If any restore operation fails, a RestoreFailedError is raised.

    Args:
        session: Session to restore.
        resources: Resource context containing snapshotable resources.
        snapshot: The composite snapshot to restore from.

    Raises:
        RestoreFailedError: If restoring any component fails.
    """
    # Restore session first
    try:
        session.restore(snapshot.session)
    except SnapshotRestoreError as error:
        raise RestoreFailedError(f"Failed to restore session: {error}") from error

    # Restore each snapshotable resource
    for resource_type, resource_snapshot in snapshot.resources.items():
        resource = resources.singleton_cache.get(resource_type)
        if resource is None:
            continue

        if isinstance(resource, Snapshotable):  # pragma: no branch - tested separately
            try:
                resource.restore(resource_snapshot)
            except Exception as error:
                raise RestoreFailedError(
                    f"Failed to restore {resource_type.__qualname__}: {error}"
                ) from error


@contextmanager
def tool_transaction(
    session: SessionProtocol,
    resources: ScopedResourceContext,
    *,
    tag: str | None = None,
) -> Iterator[CompositeSnapshot]:
    """Context manager for transactional tool execution.

    Takes a snapshot before the block executes. On any exception, the
    snapshot is automatically restored before re-raising. For explicit
    failure handling (e.g., `result.success == False`), the caller can
    restore manually using the yielded snapshot.

    Example usage::

        # Pass prompt.resources.context (the ScopedResourceContext)
        with tool_transaction(session, prompt.resources.context, tag="my_tool") as snapshot:
            result = execute_tool(...)
            if not result.success:
                restore_snapshot(session, prompt.resources.context, snapshot)
            return result

    Args:
        session: Session to snapshot and potentially restore.
        resources: Resource context containing snapshotable resources.
        tag: Optional human-readable label for the snapshot.

    Yields:
        CompositeSnapshot that can be used for manual restoration.

    Raises:
        Any exception from the block, after state restoration.
    """
    snapshot = create_snapshot(session, resources, tag=tag)
    try:
        yield snapshot
    except Exception:
        restore_snapshot(session, resources, snapshot)
        raise


@dataclass(slots=True)
class PendingToolTracker:
    """Tracks pending tool executions for hook-based transaction management.

    Used by hooks to manage tool execution state across pre_tool_use and
    post_tool_use hook calls. Provides thread-safe begin/end/abort methods.

    Attributes:
        session: Session for snapshot/restore operations.
        resources: Resource context for snapshot/restore operations.
    """

    session: SessionProtocol
    resources: ScopedResourceContext
    _pending_tools: dict[str, PendingToolExecution] = field(
        default_factory=lambda: {}, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def begin_tool_execution(
        self,
        tool_use_id: str,
        tool_name: str,
    ) -> None:
        """Take snapshot before native tool execution.

        Called by pre_tool_use_hook before a native tool runs.
        Stores the snapshot internally for potential rollback in end_tool_execution.

        Thread Safety:
            This method acquires the internal lock to atomically capture
            the snapshot and register the pending execution.

        Args:
            tool_use_id: Unique identifier for this tool invocation.
            tool_name: Name of the tool being executed.
        """
        with self._lock:
            snapshot = create_snapshot(
                self.session, self.resources, tag=f"pre:{tool_name}:{tool_use_id}"
            )
            self._pending_tools[tool_use_id] = PendingToolExecution(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                snapshot=snapshot,
                started_at=datetime.now(UTC),
            )

    def end_tool_execution(
        self,
        tool_use_id: str,
        *,
        success: bool,
    ) -> bool:
        """Complete tool execution, restoring on failure.

        Called by post_tool_use_hook after a native tool completes.
        If success is False, automatically restores state from the pre-execution
        snapshot.

        Thread Safety:
            This method acquires the internal lock to atomically pop the
            pending execution and optionally restore state.

        Args:
            tool_use_id: Unique identifier for this tool invocation.
            success: Whether the tool execution succeeded.

        Returns:
            True if state was restored (i.e., tool failed), False otherwise.
        """
        with self._lock:
            pending = self._pending_tools.pop(tool_use_id, None)
            if pending is None:
                return False

            if not success:
                restore_snapshot(self.session, self.resources, pending.snapshot)
                return True

            return False

    def abort_tool_execution(self, tool_use_id: str) -> bool:
        """Abort tool execution and restore state.

        Used for timeouts, interrupts, or other abnormal termination.
        Always restores state from the pre-execution snapshot.

        Thread Safety:
            This method acquires the internal lock to atomically pop the
            pending execution and restore state.

        Args:
            tool_use_id: Unique identifier for this tool invocation.

        Returns:
            True if a pending execution was found and restored, False otherwise.
        """
        with self._lock:
            pending = self._pending_tools.pop(tool_use_id, None)
            if pending is None:
                return False

            restore_snapshot(self.session, self.resources, pending.snapshot)
            return True

    @property
    def pending_tool_executions(self) -> Mapping[str, PendingToolExecution]:
        """Read-only view of pending tool executions.

        Useful for debugging and monitoring in-flight tool calls.

        Thread Safety:
            Returns a snapshot copy wrapped in MappingProxyType under the lock
            to ensure a consistent view.
        """
        with self._lock:
            return types.MappingProxyType(dict(self._pending_tools))


__all__ = [
    "CompositeSnapshot",
    "PendingToolExecution",
    "PendingToolTracker",
    "SnapshotMetadata",
    "create_snapshot",
    "restore_snapshot",
    "tool_transaction",
]
