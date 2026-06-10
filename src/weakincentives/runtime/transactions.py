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

This module provides transactional semantics for tool execution over the
pair **(session, sandbox)**: a composite snapshot captures the session
slices and, when a sandbox is in play, a filesystem snapshot of the
sandbox. On tool failure both are restored atomically, so partial state
never leaks.

Example usage::

    from weakincentives.runtime.transactions import tool_transaction, restore_snapshot

    # Use as context manager for automatic rollback on exception
    with tool_transaction(session, sandbox, tag="my_tool") as snapshot:
        result = execute_tool(...)
        if not result.success:
            restore_snapshot(session, sandbox, snapshot)
        return result

For hook-based native tool execution, use the PendingToolTracker class to
manage snapshots across pre_tool_use and post_tool_use hooks.
"""

from __future__ import annotations

import json
import threading
import types
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from ..clock import SYSTEM_CLOCK
from ..dataclasses import FrozenDataclass
from ..errors import RestoreFailedError
from ..filesystem import SnapshotRef
from ..serde import dump, parse
from ..types import JSONValue
from .session.protocols import SessionProtocol
from .session.snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
)

if TYPE_CHECKING:
    from ..sandbox import Sandbox

# Schema version for composite snapshot serialization
COMPOSITE_SNAPSHOT_SCHEMA_VERSION = "2"


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
    """Consistent snapshot of the (session, sandbox) pair.

    CompositeSnapshot captures a point-in-time view of session slices and,
    when a sandbox participates in the transaction, a snapshot reference
    for the sandbox's filesystem state. This enables atomic rollback on
    tool failure.

    Attributes:
        snapshot_id: Unique identifier for this snapshot.
        created_at: Timestamp when the snapshot was taken.
        session: Snapshot of session slice state.
        sandbox: Snapshot reference for the sandbox, if one was captured.
        metadata: Optional context about when/why this snapshot was taken.
    """

    snapshot_id: UUID
    created_at: datetime
    session: Snapshot
    sandbox: SnapshotRef | None = None
    metadata: SnapshotMetadata | None = None

    def to_json(self) -> str:
        """Serialize the composite snapshot to a JSON string.

        Returns:
            JSON representation of the composite snapshot.

        Raises:
            SnapshotSerializationError: If serialization fails.
        """
        try:
            session_json = self.session.to_json()

            sandbox_payload: JSONValue = None
            if self.sandbox is not None:
                sandbox_payload = cast(JSONValue, dump(self.sandbox))

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
                "sandbox": sandbox_payload,
                "metadata": metadata_payload,
            }
            return json.dumps(payload, sort_keys=True)
        except Exception as error:  # pragma: no cover - defensive
            msg = "Failed to serialize composite snapshot"
            raise SnapshotSerializationError(msg) from error

    @classmethod
    def from_json(cls, raw: str) -> CompositeSnapshot:
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

        version = payload.get("version")
        if version != COMPOSITE_SNAPSHOT_SCHEMA_VERSION:
            msg = (
                f"Composite snapshot schema version mismatch: "
                f"expected {COMPOSITE_SNAPSHOT_SCHEMA_VERSION}, got {version!r}"
            )
            raise SnapshotRestoreError(msg)

        snapshot_id = _parse_snapshot_id(payload)
        created_at = _parse_created_at(payload)
        session_snapshot = _parse_session_snapshot(payload)
        sandbox_ref = _parse_sandbox_ref(payload)
        metadata = _parse_snapshot_metadata(payload)

        return cls(
            snapshot_id=snapshot_id,
            created_at=created_at,
            session=session_snapshot,
            sandbox=sandbox_ref,
            metadata=metadata,
        )


def _parse_snapshot_id(payload: Mapping[str, JSONValue]) -> UUID:
    snapshot_id_str = payload.get("snapshot_id")
    if not isinstance(snapshot_id_str, str):  # pragma: no cover - defensive
        raise SnapshotRestoreError("Composite snapshot snapshot_id must be a string")
    try:
        return UUID(snapshot_id_str)
    except ValueError as error:
        raise SnapshotRestoreError("Invalid snapshot_id") from error


def _parse_created_at(payload: Mapping[str, JSONValue]) -> datetime:
    created_at_str = payload.get("created_at")
    if not isinstance(created_at_str, str):  # pragma: no cover - defensive
        raise SnapshotRestoreError("Composite snapshot created_at must be a string")
    try:
        return datetime.fromisoformat(created_at_str)
    except ValueError as error:
        raise SnapshotRestoreError("Invalid created_at timestamp") from error


def _parse_session_snapshot(payload: Mapping[str, JSONValue]) -> Snapshot:
    session_payload = payload.get("session")
    if not isinstance(session_payload, Mapping):
        raise SnapshotRestoreError("Session snapshot must be an object")
    try:
        return Snapshot.from_json(json.dumps(session_payload))
    except Exception as error:  # pragma: no cover - defensive
        raise SnapshotRestoreError("Failed to parse session snapshot") from error


def _parse_sandbox_ref(payload: Mapping[str, JSONValue]) -> SnapshotRef | None:
    sandbox_payload = payload.get("sandbox")
    if sandbox_payload is None:
        return None
    if not isinstance(sandbox_payload, Mapping):
        raise SnapshotRestoreError("Sandbox snapshot must be an object")
    try:
        return parse(SnapshotRef, sandbox_payload)
    except Exception as error:
        raise SnapshotRestoreError("Failed to parse sandbox snapshot ref") from error


def _parse_snapshot_metadata(
    payload: Mapping[str, JSONValue],
) -> SnapshotMetadata | None:
    metadata_payload = payload.get("metadata")
    if metadata_payload is None:
        return None
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
    if tool_name is not None and not isinstance(tool_name, str):  # pragma: no cover
        raise SnapshotRestoreError("Metadata tool_name must be a string")
    if phase not in {"pre_tool", "post_tool", "manual"}:
        raise SnapshotRestoreError("Metadata phase must be valid")

    return SnapshotMetadata(
        tag=tag,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        phase=cast(Literal["pre_tool", "post_tool", "manual"], phase),
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
    sandbox: Sandbox | None = None,
    *,
    tag: str | None = None,
) -> CompositeSnapshot:
    """Capture a consistent snapshot of the (session, sandbox) pair.

    Takes a point-in-time snapshot of the session state and, when a
    sandbox is provided, of the sandbox's filesystem state.

    Args:
        session: Session to snapshot.
        sandbox: Sandbox participating in the transaction, if any.
        tag: Optional human-readable label for the snapshot.

    Returns:
        CompositeSnapshot containing the session and sandbox snapshots.
    """
    sandbox_ref = sandbox.snapshot(tag=tag) if sandbox is not None else None
    session_snapshot = session.snapshot()

    return CompositeSnapshot(
        snapshot_id=uuid4(),
        created_at=SYSTEM_CLOCK.utcnow(),
        session=session_snapshot,
        sandbox=sandbox_ref,
        metadata=SnapshotMetadata(tag=tag),
    )


def restore_snapshot(
    session: SessionProtocol,
    sandbox: Sandbox | None,
    snapshot: CompositeSnapshot,
) -> None:
    """Restore the (session, sandbox) pair from a composite snapshot.

    Restores the session state first, then the sandbox state when the
    snapshot captured one. If any restore operation fails, a
    RestoreFailedError is raised.

    Args:
        session: Session to restore.
        sandbox: Sandbox to restore, if one participated in the snapshot.
        snapshot: The composite snapshot to restore from.

    Raises:
        RestoreFailedError: If restoring any component fails.
    """
    try:
        session.restore(snapshot.session)
    except SnapshotRestoreError as error:
        raise RestoreFailedError(f"Failed to restore session: {error}") from error

    if snapshot.sandbox is not None and sandbox is not None:
        try:
            sandbox.restore(snapshot.sandbox)
        except Exception as error:
            raise RestoreFailedError(f"Failed to restore sandbox: {error}") from error


@contextmanager
def tool_transaction(
    session: SessionProtocol,
    sandbox: Sandbox | None = None,
    *,
    tag: str | None = None,
) -> Generator[CompositeSnapshot]:
    """Context manager for transactional tool execution.

    Takes a snapshot before the block executes. On any exception, the
    snapshot is automatically restored before re-raising. For explicit
    failure handling (e.g., `result.success == False`), the caller can
    restore manually using the yielded snapshot.

    Example usage::

        with tool_transaction(session, sandbox, tag="my_tool") as snapshot:
            result = execute_tool(...)
            if not result.success:
                restore_snapshot(session, sandbox, snapshot)
            return result

    Args:
        session: Session to snapshot and potentially restore.
        sandbox: Sandbox participating in the transaction, if any.
        tag: Optional human-readable label for the snapshot.

    Yields:
        CompositeSnapshot that can be used for manual restoration.

    Raises:
        Any exception from the block, after state restoration.
    """
    snapshot = create_snapshot(session, sandbox, tag=tag)
    try:
        yield snapshot
    except Exception:
        restore_snapshot(session, sandbox, snapshot)
        raise


@dataclass(slots=True)
class PendingToolTracker:
    """Tracks pending tool executions for hook-based transaction management.

    Used by hooks to manage tool execution state across pre_tool_use and
    post_tool_use hook calls. Provides thread-safe begin/end/abort methods.

    Attributes:
        session: Session for snapshot/restore operations.
        sandbox: Sandbox participating in transactions, if any.
    """

    session: SessionProtocol
    sandbox: Sandbox | None = None
    _pending_tools: dict[str, PendingToolExecution] = field(
        default_factory=dict[str, PendingToolExecution], repr=False
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
                self.session, self.sandbox, tag=f"pre:{tool_name}:{tool_use_id}"
            )
            self._pending_tools[tool_use_id] = PendingToolExecution(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                snapshot=snapshot,
                started_at=SYSTEM_CLOCK.utcnow(),
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
                restore_snapshot(self.session, self.sandbox, pending.snapshot)
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

            restore_snapshot(self.session, self.sandbox, pending.snapshot)
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
