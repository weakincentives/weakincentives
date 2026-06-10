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

"""Tests for transactions over the (session, sandbox) pair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weakincentives.errors import RestoreFailedError
from weakincentives.filesystem import Filesystem
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.snapshots import Snapshot, SnapshotRestoreError
from weakincentives.runtime.transactions import (
    CompositeSnapshot,
    PendingToolTracker,
    create_snapshot,
    restore_snapshot,
    tool_transaction,
)
from weakincentives.sandbox import LocalSandbox, LocalShell, Sandbox


def _make_session() -> Session:
    return Session(dispatcher=InProcessDispatcher())


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    """In-memory sandbox: fast snapshot/restore without git plumbing."""
    return LocalSandbox(
        root=tmp_path,
        filesystem=Filesystem.in_memory(),
        shell=LocalShell(tmp_path),
    )


class TestCompositeSnapshotSerialization:
    """Tests for CompositeSnapshot.to_json() and from_json() methods."""

    def test_to_json_basic(self, sandbox: Sandbox) -> None:
        """Test serializing a composite snapshot to JSON."""
        session = _make_session()
        _ = sandbox.filesystem.write("test.txt", "content")

        snapshot = create_snapshot(session, sandbox, tag="test")
        json_str = snapshot.to_json()

        # Verify it's valid JSON
        payload = json.loads(json_str)
        assert payload["version"] == "2"
        assert "snapshot_id" in payload
        assert "created_at" in payload
        assert "session" in payload
        assert payload["sandbox"]["token"] == snapshot.sandbox.token  # type: ignore[union-attr]
        assert payload["metadata"]["tag"] == "test"

    def test_to_json_without_sandbox(self) -> None:
        """Test serializing a snapshot without a sandbox."""
        session = _make_session()

        snapshot = create_snapshot(session, tag="no-sandbox")
        json_str = snapshot.to_json()

        payload = json.loads(json_str)
        assert payload["sandbox"] is None
        assert payload["metadata"]["tag"] == "no-sandbox"

    def test_to_json_without_metadata(self) -> None:
        """Test serializing a snapshot without a tag."""
        session = _make_session()

        snapshot = create_snapshot(session)
        json_str = snapshot.to_json()

        payload = json.loads(json_str)
        assert payload["metadata"] is not None  # Still has metadata with None tag

    def test_from_json_basic(self, sandbox: Sandbox) -> None:
        """Test deserializing a composite snapshot from JSON."""
        session = _make_session()
        _ = sandbox.filesystem.write("test.txt", "original")

        # Create and serialize
        original = create_snapshot(session, sandbox, tag="roundtrip")
        json_str = original.to_json()

        # Deserialize
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert restored.sandbox == original.sandbox
        assert restored.metadata is not None
        assert restored.metadata.tag == "roundtrip"

    def test_from_json_invalid_json(self) -> None:
        """Test that invalid JSON raises SnapshotRestoreError."""
        with pytest.raises(
            SnapshotRestoreError, match="Invalid composite snapshot JSON"
        ):
            CompositeSnapshot.from_json("not valid json")

    def test_from_json_not_object(self) -> None:
        """Test that non-object JSON raises SnapshotRestoreError."""
        with pytest.raises(SnapshotRestoreError, match="must be an object"):
            CompositeSnapshot.from_json('"just a string"')

    def test_from_json_wrong_version(self) -> None:
        """Test that wrong schema version raises SnapshotRestoreError."""
        payload = {"version": "99", "snapshot_id": "123", "created_at": "2024-01-01"}
        with pytest.raises(SnapshotRestoreError, match="schema version mismatch"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_snapshot_id(self) -> None:
        """Test that invalid snapshot_id raises SnapshotRestoreError."""
        payload = {
            "version": "2",
            "snapshot_id": "not-a-uuid",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        with pytest.raises(SnapshotRestoreError, match="Invalid snapshot_id"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_created_at(self) -> None:
        """Test that invalid created_at raises SnapshotRestoreError."""
        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "not-a-timestamp",
        }
        with pytest.raises(SnapshotRestoreError, match="Invalid created_at"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_session(self) -> None:
        """Test that invalid session raises SnapshotRestoreError."""
        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": "not an object",
        }
        with pytest.raises(
            SnapshotRestoreError, match="Session snapshot must be an object"
        ):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_sandbox_not_object(self) -> None:
        """Test that a non-object sandbox payload raises SnapshotRestoreError."""
        session = _make_session()
        session_snapshot = session.snapshot()

        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "sandbox": "not an object",
        }
        with pytest.raises(
            SnapshotRestoreError, match="Sandbox snapshot must be an object"
        ):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_sandbox_payload(self) -> None:
        """Test that an unparsable sandbox ref raises SnapshotRestoreError."""
        session = _make_session()
        session_snapshot = session.snapshot()

        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "sandbox": {"unexpected": "shape"},
        }
        with pytest.raises(
            SnapshotRestoreError, match="Failed to parse sandbox snapshot ref"
        ):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_no_metadata(self) -> None:
        """Test deserializing a snapshot with no metadata key."""
        session = _make_session()
        session_snapshot = session.snapshot()

        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "sandbox": None,
        }
        restored = CompositeSnapshot.from_json(json.dumps(payload))
        assert restored.metadata is None
        assert restored.sandbox is None

    def test_from_json_invalid_metadata_not_object(self) -> None:
        """Test that invalid metadata raises SnapshotRestoreError."""
        session = _make_session()
        session_snapshot = session.snapshot()

        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "sandbox": None,
            "metadata": "not an object",
        }
        with pytest.raises(SnapshotRestoreError, match="Metadata must be an object"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_metadata_phase(self) -> None:
        """Test that invalid metadata phase raises SnapshotRestoreError."""
        session = _make_session()
        session_snapshot = session.snapshot()

        payload = {
            "version": "2",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "sandbox": None,
            "metadata": {"tag": None, "phase": "invalid_phase"},
        }
        with pytest.raises(SnapshotRestoreError, match="Metadata phase must be valid"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_roundtrip_with_sandbox(self, sandbox: Sandbox) -> None:
        """Test full roundtrip serialization with a sandbox snapshot ref."""
        session = _make_session()
        _ = sandbox.filesystem.write("test.txt", "original")

        original = create_snapshot(session, sandbox, tag="roundtrip")
        json_str = original.to_json()

        restored = CompositeSnapshot.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert restored.sandbox == original.sandbox


class TestRestoreSnapshotErrors:
    """Tests for error handling in restore_snapshot()."""

    def test_restore_handles_session_restore_failure(self) -> None:
        """Test that session restore failure raises RestoreFailedError."""
        session = _make_session()

        snapshot = create_snapshot(session, tag="test")

        class FailingSession:
            def restore(self, snapshot: Snapshot) -> None:
                raise SnapshotRestoreError("Session restore failed")

        with pytest.raises(RestoreFailedError, match="Failed to restore session"):
            restore_snapshot(
                FailingSession(),  # type: ignore[arg-type]
                None,
                snapshot,
            )

    def test_restore_skips_sandbox_when_none_present(self, sandbox: Sandbox) -> None:
        """A sandbox snapshot is skipped when no sandbox is supplied."""
        session = _make_session()
        _ = sandbox.filesystem.write("test.txt", "original")

        snapshot = create_snapshot(session, sandbox, tag="test")
        _ = sandbox.filesystem.write("test.txt", "modified")

        # Should not raise - silently skips the sandbox restore
        restore_snapshot(session, None, snapshot)
        assert sandbox.filesystem.read("test.txt").content == "modified"

    def test_restore_handles_sandbox_restore_failure(self, sandbox: Sandbox) -> None:
        """Test that sandbox restore failure raises RestoreFailedError."""
        session = _make_session()
        snapshot = create_snapshot(session, sandbox, tag="test")

        class FailingSandbox:
            def restore(self, ref: object) -> None:
                raise SnapshotRestoreError("Restore failed!")

        with pytest.raises(RestoreFailedError, match="Failed to restore sandbox"):
            restore_snapshot(
                session,
                FailingSandbox(),  # type: ignore[arg-type]
                snapshot,
            )


class TestPendingToolTracker:
    """Tests for PendingToolTracker class."""

    def test_abort_tool_execution_restores_state(self, sandbox: Sandbox) -> None:
        """Test that abort_tool_execution restores state."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        tracker = PendingToolTracker(session=session, sandbox=sandbox)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        _ = fs.write("test.txt", "modified")
        assert fs.read("test.txt").content == "modified"

        # Abort should restore
        result = tracker.abort_tool_execution("call-1")
        assert result is True
        assert fs.read("test.txt").content == "original"

    def test_abort_tool_execution_unknown_id_returns_false(self) -> None:
        """Test that aborting unknown tool returns False."""
        session = _make_session()

        tracker = PendingToolTracker(session=session)

        result = tracker.abort_tool_execution("unknown-id")
        assert result is False

    def test_pending_tool_executions_property(self) -> None:
        """Test the pending_tool_executions property."""
        session = _make_session()

        tracker = PendingToolTracker(session=session)

        # Initially empty
        assert len(tracker.pending_tool_executions) == 0

        # Add some pending executions
        tracker.begin_tool_execution("call-1", "tool_a")
        tracker.begin_tool_execution("call-2", "tool_b")

        pending = tracker.pending_tool_executions
        assert len(pending) == 2
        assert "call-1" in pending
        assert "call-2" in pending
        assert pending["call-1"].tool_name == "tool_a"
        assert pending["call-2"].tool_name == "tool_b"

        # Property should be read-only (MappingProxyType)
        with pytest.raises(TypeError):
            pending["call-3"] = None  # type: ignore[index]

    def test_end_tool_execution_returns_false_for_unknown(self) -> None:
        """Test that ending unknown tool returns False."""
        session = _make_session()

        tracker = PendingToolTracker(session=session)

        result = tracker.end_tool_execution("unknown-id", success=True)
        assert result is False

    def test_end_tool_execution_restores_on_failure(self, sandbox: Sandbox) -> None:
        """Test that end_tool_execution restores state on failure."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        tracker = PendingToolTracker(session=session, sandbox=sandbox)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        _ = fs.write("test.txt", "modified")

        # End with failure should restore
        result = tracker.end_tool_execution("call-1", success=False)
        assert result is True
        assert fs.read("test.txt").content == "original"

    def test_end_tool_execution_preserves_on_success(self, sandbox: Sandbox) -> None:
        """Test that end_tool_execution preserves state on success."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        tracker = PendingToolTracker(session=session, sandbox=sandbox)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        _ = fs.write("test.txt", "modified")

        # End with success should preserve changes
        result = tracker.end_tool_execution("call-1", success=True)
        assert result is False
        assert fs.read("test.txt").content == "modified"


class TestToolTransaction:
    """Tests for tool_transaction context manager."""

    def test_restores_on_exception(self, sandbox: Sandbox) -> None:
        """Test that tool_transaction restores state on exception."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        with pytest.raises(RuntimeError, match="Tool failed"):
            with tool_transaction(session, sandbox, tag="failing"):
                _ = fs.write("test.txt", "modified")
                raise RuntimeError("Tool failed")

        assert fs.read("test.txt").content == "original"

    def test_preserves_on_success(self, sandbox: Sandbox) -> None:
        """Test that tool_transaction preserves state on success."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        with tool_transaction(session, sandbox, tag="success"):
            _ = fs.write("test.txt", "modified")

        assert fs.read("test.txt").content == "modified"

    def test_yields_snapshot_for_manual_restore(self, sandbox: Sandbox) -> None:
        """Test that tool_transaction yields snapshot for manual restore."""
        session = _make_session()
        fs = sandbox.filesystem
        _ = fs.write("test.txt", "original")

        with tool_transaction(session, sandbox, tag="manual") as snapshot:
            _ = fs.write("test.txt", "modified")
            assert fs.read("test.txt").content == "modified"

            # Manual restore
            restore_snapshot(session, sandbox, snapshot)
            assert fs.read("test.txt").content == "original"

    def test_session_only_transaction(self) -> None:
        """A transaction without a sandbox snapshots the session alone."""
        session = _make_session()

        with tool_transaction(session, tag="session-only") as snapshot:
            assert snapshot.sandbox is None


class TestCompositeSnapshotErrors:
    """Tests for CompositeSnapshot error handling paths."""

    def test_snapshot_with_metadata_roundtrip(self) -> None:
        """Test that snapshot with metadata serializes and deserializes."""
        session = _make_session()

        snapshot = create_snapshot(session, tag="test-tag")

        # Verify metadata was set
        assert snapshot.metadata is not None
        assert snapshot.metadata.tag == "test-tag"

        # Roundtrip
        json_str = snapshot.to_json()
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.metadata is not None
        assert restored.metadata.tag == "test-tag"
