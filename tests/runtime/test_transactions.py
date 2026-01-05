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

"""Tests for transactions module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.errors import RestoreFailedError
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.snapshots import Snapshot, SnapshotRestoreError
from weakincentives.runtime.snapshotable import Snapshotable
from weakincentives.runtime.transactions import (
    CompositeSnapshot,
    PendingToolTracker,
    create_snapshot,
    restore_snapshot,
    tool_transaction,
)


def _make_prompt_with_fs(fs: InMemoryFilesystem) -> Prompt[object]:
    """Create a prompt with filesystem bound in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="transactions-test"))
    prompt = prompt.bind(resources={Filesystem: fs})
    prompt.__enter__()
    return prompt


def _make_prompt() -> Prompt[object]:
    """Create a prompt in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="transactions-test"))
    prompt.__enter__()
    return prompt


class TestCompositeSnapshotSerialization:
    """Tests for CompositeSnapshot.to_json() and from_json() methods."""

    def test_to_json_basic(self) -> None:
        """Test serializing a composite snapshot to JSON."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "content")
        prompt = _make_prompt_with_fs(fs)

        snapshot = create_snapshot(session, prompt.resources, tag="test")
        json_str = snapshot.to_json()

        # Verify it's valid JSON
        payload = json.loads(json_str)
        assert payload["version"] == "1"
        assert "snapshot_id" in payload
        assert "created_at" in payload
        assert "session" in payload
        assert "resources" in payload
        assert payload["metadata"]["tag"] == "test"

    def test_to_json_without_resources(self) -> None:
        """Test serializing a snapshot without snapshotable resources."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        snapshot = create_snapshot(session, prompt.resources, tag="no-resources")
        json_str = snapshot.to_json()

        payload = json.loads(json_str)
        assert payload["resources"] == []
        assert payload["metadata"]["tag"] == "no-resources"

    def test_to_json_without_metadata(self) -> None:
        """Test serializing a snapshot without metadata."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        # Create snapshot without tag to have minimal metadata
        snapshot = create_snapshot(session, prompt.resources)
        json_str = snapshot.to_json()

        payload = json.loads(json_str)
        assert payload["metadata"] is not None  # Still has metadata with None tag

    def test_from_json_basic(self) -> None:
        """Test deserializing a composite snapshot from JSON."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        # Create and serialize
        original = create_snapshot(session, prompt.resources, tag="roundtrip")
        json_str = original.to_json()

        # Deserialize
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
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
            "version": "1",
            "snapshot_id": "not-a-uuid",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        with pytest.raises(SnapshotRestoreError, match="Invalid snapshot_id"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_created_at(self) -> None:
        """Test that invalid created_at raises SnapshotRestoreError."""
        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "not-a-timestamp",
        }
        with pytest.raises(SnapshotRestoreError, match="Invalid created_at"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_session(self) -> None:
        """Test that invalid session raises SnapshotRestoreError."""
        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": "not an object",
        }
        with pytest.raises(
            SnapshotRestoreError, match="Session snapshot must be an object"
        ):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_resources_not_list(self) -> None:
        """Test that resources not being a list raises SnapshotRestoreError."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session_snapshot = session.snapshot()

        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "resources": "not a list",
        }
        with pytest.raises(SnapshotRestoreError, match="Resources must be a list"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_resource_entry(self) -> None:
        """Test that invalid resource entry raises SnapshotRestoreError."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session_snapshot = session.snapshot()

        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "resources": ["not an object"],
        }
        with pytest.raises(
            SnapshotRestoreError, match="Resource entry must be an object"
        ):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_metadata_not_object(self) -> None:
        """Test that invalid metadata raises SnapshotRestoreError."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session_snapshot = session.snapshot()

        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "resources": [],
            "metadata": "not an object",
        }
        with pytest.raises(SnapshotRestoreError, match="Metadata must be an object"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_from_json_invalid_metadata_phase(self) -> None:
        """Test that invalid metadata phase raises SnapshotRestoreError."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session_snapshot = session.snapshot()

        payload = {
            "version": "1",
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "created_at": "2024-01-01T00:00:00+00:00",
            "session": json.loads(session_snapshot.to_json()),
            "resources": [],
            "metadata": {"tag": None, "phase": "invalid_phase"},
        }
        with pytest.raises(SnapshotRestoreError, match="Metadata phase must be valid"):
            CompositeSnapshot.from_json(json.dumps(payload))

    def test_roundtrip_with_filesystem_resource(self) -> None:
        """Test full roundtrip serialization with filesystem resource."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        # Create and serialize
        original = create_snapshot(session, prompt.resources, tag="roundtrip")
        json_str = original.to_json()

        # Deserialize
        restored = CompositeSnapshot.from_json(json_str)

        # Verify all fields match
        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert len(restored.resources) == len(original.resources)


class TestRestoreSnapshotErrors:
    """Tests for error handling in restore_snapshot()."""

    def test_restore_handles_session_restore_failure(self) -> None:
        """Test that session restore failure raises RestoreFailedError."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        snapshot = create_snapshot(session, prompt.resources, tag="test")

        # Create a corrupted session snapshot
        class FailingSession:
            def restore(self, snapshot: Snapshot) -> None:
                raise SnapshotRestoreError("Session restore failed")

        with pytest.raises(RestoreFailedError, match="Failed to restore session"):
            restore_snapshot(
                FailingSession(),  # type: ignore[arg-type]
                prompt.resources,
                snapshot,
            )

    def test_restore_skips_missing_resources(self) -> None:
        """Test that restore skips resources not in current context."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt_with_fs = _make_prompt_with_fs(fs)

        # Create snapshot with filesystem resource
        snapshot = create_snapshot(session, prompt_with_fs.resources, tag="test")

        # Create a new prompt without filesystem
        prompt_without_fs = _make_prompt()

        # Should not raise - silently skips missing resources
        restore_snapshot(session, prompt_without_fs.resources, snapshot)

    def test_restore_handles_resource_restore_failure(self) -> None:
        """Test that resource restore failure raises RestoreFailedError."""

        class FailingFilesystem(Snapshotable):
            """Filesystem that fails on restore."""

            def snapshot(self, *, tag: str | None = None) -> Any:  # noqa: ANN401
                return {"state": "snapshot"}

            def restore(self, snapshot: Any) -> None:  # noqa: ANN401
                raise SnapshotRestoreError("Restore failed!")

        bus = InProcessDispatcher()
        session = Session(bus=bus)

        # Create prompt with failing filesystem
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="tests", key="failing-fs-test")
        )
        prompt = prompt.bind(resources={Filesystem: FailingFilesystem()})
        prompt.__enter__()

        snapshot = create_snapshot(session, prompt.resources, tag="test")

        with pytest.raises(RestoreFailedError, match="Failed to restore"):
            restore_snapshot(session, prompt.resources, snapshot)


class TestPendingToolTracker:
    """Tests for PendingToolTracker class."""

    def test_abort_tool_execution_restores_state(self) -> None:
        """Test that abort_tool_execution restores state."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        fs.write("/test.txt", "modified")
        assert fs.read("/test.txt").content == "modified"

        # Abort should restore
        result = tracker.abort_tool_execution("call-1")
        assert result is True
        assert fs.read("/test.txt").content == "original"

    def test_abort_tool_execution_unknown_id_returns_false(self) -> None:
        """Test that aborting unknown tool returns False."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

        result = tracker.abort_tool_execution("unknown-id")
        assert result is False

    def test_pending_tool_executions_property(self) -> None:
        """Test the pending_tool_executions property."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

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
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

        result = tracker.end_tool_execution("unknown-id", success=True)
        assert result is False

    def test_end_tool_execution_restores_on_failure(self) -> None:
        """Test that end_tool_execution restores state on failure."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        fs.write("/test.txt", "modified")

        # End with failure should restore
        result = tracker.end_tool_execution("call-1", success=False)
        assert result is True
        assert fs.read("/test.txt").content == "original"

    def test_end_tool_execution_preserves_on_success(self) -> None:
        """Test that end_tool_execution preserves state on success."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        tracker = PendingToolTracker(session=session, resources=prompt.resources)

        # Begin tool execution
        tracker.begin_tool_execution("call-1", "my_tool")

        # Modify state
        fs.write("/test.txt", "modified")

        # End with success should preserve changes
        result = tracker.end_tool_execution("call-1", success=True)
        assert result is False
        assert fs.read("/test.txt").content == "modified"


class TestToolTransaction:
    """Tests for tool_transaction context manager."""

    def test_restores_on_exception(self) -> None:
        """Test that tool_transaction restores state on exception."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        with pytest.raises(RuntimeError, match="Tool failed"):
            with tool_transaction(session, prompt.resources, tag="failing"):
                fs.write("/test.txt", "modified")
                raise RuntimeError("Tool failed")

        assert fs.read("/test.txt").content == "original"

    def test_preserves_on_success(self) -> None:
        """Test that tool_transaction preserves state on success."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        with tool_transaction(session, prompt.resources, tag="success"):
            fs.write("/test.txt", "modified")

        assert fs.read("/test.txt").content == "modified"

    def test_yields_snapshot_for_manual_restore(self) -> None:
        """Test that tool_transaction yields snapshot for manual restore."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")
        prompt = _make_prompt_with_fs(fs)

        with tool_transaction(session, prompt.resources, tag="manual") as snapshot:
            fs.write("/test.txt", "modified")
            assert fs.read("/test.txt").content == "modified"

            # Manual restore
            restore_snapshot(session, prompt.resources, snapshot)
            assert fs.read("/test.txt").content == "original"


class TestCompositeSnapshotErrors:
    """Tests for CompositeSnapshot error handling paths."""

    def test_snapshot_with_metadata_roundtrip(self) -> None:
        """Test that snapshot with metadata serializes and deserializes."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        prompt = _make_prompt()

        snapshot = create_snapshot(
            session,
            prompt.resources,
            tag="test-tag",
        )

        # Verify metadata was set
        assert snapshot.metadata is not None
        assert snapshot.metadata.tag == "test-tag"

        # Roundtrip
        json_str = snapshot.to_json()
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.metadata is not None
        assert restored.metadata.tag == "test-tag"


class TestRestoreSnapshotEdgeCases:
    """Tests for restore_snapshot edge cases."""

    def test_restore_skips_non_snapshotable_resources(self) -> None:
        """Test that restore_snapshot skips resources without Snapshotable."""
        bus = InProcessDispatcher()
        session = Session(bus=bus)

        # Create a simple non-snapshotable resource
        class SimpleResource:
            def __init__(self) -> None:
                self.value = "original"

        resource = SimpleResource()
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="tests", key="non-snapshotable-test")
        )
        prompt = prompt.bind(resources={SimpleResource: resource})
        prompt.__enter__()

        # Create snapshot (won't include non-snapshotable resource)
        snapshot = create_snapshot(session, prompt.resources, tag="test")

        # Modify the resource
        resource.value = "modified"

        # Restore should not fail (just skip the non-snapshotable resource)
        restore_snapshot(session, prompt.resources, snapshot)

        # Resource should still be modified since it wasn't snapshotable
        assert resource.value == "modified"
