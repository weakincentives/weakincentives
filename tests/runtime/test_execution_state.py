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

"""Tests for ExecutionState and related classes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.filesystem import Filesystem
from weakincentives.resources import ResourceRegistry
from weakincentives.runtime.execution_state import (
    CompositeSnapshot,
    ExecutionState,
    SnapshotMetadata,
)
from weakincentives.runtime.session import Session, Snapshot
from weakincentives.runtime.session.snapshots import SnapshotRestoreError
from weakincentives.runtime.snapshotable import Snapshotable

# --- Test Data ---


@dataclass(frozen=True)
class SamplePlan:
    """Test dataclass for session state."""

    objective: str
    status: str = "active"


# --- SnapshotMetadata Tests ---


class TestSnapshotMetadata:
    def test_default_values(self) -> None:
        metadata = SnapshotMetadata()
        assert metadata.tag is None
        assert metadata.tool_call_id is None
        assert metadata.tool_name is None
        assert metadata.phase == "manual"

    def test_custom_values(self) -> None:
        metadata = SnapshotMetadata(
            tag="before-edit",
            tool_call_id="call_123",
            tool_name="edit_file",
            phase="pre_tool",
        )
        assert metadata.tag == "before-edit"
        assert metadata.tool_call_id == "call_123"
        assert metadata.tool_name == "edit_file"
        assert metadata.phase == "pre_tool"


# --- CompositeSnapshot Tests ---


class TestCompositeSnapshot:
    def test_create_with_defaults(self) -> None:
        session = Session()
        session_snapshot = session.snapshot()

        composite = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
        )

        assert composite.snapshot_id == UUID("12345678-1234-5678-1234-567812345678")
        assert len(composite.resources) == 0
        assert composite.metadata is None

    def test_create_with_metadata(self) -> None:
        session = Session()
        session_snapshot = session.snapshot()
        metadata = SnapshotMetadata(tag="test", phase="pre_tool")

        composite = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
            metadata=metadata,
        )

        assert composite.metadata is not None
        assert composite.metadata.tag == "test"
        assert composite.metadata.phase == "pre_tool"

    def test_to_json_roundtrip(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test")])
        session_snapshot = session.snapshot()

        original = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            session=session_snapshot,
            metadata=SnapshotMetadata(
                tag="test-tag",
                tool_call_id="call_123",
                tool_name="edit",
                phase="pre_tool",
            ),
        )

        json_str = original.to_json()
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert restored.metadata is not None
        assert restored.metadata.tag == "test-tag"
        assert restored.metadata.tool_call_id == "call_123"
        assert restored.metadata.tool_name == "edit"
        assert restored.metadata.phase == "pre_tool"

    def test_from_json_rejects_invalid_json(self) -> None:
        with pytest.raises(SnapshotRestoreError):
            CompositeSnapshot.from_json("not-json")

    def test_from_json_rejects_non_object(self) -> None:
        with pytest.raises(SnapshotRestoreError):
            CompositeSnapshot.from_json('"string"')

    def test_from_json_rejects_wrong_version(self) -> None:
        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        # Modify version
        import json

        data = json.loads(json_str)
        data["version"] = "999"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)

        assert "version mismatch" in str(exc_info.value)


# --- ExecutionState Tests ---


class TestExecutionState:
    def test_create_minimal(self) -> None:
        session = Session()
        state = ExecutionState(session=session)

        assert state.session is session
        assert isinstance(state.resources, ResourceRegistry)

    def test_create_with_resources(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})

        state = ExecutionState(session=session, resources=resources)

        assert state.resources.get(Filesystem) is fs

    def test_snapshot_captures_session(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="original")])
        state = ExecutionState(session=session)

        snapshot = state.snapshot(tag="test")

        assert snapshot.metadata is not None
        assert snapshot.metadata.tag == "test"
        assert isinstance(snapshot.session, Snapshot)

    def test_snapshot_captures_filesystem(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "hello")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot(tag="with-fs")

        assert Filesystem in snapshot.resources

    def test_restore_restores_session(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="original")])
        state = ExecutionState(session=session)

        snapshot = state.snapshot()

        # Modify session
        session[SamplePlan].seed([SamplePlan(objective="modified")])
        assert session[SamplePlan].latest().objective == "modified"

        # Restore
        state.restore(snapshot)

        assert session[SamplePlan].latest().objective == "original"

    def test_restore_restores_filesystem(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot()

        # Modify filesystem
        fs.write("test.txt", "modified")
        assert fs.read("test.txt").content == "modified"

        # Restore
        state.restore(snapshot)

        assert fs.read("test.txt").content == "original"

    def test_tool_failure_restores_state(self) -> None:
        """Tool failure does not change state - acceptance criteria."""
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test", status="active")])
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Capture pre-execution snapshot
        pre_snapshot = state.snapshot()

        # Simulate failed tool execution that modifies state
        fs.write("file.txt", "modified")
        session[SamplePlan].seed([SamplePlan(objective="changed")])

        # Restore on failure
        state.restore(pre_snapshot)

        assert fs.read("file.txt").content == "original"
        assert session[SamplePlan].latest().objective == "test"


# --- Snapshotable Protocol Tests ---


class TestSnapshotableProtocol:
    def test_in_memory_filesystem_is_snapshotable(self) -> None:
        fs = InMemoryFilesystem()
        assert isinstance(fs, Snapshotable)

    def test_resource_registry_finds_snapshotable_resources(self) -> None:
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})

        snapshotable = resources.get_all(lambda x: isinstance(x, Snapshotable))

        assert Filesystem in snapshotable
        assert snapshotable[Filesystem] is fs


# --- Integration Tests ---


class TestExecutionStateIntegration:
    def test_composite_snapshot_roundtrip_with_filesystem(self) -> None:
        """Composite snapshot includes both session and filesystem."""
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test")])
        fs = InMemoryFilesystem()
        fs.write("config.py", "DEBUG = True")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Snapshot
        snapshot = state.snapshot(tag="pre:tool_call")

        # Modify state
        session[SamplePlan].seed([SamplePlan(objective="changed")])
        fs.write("config.py", "DEBUG = False")

        # Restore
        state.restore(snapshot)

        assert session[SamplePlan].latest().objective == "test"
        assert fs.read("config.py").content == "DEBUG = True"

    def test_multiple_snapshots_isolated(self) -> None:
        """Multiple snapshots don't interfere with each other."""
        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # First state
        fs.write("file.txt", "v1")
        snapshot1 = state.snapshot()

        # Second state
        fs.write("file.txt", "v2")
        snapshot2 = state.snapshot()

        # Third state
        fs.write("file.txt", "v3")

        # Restore to first
        state.restore(snapshot1)
        assert fs.read("file.txt").content == "v1"

        # Restore to second
        state.restore(snapshot2)
        assert fs.read("file.txt").content == "v2"


# --- CompositeSnapshot Error Handling Tests ---


class TestCompositeSnapshotFromJsonErrors:
    """Tests for from_json error handling paths."""

    def test_from_json_rejects_invalid_snapshot_id_type(self) -> None:
        """snapshot_id must be a string."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["snapshot_id"] = 12345  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "snapshot_id must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_snapshot_id_uuid(self) -> None:
        """snapshot_id must be a valid UUID."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["snapshot_id"] = "not-a-uuid"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Invalid snapshot_id" in str(exc_info.value)

    def test_from_json_rejects_invalid_created_at_type(self) -> None:
        """created_at must be a string."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["created_at"] = 12345  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "created_at must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_created_at_timestamp(self) -> None:
        """created_at must be a valid timestamp."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["created_at"] = "not-a-timestamp"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Invalid created_at timestamp" in str(exc_info.value)

    def test_from_json_rejects_invalid_session_type(self) -> None:
        """session must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["session"] = "not-an-object"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Session snapshot must be an object" in str(exc_info.value)

    def test_from_json_rejects_invalid_session_parse(self) -> None:
        """Session snapshot must be parseable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["session"] = {"invalid": "session-data"}  # Invalid structure
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Failed to parse session snapshot" in str(exc_info.value)

    def test_from_json_rejects_non_list_resources(self) -> None:
        """resources must be a list."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = {"not": "a list"}
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resources must be a list" in str(exc_info.value)

    def test_from_json_rejects_non_object_resource_entry(self) -> None:
        """Resource entry must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = ["not-an-object"]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource entry must be an object" in str(exc_info.value)

    def test_from_json_rejects_missing_resource_type(self) -> None:
        """Resource entry must have resource_type."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [{"snapshot": {}}]  # Missing resource_type
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource type must be a string" in str(exc_info.value)

    def test_from_json_rejects_unknown_resource_type(self) -> None:
        """Resource type must be resolvable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {"resource_type": "unknown.module:UnknownType", "snapshot": {}}
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Unknown resource type" in str(exc_info.value)

    def test_from_json_rejects_non_object_resource_snapshot(self) -> None:
        """Resource snapshot must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.filesystem:Filesystem",
                "snapshot": "not-an-object",
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource snapshot must be an object" in str(exc_info.value)

    def test_from_json_rejects_missing_type_reference(self) -> None:
        """Resource snapshot must include type reference."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.filesystem:Filesystem",
                "snapshot": {},  # Missing __type__
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "must include type reference" in str(exc_info.value)

    def test_from_json_rejects_unknown_snapshot_type(self) -> None:
        """Snapshot type must be resolvable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.filesystem:Filesystem",
                "snapshot": {"__type__": "unknown.module:UnknownSnapshot"},
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Unknown snapshot type" in str(exc_info.value)

    def test_from_json_rejects_invalid_resource_snapshot_parse(self) -> None:
        """Resource snapshot must be parseable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.filesystem:Filesystem",
                "snapshot": {
                    "__type__": ("weakincentives.filesystem:FilesystemSnapshot"),
                    "invalid_field": "missing required fields",
                },
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Failed to parse resource snapshot" in str(exc_info.value)

    def test_from_json_rejects_non_object_metadata(self) -> None:
        """metadata must be an object if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = "not-an-object"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata must be an object" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tag(self) -> None:
        """metadata.tag must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tag": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tag must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tool_call_id(self) -> None:
        """metadata.tool_call_id must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tool_call_id": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tool_call_id must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tool_name(self) -> None:
        """metadata.tool_name must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tool_name": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tool_name must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_phase(self) -> None:
        """metadata.phase must be valid."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"phase": "invalid_phase"}
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata phase must be valid" in str(exc_info.value)


class TestCompositeSnapshotToJsonWithResources:
    """Tests for to_json with resources."""

    def test_to_json_serializes_filesystem_snapshot(self) -> None:
        """to_json serializes filesystem snapshots."""
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "content")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot()
        json_str = snapshot.to_json()

        # Verify we can roundtrip
        restored = CompositeSnapshot.from_json(json_str)
        assert Filesystem in restored.resources


class TestExecutionStateRestoreErrors:
    """Tests for restore error handling."""

    def test_restore_skips_missing_resource(self) -> None:
        """restore skips resources not in registry."""
        import types

        session = Session()
        state = ExecutionState(session=session)  # No filesystem

        # Create snapshot with filesystem resource
        fs = InMemoryFilesystem()
        fs.write("test.txt", "content")
        fs_snapshot = fs.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            resources=types.MappingProxyType({Filesystem: fs_snapshot}),
        )

        # Should not raise - just skips missing resource
        state.restore(snapshot)

    def test_restore_raises_on_resource_restore_failure(self) -> None:
        """restore raises RestoreFailedError on resource restore failure."""
        import types

        from weakincentives.errors import RestoreFailedError
        from weakincentives.runtime.snapshotable import Snapshotable

        class FailingResource:
            """Resource that fails to restore."""

            def snapshot(self, *, tag: str | None = None) -> str:
                return "snapshot"

            def restore(self, snapshot: object) -> None:
                raise RuntimeError("Restore failed")

        assert isinstance(FailingResource(), Snapshotable)

        session = Session()
        resource = FailingResource()
        resources = ResourceRegistry.build({FailingResource: resource})
        state = ExecutionState(session=session, resources=resources)

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            resources=types.MappingProxyType({FailingResource: "snapshot"}),
        )

        with pytest.raises(RestoreFailedError) as exc_info:
            state.restore(snapshot)
        assert "Failed to restore" in str(exc_info.value)
        assert "FailingResource" in str(exc_info.value)

    def test_restore_raises_on_session_rollback_failure(self) -> None:
        """restore raises RestoreFailedError when session.rollback fails."""
        import types
        from unittest.mock import MagicMock

        from weakincentives.errors import RestoreFailedError

        session = MagicMock()
        session.restore.side_effect = SnapshotRestoreError("Session rollback failed")

        state = ExecutionState(session=session)

        # Use a real session for the snapshot
        real_session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=real_session.snapshot(),
            resources=types.MappingProxyType({}),
        )

        with pytest.raises(RestoreFailedError) as exc_info:
            state.restore(snapshot)
        assert "Failed to restore session" in str(exc_info.value)


class TestCompositeSnapshotToJsonErrors:
    """Tests for to_json error handling."""

    def test_to_json_raises_on_resource_serialization_error(self) -> None:
        """to_json raises SnapshotSerializationError when resource fails to serialize."""
        import types

        from weakincentives.runtime.session.snapshots import SnapshotSerializationError

        # Create a mock snapshot that can't be serialized
        class UnserializableSnapshot:
            pass

        session = Session()
        session_snapshot = session.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
            resources=types.MappingProxyType({Filesystem: UnserializableSnapshot()}),
        )

        with pytest.raises(SnapshotSerializationError) as exc_info:
            snapshot.to_json()
        assert "Failed to serialize resource snapshot" in str(exc_info.value)

    def test_to_json_raises_on_general_error(self) -> None:
        """to_json raises SnapshotSerializationError on general errors."""
        from unittest.mock import patch

        from weakincentives.runtime.session.snapshots import SnapshotSerializationError

        session = Session()
        session_snapshot = session.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
        )

        # Mock json.dumps to raise an exception after session serialization
        original_dumps = __import__("json").dumps

        def failing_dumps(obj: object, **kwargs: object) -> str:
            if isinstance(obj, dict) and "version" in obj:
                raise RuntimeError("Unexpected error")
            return original_dumps(obj, **kwargs)

        with patch("json.dumps", side_effect=failing_dumps):
            with pytest.raises(SnapshotSerializationError) as exc_info:
                snapshot.to_json()
            assert "Failed to serialize composite snapshot" in str(exc_info.value)


class TestToolTransactionMethods:
    """Tests for ExecutionState tool transaction methods."""

    def test_begin_tool_execution_creates_snapshot(self) -> None:
        """begin_tool_execution takes a snapshot and stores pending execution."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Check pending execution exists
        assert "abc123" in state.pending_tool_executions
        pending = state.pending_tool_executions["abc123"]
        assert pending.tool_use_id == "abc123"
        assert pending.tool_name == "Edit"
        assert pending.snapshot is not None

    def test_end_tool_execution_success_no_restore(self) -> None:
        """end_tool_execution with success=True doesn't restore state."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # End with success
        restored = state.end_tool_execution(tool_use_id="abc123", success=True)

        assert restored is False
        assert "abc123" not in state.pending_tool_executions
        # State should NOT be restored
        assert fs.read("/test.txt").content == "modified"

    def test_end_tool_execution_failure_restores(self) -> None:
        """end_tool_execution with success=False restores state."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # End with failure
        restored = state.end_tool_execution(tool_use_id="abc123", success=False)

        assert restored is True
        assert "abc123" not in state.pending_tool_executions
        # State should be restored
        assert fs.read("/test.txt").content == "initial"

    def test_end_tool_execution_unknown_id_returns_false(self) -> None:
        """end_tool_execution with unknown ID returns False."""
        session = Session()
        state = ExecutionState(session=session)

        restored = state.end_tool_execution(tool_use_id="unknown", success=False)

        assert restored is False

    def test_abort_tool_execution_restores(self) -> None:
        """abort_tool_execution always restores state."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # Abort
        restored = state.abort_tool_execution(tool_use_id="abc123")

        assert restored is True
        assert "abc123" not in state.pending_tool_executions
        # State should be restored
        assert fs.read("/test.txt").content == "initial"

    def test_abort_tool_execution_unknown_id_returns_false(self) -> None:
        """abort_tool_execution with unknown ID returns False."""
        session = Session()
        state = ExecutionState(session=session)

        restored = state.abort_tool_execution(tool_use_id="unknown")

        assert restored is False

    def test_pending_tool_executions_is_read_only(self) -> None:
        """pending_tool_executions returns read-only mapping."""
        import types as types_module

        session = Session()
        state = ExecutionState(session=session)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        pending = state.pending_tool_executions
        assert isinstance(pending, types_module.MappingProxyType)
        # Should not be able to modify
        with pytest.raises(TypeError):
            pending["new"] = None  # type: ignore[index]


class TestToolTransactionContextManager:
    """Tests for ExecutionState.tool_transaction context manager."""

    def test_tool_transaction_yields_snapshot(self) -> None:
        """tool_transaction yields a CompositeSnapshot."""
        session = Session()
        state = ExecutionState(session=session)

        with state.tool_transaction(tag="test-tag") as snapshot:
            assert isinstance(snapshot, CompositeSnapshot)
            assert snapshot.metadata is not None
            assert snapshot.metadata.tag == "test-tag"

    def test_tool_transaction_restores_on_exception(self) -> None:
        """tool_transaction restores state when an exception is raised."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        with pytest.raises(RuntimeError, match="Tool failed"):
            with state.tool_transaction(tag="pre:tool"):
                fs.write("/test.txt", "modified")
                raise RuntimeError("Tool failed")

        # State should be restored after exception
        assert fs.read("/test.txt").content == "initial"

    def test_tool_transaction_preserves_state_on_success(self) -> None:
        """tool_transaction preserves state changes on successful exit."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        with state.tool_transaction(tag="pre:tool"):
            fs.write("/test.txt", "modified")

        # State should be preserved after successful exit
        assert fs.read("/test.txt").content == "modified"

    def test_tool_transaction_manual_restore_on_failure_result(self) -> None:
        """tool_transaction allows manual restore for result.success=False."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        with state.tool_transaction(tag="pre:tool") as snapshot:
            fs.write("/test.txt", "modified")
            # Simulate tool failure result (not exception)
            success = False
            if not success:
                state.restore(snapshot)

        # State should be restored due to manual restore
        assert fs.read("/test.txt").content == "initial"

    def test_tool_transaction_reraises_exception(self) -> None:
        """tool_transaction re-raises the original exception."""
        session = Session()
        state = ExecutionState(session=session)

        with pytest.raises(ValueError, match="specific error"):
            with state.tool_transaction():
                raise ValueError("specific error")


class TestCompositeSnapshotNullMetadata:
    """Tests for CompositeSnapshot with None metadata."""

    def test_from_json_accepts_null_metadata(self) -> None:
        """from_json accepts null metadata field."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            metadata=None,  # Explicitly None
        )
        json_str = snapshot.to_json()

        # Verify null metadata in JSON
        data = json.loads(json_str)
        assert data["metadata"] is None

        # Roundtrip should work
        restored = CompositeSnapshot.from_json(json_str)
        assert restored.metadata is None


class TestExecutionStateNonSnapshotableResource:
    """Tests for ExecutionState with non-snapshotable resources."""

    def test_restore_skips_non_snapshotable_resource(self) -> None:
        """restore skips resources that are not Snapshotable."""
        import types as types_module

        class NonSnapshotableResource:
            """Resource that does not implement Snapshotable."""

            value: str = "initial"

        session = Session()
        resource = NonSnapshotableResource()
        resources = ResourceRegistry.build({NonSnapshotableResource: resource})
        state = ExecutionState(session=session, resources=resources)

        # Modify resource
        resource.value = "modified"

        # Create a fake snapshot with the resource
        snapshot2 = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            resources=types_module.MappingProxyType(
                {NonSnapshotableResource: "fake-snapshot"}
            ),
        )

        # Restore should skip non-snapshotable resource (no error)
        state.restore(snapshot2)

        # Resource should still be modified (not restored)
        assert resource.value == "modified"


# --- Concurrency Tests ---


class TestExecutionStateConcurrency:
    """Concurrency-focused tests for ExecutionState thread safety."""

    def test_concurrent_begin_tool_execution_no_lost_snapshots(self) -> None:
        """Multiple threads calling begin_tool_execution don't lose snapshots."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        num_threads = 10
        tool_ids = [f"tool_{i}" for i in range(num_threads)]
        errors: list[Exception] = []
        barrier = threading.Barrier(num_threads)

        def begin_tool(tool_id: str) -> None:
            try:
                barrier.wait()  # Synchronize all threads to start together
                state.begin_tool_execution(tool_use_id=tool_id, tool_name="TestTool")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=begin_tool, args=(tool_id,)) for tool_id in tool_ids
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        # All tool executions should be registered
        pending = state.pending_tool_executions
        assert len(pending) == num_threads
        for tool_id in tool_ids:
            assert tool_id in pending
            assert pending[tool_id].snapshot is not None

    def test_concurrent_end_tool_execution_deterministic_restore(self) -> None:
        """Multiple threads calling end_tool_execution restore deterministically."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        num_threads = 10
        tool_ids = [f"tool_{i}" for i in range(num_threads)]
        results: dict[str, bool] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(num_threads)
        results_lock = threading.Lock()

        # Setup: begin all tool executions first
        for i, tool_id in enumerate(tool_ids):
            fs.write(f"/file_{i}.txt", f"initial_{i}")
            state.begin_tool_execution(tool_use_id=tool_id, tool_name="TestTool")
            # Modify fs after each begin to have something to restore
            fs.write(f"/file_{i}.txt", f"modified_{i}")

        def end_tool(tool_id: str, index: int) -> None:
            try:
                barrier.wait()  # Synchronize all threads to start together
                restored = state.end_tool_execution(
                    tool_use_id=tool_id,
                    success=False,  # Should trigger restore
                )
                with results_lock:
                    results[tool_id] = restored
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=end_tool, args=(tool_id, i))
            for i, tool_id in enumerate(tool_ids)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        # All tool executions should have been ended with restore
        assert len(results) == num_threads
        for tool_id in tool_ids:
            assert results[tool_id] is True

        # No pending executions should remain
        assert len(state.pending_tool_executions) == 0

    def test_concurrent_abort_tool_execution(self) -> None:
        """Multiple threads calling abort_tool_execution don't corrupt state."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        num_threads = 10
        tool_ids = [f"tool_{i}" for i in range(num_threads)]
        results: dict[str, bool] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(num_threads)
        results_lock = threading.Lock()

        # Setup: begin all tool executions first
        for tool_id in tool_ids:
            state.begin_tool_execution(tool_use_id=tool_id, tool_name="TestTool")

        def abort_tool(tool_id: str) -> None:
            try:
                barrier.wait()  # Synchronize all threads to start together
                restored = state.abort_tool_execution(tool_use_id=tool_id)
                with results_lock:
                    results[tool_id] = restored
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=abort_tool, args=(tool_id,)) for tool_id in tool_ids
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        # All aborts should have succeeded
        assert len(results) == num_threads
        for tool_id in tool_ids:
            assert results[tool_id] is True

        # No pending executions should remain
        assert len(state.pending_tool_executions) == 0

    def test_mixed_concurrent_hook_calls(self) -> None:
        """Mixed begin/end/abort calls from multiple threads don't corrupt state."""
        import threading
        import time

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        errors: list[Exception] = []
        operations_completed = threading.Event()
        counter = {"value": 0}
        counter_lock = threading.Lock()

        num_operations = 100

        def run_operation(op_id: int) -> None:
            try:
                tool_id = f"tool_{op_id}"
                state.begin_tool_execution(tool_use_id=tool_id, tool_name="TestTool")

                # Small delay to increase chance of overlap
                time.sleep(0.001)

                # Alternate between end with success, end with failure, and abort
                if op_id % 3 == 0:
                    state.end_tool_execution(tool_use_id=tool_id, success=True)
                elif op_id % 3 == 1:
                    state.end_tool_execution(tool_use_id=tool_id, success=False)
                else:
                    state.abort_tool_execution(tool_use_id=tool_id)

                with counter_lock:
                    counter["value"] += 1
                    if counter["value"] == num_operations:
                        operations_completed.set()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_operation, args=(i,))
            for i in range(num_operations)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        # All operations should complete
        assert counter["value"] == num_operations

        # No pending executions should remain
        assert len(state.pending_tool_executions) == 0

    def test_concurrent_pending_tool_executions_read(self) -> None:
        """Reading pending_tool_executions while modifying is thread-safe."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        errors: list[Exception] = []
        stop_event = threading.Event()
        reads_completed = {"count": 0}
        reads_lock = threading.Lock()

        # Reader thread that continuously reads pending_tool_executions
        def reader() -> None:
            try:
                while not stop_event.is_set():
                    pending = state.pending_tool_executions
                    # Just accessing the data should not raise
                    _ = len(pending)
                    for v in pending.values():
                        _ = v.tool_use_id
                        _ = v.snapshot
                    with reads_lock:
                        reads_completed["count"] += 1
            except Exception as e:
                errors.append(e)

        # Writer thread that adds and removes tool executions
        def writer() -> None:
            try:
                for i in range(50):
                    tool_id = f"tool_{i}"
                    state.begin_tool_execution(
                        tool_use_id=tool_id, tool_name="TestTool"
                    )
                    state.end_tool_execution(tool_use_id=tool_id, success=True)
            except Exception as e:
                errors.append(e)
            finally:
                stop_event.set()

        reader_threads = [threading.Thread(target=reader) for _ in range(5)]
        writer_thread = threading.Thread(target=writer)

        for r in reader_threads:
            r.start()
        writer_thread.start()

        writer_thread.join()
        for r in reader_threads:
            r.join()

        assert not errors, f"Errors occurred: {errors}"
        # Readers should have completed at least some reads
        assert reads_completed["count"] > 0

    def test_concurrent_double_end_returns_false(self) -> None:
        """Concurrent calls to end_tool_execution for same ID return False once."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        tool_id = "unique_tool"
        state.begin_tool_execution(tool_use_id=tool_id, tool_name="TestTool")

        results: list[bool] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(10)
        results_lock = threading.Lock()

        def end_tool() -> None:
            try:
                barrier.wait()
                restored = state.end_tool_execution(tool_use_id=tool_id, success=False)
                with results_lock:
                    results.append(restored)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=end_tool) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        # Only one thread should have successfully restored
        assert results.count(True) == 1
        # Others should return False
        assert results.count(False) == 9

    def test_snapshot_preserved_across_concurrent_operations(self) -> None:
        """Snapshots captured in begin_tool_execution are preserved correctly."""
        import threading

        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem
        from weakincentives.resources import ResourceRegistry

        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="initial")])
        fs = InMemoryFilesystem()
        fs.write("/data.txt", "original_content")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def tool_that_fails() -> None:
            """Tool that modifies state then fails."""
            try:
                barrier.wait()
                state.begin_tool_execution(
                    tool_use_id="fail_tool", tool_name="FailTool"
                )
                # Modify state
                fs.write("/data.txt", "modified_by_fail_tool")
                session[SamplePlan].seed(
                    [SamplePlan(objective="modified_by_fail_tool")]
                )
                # Fail and restore
                state.end_tool_execution(tool_use_id="fail_tool", success=False)
            except Exception as e:
                errors.append(e)

        def tool_that_succeeds() -> None:
            """Tool that modifies state and succeeds."""
            try:
                barrier.wait()
                state.begin_tool_execution(
                    tool_use_id="success_tool", tool_name="SuccessTool"
                )
                # Modify state
                fs.write("/other.txt", "success_content")
                session[SamplePlan].seed([SamplePlan(objective="modified_by_success")])
                # Succeed
                state.end_tool_execution(tool_use_id="success_tool", success=True)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=tool_that_fails)
        t2 = threading.Thread(target=tool_that_succeeds)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors occurred: {errors}"
        # No pending executions should remain
        assert len(state.pending_tool_executions) == 0
